from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .walk_forward import timestamp_walk_forward_splits


RANKING_VARIANTS = ("legacy", "listmle", "lambda")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hash() -> str:
    training = Path(__file__).resolve().parent
    package = training.parent
    files = (
        training / "ablation.py",
        training / "ablation_worker.py",
        training / "ranking.py",
        training / "train.py",
        training / "walk_forward.py",
        training / "gradient_methods.py",
        training / "phase1e.py",
        training / "phase2b.py",
        training / "robust_training.py",
        package / "cli.py",
        package / "features" / "scaling.py",
        package / "models" / "temporal_transformer.py",
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(package).as_posix().encode("ascii"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _frozen_training_config(config: dict) -> dict:
    frozen = copy.deepcopy(config)
    frozen.get("training", {}).pop("output", None)
    frozen.get("ranking", {}).pop("loss_variant", None)
    frozen.pop("phase1b_ablation", None)
    return frozen


def _mean_metrics(folds: list[dict]) -> dict:
    keys = (
        "ndcg_at_cutoff", "rank_ic", "precision_at_cutoff", "top_k_overlap",
        "top_k_turnover", "top_bottom_utility_spread",
    )
    return {key: float(np.mean([fold[key] for fold in folds])) for key in keys}


def _mean_head_metrics(folds: list[dict]) -> dict:
    return {
        key: float(np.mean([fold[key] for fold in folds]))
        for key in folds[0]
    }


def _checkpoint_head_metrics(checkpoint_path: Path, dataset_path: Path) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("checkpoint 主头评估需要安装 PyTorch") from exc
    from ..models.temporal_transformer import (
        TemporalTransformerConfig,
        TemporalTransformerV1,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TemporalTransformerV1(
        TemporalTransformerConfig(**checkpoint["model_config"])
    )
    from ..models.temporal_transformer import load_temporal_transformer_state_dict
    load_temporal_transformer_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    with np.load(dataset_path, allow_pickle=False) as data:
        timestamps = data["timestamps"]
        selected = np.flatnonzero(np.isin(
            timestamps, np.asarray(checkpoint["split"]["test_timestamps"])
        ))
        features = data["features"][selected].astype(np.float32, copy=False)
        valid_mask = data["valid_mask"][selected].astype(np.uint8, copy=False)
        expected = data["expected_return"][selected]
        direction = data["direction"][selected]
        volatility = data["realized_volatility"][selected]
    outputs = {name: [] for name in (
        "expected_return", "direction_probability", "expected_volatility",
        "lower_quantile", "upper_quantile",
    )}
    with torch.no_grad():
        for start in range(0, selected.size, 256):
            prediction = model(
                torch.from_numpy(features[start:start + 256]),
                torch.from_numpy(valid_mask[start:start + 256]),
            )
            for name in outputs:
                outputs[name].append(prediction[name].numpy())
    values = {name: np.concatenate(parts) for name, parts in outputs.items()}
    return {
        "return_mae": float(np.mean(np.abs(values["expected_return"] - expected))),
        "direction_brier": float(np.mean(np.square(
            values["direction_probability"] - direction
        ))),
        "volatility_mae": float(np.mean(np.abs(
            values["expected_volatility"] - volatility
        ))),
        "interval_coverage": float(np.mean(
            (expected >= values["lower_quantile"])
            & (expected <= values["upper_quantile"])
        )),
        "interval_width": float(np.mean(
            values["upper_quantile"] - values["lower_quantile"]
        )),
    }


def _promotion_decision(aggregates: dict, config: dict) -> dict:
    gate = config.get("promotion_gate", {})
    baseline = aggregates["legacy"]
    eligible = ["legacy"]
    reasons: dict[str, list[str]] = {}
    for variant in RANKING_VARIANTS[1:]:
        candidate = aggregates[variant]
        failures = []
        if candidate["ndcg_at_cutoff"] - baseline["ndcg_at_cutoff"] < float(
            gate.get("minimum_ndcg_improvement", 0.0)
        ):
            failures.append("ndcg_gate")
        if candidate["top_bottom_utility_spread"] < baseline[
            "top_bottom_utility_spread"
        ] * float(gate.get("minimum_utility_spread_ratio", 1.0)):
            failures.append("utility_spread_gate")
        if candidate["top_k_turnover"] - baseline["top_k_turnover"] > float(
            gate.get("maximum_turnover_increase", 0.0)
        ):
            failures.append("turnover_gate")
        reasons[variant] = failures
        if not failures:
            eligible.append(variant)
    winner = max(
        eligible,
        key=lambda variant: (aggregates[variant]["ndcg_at_cutoff"], -RANKING_VARIANTS.index(variant)),
    )
    economic = config.get("cpp_economic_gate")
    economic_passed = isinstance(economic, dict) and economic.get("passed") is True
    incumbent_retained = winner == "legacy" and eligible == ["legacy"]
    frozen = incumbent_retained or economic_passed
    return {
        "model_metric_winner": winner,
        "eligible_variants": eligible,
        "failed_gates": reasons,
        "cpp_economic_gate_present": isinstance(economic, dict),
        "cpp_economic_gate_passed": economic_passed,
        "winner_frozen": frozen,
        "frozen_winner": winner if frozen else None,
        "status": (
            "incumbent_retained" if incumbent_retained
            else "challenger_frozen" if economic_passed
            else "awaiting_cpp_economic_gate"
        ),
    }


def _run_training_job(job):
    train_fn, run_config, dataset_path, run_path, fold = job
    train_fn(run_config, dataset_path, run_path, _split_override=fold)
    return run_path


def _run_subprocess_jobs(jobs, job_root: Path, workers: int, code_hash: str) -> None:
    job_root.mkdir()
    commands = []
    for index, (_, run_config, dataset_path, run_path, fold) in enumerate(jobs):
        job_path = job_root / f"job-{index + 1}.json"
        job_path.write_text(json.dumps({
            "config": run_config,
            "dataset": dataset_path,
            "output": run_path,
            "implementation_sha256": code_hash,
            "split": {
                "train_first": int(fold.train_timestamps[0]),
                "train_last": int(fold.train_timestamps[-1]),
                "validation_first": int(fold.validation_timestamps[0]),
                "validation_last": int(fold.validation_timestamps[-1]),
                "test_first": int(fold.test_timestamps[0]),
                "test_last": int(fold.test_timestamps[-1]),
            },
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        commands.append([sys.executable, "-m", "python.qbt_ml.training.ablation_worker",
                         str(job_path)])
    for start in range(0, len(commands), workers):
        processes = [subprocess.Popen(command) for command in commands[start:start + workers]]
        return_codes = [process.wait() for process in processes]
        if any(code != 0 for code in return_codes):
            raise RuntimeError(f"Phase 1B worker 失败，退出码: {return_codes}")


def run_phase1b_ablation(config: dict, dataset_path: str | Path, output: str | Path, train_fn):
    settings = config.get("phase1b_ablation", {})
    if settings.get("enabled") is not True:
        raise RuntimeError("Phase 1B ablation 默认关闭；需显式设置 enabled=true")
    variants = tuple(settings.get("variants", RANKING_VARIANTS))
    if variants != RANKING_VARIANTS:
        raise ValueError("Phase 1B 首轮必须且只能按 legacy/listmle/lambda 顺序配对")
    dataset_path = Path(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as data:
        timestamps = data["timestamps"]
        label_hash = str(data["label_spec_sha256"])
        ranking_hash = str(data["ranking_score_spec_sha256"])
    folds = list(timestamp_walk_forward_splits(
        timestamps,
        train_size=int(settings.get("train_timestamps", 504)),
        validation_size=int(settings.get("validation_timestamps", 126)),
        test_size=int(settings.get("test_timestamps", 126)),
        step=int(settings.get("step_timestamps", 126)),
        purge_timestamps=int(settings.get("purge_timestamps", 6)),
        embargo_timestamps=int(settings.get("embargo_timestamps", 5)),
    ))
    window_count = int(settings.get("window_count", 3))
    if window_count < 3 or len(folds) < window_count:
        raise ValueError("Phase 1B 至少需要三个可用 purged OOS 窗口")
    if len(folds) != window_count:
        raise ValueError(
            f"当前窗口参数产生 {len(folds)} 个 fold；请冻结窗口边界使其恰为 window_count={window_count}"
        )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    contract = {
        "schema_version": 1,
        "variants": list(variants),
        "seed": int(config.get("seed", 20260724)),
        "dataset_sha256": _file_hash(dataset_path),
        "label_spec_sha256": label_hash,
        "ranking_score_spec_sha256": ranking_hash,
        "frozen_training_config_sha256": _canonical_hash(_frozen_training_config(config)),
        "implementation_sha256": implementation_hash(),
        "window_count": window_count,
    }
    fold_reports: list[dict] = []
    by_variant: dict[str, list[dict]] = {variant: [] for variant in variants}
    jobs = []
    for fold_index, fold in enumerate(folds):
        for variant in variants:
            run_config = copy.deepcopy(config)
            run_config.setdefault("ranking", {})["loss_variant"] = variant
            run_path = output / f"fold-{fold_index + 1}" / variant
            jobs.append((train_fn, run_config, str(dataset_path), str(run_path), fold))

    workers = int(settings.get("parallel_workers", 1))
    if workers <= 0:
        raise ValueError("phase1b_ablation.parallel_workers 必须为正数")
    if workers == 1:
        for job in jobs:
            _run_training_job(job)
    else:
        _run_subprocess_jobs(
            jobs, output / "jobs", workers, contract["implementation_sha256"]
        )

    for fold_index, fold in enumerate(folds):
        boundaries = {
            "train_first": int(fold.train_timestamps[0]),
            "train_last": int(fold.train_timestamps[-1]),
            "validation_first": int(fold.validation_timestamps[0]),
            "validation_last": int(fold.validation_timestamps[-1]),
            "test_first": int(fold.test_timestamps[0]),
            "test_last": int(fold.test_timestamps[-1]),
        }
        variants_report = {}
        for variant in variants:
            run_path = output / f"fold-{fold_index + 1}" / variant
            metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
            result = metrics["test_ranking"]
            variants_report[variant] = result
            by_variant[variant].append(result)
        fold_reports.append({"fold": fold_index + 1, "split": boundaries,
                             "variants": variants_report})
    aggregates = {variant: _mean_metrics(results) for variant, results in by_variant.items()}
    report = {
        "contract": contract,
        "folds": fold_reports,
        "aggregate_timestamp_equal_weighted": aggregates,
        "promotion": _promotion_decision(aggregates, settings),
    }
    (output / "paired_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def run_kendall_ablation(
    config: dict,
    dataset_path: str | Path,
    fixed_report_path: str | Path,
    output: str | Path,
    train_fn,
):
    settings = config.get("kendall_ablation", {})
    if settings.get("enabled") is not True:
        raise RuntimeError("Kendall ablation 默认关闭；需显式设置 enabled=true")
    dataset_path = Path(dataset_path)
    fixed_report_path = Path(fixed_report_path)
    fixed_root = fixed_report_path.parent
    fixed_report = json.loads(fixed_report_path.read_text(encoding="utf-8"))
    fixed_decision = _promotion_decision(
        fixed_report["aggregate_timestamp_equal_weighted"],
        config.get("phase1b_ablation", {}),
    )
    frozen_loss = str(settings.get("frozen_ranking_loss", ""))
    if not fixed_decision["winner_frozen"] or fixed_decision["frozen_winner"] != frozen_loss:
        raise RuntimeError("Kendall 只能绑定已冻结的 ranking loss")
    if _file_hash(dataset_path) != fixed_report["contract"]["dataset_sha256"]:
        raise RuntimeError("Kendall dataset hash 与固定权重配对不一致")

    with np.load(dataset_path, allow_pickle=False) as data:
        timestamps = data["timestamps"]
    folds = list(timestamp_walk_forward_splits(
        timestamps,
        train_size=int(config["phase1b_ablation"].get("train_timestamps", 504)),
        validation_size=int(config["phase1b_ablation"].get("validation_timestamps", 126)),
        test_size=int(config["phase1b_ablation"].get("test_timestamps", 126)),
        step=int(config["phase1b_ablation"].get("step_timestamps", 126)),
        purge_timestamps=int(config["phase1b_ablation"].get("purge_timestamps", 6)),
        embargo_timestamps=int(config["phase1b_ablation"].get("embargo_timestamps", 5)),
    ))
    if len(folds) != 3:
        raise RuntimeError("Kendall 配对必须复用三个固定 OOS fold")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for fold_index, fold in enumerate(folds):
        run_config = copy.deepcopy(config)
        run_config.setdefault("ranking", {})["loss_variant"] = frozen_loss
        run_config["multitask_weighting"] = {
            "mode": "kendall",
            "frozen_ranking_loss": frozen_loss,
            "initial_log_variance": float(settings.get("initial_log_variance", 0.0)),
            "regularizer": float(settings.get("regularizer", 0.5)),
            "minimum_log_variance": float(settings.get("minimum_log_variance", -6.0)),
            "maximum_log_variance": float(settings.get("maximum_log_variance", 6.0)),
        }
        jobs.append((
            train_fn, run_config, str(dataset_path),
            str(output / f"fold-{fold_index + 1}" / "kendall"), fold,
        ))
    workers = int(settings.get("parallel_workers", 3))
    _run_subprocess_jobs(jobs, output / "jobs", workers, implementation_hash())

    fixed_ranking = []
    dynamic_ranking = []
    fixed_heads = []
    dynamic_heads = []
    fold_reports = []
    boundary_failures = []
    for fold_index in range(3):
        fixed_metrics = fixed_report["folds"][fold_index]["variants"][frozen_loss]
        dynamic_path = output / f"fold-{fold_index + 1}" / "kendall"
        dynamic_metrics = json.loads(
            (dynamic_path / "metrics.json").read_text(encoding="utf-8")
        )
        dynamic_test = dynamic_metrics["test_ranking"]
        fixed_head = _checkpoint_head_metrics(
            fixed_root / f"fold-{fold_index + 1}" / frozen_loss / "checkpoint.pt",
            dataset_path,
        )
        dynamic_head = dynamic_test["head_metrics"]
        diagnostics = dynamic_metrics["loss_weight_diagnostics"]
        minimum = float(settings.get("minimum_log_variance", -6.0))
        maximum = float(settings.get("maximum_log_variance", 6.0))
        at_boundary = any(
            abs(epoch["log_variances"][task] - boundary) <= 1e-6
            for epoch in diagnostics[-10:]
            for task in epoch["log_variances"]
            for boundary in (minimum, maximum)
        )
        boundary_failures.append(at_boundary)
        fixed_ranking.append(fixed_metrics)
        dynamic_ranking.append(dynamic_test)
        fixed_heads.append(fixed_head)
        dynamic_heads.append(dynamic_head)
        fold_reports.append({
            "fold": fold_index + 1,
            "fixed_ranking": fixed_metrics,
            "kendall_ranking": dynamic_test,
            "fixed_heads": fixed_head,
            "kendall_heads": dynamic_head,
            "final_weight_diagnostics": diagnostics[-1],
            "weight_at_boundary_last_10_epochs": at_boundary,
        })

    fixed_aggregate = _mean_metrics(fixed_ranking)
    dynamic_aggregate = _mean_metrics(dynamic_ranking)
    fixed_head_aggregate = _mean_head_metrics(fixed_heads)
    dynamic_head_aggregate = _mean_head_metrics(dynamic_heads)
    head_error_keys = ("return_mae", "direction_brier", "volatility_mae")
    heads_degraded_all_windows = [
        key for key in head_error_keys
        if all(dynamic[key] > fixed[key] for fixed, dynamic in zip(fixed_heads, dynamic_heads))
    ]
    gate = settings.get("promotion_gate", {})
    failures = []
    if dynamic_aggregate["ndcg_at_cutoff"] - fixed_aggregate[
        "ndcg_at_cutoff"
    ] < float(gate.get("minimum_ndcg_improvement", 0.0)):
        failures.append("ndcg_gate")
    if dynamic_aggregate["top_bottom_utility_spread"] < fixed_aggregate[
        "top_bottom_utility_spread"
    ] * float(gate.get("minimum_utility_spread_ratio", 1.0)):
        failures.append("utility_spread_gate")
    if dynamic_aggregate["top_k_turnover"] - fixed_aggregate[
        "top_k_turnover"
    ] > float(gate.get("maximum_turnover_increase", 0.0)):
        failures.append("turnover_gate")
    if heads_degraded_all_windows:
        failures.append("main_head_gate")
    if any(boundary_failures):
        failures.append("weight_boundary_gate")
    report = {
        "contract": {
            "schema_version": 1,
            "dataset_sha256": fixed_report["contract"]["dataset_sha256"],
            "fixed_report_sha256": _file_hash(fixed_report_path),
            "fixed_implementation_sha256": fixed_report["contract"][
                "implementation_sha256"
            ],
            "kendall_implementation_sha256": implementation_hash(),
            "frozen_ranking_loss": frozen_loss,
            "seed": int(config.get("seed", 20260724)),
            "window_count": 3,
        },
        "folds": fold_reports,
        "aggregate": {
            "fixed_ranking": fixed_aggregate,
            "kendall_ranking": dynamic_aggregate,
            "fixed_heads": fixed_head_aggregate,
            "kendall_heads": dynamic_head_aggregate,
        },
        "decision": {
            "status": "promoted" if not failures else "rejected",
            "failed_gates": failures,
            "heads_degraded_all_windows": heads_degraded_all_windows,
            "kendall_weighting_frozen": not failures,
        },
    }
    (output / "paired_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output

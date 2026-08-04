"""Phase 2B three-window clean/noisy/stress OOS runner."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from ..data.schemas import BAR_V1
from .ablation import _run_subprocess_jobs, implementation_hash
from .phase1e import phase1e_oos_splits
from .robust_training import build_stress_sets
from .walk_forward import TimestampSplit


PHASE2B_VARIANTS = ("none", "direction_apl", "latent_fgm", "feature_pgd")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_payload(fold: TimestampSplit) -> dict:
    return {
        "train_first": int(fold.train_timestamps[0]),
        "train_last": int(fold.train_timestamps[-1]),
        "validation_first": int(fold.validation_timestamps[0]),
        "validation_last": int(fold.validation_timestamps[-1]),
        "test_first": int(fold.test_timestamps[0]),
        "test_last": int(fold.test_timestamps[-1]),
    }


def _checkpoint_model(checkpoint_path: Path):
    import torch
    from ..models.temporal_transformer import (
        TemporalTransformerConfig,
        TemporalTransformerV1,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TemporalTransformerV1(
        TemporalTransformerConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _binary_cross_entropy(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.clip(probability.astype(np.float64), 1e-7, 1.0 - 1e-7)
    target = target.astype(np.float64)
    return float(np.mean(
        -(target * np.log(probability) + (1.0 - target) * np.log1p(-probability))
    ))


def _predict_metrics(
    model,
    features: np.ndarray,
    valid_mask: np.ndarray,
    expected: np.ndarray,
    direction: np.ndarray,
    volatility: np.ndarray,
    timestamps: np.ndarray,
    symbols: np.ndarray,
    utility: np.ndarray,
    relevance: np.ndarray,
    *,
    ranking_cutoff: int = 20,
    batch_size: int = 256,
) -> dict:
    import torch
    from .ranking import ranking_oos_metrics

    output_parts = {name: [] for name in (
        "expected_return", "expected_volatility", "direction_probability",
        "lower_quantile", "upper_quantile",
    )}
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            end = start + batch_size
            prediction = model(
                torch.from_numpy(features[start:end]),
                torch.from_numpy(valid_mask[start:end]),
            )
            for name in output_parts:
                output_parts[name].append(prediction[name].numpy())
    values = {name: np.concatenate(parts) for name, parts in output_parts.items()}
    ranking = ranking_oos_metrics(
        values["expected_return"], utility, relevance, timestamps, symbols,
        cutoff=int(ranking_cutoff),
    )
    return {
        "return_mae": float(np.mean(np.abs(values["expected_return"] - expected))),
        "direction_brier": float(np.mean(
            (values["direction_probability"] - direction) ** 2
        )),
        "direction_bce": _binary_cross_entropy(
            values["direction_probability"], direction,
        ),
        "volatility_mae": float(np.mean(
            np.abs(values["expected_volatility"] - volatility)
        )),
        "interval_coverage": float(np.mean(
            (expected >= values["lower_quantile"])
            & (expected <= values["upper_quantile"])
        )),
        "interval_width": float(np.mean(
            values["upper_quantile"] - values["lower_quantile"]
        )),
        "ndcg_at_cutoff": float(ranking.ndcg_at_cutoff),
        "rank_ic": float(ranking.rank_ic),
        "top_k_turnover": float(ranking.top_k_turnover),
        "composite_error": float(np.mean([
            np.mean(np.abs(values["expected_return"] - expected)),
            np.mean((values["direction_probability"] - direction) ** 2),
            np.mean(np.abs(values["expected_volatility"] - volatility)),
        ])),
    }


def _noisy_direction_metrics(
    probability: np.ndarray,
    direction: np.ndarray,
    *,
    rates: tuple[float, ...],
    seed: int,
) -> dict[str, dict]:
    metrics = {}
    for index, rate in enumerate(rates):
        generator = np.random.default_rng(seed + index)
        flip = generator.random(direction.size) < rate
        noisy = np.where(flip, 1.0 - direction, direction)
        metrics[str(rate)] = {
            "flip_rate": float(rate),
            "realized_flip_rate": float(np.mean(flip)),
            "flipped_count": int(np.sum(flip)),
            "direction_brier": float(np.mean((probability - noisy) ** 2)),
            "direction_bce": _binary_cross_entropy(probability, noisy),
        }
    return metrics


def _evaluate_fold(
    checkpoint_path: Path,
    data: Mapping[str, np.ndarray],
    fold: TimestampSplit,
    *,
    noise_rates: tuple[float, ...],
    noise_seed: int,
    ranking_cutoff: int,
    batch_size: int,
) -> dict:
    model = _checkpoint_model(checkpoint_path)
    indices = fold.test
    features = data["features"][indices].astype(np.float32, copy=False)
    valid_mask = data["valid_mask"][indices].astype(np.uint8, copy=False)
    expected = data["expected_return"][indices].astype(np.float32, copy=False)
    direction = data["direction"][indices].astype(np.float32, copy=False)
    volatility = data["realized_volatility"][indices].astype(np.float32, copy=False)
    timestamps = data["timestamps"][indices]
    symbols = data["symbols"][indices].astype(str)
    utility = data["rank_utility"][indices].astype(np.float32, copy=False)
    relevance = data["rank_relevance"][indices].astype(np.float32, copy=False)
    clean = _predict_metrics(
        model, features, valid_mask, expected, direction, volatility,
        timestamps, symbols, utility, relevance, batch_size=batch_size,
        ranking_cutoff=ranking_cutoff,
    )
    import torch
    with torch.no_grad():
        probability_parts = []
        for start in range(0, features.shape[0], batch_size):
            end = start + batch_size
            probability_parts.append(model(
                torch.from_numpy(features[start:end]),
                torch.from_numpy(valid_mask[start:end]),
            )["direction_probability"].numpy())
    probability = np.concatenate(probability_parts)
    noisy = _noisy_direction_metrics(
        probability, direction, rates=noise_rates, seed=noise_seed,
    )
    stress = {}
    for name, value in build_stress_sets(
        features, valid_mask, feature_names=BAR_V1.feature_names,
        seed=noise_seed,
    ).items():
        stress[name] = _predict_metrics(
            model, value["features"], value["valid_mask"], expected,
            direction, volatility, timestamps, symbols, utility, relevance,
            ranking_cutoff=ranking_cutoff, batch_size=batch_size,
        )
    return {"clean": clean, "noisy": noisy, "stress": stress}


def _mean_metrics(values: list[dict]) -> dict:
    keys = values[0].keys()
    return {key: float(np.mean([value[key] for value in values])) for key in keys}


def _gate_candidate(candidate: dict, baseline: dict, folds: list[dict]) -> dict:
    clean_keys = ("return_mae", "direction_brier", "volatility_mae")
    clean_pass = all(
        candidate["clean"][key] <= baseline["clean"][key] + 1e-12
        for key in clean_keys
    )
    stress_names = tuple(candidate["stress"])
    stress_delta = {
        name: candidate["stress"][name]["composite_error"] -
        baseline["stress"][name]["composite_error"]
        for name in stress_names
    }
    stress_pass = all(value <= 1e-12 for value in stress_delta.values())
    consistent = {}
    for name in stress_names:
        deltas = [
            fold["candidate"]["stress"][name]["composite_error"] -
            fold["baseline"]["stress"][name]["composite_error"]
            for fold in folds
        ]
        consistent[name] = bool(all(value <= 1e-12 for value in deltas))
    return {
        "clean_non_degraded": clean_pass,
        "stress_non_degraded": stress_pass,
        "stress_window_consistency": consistent,
        "all_three_window_consistent": all(consistent.values()),
        "research_gate_passed": clean_pass and stress_pass and all(consistent.values()),
        "promotion_eligible": False,
        "stress_delta_composite_error": stress_delta,
    }


def run_phase2b_oos(
    config: dict,
    dataset_path: str | Path,
    output: str | Path,
    train_fn: Callable,
    *,
    baseline_root: str | Path | None = None,
) -> Path:
    settings = config.get("phase2b_oos", {})
    if not isinstance(settings, dict):
        raise ValueError("phase2b_oos 必须是对象")
    if settings.get("enabled") is not True:
        raise RuntimeError("Phase 2B OOS 默认关闭；需显式 enabled=true")
    dataset_path = Path(dataset_path)
    output = Path(output)
    resume = settings.get("resume") is True
    existing_contract = None
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        contract_path = output / "preregistered_contract.json"
        if not contract_path.exists():
            raise FileNotFoundError(
                f"resume 需要已有 preregistered_contract.json: {contract_path}"
            )
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    else:
        output.mkdir(parents=True)
    with np.load(dataset_path, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in (
            "features", "valid_mask", "timestamps", "symbols",
            "expected_return", "direction", "realized_volatility",
            "rank_utility", "rank_relevance",
        )}
        timestamps = data["timestamps"]
    data_audit_path = settings.get("data_audit")
    if data_audit_path:
        audit = json.loads(Path(data_audit_path).read_text(encoding="utf-8"))
        cutoff = int(audit["phase1b_last_timestamp"])
    else:
        cutoff = int(settings["phase1b_last_timestamp"])
    folds = phase1e_oos_splits(
        timestamps, phase1b_last_timestamp=cutoff,
        train_size=int(settings.get("train_timestamps", 504)),
        validation_size=int(settings.get("validation_timestamps", 126)),
        test_size=int(settings.get("test_timestamps", 126)),
        purge_timestamps=int(settings.get("purge_timestamps", 6)),
        embargo_timestamps=int(settings.get("embargo_timestamps", 5)),
        window_count=3,
    )
    noise_rates = tuple(float(value) for value in settings.get(
        "noise_rates", (0.05, 0.10, 0.20)
    ))
    ranking_cutoff = int(settings.get("ranking_cutoff", 20))
    if ranking_cutoff <= 0:
        raise ValueError("Phase 2B ranking_cutoff 必须为正数")
    variants = tuple(settings.get("variants", PHASE2B_VARIANTS))
    if set(variants) - set(PHASE2B_VARIANTS) or "none" not in variants:
        raise ValueError("Phase 2B variants 必须包含 none/apl/FGM/PGD 合法候选")
    contract = {
        "schema_version": 1,
        "registered_before_training": True,
        "promotion_allowed": False,
        "dataset_sha256": _file_hash(dataset_path),
        "data_audit_sha256": _file_hash(data_audit_path) if data_audit_path else None,
        "phase1b_last_timestamp": cutoff,
        "window_count": 3,
        "noise_rates": list(noise_rates),
        "label_provenance": {
            "clean": {
                "direction": "dataset.direction soft label; source=phase1e_label_v2",
                "noisy": "not an input column",
            },
            "noisy": {
                "direction": "dataset.direction soft label",
                "noisy": "deterministic derived label: 1-direction on seeded flip mask",
                "seed": int(settings.get("noise_seed", 20260804)),
                "rates": list(noise_rates),
            },
            "stress": {
                "features": "deterministic derived perturbations from dataset.features",
                "modes": ["price", "volume", "missing", "extreme_volatility"],
            },
        },
        "ranking_cutoff": ranking_cutoff,
        "variants": list(variants),
        "folds": [{"fold": index + 1, **_split_payload(fold)}
                  for index, fold in enumerate(folds)],
    }
    contract["contract_sha256"] = _canonical_hash(contract)
    if existing_contract is not None:
        if existing_contract.get("contract_sha256") != contract["contract_sha256"]:
            raise ValueError("resume 的预注册合同与当前配置不一致")
    else:
        (output / "preregistered_contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    training_jobs = []
    checkpoint_paths = {}
    for variant in variants:
        for fold_index, fold in enumerate(folds):
            run_path = output / "runs" / f"fold-{fold_index + 1}" / variant
            checkpoint_path = run_path / "checkpoint.pt"
            if variant == "none" and baseline_root is not None:
                reused = Path(baseline_root) / f"fold-{fold_index + 1}" / "none"
                if (reused / "checkpoint.pt").exists():
                    checkpoint_path = reused / "checkpoint.pt"
            if not checkpoint_path.exists():
                run_config = copy.deepcopy(config)
                run_config["training"] = dict(config.get("training", {}))
                run_config["training"]["output"] = str(run_path)
                run_config["robust_training"] = {
                    "mode": variant,
                    "hypothesis_id": f"phase2b-oos-{variant}-v1",
                    "epsilon": float(settings.get("epsilon", 0.01)),
                    "beta": float(settings.get("beta", 0.5)),
                    "pgd_steps": int(settings.get("pgd_steps", 3)),
                }
                training_jobs.append((
                    train_fn, run_config, str(dataset_path), str(run_path), fold,
                ))
                checkpoint_path = run_path / "checkpoint.pt"
            checkpoint_paths[(variant, fold_index)] = checkpoint_path

    parallel_workers = int(settings.get("parallel_workers", 1))
    if parallel_workers <= 0:
        raise ValueError("phase2b_oos.parallel_workers 必须为正数")
    if training_jobs:
        if parallel_workers == 1:
            for job in training_jobs:
                train_fn(
                    job[1], job[2], job[3], _split_override=job[4],
                )
        else:
            job_root = output / "jobs"
            if resume:
                suffix = 1
                while job_root.exists():
                    suffix += 1
                    job_root = output / f"jobs-resume-{suffix}"
            _run_subprocess_jobs(
                training_jobs, job_root, parallel_workers,
                implementation_hash(),
            )

    results = {}
    for variant in variants:
        fold_results = []
        for fold_index, fold in enumerate(folds):
            checkpoint_path = checkpoint_paths[(variant, fold_index)]
            if not checkpoint_path.exists():
                raise RuntimeError(f"Phase 2B checkpoint 缺失: {checkpoint_path}")
            evaluated = _evaluate_fold(
                checkpoint_path, data, fold,
                noise_rates=noise_rates,
                noise_seed=int(settings.get("noise_seed", 20260804)) + fold_index,
                ranking_cutoff=ranking_cutoff,
                batch_size=int(config.get("training", {}).get("batch_size", 256)),
            )
            fold_results.append({
                "fold": fold_index + 1,
                "checkpoint": str(checkpoint_path),
                "split": _split_payload(fold),
                **evaluated,
            })
        results[variant] = {
            "folds": fold_results,
            "clean": _mean_metrics([fold["clean"] for fold in fold_results]),
            "stress": {
                name: _mean_metrics([fold["stress"][name] for fold in fold_results])
                for name in fold_results[0]["stress"]
            },
        }
        results[variant]["noisy"] = {
            rate: _mean_metrics([fold["noisy"][rate] for fold in fold_results])
            for rate in fold_results[0]["noisy"]
        }
    baseline = results["none"]
    gates = {}
    for variant in variants:
        if variant == "none":
            continue
        fold_pairs = [
            {"candidate": results[variant]["folds"][index],
             "baseline": baseline["folds"][index]}
            for index in range(3)
        ]
        gates[variant] = _gate_candidate(results[variant], baseline, fold_pairs)
    report = {
        "schema_version": 1,
        "status": "research_only_no_promotion",
        "contract_sha256": contract["contract_sha256"],
        "dataset_sha256": contract["dataset_sha256"],
        "window_count": 3,
        "baseline": baseline,
        "variants": results,
        "gates": gates,
    }
    report["report_sha256"] = _canonical_hash(report)
    (output / "phase2b_oos_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = ["PHASE2B_VARIANTS", "run_phase2b_oos"]

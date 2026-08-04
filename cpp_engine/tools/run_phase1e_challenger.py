#!/usr/bin/env python3
"""Run one Phase 1E challenger on three frozen purged OOS windows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _metric_summary(root: Path, label: str, folds: int) -> dict[str, Any]:
    ranking: dict[str, list[float]] = {}
    heads: dict[str, list[float]] = {}
    for index in range(1, folds + 1):
        metrics = json.loads(
            (root / f"fold-{index}" / label / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for key, value in metrics["test_ranking"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                ranking.setdefault(key, []).append(float(value))
        for key, value in metrics["test_ranking"].get("head_metrics", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                heads.setdefault(key, []).append(float(value))
        for key, value in metrics.get("test_head_metrics", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                heads.setdefault(key, []).append(float(value))
    return {
        "ranking_timestamp_equal_weighted": {
            key: _mean(values) for key, values in sorted(ranking.items())
        },
        "head_timestamp_equal_weighted": {
            key: _mean(values) for key, values in sorted(heads.items())
        },
    }


def _fold_metrics(root: Path, label: str, folds: int) -> list[dict[str, Any]]:
    results = []
    for index in range(1, folds + 1):
        path = root / f"fold-{index}" / label / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        mechanism = metrics.get("gradient_mechanism_diagnostics", [])
        results.append({
            "fold": index,
            "test_ranking": metrics.get("test_ranking", {}),
            "test_loss": metrics.get("test_loss"),
            "gradient_optimization_mode": metrics.get("gradient_optimization_mode"),
            "gradient_optimization_hypothesis_id": metrics.get(
                "gradient_optimization_hypothesis_id"
            ),
            "gradient_optimization_spec_sha256": metrics.get(
                "gradient_optimization_spec_sha256"
            ),
            "gradient_conflict_report_sha256": metrics.get(
                "gradient_conflict_report_sha256"
            ),
            "mechanism_diagnostics_count": len(mechanism),
            "mechanism_diagnostics": mechanism,
            "checkpoint_present": (path.parent / "checkpoint.pt").exists(),
        })
    return results


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("pcgrad", "gradnorm"), required=True)
    parser.add_argument("--hypothesis-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--audit")
    parser.add_argument("--diagnostics-report")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--update-every-steps", type=int, default=1)
    parser.add_argument("--weight-learning-rate", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers 必须为正数")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs 必须为正数")
    config_path = Path(args.config).resolve()
    project_root = config_path.parents[2]
    dataset_path = _resolve(project_root, args.dataset or "data/research/phase1e_pit_120_dataset.npz")
    audit_path = _resolve(project_root, args.audit or "data/research/phase1e_pit_120_audit.json")
    output = Path(args.output).resolve()
    diagnostics_path = (
        Path(args.diagnostics_report).resolve()
        if args.diagnostics_report
        else project_root / "runs/phase1e-diagnostics-real/diagnostic_report.json"
    )
    if not dataset_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("Phase 1E dataset/audit 不存在")
    if output.exists():
        raise FileExistsError(f"输出目录不可覆盖: {output}")

    python_project = str(project_root)
    if python_project not in sys.path:
        sys.path.insert(0, python_project)
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        python_project
        if not inherited_pythonpath
        else python_project + os.pathsep + inherited_pythonpath
    )
    from python.qbt_ml.cli import _train
    from python.qbt_ml.training.ablation import implementation_hash, _run_subprocess_jobs
    from python.qbt_ml.training.phase1e import phase1e_oos_splits

    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    settings = config.get("phase1e_diagnostics", {})
    if settings.get("diagnostic_only") is not True:
        raise ValueError("必须复用 diagnostic-only Phase 1E 配置")
    with np.load(dataset_path, allow_pickle=False) as data:
        timestamps = data["timestamps"]
        folds = phase1e_oos_splits(
            timestamps,
            phase1b_last_timestamp=int(audit["phase1b_last_timestamp"]),
            train_size=int(settings.get("train_timestamps", 504)),
            validation_size=int(settings.get("validation_timestamps", 126)),
            test_size=int(settings.get("test_timestamps", 126)),
            purge_timestamps=int(settings.get("purge_timestamps", 6)),
            embargo_timestamps=int(settings.get("embargo_timestamps", 5)),
            window_count=3,
        )

    output.mkdir(parents=True)
    code_hash = implementation_hash()
    preregistered = {
        "schema_version": 1,
        "registered_before_training": True,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "method": args.method,
        "hypothesis_id": args.hypothesis_id,
        "dataset_sha256": _file_hash(dataset_path),
        "data_audit_sha256": _file_hash(audit_path),
        "implementation_sha256": code_hash,
        "phase1b_last_timestamp": int(audit["phase1b_last_timestamp"]),
        "window_count": 3,
        "epochs": int(args.epochs or config.get("training", {}).get("epochs", 50)),
        "baseline_run": "none",
        "frozen_champion": "legacy + fixed",
        "scalar_weights": {
            "return": 1.0,
            "direction": 0.25,
            "volatility": 0.25,
            "quantile": 0.25,
            "legacy_rank": 0.1,
        },
        "folds": [
            {
                "fold": index + 1,
                "train_first": int(fold.train_timestamps[0]),
                "train_last": int(fold.train_timestamps[-1]),
                "validation_first": int(fold.validation_timestamps[0]),
                "validation_last": int(fold.validation_timestamps[-1]),
                "test_first": int(fold.test_timestamps[0]),
                "test_last": int(fold.test_timestamps[-1]),
            }
            for index, fold in enumerate(folds)
        ],
    }
    (output / "preregistered_contract.json").write_text(
        json.dumps(
            {**preregistered, "contract_sha256": _canonical_hash(preregistered)},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    jobs = []
    for fold_index, fold in enumerate(folds):
        baseline_config = copy.deepcopy(config)
        if args.epochs is not None:
            baseline_config.setdefault("training", {})["epochs"] = args.epochs
        baseline_config["gradient_optimization"] = {"mode": "none"}
        jobs.append((
            _train,
            baseline_config,
            str(dataset_path),
            str(output / f"fold-{fold_index + 1}" / "none"),
            fold,
        ))
        candidate_config = copy.deepcopy(config)
        if args.epochs is not None:
            candidate_config.setdefault("training", {})["epochs"] = args.epochs
        gradient_config = {
            "mode": args.method,
            "hypothesis_id": args.hypothesis_id,
            "seed": int(settings.get("sampling_seed", 20260801)),
            "fold": fold_index + 1,
            "regime": "ALL",
        }
        if args.method == "pcgrad":
            gradient_config["zero_norm_epsilon"] = 1e-12
        else:
            gradient_config.update({
                "alpha": args.alpha,
                "update_every_steps": args.update_every_steps,
                "weight_learning_rate": args.weight_learning_rate,
            })
        candidate_config["gradient_optimization"] = gradient_config
        jobs.append((
            _train,
            candidate_config,
            str(dataset_path),
            str(output / f"fold-{fold_index + 1}" / args.method),
            fold,
        ))
    _run_subprocess_jobs(jobs, output / "jobs", args.workers, code_hash)

    baseline_folds = _fold_metrics(output, "none", 3)
    candidate_folds = _fold_metrics(output, args.method, 3)
    baseline_summary = _metric_summary(output, "none", 3)
    candidate_summary = _metric_summary(output, args.method, 3)
    ranking_delta = {
        key: candidate_summary["ranking_timestamp_equal_weighted"].get(key, 0.0)
        - baseline_summary["ranking_timestamp_equal_weighted"].get(key, 0.0)
        for key in sorted(
            set(baseline_summary["ranking_timestamp_equal_weighted"])
            | set(candidate_summary["ranking_timestamp_equal_weighted"])
        )
    }
    economic_inputs = audit.get("remaining_economic_inputs", [])
    economic = {
        "status": "awaiting_cpp_economic_gate",
        "promotion_eligible": False,
        "cpp_replay_executed": False,
        "cvar_available": False,
        "missing_inputs": economic_inputs,
        "reason": "Phase 1E challenger 不能以缺失的 PIT 执行字段伪造 C++ Replay/CVaR",
    }
    references = {
        "phase1e_diagnostics_report": {
            "path": str(diagnostics_path),
            "present": diagnostics_path.is_file(),
            "sha256": _file_hash(diagnostics_path) if diagnostics_path.is_file() else None,
        },
        "phase1d_drift_baseline": {
            "present": False,
            "reason": "当前 PythonProject 未发现可复用的冻结 Phase 1D real-run report",
        },
    }
    contract = json.loads((output / "preregistered_contract.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "status": "challenger_report_only_no_promotion",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "method": args.method,
        "hypothesis_id": args.hypothesis_id,
        "contract_sha256": contract["contract_sha256"],
        "dataset_sha256": _file_hash(dataset_path),
        "data_audit_sha256": _file_hash(audit_path),
        "implementation_sha256": code_hash,
        "baseline": {
            "label": "none",
            "summary": baseline_summary,
            "folds": baseline_folds,
        },
        "challenger": {
            "label": args.method,
            "summary": candidate_summary,
            "folds": candidate_folds,
        },
        "main_metric_delta_challenger_minus_baseline": ranking_delta,
        "mechanism": {
            "baseline": [fold["mechanism_diagnostics"] for fold in baseline_folds],
            "challenger": [fold["mechanism_diagnostics"] for fold in candidate_folds],
        },
        "references": references,
        "economic_gate": economic,
        "frozen_champion": {
            "name": "legacy + fixed",
            "retained": True,
            "reason": "challenger 报告未获得 C++ economic gate；不自动晋级或覆盖冻结冠军",
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    (output / "challenger_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

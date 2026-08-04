from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from ..data import BAR_V1, BAR_V1_FEATURE_GROUPS
from .deep_walk_forward import run_transformer_walk_forward
from .walk_forward import run_walk_forward_baseline


TASK_METRICS = (
    "return_mae", "return_rmse", "return_huber", "ic", "rank_ic",
    "volatility_mae", "volatility_rmse", "direction_logloss",
    "direction_brier", "direction_auc", "direction_ece",
    "q10_pinball", "q90_pinball", "interval_coverage",
)


def _read_summary(path: Path) -> dict:
    return json.loads((path / "aggregate" / "summary.json").read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _requested_groups(config: dict) -> tuple[str, ...]:
    requested = tuple(config.get("groups", BAR_V1_FEATURE_GROUPS))
    unknown = sorted(set(requested) - set(BAR_V1_FEATURE_GROUPS))
    if unknown:
        raise ValueError("未知特征组: " + ", ".join(unknown))
    if not requested:
        raise ValueError("Feature Ablation 至少需要一个特征组")
    if len(set(requested)) != len(requested):
        raise ValueError("Feature Ablation 特征组不能重复")
    return requested


def _keep_indices(group: str) -> list[int]:
    dropped = set(BAR_V1_FEATURE_GROUPS[group])
    return [
        index for index, name in enumerate(BAR_V1.feature_names)
        if name not in dropped
    ]


def _metric_deltas(summary: dict, baseline: dict) -> dict[str, float]:
    return {
        metric: float(
            summary["metrics"][metric]["mean"]
            - baseline["metrics"][metric]["mean"]
        )
        for metric in TASK_METRICS
    }


def run_feature_ablation(
    config: dict,
    dataset_path: str | Path,
    output_path: str | Path,
) -> Path:
    if config.get("enabled") is not True:
        raise RuntimeError("Feature Ablation 配置必须显式设置 enabled=true")
    requested = _requested_groups(config)
    walk_config = copy.deepcopy(config.get("walk_forward", {}))
    walk_config["enabled"] = True
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)

    baseline_path = run_walk_forward_baseline(
        walk_config, dataset_path, output / "baseline_full"
    )
    baseline = _read_summary(baseline_path)
    results = []
    for group in requested:
        keep = _keep_indices(group)
        group_config = copy.deepcopy(walk_config)
        group_config.setdefault("model", {})["feature_indices"] = keep
        group_path = run_walk_forward_baseline(
            group_config, dataset_path, output / f"drop_{group}"
        )
        summary = _read_summary(group_path)
        results.append({
            "group": group,
            "dropped_features": list(BAR_V1_FEATURE_GROUPS[group]),
            "remaining_feature_count": len(keep),
            "metric_delta_drop_minus_full": _metric_deltas(summary, baseline),
        })

    _write(output / "task_delta.json", {"results": results})
    _write(output / "experiment_manifest.json", {
        "schema_version": 1,
        "mode": "group_drop_retrain",
        "model_family": "RidgeMultiTaskBaseline",
        "formal_transformer_conclusion": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(dataset_path).resolve()),
        "groups": list(requested),
        "feature_schema_sha256": BAR_V1.sha256,
        "interpretation": (
            "Pipeline and baseline diagnostic only; rerun with TemporalTransformerV1.1 "
            "before making model promotion decisions."
        ),
    })
    return output


def run_transformer_feature_ablation(
    config: dict,
    dataset_path: str | Path,
    output_path: str | Path,
) -> Path:
    if config.get("enabled") is not True:
        raise RuntimeError("Transformer Feature Ablation 配置必须显式设置 enabled=true")
    requested = _requested_groups(config)
    walk_config = copy.deepcopy(config.get("walk_forward", {}))
    walk_config["enabled"] = True
    walk_config.setdefault("model", {}).pop("feature_indices", None)
    seeds = tuple(int(seed) for seed in config.get(
        "seeds", [walk_config.get("seed", 20260724)]
    ))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Transformer Feature Ablation seeds 必须非空且不能重复")
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)

    seed_results = {group: [] for group in requested}
    for seed in seeds:
        seed_config = copy.deepcopy(walk_config)
        seed_config["seed"] = seed
        seed_root = output / f"seed_{seed}"
        baseline_path = run_transformer_walk_forward(
            seed_config, dataset_path, seed_root / "baseline_full"
        )
        baseline = _read_summary(baseline_path)
        for group in requested:
            keep = _keep_indices(group)
            group_config = copy.deepcopy(seed_config)
            group_config.setdefault("model", {})["feature_indices"] = keep
            group_path = run_transformer_walk_forward(
                group_config, dataset_path, seed_root / f"drop_{group}"
            )
            summary = _read_summary(group_path)
            seed_results[group].append({
                "seed": seed,
                "metric_delta_drop_minus_full": _metric_deltas(summary, baseline),
            })

    results = []
    for group in requested:
        per_seed = seed_results[group]
        mean_delta = {
            metric: float(sum(
                item["metric_delta_drop_minus_full"][metric] for item in per_seed
            ) / len(per_seed))
            for metric in TASK_METRICS
        }
        std_delta = {
            metric: float((sum(
                (item["metric_delta_drop_minus_full"][metric] - mean_delta[metric]) ** 2
                for item in per_seed
            ) / len(per_seed)) ** 0.5)
            for metric in TASK_METRICS
        }
        results.append({
            "group": group,
            "dropped_features": list(BAR_V1_FEATURE_GROUPS[group]),
            "remaining_feature_count": len(_keep_indices(group)),
            "metric_delta_drop_minus_full": mean_delta,
            "metric_delta_std_across_seeds": std_delta,
            "per_seed": per_seed,
        })

    _write(output / "task_delta.json", {"results": results})
    _write(output / "stability.json", {
        "seed_count": len(seeds),
        "seeds": list(seeds),
        "formal_minimum_seed_count": 3,
        "formal_seed_requirement_met": len(seeds) >= 3,
    })
    first_baseline_manifest = json.loads(
        (output / f"seed_{seeds[0]}" / "baseline_full" / "experiment_manifest.json")
        .read_text(encoding="utf-8")
    )
    _write(output / "experiment_manifest.json", {
        "schema_version": 1,
        "mode": "group_drop_retrain",
        "model_family": "TemporalTransformerV1.1",
        "formal_transformer_conclusion": len(seeds) >= 3,
        "formal_conclusion_scope": "prediction_tasks_only",
        "formal_model_promotion_conclusion": False,
        "model_promotion_blocker": (
            "C++ net-of-cost portfolio ablation metrics are not integrated yet."
        ),
        "artifact_policy": "analysis_only_not_for_production_export",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(dataset_path).resolve()),
        "dataset_fingerprint": first_baseline_manifest["dataset_fingerprint"],
        "execution_reference_status": first_baseline_manifest.get(
            "execution_reference_status", "UNDECLARED"
        ),
        "groups": list(requested),
        "seeds": list(seeds),
        "feature_schema_sha256": BAR_V1.sha256,
        "comparison_control": (
            "Identical seed, walk-forward windows, model architecture, and training budget; "
            "only selected input columns differ."
        ),
    })
    return output

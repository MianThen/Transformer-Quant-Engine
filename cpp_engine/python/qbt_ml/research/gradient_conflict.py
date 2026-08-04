"""Deterministic, observation-only shared-gradient diagnostics and oracles."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any


class GradientConflictArtifactValidationError(ValueError):
    """Raised when a gradient diagnostic artifact or oracle input is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_like(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _valid_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z") or "T" not in value:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo == timezone.utc


def _vector(value: Any, name: str, dimension: int | None = None) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GradientConflictArtifactValidationError(f"{name} 必须是梯度向量")
    values = tuple(float(item) for item in value if _finite(item))
    if len(values) != len(value):
        raise GradientConflictArtifactValidationError(f"{name} 必须全部为有限数值")
    if not values:
        raise GradientConflictArtifactValidationError(f"{name} 不得为空")
    if dimension is not None and len(values) != dimension:
        raise GradientConflictArtifactValidationError(f"{name} 维度不一致")
    return values


def _loss(value: Any, name: str) -> float:
    if not _finite(value):
        raise GradientConflictArtifactValidationError(f"{name} 必须为有限数值")
    return float(value)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _add_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["artifact_hash"] = _sha256_text(_canonical_json(result))
    return result


def _strip_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("artifact_hash", None)
    return result


def shared_parameter_set_sha256(parameter_names: Sequence[str]) -> str:
    names = tuple(parameter_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise GradientConflictArtifactValidationError("shared parameter 名称无效")
    if len(names) != len(set(names)):
        raise GradientConflictArtifactValidationError("shared parameter 名称不得重复")
    return _sha256_text(_canonical_json(list(names)))


@dataclass(frozen=True)
class GradientConflictSpecV1:
    schema_version: int = 1
    hypothesis_id: str = "MTL-FIXED-DIAGNOSTIC-V1"
    task_names: tuple[str, ...] = (
        "return",
        "direction",
        "volatility",
        "quantile",
        "legacy_rank",
    )
    task_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 0.1)
    shared_parameter_set_sha256: str = ""
    diagnostic_seed: int = 1
    cadence: str = "epoch"
    amp_enabled: bool = False
    amp_unscale_before_measurement: bool = True
    loss_scaling_mode: str = "none"
    diagnostic_only: bool = True
    available_at_utc: str = ""

    def validate(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise GradientConflictArtifactValidationError("GradientConflictSpec schema_version 必须为 1")
        if not self.hypothesis_id or not isinstance(self.hypothesis_id, str):
            raise GradientConflictArtifactValidationError("hypothesis_id 无效")
        if not self.task_names or any(not isinstance(name, str) or not name for name in self.task_names):
            raise GradientConflictArtifactValidationError("task_names 无效")
        if len(self.task_names) != len(set(self.task_names)):
            raise GradientConflictArtifactValidationError("task_names 不得重复")
        if len(self.task_weights) != len(self.task_names):
            raise GradientConflictArtifactValidationError("task_weights 与 task_names 长度不一致")
        if any(not _finite(weight) or float(weight) <= 0.0 for weight in self.task_weights):
            raise GradientConflictArtifactValidationError("task_weights 必须为正有限数值")
        if "legacy_rank" in self.task_names:
            rank_weight = self.task_weights[self.task_names.index("legacy_rank")]
            if not math.isclose(float(rank_weight), 0.1, rel_tol=0.0, abs_tol=1e-12):
                raise GradientConflictArtifactValidationError("legacy_rank 权重必须固定为 0.1")
        if not _digest_like(self.shared_parameter_set_sha256):
            raise GradientConflictArtifactValidationError("shared_parameter_set_sha256 格式无效")
        if isinstance(self.diagnostic_seed, bool) or self.diagnostic_seed <= 0:
            raise GradientConflictArtifactValidationError("diagnostic_seed 必须为正整数")
        if not isinstance(self.cadence, str) or not self.cadence:
            raise GradientConflictArtifactValidationError("cadence 无效")
        if not isinstance(self.amp_enabled, bool) or not isinstance(self.amp_unscale_before_measurement, bool):
            raise GradientConflictArtifactValidationError("AMP 状态必须为布尔值")
        if not isinstance(self.loss_scaling_mode, str) or not self.loss_scaling_mode:
            raise GradientConflictArtifactValidationError("loss_scaling_mode 无效")
        if self.amp_enabled and not self.amp_unscale_before_measurement:
            raise GradientConflictArtifactValidationError("AMP 梯度必须在 unscale 后测量")
        if self.diagnostic_only is not True:
            raise GradientConflictArtifactValidationError("GradientConflict V1 必须是 diagnostic_only")
        if not _valid_utc_timestamp(self.available_at_utc):
            raise GradientConflictArtifactValidationError("available_at_utc 无效")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hypothesis_id": self.hypothesis_id,
            "task_names": list(self.task_names),
            "task_weights": list(self.task_weights),
            "shared_parameter_set_sha256": self.shared_parameter_set_sha256,
            "diagnostic_seed": self.diagnostic_seed,
            "cadence": self.cadence,
            "amp_enabled": self.amp_enabled,
            "amp_unscale_before_measurement": self.amp_unscale_before_measurement,
            "loss_scaling_mode": self.loss_scaling_mode,
            "diagnostic_only": self.diagnostic_only,
            "available_at_utc": self.available_at_utc,
        }

    @property
    def spec_sha256(self) -> str:
        self.validate()
        return _sha256_text(_canonical_json(self.to_dict()))


def compute_gradient_conflict_metrics(
    task_gradients: Mapping[str, Sequence[Any]],
    task_losses: Mapping[str, Any],
    *,
    initial_losses: Mapping[str, Any] | None = None,
    task_weights: Mapping[str, Any] | None = None,
    task_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(task_gradients, Mapping) or not isinstance(task_losses, Mapping):
        raise GradientConflictArtifactValidationError("task_gradients/task_losses 必须是对象")
    names = tuple(task_names) if task_names is not None else tuple(sorted(task_gradients))
    if not names or len(names) != len(set(names)):
        raise GradientConflictArtifactValidationError("task_names 无效")
    if set(task_gradients) != set(names) or set(task_losses) != set(names):
        raise GradientConflictArtifactValidationError("task_gradients/task_losses 任务集合不一致")
    if initial_losses is not None and set(initial_losses) != set(names):
        raise GradientConflictArtifactValidationError("initial_losses 任务集合不一致")
    if task_weights is not None and set(task_weights) != set(names):
        raise GradientConflictArtifactValidationError("task_weights 任务集合不一致")
    weights = {
        name: _loss(task_weights[name], f"task_weights[{name}]")
        if task_weights is not None
        else 0.1 if name == "legacy_rank" else 1.0
        for name in names
    }
    if any(value <= 0.0 for value in weights.values()):
        raise GradientConflictArtifactValidationError("task_weights 必须为正")
    gradients: dict[str, tuple[float, ...]] = {}
    dimension: int | None = None
    for name in names:
        gradients[name] = _vector(task_gradients[name], f"task_gradients[{name}]", dimension)
        dimension = len(gradients[name])
    losses = {name: _loss(task_losses[name], f"task_losses[{name}]") for name in names}
    initials = (
        {name: _loss(initial_losses[name], f"initial_losses[{name}]") for name in names}
        if initial_losses is not None
        else {}
    )
    normalized_losses = {
        name: losses[name] / initials[name]
        if name in initials and initials[name] > 0.0
        else None
        for name in names
    }
    finite_normalized = [value for value in normalized_losses.values() if value is not None]
    normalized_mean = _mean(finite_normalized)
    relative_rates = {
        name: normalized_losses[name] / normalized_mean
        if normalized_losses[name] is not None and normalized_mean is not None and normalized_mean > 0.0
        else None
        for name in names
    }
    norms = {name: _norm(gradients[name]) for name in names}
    weighted_gradients = {
        name: tuple(weights[name] * value for value in gradients[name]) for name in names
    }
    aggregate = tuple(
        sum(weighted_gradients[name][index] for name in names)
        for index in range(dimension or 0)
    )
    aggregate_norm = _norm(aggregate)
    pairwise_dot = {
        left: {right: _dot(gradients[left], gradients[right]) for right in names}
        for left in names
    }
    pairwise_cosine: dict[str, dict[str, float | None]] = {}
    for left in names:
        pairwise_cosine[left] = {}
        for right in names:
            denominator = norms[left] * norms[right]
            pairwise_cosine[left][right] = (
                pairwise_dot[left][right] / denominator if denominator > 0.0 else None
            )
    negative_rates: dict[str, float | None] = {}
    negative_pairs = 0
    pair_count = 0
    for left in names:
        negative = 0
        compared = 0
        for right in names:
            if left == right:
                continue
            cosine = pairwise_cosine[left][right]
            if cosine is None:
                continue
            compared += 1
            negative += int(cosine < 0.0)
            if names.index(left) < names.index(right):
                pair_count += 1
                negative_pairs += int(cosine < 0.0)
        negative_rates[left] = negative / compared if compared else None
    absolute_contributions = {
        name: abs(_dot(weighted_gradients[name], aggregate)) for name in names
    }
    contribution_total = sum(absolute_contributions.values())
    task_metrics: dict[str, dict[str, Any]] = {}
    for name in names:
        absolute_values = [abs(value) for value in gradients[name]]
        projection = (
            _dot(gradients[name], aggregate) / (aggregate_norm * aggregate_norm)
            if aggregate_norm > 0.0
            else None
        )
        task_metrics[name] = {
            "raw_loss": losses[name],
            "initial_loss": initials.get(name),
            "normalized_loss": normalized_losses[name],
            "relative_training_rate": relative_rates[name],
            "scalar_weight": weights[name],
            "weighted_loss": losses[name] * weights[name],
            "gradient_l2_norm": norms[name],
            "max_abs_coordinate": max(absolute_values),
            "median_abs_coordinate": _median(absolute_values),
            "projection_on_synthetic_update": projection,
            "dominance_ratio": (
                absolute_contributions[name] / contribution_total
                if contribution_total > 0.0
                else None
            ),
        }
    positive_norms = [value for value in norms.values() if value > 0.0]
    result: dict[str, Any] = {
        "status": "OK",
        "diagnostic_only": True,
        "parameter_update_applied": False,
        "tasks": list(names),
        "dimension": dimension,
        "task_metrics": task_metrics,
        "pairwise_dot": pairwise_dot,
        "pairwise_cosine": pairwise_cosine,
        "negative_conflict_rate": negative_rates,
        "aggregate": {
            "gradient_l2_norm": aggregate_norm,
            "max_abs_coordinate": max(abs(value) for value in aggregate),
            "median_abs_coordinate": _median([abs(value) for value in aggregate]),
            "max_min_gradient_norm_ratio": (
                max(positive_norms) / min(positive_norms) if positive_norms else None
            ),
            "pairwise_negative_conflict_rate": (
                negative_pairs / pair_count if pair_count else None
            ),
            "negative_conflict_pairs": negative_pairs,
            "pair_count": pair_count,
        },
    }
    return _add_hash(result)


def _aggregate_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not metrics:
        raise GradientConflictArtifactValidationError("不能聚合空 metrics")
    first = metrics[0]
    tasks = tuple(first["tasks"])

    def average_path(values: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float | None:
        numbers: list[float] = []
        for value in values:
            current: Any = value
            for key in path:
                if not isinstance(current, Mapping) or key not in current:
                    current = None
                    break
                current = current[key]
            if _finite(current):
                numbers.append(float(current))
        return _mean(numbers)

    task_fields = (
        "raw_loss",
        "initial_loss",
        "normalized_loss",
        "relative_training_rate",
        "scalar_weight",
        "weighted_loss",
        "gradient_l2_norm",
        "max_abs_coordinate",
        "median_abs_coordinate",
        "projection_on_synthetic_update",
        "dominance_ratio",
    )
    task_metrics = {
        task: {
            field: average_path(metrics, ("task_metrics", task, field))
            for field in task_fields
        }
        for task in tasks
    }
    pairwise_dot = {
        left: {
            right: average_path(metrics, ("pairwise_dot", left, right))
            for right in tasks
        }
        for left in tasks
    }
    pairwise_cosine = {
        left: {
            right: average_path(metrics, ("pairwise_cosine", left, right))
            for right in tasks
        }
        for left in tasks
    }
    negative_rates = {
        task: average_path(metrics, ("negative_conflict_rate", task)) for task in tasks
    }
    aggregate_fields = (
        "gradient_l2_norm",
        "max_abs_coordinate",
        "median_abs_coordinate",
        "max_min_gradient_norm_ratio",
        "pairwise_negative_conflict_rate",
    )
    return {
        "status": "OK" if all(item.get("status") == "OK" for item in metrics) else "INSUFFICIENT_DATA",
        "diagnostic_only": True,
        "parameter_update_applied": False,
        "observations": len(metrics),
        "tasks": list(tasks),
        "task_metrics": task_metrics,
        "pairwise_dot": pairwise_dot,
        "pairwise_cosine": pairwise_cosine,
        "negative_conflict_rate": negative_rates,
        "aggregate": {
            field: average_path(metrics, ("aggregate", field)) for field in aggregate_fields
        },
    }


def build_gradient_conflict_artifact(
    spec: GradientConflictSpecV1,
    observations: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> str:
    spec.validate()
    records_input = [observations] if isinstance(observations, Mapping) else list(observations)
    if not records_input:
        raise GradientConflictArtifactValidationError("observations 不得为空")
    records: list[dict[str, Any]] = []
    metric_values: list[Mapping[str, Any]] = []
    for index, observation in enumerate(records_input):
        if not isinstance(observation, Mapping):
            raise GradientConflictArtifactValidationError("observation 必须是对象")
        fold = observation.get("fold")
        regime = observation.get("market_regime")
        epoch = observation.get("epoch")
        if not isinstance(fold, str) or not fold:
            raise GradientConflictArtifactValidationError(f"observation[{index}] fold 无效")
        if not isinstance(regime, str) or not regime:
            raise GradientConflictArtifactValidationError(f"observation[{index}] market_regime 无效")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise GradientConflictArtifactValidationError(f"observation[{index}] epoch 无效")
        overhead_ms = observation.get("overhead_ms")
        if overhead_ms is not None and (not _finite(overhead_ms) or float(overhead_ms) < 0.0):
            raise GradientConflictArtifactValidationError(f"observation[{index}] overhead_ms 无效")
        observation_weights = observation.get("task_weights")
        expected_weights = dict(zip(spec.task_names, spec.task_weights))
        if observation_weights is not None:
            if not isinstance(observation_weights, Mapping) or set(observation_weights) != set(spec.task_names):
                raise GradientConflictArtifactValidationError(
                    f"observation[{index}] task_weights 与 spec 不一致"
                )
            for task in spec.task_names:
                if not _finite(observation_weights[task]) or not math.isclose(
                    float(observation_weights[task]),
                    expected_weights[task],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise GradientConflictArtifactValidationError(
                        f"observation[{index}] task_weights 不得覆盖固定 spec"
                    )
        metrics = compute_gradient_conflict_metrics(
            observation.get("task_gradients", {}),
            observation.get("task_losses", {}),
            initial_losses=observation.get("initial_losses"),
            task_weights=observation_weights if observation_weights is not None else expected_weights,
            task_names=spec.task_names,
        )
        if metrics["tasks"] != list(spec.task_names):
            raise GradientConflictArtifactValidationError("observation task 顺序与 spec 不一致")
        record = {
            "fold": fold,
            "epoch": epoch,
            "market_regime": regime,
            "overhead_ms": float(overhead_ms) if overhead_ms is not None else None,
            "metrics": metrics,
        }
        records.append(record)
        metric_values.append(metrics)
    by_fold: dict[str, list[Mapping[str, Any]]] = {}
    by_regime: dict[str, list[Mapping[str, Any]]] = {}
    by_epoch: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_fold.setdefault(record["fold"], []).append(record["metrics"])
        by_regime.setdefault(record["market_regime"], []).append(record["metrics"])
        by_epoch.setdefault(str(record["epoch"]), []).append(record["metrics"])
    aggregates = {
        "overall": _aggregate_metrics(metric_values),
        "by_fold": {key: _aggregate_metrics(by_fold[key]) for key in sorted(by_fold)},
        "by_market_regime": {
            key: _aggregate_metrics(by_regime[key]) for key in sorted(by_regime)
        },
        "by_epoch": {key: _aggregate_metrics(by_epoch[key]) for key in sorted(by_epoch, key=int)},
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": "gradient_conflict_artifact_v1",
        "hypothesis_id": spec.hypothesis_id,
        "spec_sha256": spec.spec_sha256,
        "task_names": list(spec.task_names),
        "task_weights": list(spec.task_weights),
        "shared_parameter_set_sha256": spec.shared_parameter_set_sha256,
        "diagnostic_seed": spec.diagnostic_seed,
        "cadence": spec.cadence,
        "amp_enabled": spec.amp_enabled,
        "amp_unscale_before_measurement": spec.amp_unscale_before_measurement,
        "loss_scaling_mode": spec.loss_scaling_mode,
        "available_at_utc": spec.available_at_utc,
        "diagnostic_only": True,
        "parameter_update_applied": False,
        "observations": records,
        "observations_sha256": _sha256_text(_canonical_json(records)),
        "aggregates": aggregates,
        "aggregates_sha256": _sha256_text(_canonical_json(aggregates)),
    }
    unsigned = _canonical_json(payload)
    return unsigned[:-1] + ',"report_sha256":"' + _sha256_text(unsigned) + '"}'


def _parse_artifact(value: str | Path | Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(value, Path):
        try:
            raw = value.read_text(encoding="utf-8")
        except OSError as exc:
            raise GradientConflictArtifactValidationError("无法读取 GradientConflictArtifact") from exc
    elif isinstance(value, str):
        if value.lstrip().startswith("{"):
            raw = value
        else:
            try:
                candidate = Path(value)
                raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
            except OSError:
                raw = value
    elif isinstance(value, Mapping):
        raw = _canonical_json(value)
    else:
        raise GradientConflictArtifactValidationError("GradientConflictArtifact 类型无效")
    try:
        parsed = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GradientConflictArtifactValidationError("GradientConflictArtifact JSON 无效") from exc
    if not isinstance(parsed, dict):
        raise GradientConflictArtifactValidationError("GradientConflictArtifact 必须是对象")
    return raw, parsed


def _validate_metrics(metrics: Any, task_names: Sequence[str]) -> None:
    if not isinstance(metrics, dict):
        raise GradientConflictArtifactValidationError("metrics 必须是对象")
    if metrics.get("status") != "OK" or metrics.get("diagnostic_only") is not True:
        raise GradientConflictArtifactValidationError("metrics 状态或 diagnostic_only 无效")
    if metrics.get("parameter_update_applied") is not False:
        raise GradientConflictArtifactValidationError("诊断 metrics 不得应用参数更新")
    if metrics.get("tasks") != list(task_names):
        raise GradientConflictArtifactValidationError("metrics tasks 不一致")
    metric_hash = metrics.get("artifact_hash")
    if not _digest_like(metric_hash) or _sha256_text(_canonical_json(_strip_hash(metrics))) != metric_hash:
        raise GradientConflictArtifactValidationError("metrics artifact_hash 校验失败")
    for section in ("task_metrics", "pairwise_dot", "pairwise_cosine", "negative_conflict_rate"):
        if not isinstance(metrics.get(section), dict):
            raise GradientConflictArtifactValidationError(f"metrics {section} 无效")
    if not isinstance(metrics.get("aggregate"), dict):
        raise GradientConflictArtifactValidationError("metrics aggregate 无效")


def validate_gradient_conflict_artifact(
    value: str | Path | Mapping[str, Any],
) -> Mapping[str, Any]:
    raw, parsed = _parse_artifact(value)
    if parsed.get("schema_version") != 1 or parsed.get("role") != "gradient_conflict_artifact_v1":
        raise GradientConflictArtifactValidationError("GradientConflictArtifact schema 不受支持")
    report_hash = parsed.get("report_sha256")
    if not _digest_like(report_hash):
        raise GradientConflictArtifactValidationError("report_sha256 格式无效")
    marker = ',"report_sha256":"'
    marker_start = raw.rfind(marker)
    if marker_start < 0 or not raw.endswith("}"):
        raise GradientConflictArtifactValidationError("report hash 尾部无效")
    if _sha256_text(raw[:marker_start] + "}") != report_hash:
        raise GradientConflictArtifactValidationError("report SHA-256 校验失败")
    if parsed.get("diagnostic_only") is not True or parsed.get("parameter_update_applied") is not False:
        raise GradientConflictArtifactValidationError("Artifact 必须保持 observation-only")
    task_names = parsed.get("task_names")
    task_weights = parsed.get("task_weights")
    if not isinstance(task_names, list) or not task_names or any(
        not isinstance(name, str) or not name for name in task_names
    ) or len(task_names) != len(set(task_names)):
        raise GradientConflictArtifactValidationError("Artifact task_names 无效")
    if not isinstance(task_weights, list) or len(task_weights) != len(task_names) or any(
        not _finite(weight) or float(weight) <= 0.0 for weight in task_weights
    ):
        raise GradientConflictArtifactValidationError("Artifact task_weights 无效")
    if "legacy_rank" in task_names:
        rank_weight = task_weights[task_names.index("legacy_rank")]
        if not math.isclose(float(rank_weight), 0.1, rel_tol=0.0, abs_tol=1e-12):
            raise GradientConflictArtifactValidationError("Artifact legacy_rank 权重必须为 0.1")
    for key in ("spec_sha256", "shared_parameter_set_sha256"):
        if not _digest_like(parsed.get(key)):
            raise GradientConflictArtifactValidationError(f"Artifact {key} 无效")
    if isinstance(parsed.get("diagnostic_seed"), bool) or not isinstance(parsed.get("diagnostic_seed"), int) or parsed["diagnostic_seed"] <= 0:
        raise GradientConflictArtifactValidationError("Artifact diagnostic_seed 无效")
    if not isinstance(parsed.get("amp_enabled"), bool) or not isinstance(parsed.get("amp_unscale_before_measurement"), bool):
        raise GradientConflictArtifactValidationError("Artifact AMP 字段无效")
    if parsed["amp_enabled"] and not parsed["amp_unscale_before_measurement"]:
        raise GradientConflictArtifactValidationError("AMP Artifact 必须记录 unscale 后梯度")
    if not isinstance(parsed.get("loss_scaling_mode"), str) or not parsed["loss_scaling_mode"]:
        raise GradientConflictArtifactValidationError("Artifact loss_scaling_mode 无效")
    if not _valid_utc_timestamp(parsed.get("available_at_utc")):
        raise GradientConflictArtifactValidationError("Artifact available_at_utc 无效")
    observations = parsed.get("observations")
    if not isinstance(observations, list) or not observations:
        raise GradientConflictArtifactValidationError("Artifact observations 无效")
    observation_hash = parsed.get("observations_sha256")
    if not _digest_like(observation_hash) or _sha256_text(_canonical_json(observations)) != observation_hash:
        raise GradientConflictArtifactValidationError("observations_sha256 校验失败")
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise GradientConflictArtifactValidationError(f"observation[{index}] 无效")
        if not isinstance(observation.get("fold"), str) or not observation["fold"]:
            raise GradientConflictArtifactValidationError(f"observation[{index}] fold 无效")
        if not isinstance(observation.get("market_regime"), str) or not observation["market_regime"]:
            raise GradientConflictArtifactValidationError(f"observation[{index}] market_regime 无效")
        if isinstance(observation.get("epoch"), bool) or not isinstance(observation.get("epoch"), int) or observation["epoch"] < 0:
            raise GradientConflictArtifactValidationError(f"observation[{index}] epoch 无效")
        overhead = observation.get("overhead_ms")
        if overhead is not None and (not _finite(overhead) or float(overhead) < 0.0):
            raise GradientConflictArtifactValidationError(f"observation[{index}] overhead_ms 无效")
        _validate_metrics(observation.get("metrics"), task_names)
    aggregates = parsed.get("aggregates")
    aggregates_hash = parsed.get("aggregates_sha256")
    if not isinstance(aggregates, dict) or not _digest_like(aggregates_hash) or _sha256_text(_canonical_json(aggregates)) != aggregates_hash:
        raise GradientConflictArtifactValidationError("aggregates_sha256 校验失败")
    return MappingProxyType(dict(parsed))


def _fisher_yates(values: Sequence[str], state: int) -> tuple[list[str], int]:
    result = list(values)
    for index in range(len(result) - 1, 0, -1):
        state, random_value = _splitmix64(state)
        swap = random_value % (index + 1)
        result[index], result[swap] = result[swap], result[index]
    return result, state


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, (value ^ (value >> 31)) & ((1 << 64) - 1)


def pcgrad_project(
    task_gradients: Mapping[str, Sequence[Any]],
    *,
    task_order: Sequence[str] | None = None,
    seed: int = 1,
    task_weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(seed, bool) or seed <= 0:
        raise GradientConflictArtifactValidationError("PCGrad seed 必须为正整数")
    if not isinstance(task_gradients, Mapping) or not task_gradients:
        raise GradientConflictArtifactValidationError("PCGrad task_gradients 不得为空")
    names = tuple(task_order) if task_order is not None else tuple(sorted(task_gradients))
    if set(names) != set(task_gradients) or len(names) != len(set(names)):
        raise GradientConflictArtifactValidationError("PCGrad task_order 不一致")
    gradients: dict[str, tuple[float, ...]] = {}
    dimension: int | None = None
    for name in names:
        gradients[name] = _vector(task_gradients[name], f"task_gradients[{name}]", dimension)
        dimension = len(gradients[name])
    weights = {
        name: _loss(task_weights[name], f"task_weights[{name}]")
        if task_weights is not None
        else 0.1 if name == "legacy_rank" else 1.0
        for name in names
    }
    if any(value <= 0.0 for value in weights.values()):
        raise GradientConflictArtifactValidationError("PCGrad task_weights 必须为正")
    if "legacy_rank" in names and not math.isclose(
        weights["legacy_rank"], 0.1, rel_tol=0.0, abs_tol=1e-12
    ):
        raise GradientConflictArtifactValidationError("PCGrad legacy rank 权重必须固定为 0.1")
    projected = {name: list(gradients[name]) for name in names}
    peer_orders: dict[str, list[str]] = {}
    projection_count = 0
    conflict_count = 0
    zero_norm_skips = 0
    state = seed
    for name in names:
        peers, state = _fisher_yates([candidate for candidate in names if candidate != name], state)
        peer_orders[name] = peers
        for peer in peers:
            dot = _dot(projected[name], gradients[peer])
            peer_norm_squared = _dot(gradients[peer], gradients[peer])
            if peer_norm_squared <= 0.0:
                zero_norm_skips += 1
                continue
            if dot < 0.0:
                conflict_count += 1
                coefficient = dot / peer_norm_squared
                projected[name] = [
                    value - coefficient * peer_value
                    for value, peer_value in zip(projected[name], gradients[peer])
                ]
                projection_count += 1
    aggregate = [
        sum(weights[name] * projected[name][index] for name in names)
        for index in range(dimension or 0)
    ]
    return {
        "algorithm": "PCGRAD",
        "scope": "shared_backbone_only",
        "task_specific_heads_unchanged": True,
        "diagnostic_only": True,
        "parameter_update_applied": False,
        "seed": seed,
        "task_order": list(names),
        "peer_orders": peer_orders,
        "task_weights": weights,
        "projected_gradients": {name: projected[name] for name in names},
        "aggregate_gradient": aggregate,
        "projection_count": projection_count,
        "negative_pair_count": conflict_count,
        "zero_norm_skips": zero_norm_skips,
    }


def renormalize_gradnorm_weights(
    task_weights: Mapping[str, Any],
    *,
    task_names: Sequence[str] | None = None,
    rank_task: str = "legacy_rank",
    rank_weight: float = 0.1,
    total_weight: float | None = None,
) -> dict[str, float]:
    if not _finite(rank_weight) or rank_weight <= 0.0:
        raise GradientConflictArtifactValidationError("GradNorm rank_weight 必须为正")
    names = tuple(task_names) if task_names is not None else tuple(sorted(task_weights))
    if not names or set(names) != set(task_weights) or len(names) != len(set(names)):
        raise GradientConflictArtifactValidationError("GradNorm task_names 不一致")
    raw = {name: _loss(task_weights[name], f"task_weights[{name}]") for name in names}
    if any(value <= 0.0 for value in raw.values()):
        raise GradientConflictArtifactValidationError("GradNorm 权重必须为正")
    target_total = sum(raw.values()) if total_weight is None else _loss(total_weight, "total_weight")
    if target_total <= 0.0:
        raise GradientConflictArtifactValidationError("GradNorm total_weight 必须为正")
    if rank_task not in names:
        scale = target_total / sum(raw.values())
        return {name: raw[name] * scale for name in names}
    if target_total <= rank_weight:
        raise GradientConflictArtifactValidationError("GradNorm total_weight 不足以固定 rank 权重")
    remainder = target_total - rank_weight
    non_rank = [name for name in names if name != rank_task]
    if not non_rank:
        return {rank_task: target_total}
    raw_non_rank_total = sum(raw[name] for name in non_rank)
    return {
        name: rank_weight if name == rank_task else remainder * raw[name] / raw_non_rank_total
        for name in names
    }


def gradnorm_targets(
    task_losses: Mapping[str, Any],
    initial_losses: Mapping[str, Any],
    weighted_gradient_norms: Mapping[str, Any],
    *,
    alpha: float = 1.5,
    task_weights: Mapping[str, Any] | None = None,
    task_names: Sequence[str] | None = None,
    rank_task: str = "legacy_rank",
    rank_weight: float = 0.1,
) -> dict[str, Any]:
    if not _finite(alpha) or float(alpha) < 0.0:
        raise GradientConflictArtifactValidationError("GradNorm alpha 必须为非负有限值")
    names = tuple(task_names) if task_names is not None else tuple(sorted(task_losses))
    if not names or set(names) != set(task_losses) or set(names) != set(initial_losses) or set(names) != set(weighted_gradient_norms):
        raise GradientConflictArtifactValidationError("GradNorm 任务集合不一致")
    losses = {name: _loss(task_losses[name], f"task_losses[{name}]") for name in names}
    initials = {name: _loss(initial_losses[name], f"initial_losses[{name}]") for name in names}
    norms = {name: _loss(weighted_gradient_norms[name], f"weighted_gradient_norms[{name}]") for name in names}
    if any(value < 0.0 for value in losses.values()) or any(value < 0.0 for value in norms.values()):
        raise GradientConflictArtifactValidationError("GradNorm loss/norm 不得为负")
    if task_weights is not None:
        if set(task_weights) != set(names):
            raise GradientConflictArtifactValidationError("GradNorm task_weights 任务集合不一致")
        if rank_task in names:
            if not math.isclose(float(task_weights[rank_task]), rank_weight, rel_tol=0.0, abs_tol=1e-12):
                raise GradientConflictArtifactValidationError("GradNorm legacy rank 权重必须固定")
    invalid_initial = [name for name in names if initials[name] <= 0.0]
    if invalid_initial:
        return {
            "status": "INSUFFICIENT_DATA",
            "diagnostic_only": True,
            "parameter_update_applied": False,
            "alpha": float(alpha),
            "invalid_initial_loss_tasks": invalid_initial,
            "normalized_losses": {name: None for name in names},
            "relative_training_rates": {name: None for name in names},
            "gradient_norms": norms,
            "targets": {name: None for name in names},
            "gradnorm_loss": None,
        }
    normalized = {name: losses[name] / initials[name] for name in names}
    mean_normalized = sum(normalized.values()) / len(names)
    rates = {
        name: normalized[name] / mean_normalized if mean_normalized > 0.0 else None
        for name in names
    }
    mean_norm = sum(norms.values()) / len(names)
    targets = {
        name: mean_norm * (rates[name] ** float(alpha)) if rates[name] is not None else None
        for name in names
    }
    return {
        "status": "OK" if all(value is not None for value in rates.values()) else "INSUFFICIENT_DATA",
        "diagnostic_only": True,
        "parameter_update_applied": False,
        "alpha": float(alpha),
        "normalized_losses": normalized,
        "relative_training_rates": rates,
        "gradient_norms": norms,
        "targets": targets,
        "gradnorm_loss": sum(abs(norms[name] - targets[name]) for name in names if targets[name] is not None),
        "renormalized_weights": (
            renormalize_gradnorm_weights(
                task_weights,
                task_names=names,
                rank_task=rank_task,
                rank_weight=rank_weight,
            )
            if task_weights is not None
            else None
        ),
    }


__all__ = [
    "GradientConflictArtifactValidationError",
    "GradientConflictSpecV1",
    "build_gradient_conflict_artifact",
    "compute_gradient_conflict_metrics",
    "gradnorm_targets",
    "pcgrad_project",
    "renormalize_gradnorm_weights",
    "shared_parameter_set_sha256",
    "validate_gradient_conflict_artifact",
]

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Sequence

import numpy as np


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite_vector(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是一维非空有限数组")
    return array


def _timestamps(
    values: Sequence[int], *, allow_equal: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    differences = np.diff(array)
    if (
        array.ndim != 1
        or array.size == 0
        or np.any(array <= 0)
        or np.any(differences < 0)
        or (not allow_equal and np.any(differences == 0))
    ):
        order = "单调不减" if allow_equal else "严格递增"
        raise ValueError(f"timestamps 必须{order}且为正数")
    return array


@dataclass(frozen=True)
class RollingCqrSpec:
    target_coverage: float = 0.90
    minimum_calibration_observations: int = 40
    maximum_calibration_observations: int = 252
    exponential_decay: float = 1.0
    config_hash: int = 0

    def validate(self) -> None:
        if not 0.5 < self.target_coverage < 1.0:
            raise ValueError("target_coverage 必须位于 (0.5,1)")
        if self.minimum_calibration_observations < 20:
            raise ValueError("minimum_calibration_observations 不得小于 20")
        if self.maximum_calibration_observations < self.minimum_calibration_observations:
            raise ValueError("maximum_calibration_observations 小于最小校准样本")
        if not 0.0 < self.exponential_decay <= 1.0:
            raise ValueError("exponential_decay 必须位于 (0,1]")
        if self.config_hash <= 0:
            raise ValueError("config_hash 必须冻结为正整数")

    @property
    def spec_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


@dataclass(frozen=True)
class CqrCalibrationArtifact:
    schema_version: int
    method: str
    target_coverage: float
    score_definition: str
    quantile_policy: str
    raw_q_hat: float
    q_hat: float
    calibration_count: int
    effective_calibration_count: float
    fit_start: int
    fit_end: int
    available_at: int
    score_sha256: str
    spec_sha256: str
    artifact_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RollingCqrBacktestResult:
    schema_version: int
    role: str
    status: str
    target_coverage: float
    forecast_count: int
    first_forecast_timestamp: int
    last_forecast_timestamp: int
    empirical_coverage: float
    lower_miss_rate: float
    upper_miss_rate: float
    mean_interval_width: float
    q_hat_values: tuple[float, ...]
    calibrated_lower: tuple[float, ...]
    calibrated_upper: tuple[float, ...]
    realization_timestamps: tuple[int, ...]
    spec_sha256: str
    input_sha256: str
    artifact_sha256: str
    formal_distribution_free_claim: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _conformity_scores(
    lower_quantile: np.ndarray,
    upper_quantile: np.ndarray,
    realized_target: np.ndarray,
) -> np.ndarray:
    if not (
        lower_quantile.size == upper_quantile.size == realized_target.size
    ):
        raise ValueError("CQR calibration 数组长度不一致")
    if np.any(lower_quantile > upper_quantile):
        raise ValueError("lower_quantile 不得高于 upper_quantile")
    return np.maximum(
        lower_quantile - realized_target,
        realized_target - upper_quantile,
    )


def _calibration_quantile(
    scores: np.ndarray,
    spec: RollingCqrSpec,
) -> tuple[float, float, str]:
    indices = np.arange(scores.size, dtype=np.int64)
    order = np.lexsort((indices, scores))
    ordered_scores = scores[order]
    if spec.exponential_decay == 1.0:
        rank = math.ceil((scores.size + 1) * spec.target_coverage)
        selected = min(scores.size, max(1, rank)) - 1
        return float(ordered_scores[selected]), float(scores.size), (
            "ROLLING_CQR_UNIFORM_FINITE_SAMPLE_HIGHER_V1"
        )
    ages = scores.size - 1 - indices
    weights = np.power(spec.exponential_decay, ages, dtype=np.float64)
    ordered_weights = weights[order]
    normalized = ordered_weights / ordered_weights.sum()
    cumulative = np.cumsum(normalized)
    selected = min(
        ordered_scores.size - 1,
        int(np.searchsorted(cumulative, spec.target_coverage, side="left")),
    )
    effective = float(1.0 / np.square(normalized).sum())
    return float(ordered_scores[selected]), effective, (
        "ROLLING_CQR_EXPONENTIALLY_WEIGHTED_HIGHER_V1"
    )


def fit_cqr_calibrator(
    lower_quantile: Sequence[float],
    upper_quantile: Sequence[float],
    realized_target: Sequence[float],
    timestamps: Sequence[int],
    *,
    available_at: int,
    spec: RollingCqrSpec,
) -> CqrCalibrationArtifact:
    spec.validate()
    lower = _finite_vector("lower_quantile", lower_quantile)
    upper = _finite_vector("upper_quantile", upper_quantile)
    target = _finite_vector("realized_target", realized_target)
    times = _timestamps(timestamps, allow_equal=True)
    if not (lower.size == upper.size == target.size == times.size):
        raise ValueError("CQR calibration 输入未对齐")
    if times[-1] > available_at:
        raise ValueError("校准目标在 available_at 之后，存在未来数据")
    if lower.size < spec.minimum_calibration_observations:
        raise ValueError("CQR calibration 样本不足")
    if lower.size > spec.maximum_calibration_observations:
        start = lower.size - spec.maximum_calibration_observations
        lower = lower[start:]
        upper = upper[start:]
        target = target[start:]
        times = times[start:]
    scores = _conformity_scores(lower, upper, target)
    raw_q_hat, effective, method = _calibration_quantile(scores, spec)
    q_hat = max(0.0, raw_q_hat)
    unsigned = {
        "schema_version": 1,
        "method": method,
        "target_coverage": spec.target_coverage,
        "score_definition": (
            "max(lower_quantile-realized_target,"
            "realized_target-upper_quantile)"
        ),
        "quantile_policy": "HIGHER_WITH_NONNEGATIVE_INTERVAL_GUARD_V1",
        "raw_q_hat": raw_q_hat,
        "q_hat": q_hat,
        "calibration_count": int(scores.size),
        "effective_calibration_count": effective,
        "fit_start": int(times[0]),
        "fit_end": int(times[-1]),
        "available_at": int(available_at),
        "score_sha256": _sha256(scores.tolist()),
        "spec_sha256": spec.spec_sha256,
    }
    return CqrCalibrationArtifact(
        **unsigned, artifact_sha256=_sha256(unsigned),
    )


def apply_cqr_calibrator(
    artifact: CqrCalibrationArtifact,
    lower_quantile: Sequence[float],
    upper_quantile: Sequence[float],
    forecast_timestamps: Sequence[int],
    *,
    forecast_available_at: int,
) -> tuple[np.ndarray, np.ndarray]:
    lower = _finite_vector("lower_quantile", lower_quantile)
    upper = _finite_vector("upper_quantile", upper_quantile)
    times = _timestamps(forecast_timestamps, allow_equal=True)
    if lower.size != upper.size or lower.size != times.size:
        raise ValueError("CQR forecast 输入未对齐")
    if np.any(lower > upper):
        raise ValueError("lower_quantile 不得高于 upper_quantile")
    if forecast_available_at < artifact.fit_end:
        raise ValueError("forecast_available_at 早于校准结束")
    if times[0] <= forecast_available_at:
        raise ValueError("forecast realization 必须晚于 forecast_available_at")
    calibrated_lower = lower - artifact.q_hat
    calibrated_upper = upper + artifact.q_hat
    if np.any(calibrated_lower > calibrated_upper):
        raise ValueError("CQR 校正后区间发生 crossing")
    return calibrated_lower, calibrated_upper


def rolling_cqr_backtest(
    lower_quantile: Sequence[float],
    upper_quantile: Sequence[float],
    realized_target: Sequence[float],
    timestamps: Sequence[int],
    *,
    spec: RollingCqrSpec,
) -> RollingCqrBacktestResult:
    spec.validate()
    lower = _finite_vector("lower_quantile", lower_quantile)
    upper = _finite_vector("upper_quantile", upper_quantile)
    target = _finite_vector("realized_target", realized_target)
    times = _timestamps(timestamps)
    if not (lower.size == upper.size == target.size == times.size):
        raise ValueError("rolling CQR 输入未对齐")
    if np.any(lower > upper):
        raise ValueError("lower_quantile 不得高于 upper_quantile")
    if lower.size <= spec.minimum_calibration_observations:
        raise ValueError("rolling CQR 没有可回测预测")

    calibrated_lower: list[float] = []
    calibrated_upper: list[float] = []
    q_hat_values: list[float] = []
    realization_timestamps: list[int] = []
    first = spec.minimum_calibration_observations
    for index in range(first, lower.size):
        start = max(0, index - spec.maximum_calibration_observations)
        artifact = fit_cqr_calibrator(
            lower[start:index], upper[start:index], target[start:index],
            times[start:index], available_at=int(times[index - 1]), spec=spec,
        )
        adjusted_lower, adjusted_upper = apply_cqr_calibrator(
            artifact, lower[index:index + 1], upper[index:index + 1],
            times[index:index + 1], forecast_available_at=int(times[index - 1]),
        )
        calibrated_lower.append(float(adjusted_lower[0]))
        calibrated_upper.append(float(adjusted_upper[0]))
        q_hat_values.append(artifact.q_hat)
        realization_timestamps.append(int(times[index]))

    evaluation_target = target[first:]
    lower_array = np.asarray(calibrated_lower)
    upper_array = np.asarray(calibrated_upper)
    covered = (evaluation_target >= lower_array) & (
        evaluation_target <= upper_array
    )
    lower_miss = evaluation_target < lower_array
    upper_miss = evaluation_target > upper_array
    input_payload = {
        "lower_quantile": lower.tolist(),
        "upper_quantile": upper.tolist(),
        "realized_target": target.tolist(),
        "timestamps": times.tolist(),
        "spec_sha256": spec.spec_sha256,
    }
    unsigned = {
        "schema_version": 1,
        "role": "phase3c_rolling_cqr_backtest",
        "status": "COMPLETE",
        "target_coverage": spec.target_coverage,
        "forecast_count": len(q_hat_values),
        "first_forecast_timestamp": realization_timestamps[0],
        "last_forecast_timestamp": realization_timestamps[-1],
        "empirical_coverage": float(covered.mean()),
        "lower_miss_rate": float(lower_miss.mean()),
        "upper_miss_rate": float(upper_miss.mean()),
        "mean_interval_width": float((upper_array - lower_array).mean()),
        "q_hat_values": tuple(q_hat_values),
        "calibrated_lower": tuple(calibrated_lower),
        "calibrated_upper": tuple(calibrated_upper),
        "realization_timestamps": tuple(realization_timestamps),
        "spec_sha256": spec.spec_sha256,
        "input_sha256": _sha256(input_payload),
        "formal_distribution_free_claim": False,
    }
    return RollingCqrBacktestResult(
        **unsigned, artifact_sha256=_sha256(unsigned),
    )


__all__ = [
    "CqrCalibrationArtifact",
    "RollingCqrBacktestResult",
    "RollingCqrSpec",
    "apply_cqr_calibrator",
    "fit_cqr_calibrator",
    "rolling_cqr_backtest",
]

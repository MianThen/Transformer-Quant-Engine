"""Deterministic, validation-only probability calibration.

The public fit functions consume probabilities from a frozen direction head.
Platt scaling is fitted on their logits, while isotonic regression uses a
weighted pool-adjacent-violators (PAV) fit and a deterministic linear knot
interpolation rule.  Both methods return the same auditable artifact shape.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np


class ProbabilityCalibrationValidationError(ValueError):
    """Raised when a probability calibration input fails closed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z") or "T" not in value:
        raise ProbabilityCalibrationValidationError(f"{name} 必须是 UTC 时间")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ProbabilityCalibrationValidationError(f"{name} 格式无效") from exc
    if parsed.tzinfo != timezone.utc:
        raise ProbabilityCalibrationValidationError(f"{name} 必须带 UTC 时区")
    return parsed


def _one_dimensional_floats(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProbabilityCalibrationValidationError(f"{name} 必须是数值向量") from exc
    if result.ndim != 1 or result.size == 0:
        raise ProbabilityCalibrationValidationError(f"{name} 必须是一维非空向量")
    if not np.all(np.isfinite(result)):
        raise ProbabilityCalibrationValidationError(f"{name} 必须全部为有限数值")
    return result


def _probabilities(value: Any, name: str) -> np.ndarray:
    result = _one_dimensional_floats(value, name)
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ProbabilityCalibrationValidationError(f"{name} 必须位于 [0, 1]")
    return result


def _labels(value: Any, name: str = "labels") -> np.ndarray:
    result = _one_dimensional_floats(value, name)
    if not np.all((result == 0.0) | (result == 1.0)):
        raise ProbabilityCalibrationValidationError(f"{name} 必须是二元标签 0/1")
    return result


def _weights(value: Any, size: int) -> np.ndarray:
    if value is None:
        return np.ones(size, dtype=np.float64)
    result = _one_dimensional_floats(value, "sample_weight")
    if result.size != size:
        raise ProbabilityCalibrationValidationError("sample_weight 长度不一致")
    if np.any(result <= 0.0):
        raise ProbabilityCalibrationValidationError("sample_weight 必须严格为正")
    return result


def _array_hash(values: np.ndarray) -> str:
    return _sha256([float(value) for value in values])


def _effective_sample_size(weights: np.ndarray) -> float:
    weight_sum = float(np.sum(weights))
    return weight_sum * weight_sum / float(np.dot(weights, weights))


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    result[~nonnegative] = exponent / (1.0 + exponent)
    return result


def _logit(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


@dataclass(frozen=True)
class CalibrationMetricsV1:
    """Weighted binary-probability metrics and reliability bins."""

    sample_count: int
    weight_sum: float
    effective_sample_count: float
    brier: float
    nll: float
    ece: float
    ece_bin_count: int
    reliability_bins: tuple[Mapping[str, Any], ...]

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "weight_sum": self.weight_sum,
            "effective_sample_count": self.effective_sample_count,
            "brier": self.brier,
            "nll": self.nll,
            "ece": self.ece,
            "ece_bin_count": self.ece_bin_count,
            "reliability_bins": [dict(item) for item in self.reliability_bins],
        }

    @property
    def metrics_sha256(self) -> str:
        return _sha256(self._unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        result = self._unsigned_dict()
        result["metrics_sha256"] = self.metrics_sha256
        return result


def calibration_metrics(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
    *,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    ece_bins: int = 10,
    nll_epsilon: float = 1e-12,
) -> CalibrationMetricsV1:
    """Compute deterministic weighted Brier, NLL, ECE, and reliability bins."""

    predicted = _probabilities(probabilities, "probabilities")
    observed = _labels(labels)
    if predicted.size != observed.size:
        raise ProbabilityCalibrationValidationError("probabilities 与 labels 长度不一致")
    weights = _weights(sample_weight, predicted.size)
    if isinstance(ece_bins, bool) or not isinstance(ece_bins, int) or ece_bins < 2:
        raise ProbabilityCalibrationValidationError("ece_bins 必须是至少为 2 的整数")
    if not isinstance(nll_epsilon, (float, int)) or isinstance(nll_epsilon, bool):
        raise ProbabilityCalibrationValidationError("nll_epsilon 无效")
    nll_epsilon = float(nll_epsilon)
    if not math.isfinite(nll_epsilon) or not 0.0 < nll_epsilon < 0.5:
        raise ProbabilityCalibrationValidationError("nll_epsilon 必须位于 (0, 0.5)")

    weight_sum = float(np.sum(weights))
    brier = float(np.dot(weights, np.square(predicted - observed)) / weight_sum)
    clipped = np.clip(predicted, nll_epsilon, 1.0 - nll_epsilon)
    nll_terms = -(observed * np.log(clipped) + (1.0 - observed) * np.log1p(-clipped))
    nll = float(np.dot(weights, nll_terms) / weight_sum)

    bin_indices = np.minimum((predicted * ece_bins).astype(np.int64), ece_bins - 1)
    reliability: list[Mapping[str, Any]] = []
    ece = 0.0
    for bin_index in range(ece_bins):
        selected = bin_indices == bin_index
        count = int(np.count_nonzero(selected))
        lower = float(bin_index / ece_bins)
        upper = float((bin_index + 1) / ece_bins)
        if count == 0:
            reliability.append(
                {
                    "bin_index": bin_index,
                    "lower": lower,
                    "upper": upper,
                    "sample_count": 0,
                    "weight_sum": 0.0,
                    "mean_probability": None,
                    "positive_rate": None,
                    "absolute_gap": None,
                }
            )
            continue
        selected_weights = weights[selected]
        selected_weight_sum = float(np.sum(selected_weights))
        mean_probability = float(
            np.dot(selected_weights, predicted[selected]) / selected_weight_sum
        )
        positive_rate = float(
            np.dot(selected_weights, observed[selected]) / selected_weight_sum
        )
        gap = abs(mean_probability - positive_rate)
        ece += selected_weight_sum * gap / weight_sum
        reliability.append(
            {
                "bin_index": bin_index,
                "lower": lower,
                "upper": upper,
                "sample_count": count,
                "weight_sum": selected_weight_sum,
                "mean_probability": mean_probability,
                "positive_rate": positive_rate,
                "absolute_gap": gap,
            }
        )

    return CalibrationMetricsV1(
        sample_count=int(predicted.size),
        weight_sum=weight_sum,
        effective_sample_count=_effective_sample_size(weights),
        brier=brier,
        nll=nll,
        ece=float(ece),
        ece_bin_count=ece_bins,
        reliability_bins=tuple(reliability),
    )


@dataclass(frozen=True)
class ProbabilityCalibrationArtifactV1:
    """Frozen calibration fit with provenance, metrics, and prediction guard."""

    method: str
    split_role: str
    fit_start_utc: str
    fit_end_utc: str
    available_at_utc: str
    sample_count: int
    weight_sum: float
    effective_sample_count: float
    input_probability_min: float
    input_probability_max: float
    platt_slope: float | None
    platt_intercept: float | None
    knot_scores: tuple[float, ...]
    knot_probabilities: tuple[float, ...]
    pav_block_count: int | None
    uncalibrated_metrics: CalibrationMetricsV1
    calibrated_metrics: CalibrationMetricsV1
    source_probabilities_sha256: str
    source_labels_sha256: str
    sample_weight_sha256: str

    @property
    def fit_range(self) -> tuple[str, str]:
        return self.fit_start_utc, self.fit_end_utc

    @property
    def input_range(self) -> tuple[float, float]:
        return self.input_probability_min, self.input_probability_max

    @property
    def knot_table(self) -> tuple[dict[str, float], ...]:
        return tuple(
            {"input_probability": score, "calibrated_probability": probability}
            for score, probability in zip(self.knot_scores, self.knot_probabilities)
        )

    def _model_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": self.method,
            "input_transform": "LOGIT_CLIPPED" if self.method == "PLATT_LOGIT_V1" else "IDENTITY",
            "boundary_rule": "CLIP_TO_FIT_ENDPOINTS",
            "interpolation": "SIGMOID" if self.method == "PLATT_LOGIT_V1" else "LINEAR_KNOTS",
            "platt_slope": self.platt_slope,
            "platt_intercept": self.platt_intercept,
            "pav_block_count": self.pav_block_count,
            "knot_table": list(self.knot_table),
        }

    @property
    def model_sha256(self) -> str:
        return _sha256(self._model_dict())

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": self.method,
            "split_role": self.split_role,
            "fit_range": {
                "start_utc": self.fit_start_utc,
                "end_utc": self.fit_end_utc,
            },
            "available_at_utc": self.available_at_utc,
            "sample_count": self.sample_count,
            "weight_sum": self.weight_sum,
            "effective_sample_count": self.effective_sample_count,
            "input_probability_range": [
                self.input_probability_min,
                self.input_probability_max,
            ],
            "model": self._model_dict(),
            "uncalibrated_metrics": self.uncalibrated_metrics.to_dict(),
            "calibrated_metrics": self.calibrated_metrics.to_dict(),
            "source_probabilities_sha256": self.source_probabilities_sha256,
            "source_labels_sha256": self.source_labels_sha256,
            "sample_weight_sha256": self.sample_weight_sha256,
            "model_sha256": self.model_sha256,
        }

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self._unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        result = self._unsigned_dict()
        result["artifact_sha256"] = self.artifact_sha256
        return result

    def predict(
        self,
        probabilities: Sequence[float] | np.ndarray,
        *,
        decision_at_utc: str,
    ) -> np.ndarray:
        """Apply the frozen map only at or after its declared availability."""

        decision_at = _utc_datetime(decision_at_utc, "decision_at_utc")
        available_at = _utc_datetime(self.available_at_utc, "artifact available_at_utc")
        if decision_at < available_at:
            raise ProbabilityCalibrationValidationError(
                "校准 artifact 在 decision_at 尚不可用"
            )
        values = _probabilities(probabilities, "probabilities")
        if self.method == "PLATT_LOGIT_V1":
            if self.platt_slope is None or self.platt_intercept is None:
                raise ProbabilityCalibrationValidationError("Platt 参数缺失")
            transformed = _logit(values, 1e-12)
            return _stable_sigmoid(
                self.platt_slope * transformed + self.platt_intercept
            )
        if self.method == "ISOTONIC_PAV_LINEAR_V1":
            if len(self.knot_scores) < 2 or len(self.knot_scores) != len(
                self.knot_probabilities
            ):
                raise ProbabilityCalibrationValidationError("Isotonic knot table 无效")
            return np.interp(
                values,
                np.asarray(self.knot_scores, dtype=np.float64),
                np.asarray(self.knot_probabilities, dtype=np.float64),
                left=self.knot_probabilities[0],
                right=self.knot_probabilities[-1],
            )
        raise ProbabilityCalibrationValidationError("未知校准方法")


def _validate_fit_contract(
    probabilities: Any,
    labels: Any,
    sample_weight: Any,
    *,
    split_role: str,
    fit_start_utc: str,
    fit_end_utc: str,
    available_at_utc: str,
    observation_at_utc: Sequence[str] | None,
    label_available_at_utc: Sequence[str] | None,
    minimum_sample_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_role != "validation":
        raise ProbabilityCalibrationValidationError(
            "概率校准只能在 validation split 拟合"
        )
    fit_start = _utc_datetime(fit_start_utc, "fit_start_utc")
    fit_end = _utc_datetime(fit_end_utc, "fit_end_utc")
    available_at = _utc_datetime(available_at_utc, "available_at_utc")
    if fit_start > fit_end:
        raise ProbabilityCalibrationValidationError("fit range 起止顺序无效")
    if fit_end > available_at:
        raise ProbabilityCalibrationValidationError("fit_end_utc 晚于 available_at_utc")
    if (
        isinstance(minimum_sample_count, bool)
        or not isinstance(minimum_sample_count, int)
        or minimum_sample_count < 2
    ):
        raise ProbabilityCalibrationValidationError("minimum_sample_count 必须至少为 2")

    predicted = _probabilities(probabilities, "probabilities")
    observed = _labels(labels)
    if predicted.size != observed.size:
        raise ProbabilityCalibrationValidationError("probabilities 与 labels 长度不一致")
    weights = _weights(sample_weight, predicted.size)
    if predicted.size < minimum_sample_count:
        raise ProbabilityCalibrationValidationError("validation 样本量不足")
    if _effective_sample_size(weights) + 1e-12 < minimum_sample_count:
        raise ProbabilityCalibrationValidationError("validation 有效加权样本量不足")
    if np.unique(predicted).size < 2 or float(np.ptp(predicted)) <= 1e-12:
        raise ProbabilityCalibrationValidationError("输入概率退化")
    if np.unique(observed).size != 2:
        raise ProbabilityCalibrationValidationError("validation 标签必须同时包含 0 和 1")

    if observation_at_utc is not None:
        if isinstance(observation_at_utc, (str, bytes)) or len(observation_at_utc) != predicted.size:
            raise ProbabilityCalibrationValidationError("observation_at_utc 长度不一致")
        for index, timestamp in enumerate(observation_at_utc):
            observed_at = _utc_datetime(timestamp, f"observation_at_utc[{index}]")
            if observed_at < fit_start or observed_at > fit_end:
                raise ProbabilityCalibrationValidationError(
                    "fit 输入包含窗口外或未来 observation"
                )

    if label_available_at_utc is not None:
        if isinstance(label_available_at_utc, (str, bytes)) or len(label_available_at_utc) != predicted.size:
            raise ProbabilityCalibrationValidationError("label_available_at_utc 长度不一致")
        parsed_label_times: list[datetime] = []
        for index, timestamp in enumerate(label_available_at_utc):
            label_time = _utc_datetime(timestamp, f"label_available_at_utc[{index}]")
            if label_time > available_at:
                raise ProbabilityCalibrationValidationError(
                    "fit 输入包含在 artifact available_at 后才成熟的标签"
                )
            parsed_label_times.append(label_time)
        if observation_at_utc is not None:
            for index, timestamp in enumerate(observation_at_utc):
                observation_time = _utc_datetime(
                    timestamp, f"observation_at_utc[{index}]"
                )
                if parsed_label_times[index] < observation_time:
                    raise ProbabilityCalibrationValidationError(
                        "label_available_at_utc 早于 observation_at_utc"
                    )
    return predicted, observed, weights


def _artifact(
    *,
    method: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    calibrated: np.ndarray,
    split_role: str,
    fit_start_utc: str,
    fit_end_utc: str,
    available_at_utc: str,
    ece_bins: int,
    platt_slope: float | None = None,
    platt_intercept: float | None = None,
    knot_scores: Sequence[float] = (),
    knot_probabilities: Sequence[float] = (),
    pav_block_count: int | None = None,
) -> ProbabilityCalibrationArtifactV1:
    return ProbabilityCalibrationArtifactV1(
        method=method,
        split_role=split_role,
        fit_start_utc=fit_start_utc,
        fit_end_utc=fit_end_utc,
        available_at_utc=available_at_utc,
        sample_count=int(probabilities.size),
        weight_sum=float(np.sum(weights)),
        effective_sample_count=_effective_sample_size(weights),
        input_probability_min=float(np.min(probabilities)),
        input_probability_max=float(np.max(probabilities)),
        platt_slope=platt_slope,
        platt_intercept=platt_intercept,
        knot_scores=tuple(float(value) for value in knot_scores),
        knot_probabilities=tuple(float(value) for value in knot_probabilities),
        pav_block_count=pav_block_count,
        uncalibrated_metrics=calibration_metrics(
            probabilities, labels, sample_weight=weights, ece_bins=ece_bins
        ),
        calibrated_metrics=calibration_metrics(
            calibrated, labels, sample_weight=weights, ece_bins=ece_bins
        ),
        source_probabilities_sha256=_array_hash(probabilities),
        source_labels_sha256=_array_hash(labels),
        sample_weight_sha256=_array_hash(weights),
    )


def fit_weighted_platt(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
    *,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    split_role: str,
    fit_start_utc: str,
    fit_end_utc: str,
    available_at_utc: str,
    observation_at_utc: Sequence[str] | None = None,
    label_available_at_utc: Sequence[str] | None = None,
    minimum_sample_count: int = 20,
    ece_bins: int = 10,
    l2_strength: float = 1e-6,
    maximum_iterations: int = 100,
    tolerance: float = 1e-10,
) -> ProbabilityCalibrationArtifactV1:
    """Fit deterministic weighted Platt scaling on input-probability logits."""

    predicted, observed, weights = _validate_fit_contract(
        probabilities,
        labels,
        sample_weight,
        split_role=split_role,
        fit_start_utc=fit_start_utc,
        fit_end_utc=fit_end_utc,
        available_at_utc=available_at_utc,
        observation_at_utc=observation_at_utc,
        label_available_at_utc=label_available_at_utc,
        minimum_sample_count=minimum_sample_count,
    )
    if not isinstance(l2_strength, (float, int)) or isinstance(l2_strength, bool):
        raise ProbabilityCalibrationValidationError("l2_strength 无效")
    l2_strength = float(l2_strength)
    if not math.isfinite(l2_strength) or l2_strength <= 0.0:
        raise ProbabilityCalibrationValidationError("l2_strength 必须严格为正")
    if (
        isinstance(maximum_iterations, bool)
        or not isinstance(maximum_iterations, int)
        or maximum_iterations < 1
    ):
        raise ProbabilityCalibrationValidationError("maximum_iterations 无效")
    if not isinstance(tolerance, (float, int)) or isinstance(tolerance, bool):
        raise ProbabilityCalibrationValidationError("tolerance 无效")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ProbabilityCalibrationValidationError("tolerance 必须严格为正")

    logits = _logit(predicted, 1e-12)
    weight_sum = float(np.sum(weights))
    center = float(np.dot(weights, logits) / weight_sum)
    centered = logits - center
    scale = math.sqrt(float(np.dot(weights, centered * centered) / weight_sum))
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ProbabilityCalibrationValidationError("Platt logit 输入退化")
    normalized = centered / scale
    weighted_positive_rate = float(np.dot(weights, observed) / weight_sum)
    beta = np.asarray(
        [math.log(weighted_positive_rate / (1.0 - weighted_positive_rate)), 0.0],
        dtype=np.float64,
    )

    def objective(parameters: np.ndarray) -> float:
        linear = parameters[0] + parameters[1] * normalized
        logistic_loss = np.logaddexp(0.0, linear) - observed * linear
        return float(
            np.dot(weights, logistic_loss) / weight_sum
            + 0.5 * l2_strength * parameters[1] * parameters[1]
        )

    converged = False
    current_objective = objective(beta)
    for _ in range(maximum_iterations):
        linear = beta[0] + beta[1] * normalized
        fitted = _stable_sigmoid(linear)
        residual = fitted - observed
        curvature = fitted * (1.0 - fitted)
        gradient = np.asarray(
            [
                np.dot(weights, residual) / weight_sum,
                np.dot(weights, residual * normalized) / weight_sum
                + l2_strength * beta[1],
            ],
            dtype=np.float64,
        )
        if float(np.max(np.abs(gradient))) <= tolerance:
            converged = True
            break
        hessian = np.asarray(
            [
                [
                    np.dot(weights, curvature) / weight_sum,
                    np.dot(weights, curvature * normalized) / weight_sum,
                ],
                [
                    np.dot(weights, curvature * normalized) / weight_sum,
                    np.dot(weights, curvature * normalized * normalized) / weight_sum
                    + l2_strength,
                ],
            ],
            dtype=np.float64,
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ProbabilityCalibrationValidationError("Platt Hessian 奇异") from exc
        accepted = False
        step_scale = 1.0
        directional = float(np.dot(gradient, step))
        while step_scale >= 2.0**-40:
            candidate = beta - step_scale * step
            candidate_objective = objective(candidate)
            if (
                math.isfinite(candidate_objective)
                and candidate_objective
                <= current_objective - 1e-4 * step_scale * directional
            ):
                beta = candidate
                current_objective = candidate_objective
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            raise ProbabilityCalibrationValidationError("Platt line search 未收敛")
        if float(np.linalg.norm(step_scale * step, ord=np.inf)) <= tolerance * (
            1.0 + float(np.linalg.norm(beta, ord=np.inf))
        ):
            converged = True
            break
    if not converged:
        raise ProbabilityCalibrationValidationError("Platt 拟合未收敛")

    slope = float(beta[1] / scale)
    intercept = float(beta[0] - beta[1] * center / scale)
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 1e-12:
        raise ProbabilityCalibrationValidationError("Platt 映射退化或反向")
    calibrated = _stable_sigmoid(slope * logits + intercept)
    return _artifact(
        method="PLATT_LOGIT_V1",
        probabilities=predicted,
        labels=observed,
        weights=weights,
        calibrated=calibrated,
        split_role=split_role,
        fit_start_utc=fit_start_utc,
        fit_end_utc=fit_end_utc,
        available_at_utc=available_at_utc,
        ece_bins=ece_bins,
        platt_slope=slope,
        platt_intercept=intercept,
    )


def _weighted_pav(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    sorted_labels = labels[order]
    sorted_weights = weights[order]
    unique_scores, first_indices = np.unique(sorted_probabilities, return_index=True)
    unique_weights = np.add.reduceat(sorted_weights, first_indices)
    unique_positive_weight = np.add.reduceat(
        sorted_weights * sorted_labels, first_indices
    )

    blocks: list[list[float | int]] = []
    for index in range(unique_scores.size):
        block: list[float | int] = [
            index,
            index,
            float(unique_weights[index]),
            float(unique_positive_weight[index]),
        ]
        blocks.append(block)
        while len(blocks) >= 2:
            previous = blocks[-2]
            current = blocks[-1]
            previous_mean = float(previous[3]) / float(previous[2])
            current_mean = float(current[3]) / float(current[2])
            if previous_mean < current_mean:
                break
            blocks[-2:] = [
                [
                    int(previous[0]),
                    int(current[1]),
                    float(previous[2]) + float(current[2]),
                    float(previous[3]) + float(current[3]),
                ]
            ]

    fitted_unique = np.empty(unique_scores.size, dtype=np.float64)
    for start, end, block_weight, block_positive_weight in blocks:
        fitted_unique[int(start) : int(end) + 1] = float(block_positive_weight) / float(
            block_weight
        )
    return unique_scores, fitted_unique, len(blocks)


def _compress_linear_knots(
    scores: np.ndarray, fitted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if scores.size <= 2:
        return scores.copy(), fitted.copy()
    keep = np.zeros(scores.size, dtype=bool)
    keep[0] = True
    keep[-1] = True
    for index in range(1, scores.size - 1):
        keep[index] = not (
            fitted[index - 1] == fitted[index] == fitted[index + 1]
        )
    return scores[keep], fitted[keep]


def fit_weighted_isotonic(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[float] | np.ndarray,
    *,
    sample_weight: Sequence[float] | np.ndarray | None = None,
    split_role: str,
    fit_start_utc: str,
    fit_end_utc: str,
    available_at_utc: str,
    observation_at_utc: Sequence[str] | None = None,
    label_available_at_utc: Sequence[str] | None = None,
    minimum_sample_count: int = 20,
    minimum_block_count: int = 2,
    maximum_block_count: int | None = None,
    ece_bins: int = 10,
) -> ProbabilityCalibrationArtifactV1:
    """Fit weighted PAV isotonic calibration and emit a compressed knot table."""

    predicted, observed, weights = _validate_fit_contract(
        probabilities,
        labels,
        sample_weight,
        split_role=split_role,
        fit_start_utc=fit_start_utc,
        fit_end_utc=fit_end_utc,
        available_at_utc=available_at_utc,
        observation_at_utc=observation_at_utc,
        label_available_at_utc=label_available_at_utc,
        minimum_sample_count=minimum_sample_count,
    )
    if (
        isinstance(minimum_block_count, bool)
        or not isinstance(minimum_block_count, int)
        or minimum_block_count < 2
    ):
        raise ProbabilityCalibrationValidationError("minimum_block_count 必须至少为 2")
    if maximum_block_count is not None and (
        isinstance(maximum_block_count, bool)
        or not isinstance(maximum_block_count, int)
        or maximum_block_count < minimum_block_count
    ):
        raise ProbabilityCalibrationValidationError("maximum_block_count 无效")

    unique_scores, fitted_unique, block_count = _weighted_pav(
        predicted, observed, weights
    )
    distinct_levels = int(np.unique(fitted_unique).size)
    if block_count < minimum_block_count or distinct_levels < 2:
        raise ProbabilityCalibrationValidationError("Isotonic PAV 映射退化")
    if maximum_block_count is not None and block_count > maximum_block_count:
        raise ProbabilityCalibrationValidationError("Isotonic PAV 台阶过多")
    knot_scores, knot_probabilities = _compress_linear_knots(
        unique_scores, fitted_unique
    )
    if knot_scores.size < 2 or np.any(np.diff(knot_probabilities) < 0.0):
        raise ProbabilityCalibrationValidationError("Isotonic knot table 无效")
    calibrated = np.interp(
        predicted,
        knot_scores,
        knot_probabilities,
        left=float(knot_probabilities[0]),
        right=float(knot_probabilities[-1]),
    )
    return _artifact(
        method="ISOTONIC_PAV_LINEAR_V1",
        probabilities=predicted,
        labels=observed,
        weights=weights,
        calibrated=calibrated,
        split_role=split_role,
        fit_start_utc=fit_start_utc,
        fit_end_utc=fit_end_utc,
        available_at_utc=available_at_utc,
        ece_bins=ece_bins,
        knot_scores=knot_scores,
        knot_probabilities=knot_probabilities,
        pav_block_count=block_count,
    )


fit_weighted_pav_isotonic = fit_weighted_isotonic


__all__ = [
    "CalibrationMetricsV1",
    "ProbabilityCalibrationArtifactV1",
    "ProbabilityCalibrationValidationError",
    "calibration_metrics",
    "fit_weighted_isotonic",
    "fit_weighted_pav_isotonic",
    "fit_weighted_platt",
]

"""独立的 differentiable top-k 前向与梯度参考 oracle。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


class TopKStabilityOracleError(ValueError):
    """Raised when a top-k stability oracle input is invalid."""


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _validate_scores(scores: Mapping[str, Any], name: str) -> dict[str, float]:
    if not isinstance(scores, Mapping) or not scores:
        raise TopKStabilityOracleError(f"{name} 必须是非空 symbol->score 映射")
    normalized: dict[str, float] = {}
    for symbol, value in scores.items():
        if not isinstance(symbol, str) or not symbol:
            raise TopKStabilityOracleError(f"{name} 的 symbol 必须是非空字符串")
        if not _finite(value):
            raise TopKStabilityOracleError(f"{name}[{symbol}] 必须是有限数值")
        normalized[symbol] = float(value)
    if len(normalized) != len(scores):
        raise TopKStabilityOracleError(f"{name} 的 symbol 不得重复")
    return normalized


def _validate_top_k(top_k: int, size: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= size:
        raise TopKStabilityOracleError("top_k 必须满足 1 <= top_k <= universe size")


def _validate_temperature(temperature: float) -> float:
    if not _finite(temperature) or float(temperature) <= 0.0:
        raise TopKStabilityOracleError("temperature 必须为正有限数值")
    return float(temperature)


def softsort_topk_weights(
    scores: Mapping[str, Any],
    top_k: int,
    temperature: float,
) -> dict[str, float]:
    """Return normalized SoftSort top-k masses for a score cross-section.

    For sorted values ``v_j`` and raw scores ``s_i``, the reference matrix is
    ``P[j,i] = softmax_i(-abs(v_j - s_i) / temperature)``. The first ``top_k``
    rows are summed and divided by ``top_k``. Symbol ordering only resolves
    exact forward ties; non-tie fixtures are used for finite-difference checks.
    """
    normalized = _validate_scores(scores, "scores")
    _validate_top_k(top_k, len(normalized))
    temperature_value = _validate_temperature(temperature)
    symbols = tuple(sorted(normalized))
    ordered_values = tuple(sorted(
        (normalized[symbol] for symbol in symbols),
        reverse=True,
    ))
    masses = {symbol: 0.0 for symbol in symbols}
    for target_value in ordered_values[:top_k]:
        logits = {
            symbol: -abs(target_value - normalized[symbol]) / temperature_value
            for symbol in symbols
        }
        maximum = max(logits.values())
        exponentials = {
            symbol: math.exp(logit - maximum) for symbol, logit in logits.items()
        }
        denominator = sum(exponentials.values())
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise TopKStabilityOracleError("SoftSort normalization 失败")
        for symbol in symbols:
            masses[symbol] += exponentials[symbol] / denominator
    return {symbol: masses[symbol] / top_k for symbol in symbols}


def temporal_topk_stability_penalty(
    current_scores: Mapping[str, Any],
    previous_scores: Mapping[str, Any],
    top_k: int,
    temperature: float,
) -> float:
    """Return mean squared distance between consecutive normalized top-k masses."""
    current = _validate_scores(current_scores, "current_scores")
    previous = _validate_scores(previous_scores, "previous_scores")
    if set(current) != set(previous):
        raise TopKStabilityOracleError(
            "temporal stability 要求相邻截面使用相同的 symbol intersection"
        )
    current_weights = softsort_topk_weights(current, top_k, temperature)
    previous_weights = softsort_topk_weights(previous, top_k, temperature)
    penalty = sum(
        (current_weights[symbol] - previous_weights[symbol]) ** 2
        for symbol in sorted(current)
    ) / len(current)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise TopKStabilityOracleError("temporal stability penalty 无效")
    return penalty


def finite_difference_temporal_gradient(
    current_scores: Mapping[str, Any],
    previous_scores: Mapping[str, Any],
    top_k: int,
    temperature: float,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Return a central finite-difference gradient for the current scores."""
    current = _validate_scores(current_scores, "current_scores")
    _validate_scores(previous_scores, "previous_scores")
    if not _finite(epsilon) or float(epsilon) <= 0.0:
        raise TopKStabilityOracleError("epsilon 必须为正有限数值")
    epsilon_value = float(epsilon)
    gradient: dict[str, float] = {}
    for symbol in sorted(current):
        plus = dict(current)
        minus = dict(current)
        plus[symbol] += epsilon_value
        minus[symbol] -= epsilon_value
        plus_value = temporal_topk_stability_penalty(
            plus, previous_scores, top_k, temperature
        )
        minus_value = temporal_topk_stability_penalty(
            minus, previous_scores, top_k, temperature
        )
        gradient[symbol] = (plus_value - minus_value) / (2.0 * epsilon_value)
    return gradient


__all__ = [
    "TopKStabilityOracleError",
    "finite_difference_temporal_gradient",
    "softsort_topk_weights",
    "temporal_topk_stability_penalty",
]

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _validate_mapping(
    scores: Mapping[str, float], top_k: int, temperature: float
) -> tuple[list[str], np.ndarray, int, float]:
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("scores 必须是非空 mapping")
    if top_k < 1 or top_k > len(scores):
        raise ValueError("top_k 必须在 [1, len(scores)] 内")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature 必须是有限正数")
    symbols = sorted(str(symbol) for symbol in scores)
    values = np.asarray([float(scores[symbol]) for symbol in symbols], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("scores 必须全部有限")
    return symbols, values, int(top_k), float(temperature)


def _weights_from_values(values: np.ndarray, top_k: int, temperature: float) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    logits = -np.abs(sorted_values[:, None] - values[None, :]) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:top_k].mean(axis=0)


def softsort_topk_weights(
    scores: Mapping[str, float], top_k: int, temperature: float
) -> dict[str, float]:
    symbols, values, top_k, temperature = _validate_mapping(scores, top_k, temperature)
    weights = _weights_from_values(values, top_k, temperature)
    return {symbol: float(weight) for symbol, weight in zip(symbols, weights)}


def temporal_topk_stability_penalty(
    current: Mapping[str, float],
    previous: Mapping[str, float],
    top_k: int,
    temperature: float,
) -> float:
    current_weights = softsort_topk_weights(current, top_k, temperature)
    previous_weights = softsort_topk_weights(previous, top_k, temperature)
    common = sorted(set(current_weights) & set(previous_weights))
    if not common:
        return 0.0
    return float(np.mean([
        (current_weights[symbol] - previous_weights[symbol]) ** 2
        for symbol in common
    ]))


def finite_difference_temporal_gradient(
    current: Mapping[str, float],
    previous: Mapping[str, float],
    top_k: int,
    temperature: float,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon 必须是有限正数")
    symbols, _, _, _ = _validate_mapping(current, top_k, temperature)
    gradient: dict[str, float] = {}
    for symbol in symbols:
        plus = {key: float(value) for key, value in current.items()}
        minus = dict(plus)
        plus[symbol] += epsilon
        minus[symbol] -= epsilon
        gradient[symbol] = (
            temporal_topk_stability_penalty(plus, previous, top_k, temperature)
            - temporal_topk_stability_penalty(minus, previous, top_k, temperature)
        ) / (2.0 * epsilon)
    return gradient


def torch_softsort_topk_weights(scores, top_k: int, temperature: float):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError("torch_softsort_topk_weights 需要安装 PyTorch") from exc
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores 必须是一维非空 tensor")
    if top_k < 1 or top_k > scores.numel():
        raise ValueError("top_k 必须在 [1, scores.numel()] 内")
    if not torch.isfinite(scores).all() or not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("scores/temperature 必须有限且 temperature 为正")
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    logits = -(sorted_scores[:, None] - scores[None, :]).abs() / float(temperature)
    return torch.softmax(logits, dim=1)[:top_k].mean(dim=0)


def torch_temporal_topk_stability_loss(
    current_scores,
    current_symbols: Sequence[str],
    previous_scores,
    previous_symbols: Sequence[str],
    top_k: int,
    temperature: float,
):
    current_weights = torch_softsort_topk_weights(current_scores, top_k, temperature)
    with __import__("torch").no_grad():
        previous_weights = torch_softsort_topk_weights(
            previous_scores, top_k, temperature
        )
    previous_by_symbol = {
        str(symbol): previous_weights[index]
        for index, symbol in enumerate(previous_symbols)
    }
    current_indices = [
        index for index, symbol in enumerate(current_symbols)
        if str(symbol) in previous_by_symbol
    ]
    if not current_indices:
        return current_scores.sum() * 0.0
    target = __import__("torch").stack([
        previous_by_symbol[str(current_symbols[index])]
        for index in current_indices
    ])
    return __import__("torch").mean((current_weights[current_indices] - target) ** 2)

from __future__ import annotations

import numpy as np


def _pearson(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(np.square(left).sum() * np.square(right).sum())
    return 0.0 if denominator <= 1e-15 else float((left * right).sum() / denominator)


def _rank(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    order = np.argsort(value, kind="stable")
    ranks = np.empty(value.size, dtype=np.float64)
    start = 0
    while start < value.size:
        end = start + 1
        while end < value.size and value[order[end]] == value[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _auc(target, probability) -> float:
    target = np.asarray(target, dtype=np.float64)
    positives = target == 1.0
    negative_count = int((~positives).sum())
    positive_count = int(positives.sum())
    if not positive_count or not negative_count:
        return 0.5
    rank_sum = _rank(probability)[positives].sum() + positive_count
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _ece(target, probability, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(target)
    error = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability < edges[index + 1] if index + 1 < bins else probability <= 1.0
        )
        if selected.any():
            error += selected.sum() / total * abs(
                float(probability[selected].mean() - target[selected].mean())
            )
    return float(error)


def _pinball(prediction, target, quantile: float) -> float:
    error = np.asarray(target) - np.asarray(prediction)
    return float(np.maximum(quantile * error, (quantile - 1.0) * error).mean())


def prediction_metrics(
    *,
    expected_return,
    realized_return,
    expected_volatility,
    realized_volatility,
    direction_probability,
    direction,
    lower_quantile,
    upper_quantile,
    timestamps,
) -> dict[str, float]:
    mu = np.asarray(expected_return, dtype=np.float64)
    actual = np.asarray(realized_return, dtype=np.float64)
    sigma = np.asarray(expected_volatility, dtype=np.float64)
    actual_sigma = np.asarray(realized_volatility, dtype=np.float64)
    probability = np.clip(np.asarray(direction_probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    binary = np.asarray(direction, dtype=np.float64)
    lower = np.asarray(lower_quantile, dtype=np.float64)
    upper = np.asarray(upper_quantile, dtype=np.float64)
    timestamps = np.asarray(timestamps)
    values = (mu, actual, sigma, actual_sigma, probability, binary, lower, upper)
    if not values[0].size or any(value.shape != values[0].shape for value in values):
        raise ValueError("预测指标输入必须为 shape 相同的非空一维数组")
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("预测指标不接受 NaN/Inf")

    rank_ics = []
    for timestamp in np.unique(timestamps):
        selected = timestamps == timestamp
        if selected.sum() >= 2:
            rank_ics.append(_pearson(_rank(mu[selected]), _rank(actual[selected])))
    residual = mu - actual
    logloss = -np.mean(binary * np.log(probability) + (1.0 - binary) * np.log(1.0 - probability))
    return {
        "return_mae": float(np.abs(residual).mean()),
        "return_rmse": float(np.sqrt(np.square(residual).mean())),
        "return_huber": float(np.where(
            np.abs(residual) <= 1.0,
            0.5 * np.square(residual),
            np.abs(residual) - 0.5,
        ).mean()),
        "ic": _pearson(mu, actual),
        "rank_ic": float(np.mean(rank_ics)) if rank_ics else 0.0,
        "volatility_mae": float(np.abs(sigma - actual_sigma).mean()),
        "volatility_rmse": float(np.sqrt(np.square(sigma - actual_sigma).mean())),
        "direction_logloss": float(logloss),
        "direction_brier": float(np.square(probability - binary).mean()),
        "direction_auc": _auc(binary, probability),
        "direction_ece": _ece(binary, probability),
        "q10_pinball": _pinball(lower, actual, 0.10),
        "q90_pinball": _pinball(upper, actual, 0.90),
        "interval_coverage": float(((actual >= lower) & (actual <= upper)).mean()),
    }

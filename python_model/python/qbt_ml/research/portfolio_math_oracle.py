from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ledoit_wolf_linear_constant_correlation(
    returns: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Independent long-double transcription of the constant-correlation estimator."""
    values = np.asarray(returns, dtype=np.longdouble)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("returns must be a non-empty 2D matrix with at least two rows")
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite")
    centered = values - values.mean(axis=0)
    observations, assets = centered.shape
    sample = centered.T @ centered / observations
    if assets == 1:
        return np.asarray(sample, dtype=np.float64), 0.0

    variances = np.diag(sample)
    standard_deviations = np.sqrt(variances)
    correlations = [
        sample[row, col] / (standard_deviations[row] * standard_deviations[col])
        for row in range(assets)
        for col in range(row + 1, assets)
        if standard_deviations[row] * standard_deviations[col] > 0
    ]
    average_correlation = np.mean(correlations, dtype=np.longdouble) if correlations else 0
    target = average_correlation * np.outer(standard_deviations, standard_deviations)
    np.fill_diagonal(target, variances)

    squared = centered**2
    phi_matrix = squared.T @ squared / observations - sample**2
    phi = phi_matrix.sum()
    theta = (centered**3).T @ centered / observations - variances[:, None] * sample
    rho = np.trace(phi_matrix)
    for row in range(assets):
        if standard_deviations[row] == 0:
            continue
        for col in range(assets):
            if row != col:
                rho += (
                    average_correlation
                    * standard_deviations[col]
                    / standard_deviations[row]
                    * theta[row, col]
                )
    gamma = np.sum((sample - target) ** 2)
    shrinkage = np.longdouble(1) if gamma <= np.finfo(np.longdouble).eps else np.clip(
        (phi - rho) / (gamma * observations), 0, 1
    )
    covariance = shrinkage * target + (1 - shrinkage) * sample
    return np.asarray(covariance, dtype=np.float64), float(shrinkage)


def diagonal_risk_budget(variances: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    variances = np.asarray(variances, dtype=np.longdouble)
    budgets = np.asarray(budgets, dtype=np.longdouble)
    if variances.ndim != 1 or budgets.shape != variances.shape:
        raise ValueError("variances and budgets must be equal-length vectors")
    if np.any(variances <= 0) or np.any(budgets < 0) or not np.isclose(budgets.sum(), 1):
        raise ValueError("variances must be positive and budgets must be normalized")
    weights = np.sqrt(budgets / variances)
    return np.asarray(weights / weights.sum(), dtype=np.float64)


def weighted_empirical_es(
    scenario_returns: np.ndarray,
    confidence_level: float,
    probabilities: np.ndarray | None = None,
) -> tuple[float, float, float]:
    returns = np.asarray(scenario_returns, dtype=np.longdouble)
    if returns.ndim != 1 or returns.size == 0 or not 0 < confidence_level < 1:
        raise ValueError("invalid scenarios or confidence level")
    weights = (
        np.full(returns.size, 1 / np.longdouble(returns.size), dtype=np.longdouble)
        if probabilities is None
        else np.asarray(probabilities, dtype=np.longdouble)
    )
    if weights.shape != returns.shape or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("probabilities must be non-negative and normalized")
    order = np.argsort(-returns, kind="stable")
    losses = -returns[order]
    weights = weights[order]
    var_index = int(np.searchsorted(np.cumsum(weights), confidence_level, side="left"))
    value_at_risk = losses[var_index]
    expected_shortfall = value_at_risk + np.sum(
        weights * np.maximum(losses - value_at_risk, 0)
    ) / (1 - np.longdouble(confidence_level))
    return float(value_at_risk), float(expected_shortfall), float(-expected_shortfall)


def build_phase1a_fixture() -> dict[str, object]:
    returns = np.asarray([
        [0.01, 0.02, -0.01],
        [0.02, 0.01, 0.00],
        [-0.01, 0.00, 0.02],
        [0.00, 0.01, 0.01],
        [0.03, 0.02, -0.02],
    ])
    covariance, shrinkage = ledoit_wolf_linear_constant_correlation(returns)
    risk_budget_weights = diagonal_risk_budget(np.asarray([1.0, 4.0]), np.asarray([0.5, 0.5]))
    value_at_risk, expected_shortfall, return_cvar = weighted_empirical_es(
        np.asarray([-3.0, -1.0, 1.0, 2.0]), 0.75
    )
    return {
        "schema": "qbt.portfolio_math.phase1a_oracle.v2",
        "ledoit_wolf": {
            "estimator": "LW-LIN-CC",
            "returns": returns.tolist(),
            "covariance": covariance.tolist(),
            "shrinkage_intensity": shrinkage,
        },
        "risk_budget": {"weights": risk_budget_weights.tolist()},
        "tail_risk": {
            "estimator": "TAIL-EMPIRICAL-ES",
            "value_at_risk_loss": value_at_risk,
            "expected_shortfall_loss": expected_shortfall,
            "return_cvar": return_cvar,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Phase 1A portfolio-math oracle fixture")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_phase1a_fixture(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

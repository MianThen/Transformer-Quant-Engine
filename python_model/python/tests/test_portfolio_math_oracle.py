from __future__ import annotations

import numpy as np
import pytest

from python.qbt_ml.research.portfolio_math_oracle import (
    build_phase1a_fixture,
    weighted_empirical_es,
)


def test_phase1a_oracle_has_frozen_analytic_values():
    fixture = build_phase1a_fixture()
    assert fixture["schema"] == "qbt.portfolio_math.phase1a_oracle.v2"
    np.testing.assert_allclose(fixture["risk_budget"]["weights"], [2 / 3, 1 / 3])
    assert fixture["tail_risk"] == {
        "estimator": "TAIL-EMPIRICAL-ES",
        "value_at_risk_loss": 1.0,
        "expected_shortfall_loss": 3.0,
        "return_cvar": -3.0,
    }
    covariance = np.asarray(fixture["ledoit_wolf"]["covariance"])
    assert fixture["ledoit_wolf"]["estimator"] == "LW-LIN-CC"
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-15
    assert 0 <= fixture["ledoit_wolf"]["shrinkage_intensity"] <= 1


def test_oracle_rejects_unnormalized_scenario_probabilities():
    with pytest.raises(ValueError, match="normalized"):
        weighted_empirical_es(np.asarray([0.0, -1.0]), 0.95, np.asarray([0.4, 0.4]))

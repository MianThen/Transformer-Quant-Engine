from __future__ import annotations

import numpy as np
import pytest

from python.qbt_ml.calibration.conformal import (
    RollingCqrSpec,
    apply_cqr_calibrator,
    fit_cqr_calibrator,
    rolling_cqr_backtest,
)


def _spec(**updates):
    values = {
        "target_coverage": 0.9,
        "minimum_calibration_observations": 20,
        "maximum_calibration_observations": 40,
        "exponential_decay": 1.0,
        "config_hash": 301,
    }
    values.update(updates)
    return RollingCqrSpec(**values)


def test_uniform_cqr_uses_finite_sample_higher_quantile_and_hash():
    target = np.arange(20, dtype=np.float64)
    lower = target - 4.0
    upper = target + 1.0
    upper[-2:] = target[-2:] - np.asarray([2.0, 3.0])
    timestamps = np.arange(1, 21, dtype=np.int64)
    artifact = fit_cqr_calibrator(
        lower, upper, target, timestamps, available_at=20, spec=_spec(),
    )
    assert artifact.method == "ROLLING_CQR_UNIFORM_FINITE_SAMPLE_HIGHER_V1"
    assert artifact.q_hat == pytest.approx(2.0)
    assert artifact.raw_q_hat == pytest.approx(2.0)
    assert artifact.quantile_policy == (
        "HIGHER_WITH_NONNEGATIVE_INTERVAL_GUARD_V1"
    )
    assert len(artifact.artifact_sha256) == 64
    assert artifact == fit_cqr_calibrator(
        lower, upper, target, timestamps, available_at=20, spec=_spec(),
    )


def test_cqr_application_and_future_guards():
    target = np.linspace(-1.0, 1.0, 20)
    lower = target - 0.1
    upper = target + 0.1
    timestamps = np.arange(1, 21, dtype=np.int64)
    artifact = fit_cqr_calibrator(
        lower, upper, target, timestamps, available_at=20, spec=_spec(),
    )
    assert artifact.raw_q_hat < 0.0
    assert artifact.q_hat == 0.0
    adjusted = apply_cqr_calibrator(
        artifact, [0.0], [0.1], [21], forecast_available_at=20,
    )
    assert adjusted[0][0] <= adjusted[1][0]
    with pytest.raises(ValueError, match="未来数据"):
        fit_cqr_calibrator(
            lower, upper, target, timestamps, available_at=19, spec=_spec(),
        )
    with pytest.raises(ValueError, match="必须晚于"):
        apply_cqr_calibrator(
            artifact, [0.0], [0.1], [20], forecast_available_at=20,
        )


def test_exponential_cqr_emphasizes_recent_shift():
    target = np.zeros(40)
    lower = np.full(40, -0.1)
    upper = np.full(40, 0.1)
    target[-3:] = 2.0
    timestamps = np.arange(1, 41, dtype=np.int64)
    uniform = fit_cqr_calibrator(
        lower, upper, target, timestamps, available_at=40, spec=_spec(),
    )
    decayed = fit_cqr_calibrator(
        lower, upper, target, timestamps, available_at=40,
        spec=_spec(exponential_decay=0.8),
    )
    assert decayed.method == "ROLLING_CQR_EXPONENTIALLY_WEIGHTED_HIGHER_V1"
    assert decayed.q_hat > uniform.q_hat
    assert decayed.effective_calibration_count < decayed.calibration_count


def test_rolling_cqr_is_strictly_past_only_and_deterministic():
    timestamps = np.arange(1, 61, dtype=np.int64)
    target = np.sin(timestamps / 4.0)
    lower = target - 0.05
    upper = target + 0.05
    target[::7] += 0.4
    first = rolling_cqr_backtest(
        lower, upper, target, timestamps, spec=_spec(),
    )
    replay = rolling_cqr_backtest(
        lower, upper, target, timestamps, spec=_spec(),
    )
    assert first == replay
    assert first.forecast_count == 40
    assert first.formal_distribution_free_claim is False
    assert 0.0 <= first.empirical_coverage <= 1.0
    assert first.lower_miss_rate + first.upper_miss_rate == pytest.approx(
        1.0 - first.empirical_coverage,
    )

    mutated = target.copy()
    mutated[35:40] += 10.0
    changed = rolling_cqr_backtest(
        lower, upper, mutated, timestamps, spec=_spec(),
    )
    np.testing.assert_allclose(
        changed.q_hat_values[:16], first.q_hat_values[:16],
    )
    assert changed.q_hat_values[16:] != first.q_hat_values[16:]


def test_cqr_rejects_crossed_or_unfrozen_inputs():
    with pytest.raises(ValueError, match="config_hash"):
        _spec(config_hash=0).validate()
    with pytest.raises(ValueError, match="不得高于"):
        fit_cqr_calibrator(
            [1.0] * 20, [0.0] * 20, [0.5] * 20, range(1, 21),
            available_at=20, spec=_spec(),
        )


def test_fixed_cqr_accepts_cross_section_timestamp_ties_but_rolling_rejects():
    timestamps = np.repeat(np.arange(1, 11, dtype=np.int64), 2)
    target = np.linspace(-0.2, 0.2, 20)
    artifact = fit_cqr_calibrator(
        target - 0.1, target + 0.1, target, timestamps,
        available_at=10, spec=_spec(),
    )
    adjusted = apply_cqr_calibrator(
        artifact, [-0.1, -0.2], [0.1, 0.2], [11, 11],
        forecast_available_at=10,
    )
    assert adjusted[0].shape == (2,)
    with pytest.raises(ValueError, match="严格递增"):
        rolling_cqr_backtest(
            target - 0.1, target + 0.1, target, timestamps, spec=_spec(),
        )

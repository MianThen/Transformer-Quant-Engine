from __future__ import annotations

import math

import numpy as np
import pytest

from python.qbt_ml.calibration.probability import (
    ProbabilityCalibrationValidationError,
    calibration_metrics,
    fit_weighted_isotonic,
    fit_weighted_platt,
)


FIT_ARGUMENTS = {
    "split_role": "validation",
    "fit_start_utc": "2025-01-01T00:00:00Z",
    "fit_end_utc": "2025-01-31T00:00:00Z",
    "available_at_utc": "2025-02-02T00:00:00Z",
}


def test_weighted_metrics_match_manual_values_and_hash() -> None:
    probabilities = np.asarray([0.1, 0.4, 0.8, 0.9])
    labels = np.asarray([0.0, 1.0, 1.0, 0.0])
    weights = np.asarray([1.0, 2.0, 1.0, 2.0])

    result = calibration_metrics(
        probabilities, labels, sample_weight=weights, ece_bins=2
    )
    expected_brier = float(np.dot(weights, (probabilities - labels) ** 2) / 6.0)
    expected_nll = float(
        np.dot(
            weights,
            -(labels * np.log(probabilities) + (1.0 - labels) * np.log1p(-probabilities)),
        )
        / 6.0
    )

    assert result.brier == pytest.approx(expected_brier)
    assert result.nll == pytest.approx(expected_nll)
    first_bin_gap = abs((0.1 + 2.0 * 0.4) / 3.0 - 2.0 / 3.0)
    second_bin_gap = abs((0.8 + 2.0 * 0.9) / 3.0 - 1.0 / 3.0)
    assert result.ece == pytest.approx((3.0 * first_bin_gap + 3.0 * second_bin_gap) / 6.0)
    assert len(result.metrics_sha256) == 64
    assert result.to_dict()["metrics_sha256"] == result.metrics_sha256


def test_weighted_pav_has_manual_pool_monotonic_knots_and_clipped_boundaries() -> None:
    probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    labels = [0, 1, 0, 1, 1, 1]
    weights = [1, 2, 1, 1, 1, 1]

    artifact = fit_weighted_isotonic(
        probabilities,
        labels,
        sample_weight=weights,
        minimum_sample_count=5,
        ece_bins=3,
        **FIT_ARGUMENTS,
    )
    fitted = artifact.predict(
        probabilities, decision_at_utc="2025-02-03T00:00:00Z"
    )
    boundary = artifact.predict(
        [0.0, 1.0], decision_at_utc="2025-02-03T00:00:00Z"
    )

    assert artifact.sample_count == 6
    assert artifact.pav_block_count == 3
    assert fitted == pytest.approx([0.0, 2.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0])
    assert np.all(np.diff(fitted) >= 0.0)
    assert np.all(np.diff(artifact.knot_scores) > 0.0)
    assert np.all(np.diff(artifact.knot_probabilities) >= 0.0)
    assert boundary == pytest.approx([0.0, 1.0])
    assert artifact.fit_range == (
        FIT_ARGUMENTS["fit_start_utc"],
        FIT_ARGUMENTS["fit_end_utc"],
    )
    assert artifact.to_dict()["model"]["knot_table"] == list(artifact.knot_table)


def test_platt_is_deterministic_monotonic_and_improves_overconfident_input() -> None:
    base = np.linspace(0.05, 0.95, 80)
    labels = np.asarray([(index % 5) < round(5 * probability) for index, probability in enumerate(base)], dtype=float)
    probabilities = np.clip(0.5 + 1.8 * (base - 0.5), 0.001, 0.999)

    first = fit_weighted_platt(
        probabilities,
        labels,
        minimum_sample_count=20,
        ece_bins=8,
        **FIT_ARGUMENTS,
    )
    second = fit_weighted_platt(
        probabilities,
        labels,
        minimum_sample_count=20,
        ece_bins=8,
        **FIT_ARGUMENTS,
    )
    grid = np.linspace(0.0, 1.0, 101)
    calibrated = first.predict(
        grid, decision_at_utc="2025-02-02T00:00:00Z"
    )

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.model_sha256 == second.model_sha256
    assert first.platt_slope is not None and first.platt_slope > 0.0
    assert np.all(np.diff(calibrated) >= 0.0)
    assert first.calibrated_metrics.nll < first.uncalibrated_metrics.nll
    assert math.isfinite(first.calibrated_metrics.ece)


def test_validation_only_and_available_at_guards_fail_closed() -> None:
    probabilities = np.linspace(0.05, 0.95, 20)
    labels = np.tile([0.0, 1.0], 10)

    with pytest.raises(ProbabilityCalibrationValidationError, match="validation"):
        fit_weighted_isotonic(
            probabilities,
            labels,
            split_role="train",
            fit_start_utc="2025-01-01T00:00:00Z",
            fit_end_utc="2025-01-31T00:00:00Z",
            available_at_utc="2025-02-01T00:00:00Z",
        )

    observation_times = ["2025-01-15T00:00:00Z"] * 19 + [
        "2025-02-01T00:00:00Z"
    ]
    with pytest.raises(ProbabilityCalibrationValidationError, match="未来 observation"):
        fit_weighted_isotonic(
            probabilities,
            labels,
            observation_at_utc=observation_times,
            **FIT_ARGUMENTS,
        )

    label_times = ["2025-02-01T00:00:00Z"] * 19 + [
        "2025-02-03T00:00:00Z"
    ]
    with pytest.raises(ProbabilityCalibrationValidationError, match="才成熟"):
        fit_weighted_isotonic(
            probabilities,
            labels,
            label_available_at_utc=label_times,
            **FIT_ARGUMENTS,
        )


def test_prediction_before_artifact_availability_is_rejected() -> None:
    probabilities = np.linspace(0.05, 0.95, 20)
    labels = np.asarray([0.0] * 8 + [1.0] * 12)
    artifact = fit_weighted_isotonic(probabilities, labels, **FIT_ARGUMENTS)

    with pytest.raises(ProbabilityCalibrationValidationError, match="尚不可用"):
        artifact.predict([0.5], decision_at_utc="2025-02-01T23:59:59Z")


@pytest.mark.parametrize(
    ("probabilities", "labels", "weights", "message"),
    [
        ([0.1, 0.2, 0.3], [0, 1, 1], None, "样本量不足"),
        ([0.5] * 20, [0, 1] * 10, None, "输入概率退化"),
        (np.linspace(0.1, 0.9, 20), [1] * 20, None, "同时包含"),
        (np.linspace(0.1, 0.9, 20), [0, 1] * 10, [1] * 19 + [0], "严格为正"),
    ],
)
def test_invalid_short_and_degenerate_fit_inputs_fail_closed(
    probabilities: object, labels: object, weights: object, message: str
) -> None:
    with pytest.raises(ProbabilityCalibrationValidationError, match=message):
        fit_weighted_isotonic(
            probabilities,
            labels,
            sample_weight=weights,
            **FIT_ARGUMENTS,
        )


def test_hash_covers_weights_and_isotonic_step_guard() -> None:
    probabilities = np.linspace(0.05, 0.95, 24)
    labels = np.asarray([0.0] * 8 + [1.0] * 8 + [0.0, 1.0] * 4)
    first = fit_weighted_isotonic(
        probabilities,
        labels,
        sample_weight=np.ones(24),
        **FIT_ARGUMENTS,
    )
    changed = fit_weighted_isotonic(
        probabilities,
        labels,
        sample_weight=np.linspace(1.0, 1.2, 24),
        **FIT_ARGUMENTS,
    )

    assert first.sample_weight_sha256 != changed.sample_weight_sha256
    assert first.artifact_sha256 != changed.artifact_sha256
    with pytest.raises(ProbabilityCalibrationValidationError, match="台阶过多"):
        fit_weighted_isotonic(
            probabilities,
            labels,
            maximum_block_count=2,
            **FIT_ARGUMENTS,
        )

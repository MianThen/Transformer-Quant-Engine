from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from python.qbt_ml.monitoring.drift import (
    AlertState,
    DriftAlertMachine,
    DriftMonitorSpecV1,
    EmbeddingSpec,
    LayerStatus,
    ReferenceKind,
    WindowSpec,
    benjamini_hochberg,
    benjamini_yekutieli,
    build_current_windows,
    build_drift_report,
    build_reference_window,
    categorical_drift,
    continuous_drift,
    joint_drift,
    linear_cka,
    stationary_bootstrap_mean,
)


PREDICTION_COLUMNS = (
    "expected_return",
    "volatility",
    "direction_probability",
    "q10",
    "q90",
    "confidence",
)


def _frames(*, current_feature_shift: float = 0.0,
            reverse_current_signal: bool = False,
            include_future: bool = False):
    raw_rows, feature_rows, prediction_rows, label_rows = [], [], [], []
    end = 13 if include_future else 12
    for timestamp in range(1, end + 1):
        for symbol_index, symbol in enumerate(("A", "B", "C", "D")):
            centered = symbol_index - 1.5
            current = timestamp >= 7
            feature_shift = current_feature_shift if current else 0.0
            feature_one = centered + timestamp * 0.02 + feature_shift
            feature_two = centered * (1.0 if not current else -1.0) + timestamp * 0.01
            realized = centered * 0.01 + timestamp * 0.0001
            score = realized
            if current and reverse_current_signal:
                score = -realized
            common = {"timestamp": timestamp, "symbol": symbol,
                      "available_at": timestamp}
            raw_rows.append({**common, "price": 10.0 + feature_one,
                             "volume": 1000.0 + symbol_index * 100 + timestamp})
            feature_rows.append({**common, "f1": feature_one, "f2": feature_two})
            prediction_rows.append({
                **common,
                "expected_return": score,
                "volatility": 0.1 + abs(centered) * 0.01,
                "direction_probability": 1.0 / (1.0 + np.exp(-score * 100.0)),
                "q10": score - 0.02,
                "q90": score + 0.02,
                "confidence": 0.8,
                "ranking_score": score,
            })
            label_rows.append({**common, "return_raw": realized,
                               "rank_utility": realized})
    return tuple(pd.DataFrame(rows) for rows in (
        raw_rows, feature_rows, prediction_rows, label_rows))


def _windows():
    return (
        WindowSpec(1, 6, 6, ReferenceKind.TRAINING_STATIC),
        WindowSpec(7, 12, 12),
    )


def _spec(**changes):
    return DriftMonitorSpecV1(
        minimum_sessions=3,
        minimum_observations=8,
        bootstrap_replicates=99,
        mean_block_length=2.0,
        maximum_joint_samples=128,
        fast_window_sessions=3,
        confirm_window_sessions=6,
        **changes,
    )


def _report(frames, *, spec=None, labels=True, embeddings=None,
            embedding_specs=None, machine=None):
    raw, features, predictions, label_frame = frames
    reference, current = _windows()
    arguments = {}
    if embeddings is not None:
        arguments.update(
            reference_embeddings=embeddings[0],
            current_embeddings=embeddings[1],
            reference_embedding_spec=embedding_specs[0],
            current_embedding_spec=embedding_specs[1],
        )
    return build_drift_report(
        reference_raw=raw,
        current_raw=raw,
        reference_features=features,
        current_features=features,
        reference_predictions=predictions,
        current_predictions=predictions,
        reference_labels=label_frame if labels else None,
        current_labels=label_frame if labels else None,
        reference_window=reference,
        current_window=current,
        raw_columns=("price", "volume"),
        feature_columns=("f1", "f2"),
        prediction_columns=PREDICTION_COLUMNS,
        spec=spec or _spec(),
        alert_machine=machine,
        source_snapshot_hashes=("a" * 64, "b" * 64),
        **arguments,
    )


def test_spec_and_reference_edges_are_frozen_and_reproducible():
    spec = _spec()
    assert spec.sha256 == dataclasses.replace(spec).sha256
    with pytest.raises(ValueError, match="quantiles"):
        dataclasses.replace(spec, quantiles=(0.5, 0.25))
    reference = np.asarray([0.0, 1.0, 2.0, 3.0, np.nan])
    current = np.asarray([0.0, 1.0, 8.0, np.nan])
    first = continuous_drift(reference, current, spec)
    changed_current = np.append(current, 1000.0)
    second = continuous_drift(reference, changed_current, spec)
    assert first["psi_reference_edges"] == second["psi_reference_edges"]
    assert first["ks_d"] == pytest.approx(1.0 / 3.0)
    assert first["reference"]["missing_rate"] == pytest.approx(0.2)
    assert first["psi"] > 0.0


def test_reference_and_fast_confirm_window_builders_obey_available_at():
    raw, _, _, _ = _frames()
    static = build_reference_window(
        raw, reference_kind=ReferenceKind.TRAINING_STATIC,
        session_column="timestamp", available_at=12, before_session=7,
        training_start=1, training_end=6)
    rolling = build_reference_window(
        raw, reference_kind=ReferenceKind.ROLLING_RECENT,
        session_column="timestamp", available_at=12, before_session=7,
        rolling_sessions=3)
    windows = build_current_windows(
        raw[raw.timestamp >= 7], session_column="timestamp",
        available_at=12, spec=_spec())
    assert (static.start, static.end) == (1, 6)
    assert (rolling.start, rolling.end) == (4, 6)
    assert (windows["fast"].start, windows["fast"].end) == (10, 12)
    assert (windows["confirm"].start, windows["confirm"].end) == (7, 12)

    delayed = raw.copy()
    delayed.loc[delayed.timestamp == 12, "available_at"] = 13
    with pytest.raises(ValueError, match="confirm window"):
        build_current_windows(
            delayed[delayed.timestamp >= 7], session_column="timestamp",
            available_at=12, spec=_spec())


def test_joint_drift_detects_relationship_change_and_ignores_row_order():
    random = np.random.default_rng(7)
    first = random.normal(size=100)
    reference = np.column_stack((first, first + random.normal(scale=0.01, size=100)))
    current = np.column_stack((first, -first + random.normal(scale=0.01, size=100)))
    result = joint_drift(reference, current, _spec())
    shuffled = joint_drift(reference[::-1], current[random.permutation(100)], _spec())
    assert result["correlation_frobenius_distance"] > 1.0
    assert result["mmd_rbf"] == pytest.approx(shuffled["mmd_rbf"], abs=1e-14)
    assert result["reference_effective_rank"] >= 1.0
    assert 0.5 <= result["classifier_two_sample_auc"] <= 1.0


def test_categorical_drift_and_fixed_anchor_cka_oracles():
    categorical = categorical_drift(
        ["A", "A", "B", None], ["A", "C", "C", None])
    assert categorical["new_category_rate"] == pytest.approx(0.5)
    assert categorical["js_divergence"] > 0.0
    anchor = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert linear_cka(anchor, anchor * 3.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="anchor"):
        linear_cka(anchor, anchor[:-1])


def test_stationary_bootstrap_and_fdr_are_deterministic_and_conservative():
    values = np.asarray([0.1, 0.2, -0.1, 0.3, 0.0, 0.2])
    first = stationary_bootstrap_mean(
        values, replicates=99, mean_block_length=2.0, seed=17)
    second = stationary_bootstrap_mean(
        values, replicates=99, mean_block_length=2.0, seed=17)
    assert first == second
    p_values = np.asarray([0.01, 0.04, 0.20, 0.90])
    bh = benjamini_hochberg(p_values)
    by = benjamini_yekutieli(p_values)
    assert np.all(by >= bh)
    assert np.all((bh >= 0.0) & (bh <= 1.0))


def test_pending_labels_report_hash_and_future_mutation_contract():
    original = _frames(include_future=True)
    report = _report(original, labels=False)
    assert report.label_status is LayerStatus.PENDING_LABELS
    assert report.concept_status is LayerStatus.PENDING_LABELS
    assert report.alert_state is AlertState.INFO
    assert len(report.report_sha256) == 64

    changed = tuple(frame.copy() for frame in original)
    for frame in changed:
        numeric = [column for column in frame.columns
                   if column not in {"timestamp", "symbol", "available_at"}]
        frame.loc[frame.timestamp == 13, numeric] = 999999.0
    replay = _report(changed, labels=False)
    assert replay.report_sha256 == report.report_sha256
    assert replay.source_snapshot_set_sha256 == report.source_snapshot_set_sha256


def test_label_concept_drift_requires_persistence_before_critical():
    frames = _frames(current_feature_shift=2.0, reverse_current_signal=True)
    spec = _spec(
        marginal_warning_psi=0.01,
        joint_warning_mmd=0.0001,
        concept_warning_ic_drop=0.1,
        persistence_windows=2,
    )
    machine = DriftAlertMachine(spec)
    first = _report(frames, spec=spec, machine=machine)
    second = _report(frames, spec=spec, machine=machine)
    assert first.label_status is LayerStatus.OK
    assert first.concept_status is LayerStatus.OK
    assert first.concept_performance["pearson_ic_change"] < -1.0
    assert first.alert_state is AlertState.WARN
    assert second.alert_state is AlertState.CRITICAL
    assert second.persistence_count == 2
    assert second.retraining_review_recommended
    assert "CONCEPT_IC_DEGRADATION" in second.alert_reasons


def test_embedding_compatibility_and_diagnostic_only_boundary():
    frames = _frames()
    embeddings = (
        frames[1][frames[1].timestamp <= 6][["f1", "f2"]].to_numpy(),
        frames[1][frames[1].timestamp >= 7][["f1", "f2"]].to_numpy(),
    )
    base = EmbeddingSpec("a" * 64, "TemporalTransformerV1", "encoder.2",
                         "b" * 64, 2, "c" * 64)
    incompatible = dataclasses.replace(base, layer_id="encoder.1")
    rejected = _report(frames, embeddings=embeddings,
                       embedding_specs=(base, incompatible))
    assert rejected.embedding_status is LayerStatus.INCOMPATIBLE
    accepted = _report(frames, embeddings=embeddings,
                       embedding_specs=(base, base))
    assert accepted.embedding_status is LayerStatus.OK
    assert accepted.embedding_drift["diagnostic_only"] is True
    assert accepted.alert_state is AlertState.INFO


def test_available_at_future_leakage_fails_closed():
    frames = list(_frames())
    frames[1] = frames[1].copy()
    frames[1].loc[frames[1].timestamp == 8, "available_at"] = 13
    with pytest.raises(ValueError, match="available_at"):
        _report(tuple(frames))


def test_monitor_is_sidecar_and_never_mutates_source_snapshots():
    frames = _frames()
    frozen = tuple(frame.copy(deep=True) for frame in frames)
    first = _report(frames)
    second = _report(frames)
    assert first.report_sha256 == second.report_sha256
    assert first.verify_hash()
    with pytest.raises(TypeError):
        first.feature_drift["f1"]["psi"] = 0.0
    for before, after in zip(frozen, frames):
        pd.testing.assert_frame_equal(before, after)

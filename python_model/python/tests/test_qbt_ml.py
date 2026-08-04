from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from python.qbt_ml.cli import (
    _ablate_features, _ablate_transformer_features, _backtest_artifact,
    _benchmark_models, _build_dataset, _detect_leakage, _export, _walk_forward,
    _promotion_review, _train, _walk_forward_deep_baselines,
    _walk_forward_transformer,
)
from python.qbt_ml.data import BAR_V1, BAR_V1_FEATURE_GROUPS, build_windows
from python.qbt_ml.data.pit_enrichment import enrich_phase_e
from python.qbt_ml.export.manifest import ModelManifest, sha256_file, validate_artifact
from python.qbt_ml.features import build_bar_v1
from python.qbt_ml.labels import LabelSpec, build_next_open_labels
from python.qbt_ml.leakage import audit_dataset
from python.qbt_ml.models import RidgeBaseline
from python.qbt_ml.evaluation.portfolio_benchmark import (
    run_cpp_portfolio_ablation, run_cpp_portfolio_benchmark,
)
from python.qbt_ml.training import (
    CrossSectionBatchSampler, chronological_timestamp_split,
    expanding_timestamp_splits, walk_forward_splits,
)


def _bars(count=80, symbols=("A", "B")):
    rows = []
    for timestamp in range(1, count + 1):
        for offset, symbol in enumerate(symbols):
            close = 10.0 + offset + timestamp * 0.02 + np.sin(timestamp / 3) * 0.05
            rows.append({
                "timestamp": timestamp, "symbol": symbol,
                "open": close * 0.999, "high": close * 1.01,
                "low": close * 0.99, "close": close,
                "volume": 1000 + timestamp * (offset + 1),
                "is_listed": True, "is_suspended": False, "is_st": False,
                "universe_asof": timestamp,
                "reference_data_known_at_max": timestamp,
            })
    return pd.DataFrame(rows)


def _write_parquet_partition(root, relative, rows):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_phase_e_enrichment_preserves_suspended_slot_and_pit_availability(tmp_path):
    symbol = "000001"
    day1, day2, day3 = 100, 200, 300
    bars_root, state_root = tmp_path / "bars", tmp_path / "state"
    adjustment_root, action_root = tmp_path / "adjustment", tmp_path / "action"
    industry_root, output = tmp_path / "industry", tmp_path / "enriched"
    _write_parquet_partition(bars_root, f"ticker={symbol}/year=2024/part.parquet", [
        {"timestamp": day1, "symbol": symbol, "open": 10.0, "high": 11.0,
         "low": 9.0, "close": 10.5, "volume": 1000},
        {"timestamp": day3, "symbol": symbol, "open": 5.4, "high": 5.6,
         "low": 5.2, "close": 5.5, "volume": 2000},
    ])
    _write_parquet_partition(state_root, f"ticker={symbol}/year=2024/part.parquet", [
        {"timestamp": timestamp, "symbol": symbol, "is_listed": True,
         "is_suspended": timestamp == day2, "is_st": False,
         "is_tradable": timestamp != day2, "universe_asof": timestamp,
         "reference_data_known_at_max": timestamp}
        for timestamp in (day1, day2, day3)
    ])
    _write_parquet_partition(
        adjustment_root, f"ticker={symbol}/year=2024/part.parquet", [{
            "timestamp": day3, "symbol": symbol, "back_adjust_factor": 2.0,
            "known_at": day3,
        }],
    )
    _write_parquet_partition(action_root, f"ticker={symbol}/year=2024/part.parquet", [{
        "timestamp": day3, "symbol": symbol, "cash_dividend_per_share": 0.5,
        "stock_dividend_per_share": 0.0, "reserve_to_stock_per_share": 0.0,
        "share_multiplier": 1.0, "known_at": day2,
    }])
    _write_parquet_partition(industry_root, "snapshot=20240101/part.parquet", [{
        "timestamp": day1, "symbol": symbol, "industry": "Bank",
        "classification": "CSRC", "known_at": day1, "snapshot_asof": day2,
    }])
    result = enrich_phase_e({
        "execution_reference_mode": "optional_for_model_evaluation",
        "frequency": "1d", "calendar_id": "sse-szse-test-v1",
        "bars_root": str(bars_root), "state_root": str(state_root),
        "adjustment_root": str(adjustment_root),
        "corporate_action_root": str(action_root), "industry_root": str(industry_root),
        "adjustment_history_complete": True, "output": str(output),
    })
    table = pd.read_parquet(next(result.rglob("*.parquet"))).sort_values("timestamp")
    suspended = table.iloc[1]
    assert not suspended.bar_observed and suspended.volume == 0
    assert (suspended.open, suspended.high, suspended.low, suspended.close) == (10.5,) * 4
    assert table.iloc[0].industry is None
    assert table.iloc[1].industry == "Bank" and table.iloc[1].industry_known_at == day2
    assert table.iloc[2].signal_close == pytest.approx(11.0)
    report = json.loads((result / "enrichment_report.json").read_text())
    assert report["status"] == "MODEL_READY_EXECUTION_DEFERRED"
    assert report["model_evaluation_status"] == "READY"
    assert report["execution_promotion_status"] == "DEFERRED"
    assert report["promotion_eligible"] is False
    assert "upper/lower" in report["execution_blockers"][0]
    with pytest.raises(ValueError, match="正式 Phase E 富化被阻断"):
        enrich_phase_e({
            "execution_reference_mode": "required_for_promotion",
            "frequency": "1d", "calendar_id": "sse-szse-test-v1",
            "bars_root": str(bars_root), "state_root": str(state_root),
            "adjustment_root": str(adjustment_root),
            "corporate_action_root": str(action_root),
            "industry_root": str(industry_root),
            "adjustment_history_complete": True,
            "output": str(tmp_path / "formal-enriched"),
        })


def test_build_dataset_preserves_deferred_execution_reference_status(tmp_path):
    source = tmp_path / "enriched"
    source.mkdir()
    bars = _bars(80, ("A", "B"))
    bars["adjustment_factor"] = 1.0
    for name in ("open", "high", "low", "close"):
        bars[f"signal_{name}"] = bars[name]
    bars.to_parquet(source / "part.parquet", index=False)
    (source / "enrichment_report.json").write_text(json.dumps({
        "model_evaluation_status": "READY",
        "execution_promotion_status": "DEFERRED",
        "execution_reference_mode": "optional_for_model_evaluation",
        "source_fingerprint_sha256": "fixture-source",
    }))
    output = tmp_path / "dataset.npz"
    config = {
        "enabled": True, "lookback": 8, "label_horizon_bars": 5,
        "frequency": "1d", "price_adjustment_mode": "pit_adjusted_signal_raw_execution",
        "calendar_id": "test-calendar", "universe_id": "test-universe",
        "data": {
            "source": str(source), "dataset_output": str(output),
            "point_in_time_required": True,
            "execution_reference_mode": "optional_for_model_evaluation",
        },
    }
    _build_dataset(config, None)
    with np.load(output, allow_pickle=False) as dataset:
        assert str(dataset["execution_reference_status"]) == "DEFERRED"
        assert str(dataset["execution_reference_mode"]) == "optional_for_model_evaluation"
        assert str(dataset["execution_source_fingerprint"]) == "fixture-source"
    config["data"]["execution_reference_mode"] = "required_for_promotion"
    with pytest.raises(ValueError, match="execution_reference_mode 不一致"):
        _build_dataset(config, str(tmp_path / "invalid.npz"))


def test_bar_v1_is_ordered_stable_and_has_no_future_dependency():
    source = _bars()
    before = build_bar_v1(source)
    changed = source.copy()
    changed.loc[
        changed["timestamp"] == 80, ["open", "high", "low", "close"]
    ] *= 2
    after = build_bar_v1(changed)
    cutoff = before.table["timestamp"].to_numpy() < 80
    np.testing.assert_array_equal(before.values[cutoff], after.values[cutoff])
    assert before.values.dtype == np.float32
    assert before.values.shape[1] == len(BAR_V1.feature_names)
    assert before.valid_mask[before.table["timestamp"].to_numpy() == 70].all()


def test_bar_v1_windows_are_left_padded_and_masked():
    frame = build_bar_v1(_bars(3, ("A",)))
    windows = build_windows(frame, lookback=4)
    assert windows.features.shape == (3, 4, len(BAR_V1.feature_names))
    assert windows.valid_mask[0].tolist() == [0, 0, 0, 0]
    np.testing.assert_array_equal(windows.features[-1, 1:], frame.values)


@pytest.mark.parametrize("column,value", [("close", np.nan), ("volume", np.inf)])
def test_bar_v1_rejects_non_finite_ohlcv(column, value):
    source = _bars(3, ("A",))
    source[column] = source[column].astype(np.float64)
    source.loc[0, column] = value
    with pytest.raises(ValueError, match="有限"):
        build_bar_v1(source)


def test_bar_v1_rejects_invalid_ohlc_relationship():
    source = _bars(3, ("A",))
    source.loc[0, "high"] = source.loc[0, "close"] * 0.5
    with pytest.raises(ValueError, match="OHLC"):
        build_bar_v1(source)


def test_next_open_label_matches_execution_alignment():
    source = pd.DataFrame({
        "timestamp": [1, 2, 3, 4], "symbol": ["A"] * 4,
        "open": [10.0, 11.0, 12.0, 13.0],
        "close": [10.5, 11.5, 12.5, 14.0],
    })
    labels = build_next_open_labels(source, horizon_bars=1)
    first = labels.iloc[0]
    assert first.entry_open == 11.0
    assert first.exit_close == 11.5
    assert first.expected_return == pytest.approx(np.log(11.5 / 11.0))
    assert first.realized_volatility == 0.0
    assert first.label_entry_timestamp == 2
    assert first.label_exit_timestamp == 2
    assert labels["label_valid"].tolist() == [True, True, True, False]


def test_next_open_volatility_uses_holding_period_subreturns():
    source = pd.DataFrame({
        "timestamp": [1, 2, 3, 4], "symbol": ["A"] * 4,
        "open": [10.0, 11.0, 12.0, 13.0],
        "close": [10.5, 11.5, 12.5, 14.0],
    })
    spec = LabelSpec.next_open(2)
    labels = build_next_open_labels(source, 2, label_spec=spec)
    first = labels.iloc[0]
    expected_subreturns = [np.log(11.5 / 11.0), np.log(12.5 / 11.5)]
    assert first.exit_close == 12.5
    assert first.expected_return == pytest.approx(np.log(12.5 / 11.0))
    assert first.realized_volatility == pytest.approx(np.std(expected_subreturns, ddof=0))
    assert first.realized_volatility != pytest.approx(abs(first.expected_return))
    assert json.loads(spec.canonical_json)["exit_offset_bars"] == 2


def test_next_open_labels_use_adjusted_signal_prices_across_corporate_action():
    source = pd.DataFrame({
        "timestamp": [1, 2, 3, 4], "symbol": ["A"] * 4,
        "open": [10.0, 10.0, 5.0, 5.0],
        "close": [10.0, 10.0, 5.0, 5.0],
        "signal_open": [5.0, 5.0, 5.0, 5.0],
        "signal_close": [5.0, 5.0, 5.0, 5.0],
    })
    labels = build_next_open_labels(source, 2)
    assert labels.iloc[0].expected_return == pytest.approx(0.0)
    assert labels.iloc[0].realized_volatility == pytest.approx(0.0)
    assert labels.iloc[0].entry_open == 5.0
    assert labels.iloc[0].exit_close == 5.0


def test_walk_forward_purge_embargo_and_ridge_baseline():
    split = next(iter(walk_forward_splits(
        30, train_size=10, validation_size=5, test_size=5, purge=2, embargo=1
    )))
    assert split.train[-1] == 9 and split.validation[0] == 12 and split.test[0] == 20
    x = np.arange(10, dtype=np.float64).reshape(-1, 1)
    model = RidgeBaseline(alpha=0.0).fit(x, 2 * x[:, 0] + 1)
    np.testing.assert_allclose(model.predict([[10.0]]), [21.0], atol=1e-10)


def test_cross_section_batch_sampler_never_splits_timestamp():
    timestamps = np.asarray([1, 1, 1, 2, 2, 3, 3, 3, 3])
    sampler = CrossSectionBatchSampler(
        timestamps, timestamps_per_batch=2, shuffle=True, seed=7
    )
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(timestamps.size))
    for timestamp in np.unique(timestamps):
        expected = set(np.flatnonzero(timestamps == timestamp))
        containing = [set(batch) for batch in batches if expected & set(batch)]
        assert len(containing) == 1
        assert expected <= containing[0]


def test_timestamp_split_keeps_cross_sections_atomic_and_chronological():
    timestamps = np.repeat(np.arange(20), 3)
    split = chronological_timestamp_split(
        timestamps, train_fraction=0.5, validation_fraction=0.25,
        test_fraction=0.25, purge_timestamps=1, embargo_timestamps=1,
    )
    assert set(timestamps[split.train]) == set(split.train_timestamps)
    assert set(timestamps[split.validation]) == set(split.validation_timestamps)
    assert set(timestamps[split.test]) == set(split.test_timestamps)
    assert split.train_timestamps[-1] < split.validation_timestamps[0]
    assert split.validation_timestamps[-1] < split.test_timestamps[0]
    assert split.validation_timestamps[0] - split.train_timestamps[-1] == 2
    assert split.test_timestamps[0] - split.validation_timestamps[-1] == 3


def test_expanding_walk_forward_has_non_overlapping_test_windows():
    timestamps = np.repeat(np.arange(60), 3)
    windows = expanding_timestamp_splits(
        timestamps,
        minimum_train_timestamps=20,
        validation_timestamps=5,
        test_timestamps=5,
        step_timestamps=5,
        purge_timestamps=2,
        embargo_timestamps=2,
        minimum_windows=3,
    )
    assert len(windows) == 6
    assert len(windows[1].train_timestamps) > len(windows[0].train_timestamps)
    test_sets = [set(window.test_timestamps.tolist()) for window in windows]
    assert all(
        not left.intersection(right)
        for index, left in enumerate(test_sets)
        for right in test_sets[index + 1:]
    )
    with pytest.raises(ValueError, match="不能小于"):
        expanding_timestamp_splits(
            timestamps, minimum_train_timestamps=20, validation_timestamps=5,
            test_timestamps=5, step_timestamps=4,
        )


def test_dataset_builder_sorts_timestamp_symbol_and_excludes_invalid_asof(tmp_path):
    source = _bars(70, ("000002", "000001")).sample(frac=1.0, random_state=7)
    source.loc[(source["timestamp"] == 65) & (source["symbol"] == "000001"),
               "is_suspended"] = True
    source_path = tmp_path / "bars.csv"
    output_path = tmp_path / "dataset.npz"
    source.to_csv(source_path, index=False)
    _build_dataset({
        "enabled": True, "lookback": 8, "label_horizon_bars": 1,
        "calendar_id": "test-calendar", "universe_id": "test-universe",
        "data": {"source": str(source_path),
                 "dataset_output": str(output_path), "point_in_time_required": True},
    }, None)
    with np.load(output_path, allow_pickle=False) as data:
        keys = list(zip(data["timestamps"].tolist(), data["symbols"].tolist()))
        assert keys == sorted(keys)
        assert (65, "000001") not in keys
        assert "000001" in data["symbols"].tolist()
        assert data["valid_mask"][:, -1].all()
        assert str(data["label_spec_sha256"]).isalnum()
        assert (data["feature_source_max_timestamp"] <= data["signal_asof"]).all()
        assert (data["label_entry_timestamp"] > data["signal_asof"]).all()
        assert bool(data["prefix_invariance_pass"])
        assert bool(data["future_mutation_pass"])
        assert audit_dataset({name: data[name] for name in data.files}).status == "PASS"
    report = json.loads(output_path.with_suffix(".leakage_report.json").read_text())
    assert report["status"] == "PASS"


def test_dataset_builder_requires_point_in_time_provenance(tmp_path):
    source = _bars(10, ("A",)).drop(columns=["reference_data_known_at_max"])
    source_path = tmp_path / "bars.csv"
    source.to_csv(source_path, index=False)
    with pytest.raises(ValueError, match="reference_data_known_at_max"):
        _build_dataset({
            "enabled": True, "calendar_id": "test-calendar",
            "universe_id": "test-universe",
            "data": {
                "source": str(source_path),
                "dataset_output": str(tmp_path / "dataset.npz"),
                "point_in_time_required": True,
            },
        }, None)


def test_leakage_detection_rejects_future_features(tmp_path):
    source = _bars(120, ("A", "B"))
    source_path = tmp_path / "bars.csv"
    output_path = tmp_path / "dataset.npz"
    source.to_csv(source_path, index=False)
    _build_dataset({
        "enabled": True, "lookback": 8, "label_horizon_bars": 5,
        "calendar_id": "test-calendar", "universe_id": "test-universe",
        "data": {
            "source": str(source_path), "dataset_output": str(output_path),
            "point_in_time_required": True,
        },
    }, None)
    with np.load(output_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["feature_source_max_timestamp"] = arrays["signal_asof"].copy()
    arrays["feature_source_max_timestamp"][0] += 1
    report = audit_dataset(arrays)
    assert report.status == "FAIL"
    assert "FEATURE_TIME" in {item.code for item in report.violations}
    with pytest.raises(ValueError, match="Leakage Detection"):
        report.require_pass()

    split = chronological_timestamp_split(
        arrays["timestamps"], train_fraction=0.6, validation_fraction=0.2,
        test_fraction=0.2, purge_timestamps=5, embargo_timestamps=5,
    )
    fit_scope_report = audit_dataset(
        {name: data for name, data in arrays.items() if name != "dataset_fingerprint"},
        split=split,
        normalizer_fit_end=arrays["signal_asof"][split.validation].max(),
    )
    assert "NORMALIZER_FIT_SCOPE" in {
        item.code for item in fit_scope_report.violations
    }


def test_detect_leakage_cli_writes_pass_report(tmp_path):
    source = _bars(90, ("A", "B"))
    source_path = tmp_path / "bars.csv"
    dataset_path = tmp_path / "dataset.npz"
    source.to_csv(source_path, index=False)
    base = {
        "enabled": True, "lookback": 8, "label_horizon_bars": 5,
        "calendar_id": "test-calendar", "universe_id": "test-universe",
        "data": {
            "source": str(source_path), "dataset_output": str(dataset_path),
            "point_in_time_required": True,
        },
    }
    _build_dataset(base, None)
    report_path = _detect_leakage({
        "enabled": True,
        "split": {
            "train_fraction": 0.6, "validation_fraction": 0.2,
            "test_fraction": 0.2, "purge_timestamps": 5, "embargo_timestamps": 5,
        },
    }, str(dataset_path), str(tmp_path / "leakage"))
    assert json.loads(report_path.read_text())["status"] == "PASS"


def test_walk_forward_baseline_writes_three_window_oos_outputs(tmp_path):
    source = _bars(150, ("A", "B", "C"))
    source_path = tmp_path / "bars.csv"
    dataset_path = tmp_path / "dataset.npz"
    source.to_csv(source_path, index=False)
    _build_dataset({
        "enabled": True, "lookback": 8, "label_horizon_bars": 5,
        "calendar_id": "test-calendar", "universe_id": "test-universe",
        "data": {
            "source": str(source_path), "dataset_output": str(dataset_path),
            "point_in_time_required": True,
        },
    }, None)
    output = _walk_forward({
        "enabled": True,
        "windows": {
            "minimum_train_timestamps": 30,
            "validation_timestamps": 8,
            "test_timestamps": 8,
            "step_timestamps": 8,
            "purge_timestamps": 5,
            "embargo_timestamps": 5,
            "minimum_windows": 3,
        },
        "model": {
            "ridge_alpha_candidates": [0.1, 1.0],
            "normalization_clip": 8.0,
        },
    }, str(dataset_path), str(tmp_path / "walk-forward"))
    summary = json.loads((output / "aggregate" / "summary.json").read_text())
    assert summary["windows"] >= 3
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["test_evaluation_policy"] == "once_after_validation_freeze"
    test_keys = []
    for window_path in sorted(output.glob("window_*")):
        assert json.loads((window_path / "leakage_report.json").read_text())["status"] == "PASS"
        assert (window_path / "model" / "ridge_multitask.json").is_file()
        with np.load(window_path / "predictions.npz", allow_pickle=False) as predictions:
            test_keys.extend(zip(
                predictions["timestamps"].tolist(), predictions["symbols"].tolist()
            ))
    assert len(test_keys) == len(set(test_keys))


def test_feature_groups_are_complete_and_group_drop_retrains(tmp_path):
    grouped = [name for names in BAR_V1_FEATURE_GROUPS.values() for name in names]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(BAR_V1.feature_names)

    source = _bars(150, ("A", "B", "C"))
    source_path = tmp_path / "bars.csv"
    dataset_path = tmp_path / "dataset.npz"
    source.to_csv(source_path, index=False)
    _build_dataset({
        "enabled": True, "lookback": 8, "label_horizon_bars": 5,
        "calendar_id": "test-calendar", "universe_id": "test-universe",
        "data": {
            "source": str(source_path), "dataset_output": str(dataset_path),
            "point_in_time_required": True,
        },
    }, None)
    output = _ablate_features({
        "enabled": True,
        "groups": ["returns", "volume"],
        "walk_forward": {
            "windows": {
                "minimum_train_timestamps": 30,
                "validation_timestamps": 8,
                "test_timestamps": 8,
                "step_timestamps": 8,
                "purge_timestamps": 5,
                "embargo_timestamps": 5,
                "minimum_windows": 3,
            },
            "model": {"ridge_alpha_candidates": [0.1, 1.0]},
        },
    }, str(dataset_path), str(tmp_path / "ablation"))
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["mode"] == "group_drop_retrain"
    assert manifest["formal_transformer_conclusion"] is False
    deltas = json.loads((output / "task_delta.json").read_text())["results"]
    assert [item["group"] for item in deltas] == ["returns", "volume"]
    assert (output / "drop_returns" / "aggregate" / "summary.json").is_file()


def test_model_benchmark_pairs_rule_models_on_identical_windows(tmp_path):
    source = _bars(150, ("A", "B", "C"))
    source_path = tmp_path / "bars.csv"
    dataset_path = tmp_path / "dataset.npz"
    source.to_csv(source_path, index=False)
    _build_dataset({
        "enabled": True, "lookback": 8, "label_horizon_bars": 5,
        "calendar_id": "test-calendar", "universe_id": "test-universe",
        "data": {
            "source": str(source_path), "dataset_output": str(dataset_path),
            "point_in_time_required": True,
        },
    }, None)
    output = _benchmark_models({
        "enabled": True,
        "walk_forward": {
            "windows": {
                "minimum_train_timestamps": 30,
                "validation_timestamps": 8,
                "test_timestamps": 8,
                "step_timestamps": 8,
                "purge_timestamps": 5,
                "embargo_timestamps": 5,
                "minimum_windows": 3,
            },
            "model": {"ridge_alpha_candidates": [0.1, 1.0]},
        },
    }, str(dataset_path), str(tmp_path / "benchmark"))
    manifest = json.loads((output / "benchmark_manifest.json").read_text())
    assert manifest["evaluated_models"] == [
        "ridge_multitask", "momentum_20", "reversal_5", "equal_weight", "cash"
    ]
    assert "transformer_v1_1" in manifest["not_evaluated_models"]
    records = json.loads((output / "model_quality.json").read_text())["records"]
    windows = {record["window_id"] for record in records}
    assert len(records) == len(windows) * 5
    pairs = json.loads((output / "paired_window_deltas.json").read_text())["pairs"]
    assert len(pairs) == len(windows) * 4
    prediction_manifest = json.loads((output / "prediction_manifest.json").read_text())
    equal_weight_path = prediction_manifest["models"]["equal_weight"][0]["path"]
    with np.load(equal_weight_path, allow_pickle=False) as prediction:
        assert (prediction["expected_return"] > 0).all()
        assert (prediction["lower_quantile"] <= prediction["expected_return"]).all()
        assert (prediction["upper_quantile"] >= prediction["expected_return"]).all()
        assert (prediction["confidence"] == 1.0).all()


def test_model_benchmark_ingests_frozen_deep_baseline_suite(tmp_path):
    source = _bars(100, ("A", "B", "C"))
    source_path = tmp_path / "bars.csv"
    dataset_path = tmp_path / "dataset.npz"
    source.to_csv(source_path, index=False)
    _build_dataset({
        "enabled": True, "lookback": 8, "label_horizon_bars": 5,
        "calendar_id": "test-calendar", "universe_id": "test-universe",
        "data": {
            "source": str(source_path), "dataset_output": str(dataset_path),
            "point_in_time_required": True,
        },
    }, None)
    window_config = {
        "minimum_train_timestamps": 8, "validation_timestamps": 3,
        "test_timestamps": 3, "step_timestamps": 3,
        "purge_timestamps": 5, "embargo_timestamps": 5,
        "minimum_windows": 3,
    }
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    windows = expanding_timestamp_splits(arrays["timestamps"], **window_config)
    suite = tmp_path / "deep-suite"
    suite.mkdir()
    families = {
        "mlp": "MLPMultiTaskBaseline",
        "tcn": "CausalTCNMultiTaskBaseline",
        "gru": "GRUMultiTaskBaseline",
    }
    runs = {}
    for model, family in families.items():
        root = suite / model
        root.mkdir()
        (root / "experiment_manifest.json").write_text(json.dumps({
            "model_family": family,
            "dataset_fingerprint": str(arrays["dataset_fingerprint"].item()),
            "windows": len(windows),
        }))
        runs[model] = {"path": model, "model_family": family}
        for window in windows:
            destination = root / f"window_{window.window_id:03d}"
            destination.mkdir()
            realized = arrays["expected_return"][window.test]
            probability = np.clip(0.5 + realized, 1e-3, 1 - 1e-3)
            np.savez_compressed(
                destination / "predictions.npz",
                timestamps=arrays["timestamps"][window.test],
                symbols=arrays["symbols"][window.test],
                expected_return=realized,
                expected_volatility=arrays["realized_volatility"][window.test],
                direction_probability=probability,
                lower_quantile=realized - 0.01,
                upper_quantile=realized + 0.01,
            )
    (suite / "benchmark_suite_manifest.json").write_text(json.dumps({
        "models": ["mlp", "tcn", "gru"], "runs": runs,
    }))
    output = _benchmark_models({
        "enabled": True, "deep_baseline_run": str(suite),
        "walk_forward": {
            "windows": window_config,
            "model": {"ridge_alpha_candidates": [0.1, 1.0]},
        },
    }, str(dataset_path), str(tmp_path / "benchmark"))
    manifest = json.loads((output / "benchmark_manifest.json").read_text())
    assert all(name in manifest["evaluated_models"] for name in families)
    assert "transformer_v1" in manifest["unavailable_legacy_models"]
    records = json.loads((output / "model_quality.json").read_text())["records"]
    assert sum(item["model"] in families for item in records) == len(windows) * 3


def test_cpp_portfolio_benchmark_runs_all_models_windows_and_costs(tmp_path):
    models = [
        "momentum_20", "reversal_5", "equal_weight", "cash",
        "ridge_multitask", "mlp", "tcn", "gru", "transformer_v1_1",
    ]
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        dataset_fingerprint=np.asarray("portfolio-fixture"),
        timestamps=np.asarray([1, 1, 3, 3, 5, 5]),
        symbols=np.asarray(["A", "B", "A", "B", "A", "B"]),
        label_exit_timestamp=np.asarray([2, 2, 4, 4, 6, 6]),
        frequency=np.asarray("1d"), calendar_id=np.asarray("test-calendar"),
        price_adjustment_mode=np.asarray("pit_adjusted_signal_raw_execution"),
        execution_reference_status=np.asarray("READY"),
    )
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    artifacts = {name: [] for name in models}
    for window_id, timestamp in enumerate((1, 3, 5), 1):
        prediction_path = tmp_path / f"prediction-{window_id}.npz"
        np.savez_compressed(
            prediction_path,
            timestamps=np.asarray([timestamp, timestamp]),
            symbols=np.asarray(["A", "B"]),
            expected_return=np.asarray([0.02, 0.01], dtype=np.float32),
            expected_volatility=np.asarray([0.1, 0.1], dtype=np.float32),
            direction_probability=np.asarray([0.8, 0.7], dtype=np.float32),
            lower_quantile=np.asarray([-0.01, -0.01], dtype=np.float32),
            upper_quantile=np.asarray([0.03, 0.02], dtype=np.float32),
            confidence=np.asarray([0.8, 0.7], dtype=np.float32),
        )
        for name in models:
            artifacts[name].append({"window_id": window_id, "path": str(prediction_path)})
    (benchmark / "prediction_manifest.json").write_text(json.dumps({
        "dataset_fingerprint": "portfolio-fixture", "models": artifacts,
    }))
    rows = []
    for timestamp in range(1, 7):
        for symbol, industry in (("A", "I1"), ("B", "I2")):
            rows.append({
                "timestamp": timestamp, "symbol": symbol,
                "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0,
                "volume": 10000, "is_listed": True, "is_suspended": False,
                "is_st": False, "upper_limit": 11.0, "lower_limit": 9.0,
                "lot_size": 100, "min_buy_quantity": 100, "industry": industry,
                "industry_known_at": timestamp, "universe_asof": timestamp,
                "reference_data_known_at_max": timestamp,
                "signal_open": 10.0, "signal_high": 10.1,
                "signal_low": 9.9, "signal_close": 10.0,
                "adjustment_factor": 1.0,
            })
    bars_path = tmp_path / "bars.csv"
    pd.DataFrame(rows).to_csv(bars_path, index=False)
    actions_path = tmp_path / "actions.csv"
    pd.DataFrame(columns=[
        "timestamp", "symbol", "cash_dividend_per_share",
        "share_multiplier", "known_at",
    ]).to_csv(actions_path, index=False)

    class ExecutionConfig:
        pass

    class FeeSchedule:
        def __init__(self, *values): self.values = values

    class MarketSnapshot:
        pass

    class CorporateAction:
        pass

    class HistoryConfig:
        pass

    class Point:
        def __init__(self, equity): self.equity = equity

    class Engine:
        runs = 0

        def __init__(self, initial_cash, fill_timing, execution):
            Engine.runs += 1
            self.initial_cash = initial_cash
            self.points = [Point(initial_cash)]

        def set_fee_schedules(self, schedules): self.schedules = schedules
        def set_history_config(self, history): self.history = history
        def set_precomputed_prediction_strategy(self, frames, policy, risk, runtime):
            self.frames = frames
        def apply_corporate_action(self, action): pass
        def process_market_data_batch(self, batch): self.points.append(Point(self.initial_cash))
        def finalize(self, timestamp): self.timestamp = timestamp
        def get_equity_curve(self): return self.points
        def get_trade_history(self): return []
        def get_positions(self): return []
        def get_total_return(self): return 0.0
        def get_sharpe_ratio(self): return 0.0
        def get_max_drawdown(self): return 0.0
        def get_order_count(self): return 0
        def get_trade_count(self): return 0

    module = SimpleNamespace(
        __build_type__="Release", __lto_enabled__=True,
        __compiler_id__="Fixture", BacktestEngine=Engine,
        ExecutionConfig=ExecutionConfig, FeeSchedule=FeeSchedule,
        MarketSnapshot=MarketSnapshot, CorporateAction=CorporateAction,
        HistoryConfig=HistoryConfig,
        EquitySampling=SimpleNamespace(DAILY=1),
        FillTiming=SimpleNamespace(NEXT_OPEN=1),
    )
    config = {
        "enabled": True, "execution_reference_mode": "required_for_promotion",
        "required_models": models,
        "slippage_scenarios_bps": [0, 5, 10], "initial_cash": 1_000_000.0,
        "policy": {}, "risk": {}, "runtime": {"max_order_intents": 1024},
        "execution": {
            "max_volume_participation": 0.1, "slippage_bps": 5.0,
            "enforce_price_limits": True, "enforce_t_plus_one": True,
            "enforce_board_lot": True, "allow_short": False,
            "enforce_cash": True, "market_order_price_buffer_bps": 100.0,
        },
        "fees": {"point_in_time": True, "schedules": [{
            "effective_from": 0, "effective_to": None,
            "commission_rate": 0.0003, "min_commission": 5.0,
            "stamp_tax_rate": 0.0005, "transfer_fee_rate": 0.0,
        }]},
        "corporate_actions": {"point_in_time": True, "source": str(actions_path)},
    }
    destination = run_cpp_portfolio_benchmark(
        config, dataset_path, benchmark, bars_path, tmp_path / "output",
        _engine_module=module,
    )
    report = json.loads(destination.read_text())
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).resolve().parents[2] / (
        "schemas/phase_e_portfolio_backtest.schema.json"
    )
    jsonschema.validate(report, json.loads(schema_path.read_text()))
    assert Engine.runs == 9 * 3 * 3
    assert [item["slippage_bps"] for item in report["scenarios"]] == [0.0, 5.0, 10.0]
    assert all(len(item["windows"]) == 3 for item in report["scenarios"])
    assert set(report["scenarios"][1]["windows"][0]["models"]) == set(models)
    assert report["promotion_eligible"] is True

    np.savez_compressed(
        dataset_path,
        dataset_fingerprint=np.asarray("portfolio-fixture"),
        timestamps=np.asarray([1, 1, 3, 3, 5, 5]),
        symbols=np.asarray(["A", "B", "A", "B", "A", "B"]),
        label_exit_timestamp=np.asarray([2, 2, 4, 4, 6, 6]),
        frequency=np.asarray("1d"), calendar_id=np.asarray("test-calendar"),
        price_adjustment_mode=np.asarray("pit_adjusted_signal_raw_execution"),
        execution_reference_status=np.asarray("DEFERRED"),
    )
    research_bars = pd.DataFrame(rows).drop(columns=[
        "upper_limit", "lower_limit", "lot_size", "min_buy_quantity",
    ])
    research_bars.to_csv(bars_path, index=False)
    config["execution_reference_mode"] = "optional_for_model_evaluation"
    config["execution"]["enforce_price_limits"] = False
    config["execution"]["enforce_board_lot"] = False
    research = run_cpp_portfolio_benchmark(
        config, dataset_path, benchmark, bars_path, tmp_path / "research-output",
        _engine_module=module,
    )
    research_report = json.loads(research.read_text())
    jsonschema.validate(research_report, json.loads(schema_path.read_text()))
    assert research_report["promotion_eligible"] is False
    assert research_report["execution_reference_mode"] == "optional_for_model_evaluation"
    assert research_report["cost_model"]["enforce_price_limits"] is False


def test_cpp_portfolio_ablation_covers_all_seed_group_window_pairs(
    tmp_path, monkeypatch,
):
    groups = list(BAR_V1_FEATURE_GROUPS)
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        dataset_fingerprint=np.asarray("ablation-fixture"),
        timestamps=np.asarray([1, 1, 3, 3, 5, 5]),
        label_exit_timestamp=np.asarray([2, 2, 4, 4, 6, 6]),
        price_adjustment_mode=np.asarray("pit_adjusted_signal_raw_execution"),
        execution_reference_status=np.asarray("READY"),
    )
    ablation = tmp_path / "ablation"
    ablation.mkdir()
    (ablation / "experiment_manifest.json").write_text(json.dumps({
        "groups": groups, "seeds": [7, 8, 9],
        "dataset_fingerprint": "ablation-fixture",
    }))

    def write_prediction(path, timestamp):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, timestamps=np.asarray([timestamp, timestamp]),
            symbols=np.asarray(["A", "B"]),
            expected_return=np.asarray([0.02, 0.01], dtype=np.float32),
            expected_volatility=np.asarray([0.1, 0.1], dtype=np.float32),
            direction_probability=np.asarray([0.8, 0.7], dtype=np.float32),
            lower_quantile=np.asarray([-0.01, -0.01], dtype=np.float32),
            upper_quantile=np.asarray([0.03, 0.02], dtype=np.float32),
            confidence=np.asarray([0.8, 0.7], dtype=np.float32),
        )

    for seed in (7, 8, 9):
        for window_id, timestamp in enumerate((1, 3, 5), 1):
            write_prediction(
                ablation / f"seed_{seed}" / "baseline_full" /
                f"window_{window_id:03d}" / "predictions.npz",
                timestamp,
            )
            for group in groups:
                write_prediction(
                    ablation / f"seed_{seed}" / f"drop_{group}" /
                    f"window_{window_id:03d}" / "predictions.npz",
                    timestamp,
                )
    rows = []
    for timestamp in range(1, 7):
        for symbol, industry in (("A", "I1"), ("B", "I2")):
            rows.append({
                "timestamp": timestamp, "symbol": symbol,
                "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0,
                "signal_open": 10.0, "signal_high": 10.1,
                "signal_low": 9.9, "signal_close": 10.0,
                "adjustment_factor": 1.0, "volume": 10000,
                "is_listed": True, "is_suspended": False, "is_st": False,
                "upper_limit": 11.0, "lower_limit": 9.0, "lot_size": 100,
                "min_buy_quantity": 100,
                "industry": industry, "industry_known_at": timestamp,
                "universe_asof": timestamp,
                "reference_data_known_at_max": timestamp,
            })
    bars_path = tmp_path / "bars.csv"
    pd.DataFrame(rows).to_csv(bars_path, index=False)
    actions_path = tmp_path / "actions.csv"
    pd.DataFrame(columns=[
        "timestamp", "symbol", "cash_dividend_per_share",
        "share_multiplier", "known_at",
    ]).to_csv(actions_path, index=False)
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "net_return": 0.1, "sharpe": 1.2, "max_drawdown": 0.1,
            "turnover": 0.2, "cvar_95": -0.01,
            "symbol_contributions": {"A": 0.05},
            "industry_contributions": {"I1": 0.05},
        }

    monkeypatch.setattr(
        "python.qbt_ml.evaluation.portfolio_benchmark._run_one", fake_run
    )
    config = {
        "enabled": True, "execution_reference_mode": "required_for_promotion",
        "slippage_scenarios_bps": [0, 5, 10],
        "corporate_actions": {"point_in_time": True, "source": str(actions_path)},
    }
    module = SimpleNamespace(__build_type__="Release", __lto_enabled__=True)
    destination = run_cpp_portfolio_ablation(
        config, dataset_path, ablation, bars_path, tmp_path / "output",
        _engine_module=module,
    )
    report = json.loads(destination.read_text())
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).resolve().parents[2] / (
        "schemas/phase_e_portfolio_ablation.schema.json"
    )
    jsonschema.validate(report, json.loads(schema_path.read_text()))
    assert len(calls) == 27 + 189
    assert len(report["results"]) == 3 * len(groups)
    assert all(len(item["scenarios"]) == 3 for item in report["results"])


def test_deep_baseline_suite_trains_identical_walk_forward_windows(tmp_path):
    torch = pytest.importorskip("torch")
    source = _bars(110, ("A", "B"))
    source_path = tmp_path / "bars.csv"
    dataset_path = tmp_path / "dataset.npz"
    source.to_csv(source_path, index=False)
    _build_dataset({
        "enabled": True, "lookback": 8, "label_horizon_bars": 5,
        "calendar_id": "test-calendar", "universe_id": "test-universe",
        "data": {
            "source": str(source_path), "dataset_output": str(dataset_path),
            "point_in_time_required": True,
        },
    }, None)
    output = _walk_forward_deep_baselines({
        "enabled": True, "models": ["mlp", "tcn", "gru"], "seed": 11,
        "windows": {
            "minimum_train_timestamps": 10, "validation_timestamps": 3,
            "test_timestamps": 3, "step_timestamps": 3,
            "purge_timestamps": 5, "embargo_timestamps": 5,
            "minimum_windows": 3,
        },
        "training": {
            "device": "cpu", "epochs": 1, "early_stopping_patience": 1,
            "timestamps_per_batch": 2, "learning_rate": 0.001,
        },
        "model_configs": {
            "mlp": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0},
            "tcn": {
                "hidden_size": 8, "num_layers": 1,
                "kernel_size": 3, "dropout": 0.0,
            },
            "gru": {"hidden_size": 8, "num_layers": 1, "dropout": 0.0},
        },
    }, str(dataset_path), str(tmp_path / "suite"))
    suite = json.loads((output / "benchmark_suite_manifest.json").read_text())
    assert suite["models"] == ["mlp", "tcn", "gru"]
    assert len({suite["runs"][name]["windows"] for name in suite["models"]}) == 1
    for name in suite["models"]:
        checkpoint = torch.load(
            output / name / "window_001" / "checkpoint.pt",
            map_location="cpu", weights_only=True,
        )
        assert checkpoint["model_name"] == name


def test_deep_baselines_ignore_masked_token_values():
    torch = pytest.importorskip("torch")
    from python.qbt_ml.models.deep_baselines import build_deep_baseline

    features = torch.randn(3, 6, 4)
    mask = torch.tensor([
        [0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
    ], dtype=torch.uint8)
    changed = features.clone()
    changed[~mask.bool()] = 1_000_000.0
    for name in ("mlp", "tcn", "gru"):
        model = build_deep_baseline(name, 4, 6, {
            "hidden_size": 8, "num_layers": 1, "dropout": 0.0,
        }).eval()
        before = model(features, mask)
        after = model(changed, mask)
        for output_name in (
            "expected_return", "expected_volatility", "direction_probability",
            "lower_quantile", "upper_quantile", "confidence",
        ):
            torch.testing.assert_close(before[output_name], after[output_name])


def _promotion_fixture(tmp_path):
    groups = list(BAR_V1_FEATURE_GROUPS)
    transformer = tmp_path / "transformer"
    benchmark = tmp_path / "benchmark"
    ablation = tmp_path / "ablation"
    transformer.mkdir()
    benchmark.mkdir()
    ablation.mkdir()
    (transformer / "experiment_manifest.json").write_text(json.dumps({
        "model_family": "TemporalTransformerV1.1",
        "dataset_fingerprint": "fixture-dataset",
        "windows": 3,
    }))
    for window_id in range(1, 4):
        root = transformer / f"window_{window_id:03d}"
        root.mkdir()
        (root / "leakage_report.json").write_text(json.dumps({"status": "PASS"}))
    (benchmark / "model_quality.json").write_text(json.dumps({"records": [
        {"model": "transformer_v1_1", "window_id": window_id, "rank_ic": value}
        for window_id, value in enumerate((0.03, 0.04, 0.01), 1)
    ]}))
    (benchmark / "benchmark_manifest.json").write_text(json.dumps({
        "evaluated_models": ["transformer_v1_1", "ridge_multitask"],
        "dataset_fingerprint": "fixture-dataset",
    }))
    (ablation / "experiment_manifest.json").write_text(json.dumps({
        "model_family": "TemporalTransformerV1.1", "groups": groups,
        "seeds": [7, 8, 9], "dataset_fingerprint": "fixture-dataset",
        "formal_transformer_conclusion": True,
    }))
    paths = {}
    for name, value in {
        "portfolio_ablation.json": {
            "schema_version": 2, "engine": "cpp", "net_of_cost": True,
            "execution_reference_mode": "required_for_promotion",
            "promotion_eligible": True,
            "groups": groups, "dataset_fingerprint": "fixture-dataset",
            "seeds": [7, 8, 9],
            "slippage_scenarios_bps": [0, 5, 10],
            "results": [
                {
                    "seed": seed, "group": group,
                    "scenarios": [
                        {"slippage_bps": slippage, "windows": [{}, {}, {}]}
                        for slippage in (0, 5, 10)
                    ],
                }
                for seed in (7, 8, 9) for group in groups
            ],
        },
        "parity.json": {
            "status": "PASS", "model_id": "candidate",
            "model_version": "1",
        },
        "runtime.json": {
            "build_type": "Release", "lto_enabled": True,
            "benchmark_scope": "production_candidate",
            "training_dataset_fingerprint": "fixture-dataset",
            "model_id": "candidate", "model_version": "1",
            "results": [{
                "batch_size": 4096, "end_to_end": {"p99_ns": 15_000_000},
            }],
        },
    }.items():
        path = tmp_path / name
        path.write_text(json.dumps(value))
        paths[name] = path
    models = {
        "transformer_v1_1": {
            "net_return": 0.12, "sharpe": 1.5, "max_drawdown": 0.10,
            "turnover": 0.18, "cvar_95": -0.02,
            "symbol_contributions": {f"S{i}": 0.02 for i in range(6)},
            "industry_contributions": {f"I{i}": 0.04 for i in range(3)},
        },
        "ridge_multitask": {
            "net_return": 0.05, "sharpe": 1.0, "max_drawdown": 0.12,
            "turnover": 0.12, "cvar_95": -0.03,
        },
    }
    portfolio = {
        "schema_version": 2, "engine": "cpp", "net_of_cost": True,
        "execution_reference_mode": "required_for_promotion",
        "promotion_eligible": True,
        "dataset_fingerprint": "fixture-dataset",
        "cost_model": {
            "point_in_time_fees": True, "enforce_t_plus_one": True,
            "enforce_price_limits": True, "enforce_board_lot": True,
            "max_volume_participation": 0.10,
        },
        "scenarios": [
            {"slippage_bps": slippage, "windows": [
                {"window_id": window_id, "test_start": window_id * 10,
                 "test_end": window_id * 10 + 5, "models": models}
                for window_id in range(1, 4)
            ]}
            for slippage in (0, 5, 10)
        ],
    }
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(json.dumps(portfolio))
    config = {
        "enabled": True,
        "evidence": {
            "transformer_run": str(transformer),
            "model_benchmark": str(benchmark),
            "feature_ablation": str(ablation),
            "portfolio_ablation": str(paths["portfolio_ablation.json"]),
            "portfolio_backtest": str(portfolio_path),
            "ort_cpp_parity": str(paths["parity.json"]),
            "runtime_benchmark": str(paths["runtime.json"]),
        },
        "thresholds": {
            "candidate_model": "transformer_v1_1",
            "required_baselines": ["ridge_multitask"],
            "minimum_windows": 3, "min_rank_ic_median": 0.02,
            "min_net_sharpe": 1.0, "min_sharpe_improvement": 0.10,
            "max_drawdown": 0.20, "min_cvar_95": -0.05,
            "min_winning_window_fraction": 2 / 3,
            "max_period_top1_share": 0.40,
            "max_symbol_top1_share": 0.20,
            "max_symbol_top5_share": 0.90,
            "max_industry_top1_share": 0.40,
            "required_slippage_bps": [0, 5, 10],
            "primary_slippage_bps": 5,
            "runtime_batch_size": 4096,
            "max_runtime_p99_ns": 20_000_000,
        },
    }
    return config, paths


def test_promotion_review_separates_software_and_quality_gates(tmp_path):
    config, _ = _promotion_fixture(tmp_path)
    output = _promotion_review(config, str(tmp_path / "review"))
    report = json.loads((output / "promotion_report.json").read_text())
    assert report["decision"] == "PROMOTE"
    assert report["software_gate"] == "PASS"
    assert report["model_quality_gate"] == "PASS"
    assert (output / "multi_window_portfolio.json").is_file()
    assert (output / "concentration_stability.json").is_file()


def test_promotion_review_rejects_failed_software_gate(tmp_path):
    config, paths = _promotion_fixture(tmp_path)
    paths["parity.json"].write_text(json.dumps({"status": "FAIL"}))
    output = _promotion_review(config, str(tmp_path / "review"))
    report = json.loads((output / "promotion_report.json").read_text())
    assert report["decision"] == "REJECT"
    assert report["software_gate"] == "FAIL"
    assert report["model_quality_gate"] == "PASS"


def test_promotion_review_does_not_promote_missing_cpp_evidence(tmp_path):
    config, _ = _promotion_fixture(tmp_path)
    config["evidence"]["portfolio_backtest"] = "CONFIGURE_CPP_BACKTEST"
    output = _promotion_review(config, str(tmp_path / "review"))
    report = json.loads((output / "promotion_report.json").read_text())
    assert report["decision"] == "INSUFFICIENT_EVIDENCE"
    assert report["software_gate"] == "INSUFFICIENT_EVIDENCE"
    assert report["model_quality_gate"] == "INSUFFICIENT_EVIDENCE"


def test_promotion_review_treats_deferred_execution_as_insufficient(tmp_path):
    config, paths = _promotion_fixture(tmp_path)
    portfolio_path = Path(config["evidence"]["portfolio_backtest"])
    portfolio = json.loads(portfolio_path.read_text())
    portfolio["execution_reference_mode"] = "optional_for_model_evaluation"
    portfolio["promotion_eligible"] = False
    portfolio["cost_model"]["enforce_price_limits"] = False
    portfolio["cost_model"]["enforce_board_lot"] = False
    portfolio_path.write_text(json.dumps(portfolio))
    portfolio_ablation = json.loads(paths["portfolio_ablation.json"].read_text())
    portfolio_ablation["execution_reference_mode"] = "optional_for_model_evaluation"
    portfolio_ablation["promotion_eligible"] = False
    paths["portfolio_ablation.json"].write_text(json.dumps(portfolio_ablation))
    output = _promotion_review(config, str(tmp_path / "review"))
    report = json.loads((output / "promotion_report.json").read_text())
    assert report["decision"] == "INSUFFICIENT_EVIDENCE"
    assert report["software_gate"] == "INSUFFICIENT_EVIDENCE"
    assert report["model_quality_gate"] == "INSUFFICIENT_EVIDENCE"


def test_phase_e_frozen_thresholds_are_numeric_and_frequency_scoped():
    config_path = Path(__file__).resolve().parents[2] / "configs/ml/promotion_review_v1_1.json"
    thresholds = json.loads(config_path.read_text())["thresholds"]
    assert thresholds["min_cvar_95"] == -0.03
    assert thresholds["max_period_top1_share"] == 0.50
    assert thresholds["max_symbol_top1_share"] == 0.10
    assert thresholds["max_symbol_top5_share"] == 0.30
    assert thresholds["max_industry_top1_share"] == 0.30
    assert thresholds["runtime_batch_size"] == 4096
    assert thresholds["max_runtime_p99_ns"] == 50_000_000


def test_v1_1_multitask_heads_are_supervised_and_confidence_is_derived():
    torch = pytest.importorskip("torch")
    from python.qbt_ml.models.temporal_transformer import (
        TemporalTransformerConfig, TemporalTransformerV1,
    )
    from python.qbt_ml.training.train import multitask_loss_components

    model = TemporalTransformerV1(TemporalTransformerConfig(
        feature_count=3, lookback=4, d_model=8, nhead=2,
        num_layers=1, dim_feedforward=16, dropout=0.0,
    ))
    features = torch.randn(5, 4, 3)
    valid_mask = torch.tensor([
        [0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1],
        [1, 1, 1, 1], [0, 1, 1, 1],
    ], dtype=torch.uint8)
    prediction = model(features, valid_mask)
    target = {
        "timestamps": torch.tensor([1, 1, 1, 2, 2]),
        "expected_return": torch.linspace(-0.1, 0.1, 5),
        "direction": torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]),
        "realized_volatility": torch.linspace(0.01, 0.05, 5),
    }
    components = multitask_loss_components(prediction, target, rank_weight=0.1)
    components["total"].backward()
    assert set(components) == {
        "return", "direction", "volatility", "quantile", "rank", "total"
    }
    assert model.return_head.weight.grad.abs().sum() > 0
    assert model.volatility_head.weight.grad.abs().sum() > 0
    assert model.direction_head.weight.grad.abs().sum() > 0
    assert model.quantile_head.weight.grad.abs().sum() > 0
    assert torch.all(prediction["lower_quantile"] <= prediction["expected_return"])
    assert torch.all(prediction["upper_quantile"] >= prediction["expected_return"])
    torch.testing.assert_close(
        prediction["confidence"],
        torch.maximum(
            prediction["direction_probability"],
            1.0 - prediction["direction_probability"],
        ),
    )


def test_transformer_walk_forward_trains_three_windows(tmp_path):
    pytest.importorskip("torch")
    source = _bars(140, ("A", "B", "C"))
    source_path = tmp_path / "bars.csv"
    dataset_path = tmp_path / "dataset.npz"
    source.to_csv(source_path, index=False)
    _build_dataset({
        "enabled": True, "lookback": 8, "label_horizon_bars": 5,
        "calendar_id": "test-calendar", "universe_id": "test-universe",
        "data": {
            "source": str(source_path), "dataset_output": str(dataset_path),
            "point_in_time_required": True,
        },
    }, None)
    output = _walk_forward_transformer({
        "enabled": True,
        "seed": 7,
        "windows": {
            "minimum_train_timestamps": 25,
            "validation_timestamps": 6,
            "test_timestamps": 6,
            "step_timestamps": 6,
            "purge_timestamps": 5,
            "embargo_timestamps": 5,
            "minimum_windows": 3,
        },
        "model": {
            "d_model": 8, "nhead": 2, "num_layers": 1,
            "dim_feedforward": 16, "dropout": 0.0,
        },
        "training": {
            "device": "cpu", "epochs": 1, "early_stopping_patience": 1,
            "timestamps_per_batch": 2, "learning_rate": 0.001,
            "loss_weights": {"rank": 0.1},
        },
    }, str(dataset_path), str(tmp_path / "transformer-walk-forward"))
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["model_family"] == "TemporalTransformerV1.1"
    assert manifest["windows"] >= 3
    for window in sorted(output.glob("window_*")):
        assert (window / "checkpoint.pt").is_file()
        assert json.loads((window / "leakage_report.json").read_text())["status"] == "PASS"
        assert (window / "label_spec.json").is_file()
        assert (window / "normalization.json").is_file()
        calibration = json.loads((window / "calibration.json").read_text())
        assert calibration["input"] == "direction_logits"
        checkpoint = torch.load(
            window / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        assert checkpoint["training_dataset_fingerprint"] == manifest[
            "dataset_fingerprint"
        ]

    benchmark = _benchmark_models({
        "enabled": True,
        "transformer_run": str(output),
        "walk_forward": {
            "windows": {
                "minimum_train_timestamps": 25,
                "validation_timestamps": 6,
                "test_timestamps": 6,
                "step_timestamps": 6,
                "purge_timestamps": 5,
                "embargo_timestamps": 5,
                "minimum_windows": 3,
            },
            "model": {"ridge_alpha_candidates": [0.1, 1.0]},
        },
    }, str(dataset_path), str(tmp_path / "transformer-benchmark"))
    benchmark_manifest = json.loads(
        (benchmark / "benchmark_manifest.json").read_text()
    )
    assert "transformer_v1_1" in benchmark_manifest["evaluated_models"]
    assert "transformer_v1_1" not in benchmark_manifest["not_evaluated_models"]
    pairs = json.loads(
        (benchmark / "paired_window_deltas.json").read_text()
    )["pairs"]
    transformer_pairs = [
        pair for pair in pairs if pair["model"] == "transformer_v1_1"
    ]
    assert len(transformer_pairs) == manifest["windows"]


def test_transformer_group_drop_ablation_retrains_each_variant(tmp_path):
    pytest.importorskip("torch")
    source = _bars(140, ("A", "B", "C"))
    source_path = tmp_path / "bars.csv"
    dataset_path = tmp_path / "dataset.npz"
    source.to_csv(source_path, index=False)
    _build_dataset({
        "enabled": True, "lookback": 8, "label_horizon_bars": 5,
        "calendar_id": "test-calendar", "universe_id": "test-universe",
        "data": {
            "source": str(source_path), "dataset_output": str(dataset_path),
            "point_in_time_required": True,
        },
    }, None)
    output = _ablate_transformer_features({
        "enabled": True,
        "groups": ["returns"],
        "seeds": [7, 8, 9],
        "walk_forward": {
            "enabled": True,
            "seed": 7,
            "windows": {
                "minimum_train_timestamps": 25,
                "validation_timestamps": 6,
                "test_timestamps": 6,
                "step_timestamps": 6,
                "purge_timestamps": 5,
                "embargo_timestamps": 5,
                "minimum_windows": 3,
            },
            "model": {
                "d_model": 8, "nhead": 2, "num_layers": 1,
                "dim_feedforward": 16, "dropout": 0.0,
            },
            "training": {
                "device": "cpu", "epochs": 1,
                "early_stopping_patience": 1,
                "timestamps_per_batch": 2, "learning_rate": 0.001,
            },
        },
    }, str(dataset_path), str(tmp_path / "transformer-ablation"))
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["model_family"] == "TemporalTransformerV1.1"
    assert manifest["formal_transformer_conclusion"] is True
    assert manifest["formal_conclusion_scope"] == "prediction_tasks_only"
    assert manifest["formal_model_promotion_conclusion"] is False
    assert manifest["artifact_policy"] == "analysis_only_not_for_production_export"
    result = json.loads((output / "task_delta.json").read_text())["results"][0]
    assert result["group"] == "returns"
    assert result["remaining_feature_count"] == (
        len(BAR_V1.feature_names) - len(BAR_V1_FEATURE_GROUPS["returns"])
    )
    assert [item["seed"] for item in result["per_seed"]] == [7, 8, 9]
    assert set(result["metric_delta_std_across_seeds"]) == set(
        result["metric_delta_drop_minus_full"]
    )
    baseline_manifest = json.loads(
        (output / "seed_7" / "baseline_full" / "experiment_manifest.json").read_text()
    )
    dropped_manifest = json.loads(
        (output / "seed_7" / "drop_returns" / "experiment_manifest.json").read_text()
    )
    assert baseline_manifest["feature_indices"] == list(range(len(BAR_V1.feature_names)))
    assert manifest["dataset_fingerprint"] == baseline_manifest["dataset_fingerprint"]
    assert len(dropped_manifest["feature_indices"]) == result["remaining_feature_count"]
    dropped_window = output / "seed_7" / "drop_returns" / "window_001"
    assert (dropped_window / "checkpoint.pt").is_file()
    stability = json.loads((output / "stability.json").read_text())
    assert stability["formal_seed_requirement_met"] is True
    with pytest.raises(ValueError, match="仅用于分析"):
        _export({
            "enabled": True,
            "calendar_id": "test-calendar",
            "universe_id": "test-universe",
            "data_cutoff_utc": "2026-01-01T00:00:00Z",
        }, str(dropped_window), str(tmp_path / "invalid-export"))


def test_transformer_onnx_python_parity_supports_dynamic_batch(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from python.qbt_ml.export.onnx_export import export_temporal_transformer
    from python.qbt_ml.export.validate import validate_onnx_parity
    from python.qbt_ml.models.temporal_transformer import (
        TemporalTransformerConfig, TemporalTransformerV1,
    )

    model = TemporalTransformerV1(TemporalTransformerConfig(
        feature_count=3, lookback=4, d_model=8, nhead=2,
        num_layers=1, dim_feedforward=16, dropout=0.0,
    )).eval()
    model.set_normalization(
        np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
        np.asarray([1.0, 2.0, 0.5], dtype=np.float32),
    )
    model_path = export_temporal_transformer(model, tmp_path / "model.onnx")
    generator = np.random.default_rng(7)
    features = generator.normal(size=(3, 4, 3)).astype(np.float32)
    valid_mask = np.asarray([
        [0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1],
    ], dtype=np.uint8)
    outputs = validate_onnx_parity(model, model_path, features, valid_mask)
    assert all(value.shape == (3,) for value in outputs.values())


def test_training_exports_manifest_v2_with_calibrated_lineage(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    source = _bars(100, ("A", "B", "C"))
    source_path = tmp_path / "bars.csv"
    dataset_path = tmp_path / "dataset.npz"
    source.to_csv(source_path, index=False)
    _build_dataset({
        "enabled": True, "lookback": 8, "label_horizon_bars": 5,
        "calendar_id": "test-calendar", "universe_id": "test-universe",
        "data": {
            "source": str(source_path), "dataset_output": str(dataset_path),
            "point_in_time_required": True,
        },
    }, None)
    run = _train({
        "enabled": True, "seed": 7,
        "model": {
            "d_model": 8, "nhead": 2, "num_layers": 1,
            "dim_feedforward": 16, "dropout": 0.0,
        },
        "training": {
            "dataset": str(dataset_path), "output": str(tmp_path / "train"),
            "device": "cpu", "epochs": 1, "timestamps_per_batch": 2,
            "learning_rate": 0.001,
        },
        "split": {
            "train_fraction": 0.6, "validation_fraction": 0.2,
            "test_fraction": 0.2, "purge_timestamps": 5,
            "embargo_timestamps": 5,
        },
    }, None, None)
    artifact = _export({
        "enabled": True,
        "calendar_id": "test-calendar",
        "universe_id": "test-universe",
        "data_cutoff_utc": "2026-01-01T00:00:00Z",
        "model_id": "test-transformer-v1-1",
        "model_version": "test-v2",
        "minimum_valid_tokens": 2,
        "export": {"opset": 17},
    }, str(run), str(tmp_path / "artifact"))
    manifest = validate_artifact(artifact)
    import jsonschema
    schema = json.loads(
        (Path(__file__).parents[2] / "schemas" / "model_manifest.schema.json")
        .read_text(encoding="utf-8")
    )
    jsonschema.validate(
        json.loads((artifact / "manifest.json").read_text(encoding="utf-8")), schema
    )
    assert manifest.schema_version == 2
    assert manifest.model_family == "TemporalTransformer"
    assert manifest.architecture_version == "V1.1"
    assert manifest.frequency == "1d"
    assert manifest.minimum_valid_tokens == 2
    assert manifest.dynamic_batch is True
    assert manifest.calibration_method == "platt_validation_only"
    calibration = json.loads((artifact / "calibration.json").read_text())
    assert np.isfinite([calibration["intercept"], calibration["slope"]]).all()
    assert {item["name"] for item in manifest.outputs} == set(
        name for name in (
            "expected_return", "expected_volatility", "direction_probability",
            "lower_quantile", "upper_quantile", "confidence",
        )
    )
    assert (artifact / "golden" / "pytorch_output.npz").is_file()
    assert (artifact / "golden" / "onnx_output.npz").is_file()
    assert (artifact / "golden" / "expected_decisions.json").is_file()
    from python.qbt_ml.export import validate_ort_cpp_parity
    fake_runner = tmp_path / "fake-ort-runner"
    fake_runner.write_text(
        "#!/bin/sh\nprintf '%s\\n' "
        "'{\"status\":\"PASS\",\"batch_size\":8,"
        "\"max_absolute_error\":0.0,\"max_relative_error\":0.0,"
        "\"top_k\":3,\"top_k_match\":true,"
        "\"target_positions_match\":true}'\n",
        encoding="utf-8",
    )
    fake_runner.chmod(0o755)
    parity_path = validate_ort_cpp_parity(artifact, fake_runner)
    assert json.loads(parity_path.read_text())["status"] == "PASS"


def test_manifest_validates_required_files_and_hashes(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"model")
    BAR_V1.write(tmp_path / "feature_schema.json")
    assert sha256_file(tmp_path / "feature_schema.json") == BAR_V1.sha256
    (tmp_path / "metrics.json").write_text("{}\n", encoding="utf-8")
    outputs = tuple({
        "name": name, "unit": "probability" if "probability" in name or name == "confidence"
        else "log_return", "horizon_bars": 5,
    } for name in (
        "expected_return", "expected_volatility", "direction_probability",
        "lower_quantile", "upper_quantile", "confidence",
    ))
    manifest = ModelManifest(
        1, "test-model", "1", sha256_file(tmp_path / "model.onnx"), "BAR_V1",
        sha256_file(tmp_path / "feature_schema.json"), "calendar", "universe",
        "2026-07-24T00:00:00Z", 64, len(BAR_V1.feature_names), 0, outputs,
    )
    manifest.write(tmp_path / "manifest.json")
    assert validate_artifact(tmp_path).model_id == "test-model"
    value = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    value["model_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="model.onnx"):
        validate_artifact(tmp_path)


def test_backtest_artifact_streams_cross_sections_and_writes_summary(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.onnx").write_bytes(b"model")
    BAR_V1.write(artifact / "feature_schema.json")
    (artifact / "metrics.json").write_text("{}\n", encoding="utf-8")
    outputs = tuple({
        "name": name,
        "unit": "probability" if name in {"direction_probability", "confidence"}
                else "return_std" if name == "expected_volatility" else "log_return",
        "horizon_bars": 5,
    } for name in (
        "expected_return", "expected_volatility", "direction_probability",
        "lower_quantile", "upper_quantile", "confidence",
    ))
    ModelManifest(
        1, "test-model", "1", sha256_file(artifact / "model.onnx"), "BAR_V1",
        sha256_file(artifact / "feature_schema.json"), "calendar", "universe",
        "2026-07-24T00:00:00Z", 64, len(BAR_V1.feature_names), 0, outputs,
    ).write(artifact / "manifest.json")
    bars = _bars(3, ("B", "A"))
    data_path = tmp_path / "bars.csv"
    bars.to_csv(data_path, index=False)
    output_path = tmp_path / "summary.json"

    class ExecutionConfig:
        pass

    class MarketSnapshot:
        pass

    class Engine:
        def __init__(self, initial_cash, fill_timing, execution):
            self.initial_cash = initial_cash
            self.batches = []

        def set_model_strategy(self, artifact_path, policy, risk, runtime):
            self.artifact_path = artifact_path

        def process_market_data_batch(self, batch):
            self.batches.append(batch)

        def finalize(self, timestamp):
            self.final_timestamp = timestamp

        def get_cash(self): return self.initial_cash
        def get_equity(self): return self.initial_cash
        def get_total_return(self): return 0.0
        def get_sharpe_ratio(self): return 0.0
        def get_max_drawdown(self): return 0.0
        def get_order_count(self): return 0
        def get_trade_count(self): return 0

    module = SimpleNamespace(
        __ml_enabled__=True, __ml_backend__="onnxruntime",
        ExecutionConfig=ExecutionConfig, MarketSnapshot=MarketSnapshot,
        BacktestEngine=Engine, FillTiming=SimpleNamespace(NEXT_OPEN=1),
    )
    result = _backtest_artifact(
        artifact, data_path, output_path=output_path, _engine_module=module
    )
    assert result["rows"] == 6 and result["batches"] == 3 and result["symbols"] == 2
    assert json.loads(output_path.read_text(encoding="utf-8")) == result

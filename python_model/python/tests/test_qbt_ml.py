from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from python.qbt_ml.cli import _backtest_artifact, _build_dataset
from python.qbt_ml.data import BAR_V1, build_windows
from python.qbt_ml.export.manifest import ModelManifest, sha256_file, validate_artifact
from python.qbt_ml.features import build_bar_v1
from python.qbt_ml.labels import build_next_open_labels
from python.qbt_ml.models import RidgeBaseline
from python.qbt_ml.training import chronological_timestamp_split, walk_forward_splits


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
            })
    return pd.DataFrame(rows)


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
    assert first.exit_close == 12.5
    assert first.expected_return == pytest.approx(np.log(12.5 / 11.0))
    assert labels["label_valid"].tolist() == [True, True, False, False]


def test_walk_forward_purge_embargo_and_ridge_baseline():
    split = next(iter(walk_forward_splits(
        30, train_size=10, validation_size=5, test_size=5, purge=2, embargo=1
    )))
    assert split.train[-1] == 9 and split.validation[0] == 12 and split.test[0] == 20
    x = np.arange(10, dtype=np.float64).reshape(-1, 1)
    model = RidgeBaseline(alpha=0.0).fit(x, 2 * x[:, 0] + 1)
    np.testing.assert_allclose(model.predict([[10.0]]), [21.0], atol=1e-10)


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


def test_dataset_builder_sorts_timestamp_symbol_and_excludes_invalid_asof(tmp_path):
    source = _bars(70, ("000002", "000001")).sample(frac=1.0, random_state=7)
    source.loc[(source["timestamp"] == 65) & (source["symbol"] == "000001"),
               "is_suspended"] = True
    source_path = tmp_path / "bars.csv"
    output_path = tmp_path / "dataset.npz"
    source.to_csv(source_path, index=False)
    _build_dataset({
        "enabled": True, "lookback": 8, "label_horizon_bars": 1,
        "data": {"source": str(source_path),
                 "dataset_output": str(output_path)},
    }, None)
    with np.load(output_path, allow_pickle=False) as data:
        keys = list(zip(data["timestamps"].tolist(), data["symbols"].tolist()))
        assert keys == sorted(keys)
        assert (65, "000001") not in keys
        assert "000001" in data["symbols"].tolist()
        assert data["valid_mask"][:, -1].all()


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

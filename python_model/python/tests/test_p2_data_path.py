from __future__ import annotations

import csv
import os
import sys

import pytest

from python import engine_api as api
from python.benchmarks import MinuteReplayBenchmark
from python.data_feed import CSVDataFeed, DataLakeFeed
from python.events import BatchEventRunner, EventType, HistoricalReplaySource
from python.market_data import (
    AdjustmentFactor,
    AdjustmentFactorStore,
    ArrowDatasetScanner,
    DataLineage,
    DataLakeConfig,
    FeatureContext,
    FeatureDefinition,
    MaterializedFeatureCache,
    MarketReferenceData,
    MinuteBarDataLake,
    PartitionAwareIterator,
    SecurityMaster,
    SecurityState,
)


DAY = 86_400_000_000_000
BASE = 1_768_780_800_000_000_000


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("timestamp", "symbol", "open", "high", "low", "close", "volume"))
        writer.writerows(rows)


def _rows(day_offset=0):
    rows = []
    for minute in range(3):
        timestamp = BASE + day_offset * DAY + minute * 60_000_000_000
        for index, symbol in enumerate(("000001", "300001", "600000")):
            price = 10.0 + index + minute / 10
            rows.append((timestamp, symbol, price, price, price, price, 1000))
    return rows


def test_scanner_pushdown_projection_and_streaming(tmp_path):
    source = tmp_path / "bars.csv"
    _write(source, _rows())
    lake = MinuteBarDataLake(DataLakeConfig(tmp_path / "lake", bucket_count=4))
    lake.ingest(source)
    scanner = ArrowDatasetScanner(lake)

    batches = list(scanner.iter_batches(
        symbols="000001",
        start=BASE + 60_000_000_000,
        columns=("timestamp", "symbol", "close"),
        batch_size=1,
    ))
    assert sum(batch.num_rows for batch in batches) == 2
    assert batches[0].schema.names == ["timestamp", "symbol", "close"]
    assert all(value == "000001" for batch in batches
               for value in batch["symbol"].to_pylist())


def test_partition_iterator_is_sorted_and_keeps_timestamp_atomic(tmp_path):
    source = tmp_path / "bars.csv"
    _write(source, _rows() + _rows(1))
    lake = MinuteBarDataLake(DataLakeConfig(tmp_path / "lake", bucket_count=4))
    lake.ingest(source)
    replay = PartitionAwareIterator(
        ArrowDatasetScanner(lake), target_bytes=128, scanner_batch_size=2
    )
    batches = list(replay.iter_batches(start=BASE, end=BASE + DAY + 2 * 60_000_000_000))

    keys = []
    timestamp_batches = {}
    for batch_index, batch in enumerate(batches):
        for timestamp, symbol in zip(
            batch["timestamp"].to_pylist(), batch["symbol"].to_pylist()
        ):
            keys.append((timestamp, symbol))
            timestamp_batches.setdefault(timestamp, set()).add(batch_index)
    assert keys == sorted(keys)
    assert len(keys) == 18
    assert all(len(indices) == 1 for indices in timestamp_batches.values())
    assert replay.stats.rows == 18
    assert replay.stats.peak_batch_bytes >= replay.target_bytes


def test_lineage_changes_after_catalog_commit(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write(first, _rows())
    _write(second, _rows(1))
    lake = MinuteBarDataLake(DataLakeConfig(tmp_path / "lake", bucket_count=4))
    lake.ingest(first)
    before = lake.lineage()
    query = before.with_query(
        symbols="000001", start=BASE, end=BASE + DAY,
        columns=("timestamp", "symbol", "close"),
    )
    lake.ingest(second)
    after = lake.lineage()
    assert before.catalog_generation == 1
    assert after.catalog_generation == 2
    assert before.schema_hash == after.schema_hash
    assert before.dataset_fingerprint != after.dataset_fingerprint
    assert query.query_fingerprint
    assert all(item.content_hash for item in lake.catalog.files())

    copy = tmp_path / "same-content.csv"
    _write(copy, _rows())
    file_lineage = CSVDataFeed(first, symbols="000001", start=BASE).lineage()
    copied_lineage = CSVDataFeed(copy, symbols="000001", start=BASE).lineage()
    changed_query = CSVDataFeed(copy, symbols="600000", start=BASE).lineage()
    assert file_lineage.dataset_fingerprint == copied_lineage.dataset_fingerprint
    assert file_lineage.query_fingerprint == copied_lineage.query_fingerprint
    assert file_lineage.query_fingerprint != changed_query.query_fingerprint


def test_arrow_c_stream_bridge_consumes_native_batches(tmp_path):
    cpp = pytest.importorskip("cpp_engine")
    source = tmp_path / "bars.csv"
    _write(source, _rows())
    lake = MinuteBarDataLake(DataLakeConfig(tmp_path / "lake", bucket_count=4))
    lake.ingest(source)
    replay = PartitionAwareIterator(
        ArrowDatasetScanner(lake), target_bytes=128, scanner_batch_size=2
    )
    engine = cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE)
    stats = engine.process_arrow_stream(replay.reader(start=BASE))

    assert stats.rows == 9
    assert stats.batches >= 1
    assert stats.bytes > 0
    assert stats.decode_seconds >= 0.0
    assert stats.execution_seconds >= 0.0
    assert len(engine.get_equity_curve()) == 3


def test_arrow_bridge_receives_enriched_risk_columns():
    cpp = pytest.importorskip("cpp_engine")
    import pyarrow as pa

    reference = MarketReferenceData(
        security_master=SecurityMaster([
            SecurityState(
                "X", 0, None, industry="Bank", lot_size=1,
                min_buy_quantity=1, factor_exposures={"value": 0.7},
            ),
        ]),
        adjustment_factors=AdjustmentFactorStore([
            AdjustmentFactor("X", 0, 0.5),
        ]),
    )
    raw = pa.record_batch({
        "timestamp": pa.array([1], type=pa.int64()),
        "symbol": ["X"],
        "open": [10.0], "high": [10.0], "low": [10.0], "close": [10.0],
        "volume": pa.array([100], type=pa.int64()),
    })
    enriched = reference.enrich_batch(raw)
    engine = cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE)
    observed = []

    def strategy(batch):
        observed.append((batch[0].signal_close, batch[0].factor_exposures))
        order = cpp.Order()
        order.symbol = "X"
        order.side = cpp.Side.BUY
        order.quantity = 1
        return [order]

    engine.set_on_cross_section(strategy)
    reader = pa.RecordBatchReader.from_batches(enriched.schema, [enriched])
    engine.process_arrow_stream(reader)

    assert observed == [(5.0, {"value": pytest.approx(0.7)})]
    portfolio = engine.get_portfolio_snapshot()
    assert portfolio.industry_exposure["Bank"] > 0.0
    assert portfolio.factor_exposure["value"] == pytest.approx(0.0007)


def test_arrow_bridge_rejects_unsorted_input():
    cpp = pytest.importorskip("cpp_engine")
    import pyarrow as pa

    table = pa.table({
        "timestamp": pa.array([2, 1], type=pa.int64()),
        "symbol": ["X", "X"],
        "open": [10.0, 10.0], "high": [10.0, 10.0],
        "low": [10.0, 10.0], "close": [10.0, 10.0],
        "volume": pa.array([100, 100], type=pa.int64()),
    })
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    with pytest.raises(ValueError, match="strictly ordered"):
        cpp.BacktestEngine().process_arrow_stream(reader)

    optional = table.slice(1, 1).append_column(
        "upper_limit", pa.array([None], type=pa.float64())
    ).append_column(
        "lot_size", pa.array([None], type=pa.int64())
    ).append_column(
        "adjustment_factor", pa.array([None], type=pa.float64())
    )
    optional_reader = pa.RecordBatchReader.from_batches(
        optional.schema, optional.to_batches()
    )
    stats = cpp.BacktestEngine().process_arrow_stream(optional_reader)
    assert stats.rows == 1


def test_materialized_feature_cache_is_lineage_bound(tmp_path):
    import pyarrow as pa

    cache = MaterializedFeatureCache(tmp_path / "features", bucket_count=4)
    definition = FeatureDefinition(
        "momentum", "1", ("momentum_20",), {"window": 20}, "code-v1"
    )
    context = FeatureContext("calendar-v1", "universe-v1", "forward")
    lineage = DataLineage(1, "schema", "dataset-v1", ("source",), "query-v1")
    table = pa.table({
        "timestamp": pa.array([BASE, BASE], type=pa.int64()),
        "symbol": ["000001", "600000"],
        "momentum_20": [0.1, -0.2],
    })

    entry = cache.materialize(definition, table, lineage, context)
    assert not entry.skipped
    assert cache.materialize(definition, table, lineage, context).skipped
    batches = list(cache.scanner(
        entry, symbols="000001",
        columns=("timestamp", "symbol", "momentum_20"),
    ).to_batches())
    assert sum(batch.num_rows for batch in batches) == 1
    assert batches[0]["momentum_20"][0].as_py() == pytest.approx(0.1)

    changed = DataLineage(2, "schema", "dataset-v2", ("source2",), "query-v2")
    assert cache.lookup(definition, changed, context) is None
    changed_definition = FeatureDefinition(
        "momentum", "1", ("momentum_20",), {"window": 21}, "code-v1"
    )
    assert cache.key(definition, lineage, context) != cache.key(
        changed_definition, lineage, context
    )


def test_historical_event_source_replays_atomic_batches(tmp_path):
    cpp = pytest.importorskip("cpp_engine")
    source = tmp_path / "bars.csv"
    _write(source, _rows() + _rows(1))
    lake = MinuteBarDataLake(DataLakeConfig(tmp_path / "lake", bucket_count=4))
    lake.ingest(source)
    replay = PartitionAwareIterator(
        ArrowDatasetScanner(lake), target_bytes=128, scanner_batch_size=2
    )
    timer_timestamp = BASE + 60_000_000_000
    event_source = HistoricalReplaySource(
        replay, start=BASE, end=BASE + DAY + 2 * 60_000_000_000,
        timers=[(timer_timestamp, "rebalance")],
    )
    events = list(event_source.events())
    market_events = [event for event in events
                     if event.type == EventType.MARKET_DATA_BATCH]
    assert len(market_events) == 6
    assert all(event.record_batch.num_rows == 3 for event in market_events)
    assert [event.timestamp for event in events if event.type == EventType.TIMER] == [
        timer_timestamp
    ]

    observed_timers = []
    stats = BatchEventRunner(
        cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE),
        on_timer=lambda event: observed_timers.append(event.name),
    ).run(event_source)
    assert stats.rows == 18
    assert stats.market_batches == 6
    assert stats.stream_calls == 3
    assert observed_timers == ["rebalance"]


def _assert_split_timestamp_replay(engine, replay, expected_timestamps):
    observed = []
    engine.set_on_cross_section(
        lambda batch: observed.append([item.symbol for item in batch]) or []
    )
    stats = BatchEventRunner(engine).run(HistoricalReplaySource(replay))

    assert stats.rows == expected_timestamps * 3
    assert stats.market_batches == expected_timestamps
    assert observed == [["000001", "300001", "600000"]] * expected_timestamps
    assert len(engine.get_equity_curve()) == expected_timestamps


def test_data_lake_feed_keeps_timestamp_atomic_for_selected_backend(tmp_path):
    source = tmp_path / "bars.csv"
    _write(source, _rows())
    lake = MinuteBarDataLake(DataLakeConfig(tmp_path / "lake", bucket_count=4))
    lake.ingest(source)
    replay = DataLakeFeed(lake, batch_size=2)

    _assert_split_timestamp_replay(api.BacktestEngine(10_000.0), replay, 3)


def test_historical_event_source_rejects_cross_batch_key_regression():
    pa = pytest.importorskip("pyarrow")
    schema = pa.schema([
        ("timestamp", pa.int64()),
        ("symbol", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.int64()),
    ])

    class Replay:
        def stream_batches(self):
            for symbol in ("600000", "000001"):
                yield pa.RecordBatch.from_pylist([{
                    "timestamp": BASE,
                    "symbol": symbol,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 100,
                }], schema=schema)

    with pytest.raises(ValueError, match="strictly ordered"):
        list(HistoricalReplaySource(Replay()).events())


def test_benchmark_report_separates_pipeline_stages(tmp_path):
    cpp = pytest.importorskip("cpp_engine")
    source = tmp_path / "bars.csv"
    _write(source, _rows())
    lake = MinuteBarDataLake(DataLakeConfig(tmp_path / "lake", bucket_count=4))
    lake.ingest(source)
    config = cpp.ExecutionConfig()
    config.enforce_t_plus_one = False
    callback_calls = 0

    def round_trip_strategy(batch):
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls > 2:
            return []
        order = cpp.Order()
        order.symbol = batch[0].symbol
        order.side = cpp.Side.BUY if callback_calls == 1 else cpp.Side.SELL
        order.type = cpp.OrderType.MARKET
        order.quantity = 1
        return [order]

    report = MinuteReplayBenchmark(lake, target_bytes=256).run(
        start=BASE,
        engine_factory=lambda: cpp.BacktestEngine(
            10_000.0, cpp.FillTiming.CLOSE, config
        ),
        strategy_callback=round_trip_strategy,
        strategy_name="test_round_trip",
        cache_state="warm",
    )
    assert [stage.name for stage in report.stages] == [
        "storage_io", "arrow_decode", "arrow_compute", "c_stream_replay",
        "event_replay", "strategy_backtest",
    ]
    assert report.lineage["dataset_fingerprint"]
    assert report.lineage["query_fingerprint"]
    assert report.stages[1].rows == 9
    assert report.stages[0].details["cache"]["method"] == "sequential-preread"
    assert report.stages[3].details["c_stream_calls"] == 1
    assert report.stages[4].details["c_stream_calls"] == 1
    assert report.stages[4].details["market_batches"] == 3
    assert report.stages[5].details["strategy_callback_calls"] == 3
    assert report.stages[5].details["result"]["trades"] == 2
    assert report.stages[5].details["result"]["round_trips"] == 1
    assert report.environment["backend_artifact_path"]
    assert report.environment["backend_artifact_sha256"]
    assert report.environment["backend_artifact_size_bytes"] > 0
    assert report.environment["packages"]["pyarrow"]
    assert all(stage.rss_start_bytes > 0 for stage in report.stages)
    assert all(stage.rss_end_bytes > 0 for stage in report.stages)
    path = tmp_path / "benchmark.json"
    report.save(path)
    assert '"dataset_fingerprint"' in path.read_text(encoding="utf-8")


def test_benchmark_cold_cache_is_controlled_or_rejected(tmp_path):
    source = tmp_path / "bars.csv"
    _write(source, _rows())
    lake = MinuteBarDataLake(DataLakeConfig(tmp_path / "lake", bucket_count=4))
    lake.ingest(source)
    benchmark = MinuteReplayBenchmark(lake, target_bytes=256)

    if sys.platform.startswith("linux") and hasattr(os, "posix_fadvise"):
        report = benchmark.run(start=BASE, cache_state="cold")
        assert report.stages[0].details["cache"]["method"] == (
            "posix_fadvise-dontneed"
        )
    else:
        with pytest.raises(RuntimeError, match="cold_cache_command"):
            benchmark.run(start=BASE, cache_state="cold")

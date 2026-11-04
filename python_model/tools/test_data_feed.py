from __future__ import annotations

import csv

import pytest

from python.data_feed import Bar, CSVDataFeed, DataError, MarketDataStore, ParquetDataFeed
from python.market_data import MarketReferenceData, SecurityMaster, SecurityState


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("timestamp", "symbol", "open", "high", "low", "close", "volume"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _row(timestamp, symbol="AAA", close=10.0):
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 100,
    }


def test_csv_is_normalized_sorted_and_queryable(tmp_path):
    path = tmp_path / "bars.csv"
    _write_csv(path, [_row(3, "BBB"), _row(2), _row(1)])
    store = MarketDataStore()

    bars = store.load(path)
    assert [bar.timestamp for bar in bars] == [1, 2, 3]
    assert all(isinstance(bar, Bar) for bar in bars)
    assert store.load(path, symbols="AAA", start=2) == (bars[1],)
    assert store.latest(path, "AAA").timestamp == 2


def test_cache_hits_and_invalidates_when_file_changes(tmp_path):
    path = tmp_path / "bars.csv"
    _write_csv(path, [_row(1)])
    store = MarketDataStore(cache_capacity=2)

    store.load(path)
    store.load(path)
    assert store.cache_info().hits == 1

    _write_csv(path, [_row(1), _row(2)])
    assert len(store.load(path)) == 2
    info = store.cache_info()
    assert info.misses == 2
    assert info.entries == 1


def test_csv_feed_keeps_existing_engine_contract(tmp_path):
    path = tmp_path / "bars.csv"
    _write_csv(path, [_row(1)])
    snapshots = list(CSVDataFeed(path).stream())
    assert len(snapshots) == 1
    assert snapshots[0].symbol == "AAA"
    assert snapshots[0].close == 10.0


def test_file_feed_prepare_pins_content_and_lineage(tmp_path):
    path = tmp_path / "bars.csv"
    _write_csv(path, [_row(1)])
    feed = CSVDataFeed(path)

    lineage = feed.prepare()
    _write_csv(path, [_row(1), _row(2)])

    assert [snapshot.timestamp for snapshot in feed.stream()] == [1]
    assert feed.lineage() == lineage


def test_invalid_ohlc_and_duplicates_are_rejected(tmp_path):
    path = tmp_path / "bars.csv"
    invalid = _row(1)
    invalid["high"] = 9.0
    _write_csv(path, [invalid])
    with pytest.raises(DataError, match="high"):
        MarketDataStore().load(path)

    _write_csv(path, [_row(1), _row(1)])
    with pytest.raises(DataError, match="重复"):
        MarketDataStore().load(path)


def test_parquet_uses_the_same_bar_schema(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "bars.parquet"
    parquet.write_table(pyarrow.Table.from_pylist([_row(1), _row(2)]), path)

    bars = ParquetDataFeed(path).bars()
    assert [bar.timestamp for bar in bars] == [1, 2]
    assert bars[0].symbol == "AAA"


def test_optional_market_state_columns_reach_snapshot(tmp_path):
    path = tmp_path / "states.csv"
    row = _row(1)
    row.update(upper_limit=11.0, lower_limit=9.0, is_suspended="true")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)

    bar = MarketDataStore().load(path)[0]
    snapshot = list(CSVDataFeed(path).stream())[0]
    assert (bar.upper_limit, bar.lower_limit, bar.is_suspended) == (11.0, 9.0, True)
    assert snapshot.upper_limit == 11.0
    assert snapshot.lower_limit == 9.0
    assert snapshot.is_suspended is True


def test_feed_enriches_point_in_time_security_state(tmp_path):
    path = tmp_path / "bars.csv"
    _write_csv(path, [_row(100)])
    reference = MarketReferenceData(security_master=SecurityMaster([
        SecurityState("AAA", 0, None, board="MAIN", industry="Bank",
                      lot_size=100, min_buy_quantity=100),
    ]))
    snapshot = list(CSVDataFeed(path, reference_data=reference).stream())[0]
    assert snapshot.is_listed is True
    assert snapshot.board == "MAIN"
    assert snapshot.industry == "Bank"
    assert snapshot.lot_size == 100

from __future__ import annotations

import csv

import pytest

from python.market_data import DataCatalog, DataLakeConfig, MinuteBarDataLake, symbol_bucket
from python.data_feed import DataLakeFeed


def test_bucket_is_stable_and_config_is_persisted(tmp_path):
    config = DataLakeConfig(tmp_path / "lake", bucket_count=64)
    catalog = DataCatalog(config)
    assert catalog.generation == 0
    assert catalog.files() == ()
    assert symbol_bucket("000001.SZ", 64) == symbol_bucket("000001.SZ", 64)

    with pytest.raises(RuntimeError, match="bucket_count"):
        DataCatalog(DataLakeConfig(tmp_path / "lake", bucket_count=32))

    with pytest.raises(ValueError, match="必须为正数"):
        DataLakeConfig(tmp_path / "invalid", ingest_max_open_files=0)


def test_ingest_query_cache_and_global_primary_key(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    source = tmp_path / "minute.csv"
    rows = [
        [1704159060000000000, "000001", 10.0, 10.2, 9.9, 10.1, 100],
        [1704159060000000000, "600000", 8.0, 8.1, 7.9, 8.0, 200],
        [1704159120000000000, "000001", 10.1, 10.3, 10.0, 10.2, 150],
    ]
    _write_rows(source, rows)
    lake = MinuteBarDataLake(DataLakeConfig(
        tmp_path / "lake", bucket_count=4, query_cache_bytes=1024 * 1024,
        max_rows_per_file=100, row_group_size=50,
    ))

    result = lake.ingest(source)
    assert result.rows == 3
    assert result.files >= 1
    assert lake.ingest(source).skipped
    assert lake.catalog.generation == 1

    history = lake.history("000001")
    assert history.num_rows == 2
    assert history["symbol"].to_pylist() == ["000001", "000001"]
    cross_section = lake.cross_section(1704159060000000000)
    assert cross_section.num_rows == 2
    snapshots = list(DataLakeFeed(lake, symbols="000001", batch_size=1).stream())
    assert len(snapshots) == 2
    assert snapshots[0].symbol == "000001"

    overlapping = tmp_path / "overlap.csv"
    _write_rows(overlapping, [rows[0]])
    with pytest.raises(ValueError, match="主键冲突"):
        lake.ingest(overlapping)


def test_data_lake_feed_pins_catalog_snapshot_and_lineage(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_rows(first, [
        [1704159060000000000, "000001", 10.0, 10.2, 9.9, 10.1, 100],
    ])
    _write_rows(second, [
        [1704245460000000000, "000001", 11.0, 11.2, 10.9, 11.1, 100],
    ])
    lake = MinuteBarDataLake(DataLakeConfig(tmp_path / "lake", bucket_count=4))
    lake.ingest(first)

    snapshot = lake.catalog.snapshot()
    feed = DataLakeFeed(lake, symbols="000001")
    pinned_lineage = feed.lineage()
    lake.ingest(second)

    assert snapshot.generation == 1
    assert lake.catalog.generation == 2
    assert len(lake.files_for_query(snapshot=snapshot)) == 1
    assert [bar.timestamp for bar in feed.stream()] == [1704159060000000000]
    assert feed.lineage() == pinned_lineage
    assert pinned_lineage.catalog_generation == 1
    assert lake.lineage().catalog_generation == 2


def test_csv_ingest_streams_batches_without_reading_whole_table(tmp_path, monkeypatch):
    csv_module = pytest.importorskip("pyarrow.csv")
    pytest.importorskip("duckdb")
    source = tmp_path / "large.csv"
    rows = [
        [1704159060000000000 + index * 60_000_000_000, "000001",
         10.0, 10.2, 9.9, 10.1, 100]
        for index in range(400)
    ]
    _write_rows(source, rows)
    lake = MinuteBarDataLake(DataLakeConfig(
        tmp_path / "lake", bucket_count=4,
        ingest_csv_block_size_bytes=512,
        max_rows_per_file=100, row_group_size=50,
    ))
    monkeypatch.setattr(
        csv_module, "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("不应整表读取 CSV")
        ),
    )
    progress = []

    result = lake.ingest(source, progress=progress.append)

    assert result.rows == 400
    assert len(progress) > 1
    assert progress[-1].rows == 400
    assert progress[-1].batches == len(progress)
    assert lake.summary().rows == 400
    assert lake.list_symbols() == ("000001",)


def test_parquet_ingest_is_batch_bounded_and_preview_is_small(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("duckdb")
    source = tmp_path / "large.parquet"
    table = pa.table({
        "timestamp": pa.array(
            [1704159060000000000 + index * 60_000_000_000 for index in range(250)],
            type=pa.int64(),
        ),
        "symbol": ["600000"] * 250,
        "open": [8.0] * 250,
        "high": [8.1] * 250,
        "low": [7.9] * 250,
        "close": [8.0] * 250,
        "volume": pa.array([200] * 250, type=pa.int64()),
    })
    parquet.write_table(table, source, row_group_size=40)
    lake = MinuteBarDataLake(DataLakeConfig(
        tmp_path / "lake", bucket_count=4, ingest_batch_rows=37,
        max_rows_per_file=100, row_group_size=50,
    ))
    monkeypatch.setattr(
        parquet, "read_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("不应整表读取 Parquet")
        ),
    )
    observed = []
    original = lake._normalize_arrow

    def observe(value, *args):
        observed.append(value.num_rows)
        return original(value, *args)

    monkeypatch.setattr(lake, "_normalize_arrow", observe)
    preview = lake.preview(source, limit=5)
    result = lake.ingest(source)

    assert preview.num_rows == 5
    assert result.rows == 250
    assert max(observed) <= 37


def test_streaming_ingest_rejects_duplicates_across_csv_batches(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    source = tmp_path / "duplicates.csv"
    first = [1704159060000000000, "000001", 10.0, 10.2, 9.9, 10.1, 100]
    rows = [first] + [
        [1704159060000000000 + index * 60_000_000_000, "600000",
         8.0, 8.1, 7.9, 8.0, 200]
        for index in range(1, 80)
    ] + [first]
    _write_rows(source, rows)
    lake = MinuteBarDataLake(DataLakeConfig(
        tmp_path / "lake", bucket_count=4, ingest_csv_block_size_bytes=256,
    ))

    with pytest.raises(ValueError, match="重复"):
        lake.ingest(source)

    assert lake.catalog.generation == 0
    assert lake.summary().rows == 0


def test_ingest_many_streams_files_and_commits_one_catalog_generation(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    first = tmp_path / "source" / "first.csv"
    second = tmp_path / "source" / "nested" / "second.csv"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    _write_rows(first, [[
        1704159060000000000, "000001", 10.0, 10.2, 9.9, 10.1, 100,
    ]])
    _write_rows(second, [[
        1704159060000000000, "600000", 8.0, 8.1, 7.9, 8.0, 200,
    ]])
    lake = MinuteBarDataLake(DataLakeConfig(tmp_path / "lake", bucket_count=4))
    progress = []

    result = lake.ingest_many(
        [first, second], progress=progress.append,
        total_files=2, file_batch_size=2,
    )

    assert result.files_total == 2
    assert result.files_imported == 2
    assert result.files_skipped == 0
    assert result.rows == 2
    assert lake.catalog.generation == 1
    assert lake.summary().rows == 2
    assert [item.files_completed for item in progress] == [1, 2, 2]

    repeated = lake.ingest_many(
        [first, second], total_files=2, file_batch_size=2,
    )
    assert repeated.files_imported == 0
    assert repeated.files_skipped == 2
    assert lake.catalog.generation == 1


def test_ingest_limits_arrow_open_files(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    dataset = pytest.importorskip("pyarrow.dataset")
    source = tmp_path / "source.csv"
    _write_rows(source, [[
        1704159060000000000, "000001", 10.0, 10.2, 9.9, 10.1, 100,
    ]])
    lake = MinuteBarDataLake(DataLakeConfig(
        tmp_path / "lake", bucket_count=4, ingest_max_open_files=7,
    ))
    observed = []
    original = dataset.write_dataset

    def write_dataset(*args, **kwargs):
        observed.append(kwargs.get("max_open_files"))
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset, "write_dataset", write_dataset)

    lake.ingest(source)

    assert observed == [7]
    assert lake.summary().rows == 1


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("timestamp", "symbol", "open", "high", "low", "close", "volume"))
        writer.writerows(rows)

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from python.market_data.catalog import CatalogFile
from python.market_data.catalog_publish import (
    copy_catalog,
    publish_replacement,
    read_generation,
)
from python.market_data.lake_fingerprint import (
    compare_fingerprints,
    compute_fingerprint,
    fingerprint_summary,
)
from python.market_data.partition_merge import merge_partition
from python.market_data.partition_planner import (
    SUPPORTED_LAYOUTS,
    normalize_layout,
    partition_key,
    plan_partitions,
    plan_summary,
)

COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


def _write_part(path: Path, rows: list[tuple]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "timestamp": pa.array([r[0] for r in rows], pa.int64()),
            "symbol": pa.array([r[1] for r in rows], pa.string()),
            "open": pa.array([r[2] for r in rows], pa.float64()),
            "high": pa.array([r[3] for r in rows], pa.float64()),
            "low": pa.array([r[4] for r in rows], pa.float64()),
            "close": pa.array([r[5] for r in rows], pa.float64()),
            "volume": pa.array([r[6] for r in rows], pa.float64()),
        }
    )
    pq.write_table(table, path)
    return path


def _record(path: str, year: int, month: int, bucket: int, rows: int, lo: int, hi: int) -> dict:
    return {
        "path": path,
        "year": year,
        "month": month,
        "bucket": bucket,
        "rows": rows,
        "min_timestamp": lo,
        "max_timestamp": hi,
    }


def _catalog(path: Path, generation: int, files: list[dict]) -> Path:
    payload = {
        "schema_version": 1,
        "bucket_count": 64,
        "generation": generation,
        "files": [
            {
                "path": item["path"],
                "year": item["year"],
                "month": item["month"],
                "bucket": item["bucket"],
                "rows": item["rows"],
                "min_timestamp": item["min_timestamp"],
                "max_timestamp": item["max_timestamp"],
                "content_hash": item.get("content_hash", ""),
            }
            for item in files
        ],
        "sources": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_lake(root: Path, partitions: list[tuple]) -> Path:
    """partitions: [(year, month, bucket, [[rows...], [rows...]])]"""
    records = []
    for year, month, bucket, parts in partitions:
        all_ts = []
        for index, rows in enumerate(parts):
            relative = f"bars/year={year}/month={month}/bucket={bucket}/part-{index}.parquet"
            _write_part(root / relative, rows)
            all_ts.extend(row[0] for row in rows)
            records.append(
                _record(relative, year, month, bucket, len(rows), min(r[0] for r in rows), max(r[0] for r in rows))
            )
    _catalog(root / "catalog.json", 1, records)
    return root


def test_layouts_and_normalize() -> None:
    assert normalize_layout("Year_Month_Bucket") == "year_month_bucket"
    assert set(SUPPORTED_LAYOUTS) == {"year_month_bucket", "year_bucket", "bucket", "year"}
    with pytest.raises(ValueError):
        normalize_layout("week")


def test_partition_key_per_layout() -> None:
    record = {"year": 2024, "month": 3, "bucket": 7, "rows": 1, "min_timestamp": 0, "max_timestamp": 0}
    assert partition_key(record, "year_month_bucket") == (2024, 3, 7)
    assert partition_key(record, "year_bucket") == (2024, 7)
    assert partition_key(record, "bucket") == (7,)
    assert partition_key(record, "year") == (2024,)


def test_plan_partitions_groups_and_targets() -> None:
    records = [
        _record("a.parquet", 2024, 3, 7, 10, 100, 200),
        _record("b.parquet", 2024, 3, 7, 5, 150, 300),
        _record("c.parquet", 2024, 3, 8, 7, 120, 260),
    ]
    plans = plan_partitions(records, "year_month_bucket")
    assert len(plans) == 2
    first = plans[0]
    assert first.key == (2024, 3, 7)
    assert first.paths == ("a.parquet", "b.parquet")
    assert first.rows == 15
    assert first.min_timestamp == 100
    assert first.max_timestamp == 300
    assert first.target_relative_path == (
        "bars_compact/year=2024/month=3/bucket=7/part-compact-0.parquet"
    )
    summary = plan_summary(plans)
    assert summary == {"partitions": 2, "source_files": 3, "rows": 22}


def test_merge_partition_sorts_and_reports(tmp_path: Path) -> None:
    # 交错时间戳：直接拼接会得到乱序，只有真正排序才正确
    left = _write_part(tmp_path / "left.parquet", [
        (1, "AAA", 1.0, 1.0, 1.0, 1.0, 1.0),
        (3, "AAA", 3.0, 3.0, 3.0, 3.0, 3.0),
    ])
    right = _write_part(tmp_path / "right.parquet", [
        (2, "AAA", 2.0, 2.0, 2.0, 2.0, 2.0),
        (4, "AAA", 4.0, 4.0, 4.0, 4.0, 4.0),
    ])
    output = tmp_path / "merged.parquet"
    result = merge_partition(
        source_paths=[str(left), str(right)],
        output_path=output,
        compression="snappy",
    )
    assert result["rows"] == 4
    assert result["min_timestamp"] == 1
    assert result["max_timestamp"] == 4
    assert result["source_files"] == 2
    table = pq.read_table(output, columns=["timestamp"])
    assert table.column("timestamp").to_pylist() == [1, 2, 3, 4]
    metadata = pq.ParquetFile(output).metadata
    assert metadata.row_group(0).column(0).compression == "SNAPPY"


def test_merge_partition_rejects_duplicate_keys(tmp_path: Path) -> None:
    left = _write_part(tmp_path / "left.parquet", [(1, "AAA", 1.0, 1.0, 1.0, 1.0, 1.0)])
    right = _write_part(tmp_path / "right.parquet", [(1, "AAA", 2.0, 2.0, 2.0, 2.0, 2.0)])
    with pytest.raises(ValueError):
        merge_partition(
            source_paths=[str(left), str(right)],
            output_path=tmp_path / "merged.parquet",
        )


def test_publish_replacement_updates_generation_and_backup(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "catalog.json", 9, [_record("old.parquet", 2024, 1, 1, 1, 0, 1)])
    files = [
        CatalogFile(
            path="new.parquet",
            year=2024,
            month=1,
            bucket=1,
            rows=4,
            min_timestamp=10,
            max_timestamp=40,
            content_hash="abc",
        )
    ]
    result = publish_replacement(catalog, files, expected_generation=9)
    assert result["generation"] == 10
    assert result["previous_generation"] == 9
    assert result["files"] == 1
    assert result["rows"] == 4
    manifest = json.loads(catalog.read_text(encoding="utf-8"))
    assert manifest["generation"] == 10
    assert [item["path"] for item in manifest["files"]] == ["new.parquet"]
    assert Path(result["backup"]).exists()
    assert read_generation(catalog) == 10


def test_publish_replacement_rejects_stale_generation(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "catalog.json", 9, [])
    files = [CatalogFile(path="x.parquet", year=2024, month=1, bucket=1, rows=1,
                         min_timestamp=0, max_timestamp=1, content_hash="h")]
    with pytest.raises(RuntimeError):
        publish_replacement(catalog, files, expected_generation=8)
    assert read_generation(catalog) == 9
    assert not list(tmp_path.glob("catalog.json.bak.*"))


def test_publish_replacement_keeps_extra_metadata(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path / "catalog.json", 1, [])
    files = [CatalogFile(path="x.parquet", year=2024, month=1, bucket=1, rows=1,
                         min_timestamp=0, max_timestamp=1, content_hash="h")]
    publish_replacement(
        catalog,
        files,
        expected_generation=1,
        extra_metadata={"layout": "year_month_bucket", "compacted_files": 1},
    )
    manifest = json.loads(catalog.read_text(encoding="utf-8"))
    assert manifest["layout"] == "year_month_bucket"
    assert manifest["compacted_files"] == 1


def test_copy_catalog_copies_content(tmp_path: Path) -> None:
    source = _catalog(tmp_path / "source" / "catalog.json", 3, [])
    target = copy_catalog(source, tmp_path / "target")
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8"))["generation"] == 3


def test_fingerprint_detects_identical_and_changed_lakes(tmp_path: Path) -> None:
    lake = _make_lake(tmp_path / "lake", [
        (2024, 1, 1, [
            [(1, "AAA", 1.0, 1.0, 1.0, 1.0, 10.0), (3, "BBB", 3.0, 3.0, 3.0, 3.0, 30.0)],
            [(2, "AAA", 2.0, 2.0, 2.0, 2.0, 20.0)],
        ]),
    ])
    copy_root = tmp_path / "copy"
    copy_root.mkdir()
    for item in (lake / "bars").rglob("*.parquet"):
        destination = copy_root / item.relative_to(lake)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(item.read_bytes())
    (copy_root / "catalog.json").write_text((lake / "catalog.json").read_text(encoding="utf-8"), encoding="utf-8")

    before = compute_fingerprint(lake, per_partition=True)
    after = compute_fingerprint(copy_root, per_partition=True)
    assert fingerprint_summary(before)["rows"] == 3
    assert fingerprint_summary(before)["symbols"] == 2
    assert compare_fingerprints(before, after)["equal"] is True

    target = copy_root / "bars/year=2024/month=1/bucket=1/part-0.parquet"
    _write_part(target, [(1, "AAA", 9.0, 1.0, 1.0, 1.0, 10.0), (3, "BBB", 3.0, 3.0, 3.0, 3.0, 30.0)])
    changed = compute_fingerprint(copy_root, per_partition=True)
    comparison = compare_fingerprints(before, changed)
    assert comparison["equal"] is False
    fields = {entry["field"] for entry in comparison["differences"]}
    assert "row_hash" in fields
    assert any(field.startswith("partition:2024-1-1") for field in fields)


def test_fingerprint_on_empty_lake(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    _catalog(root / "catalog.json", 0, [])
    result = compute_fingerprint(root)
    assert result["rows"] == 0
    assert result["files"] == 0
    assert result["partitions"] == []


def test_compaction_layout_keeps_month_partitioning(tmp_path: Path) -> None:
    """year_month_bucket 压实后仍保留 (year, month) 分区，scanner 的月份窗口逻辑不受影响。"""
    lake = _make_lake(tmp_path / "lake", [
        (2024, 1, 1, [[(1, "AAA", 1.0, 1.0, 1.0, 1.0, 1.0)], [(2, "AAA", 2.0, 2.0, 2.0, 2.0, 2.0)]]),
        (2024, 2, 1, [[(3, "AAA", 3.0, 3.0, 3.0, 3.0, 3.0)]]),
    ])
    records = json.loads((lake / "catalog.json").read_text(encoding="utf-8"))["files"]
    plans = plan_partitions(records, "year_month_bucket")
    assert {(plan.key[0], plan.key[1]) for plan in plans} == {(2024, 1), (2024, 2)}
    assert all(plan.key[2] == 1 for plan in plans)
    assert all(plan.target_relative_path.startswith("bars_compact/year=2024/month=") for plan in plans)

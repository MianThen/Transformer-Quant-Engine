from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from python.market_data.catalog import CatalogFile
from python.market_data.catalog_publish import copy_catalog, publish_replacement
from python.market_data.lake_fingerprint import compare_fingerprints, compute_fingerprint
from python.market_data.partition_merge import merge_partition
from python.market_data.partition_planner import plan_partitions, plan_summary


def _load_records(source_root: Path) -> tuple[list[dict], int]:
    manifest = json.loads((source_root / "catalog.json").read_text(encoding="utf-8"))
    records = []
    for item in manifest.get("files", []):
        records.append({
            "path": str(item["path"]),
            "year": int(item["year"]),
            "month": int(item["month"]),
            "bucket": int(item["bucket"]),
            "rows": int(item["rows"]),
            "min_timestamp": int(item["min_timestamp"]),
            "max_timestamp": int(item["max_timestamp"]),
        })
    return records, int(manifest.get("generation", 0))


@dataclass(frozen=True)
class _Merged:
    year: int
    month: int
    bucket: int
    path: str
    rows: int
    min_timestamp: int
    max_timestamp: int
    content_hash: str
    source_files: int
    size_bytes: int


def _merge_one(plan, source_root: Path, target_root: Path, options) -> _Merged:
    source_paths = [str(source_root / item) for item in plan.paths]
    output_path = target_root / plan.target_relative_path

    result = merge_partition(
        source_paths=source_paths,
        output_path=output_path,
        row_group_size=options.row_group_size,
        compression=options.compression,
        memory_limit_mb=options.memory_limit_mb,
        temp_dir=options.temp_dir,
    )

    rows = result["rows"]
    min_timestamp = result["min_timestamp"]
    max_timestamp = result["max_timestamp"]
    content_hash = result["content_hash"]
    size_bytes = result["size_bytes"]
    source_files = result["source_files"]

    key = plan.key
    layout = plan.layout

    if len(key) == 3:
        year, month, bucket = int(key[0]), int(key[1]), int(key[2])
    elif len(key) == 2:
        year, bucket = int(key[0]), int(key[1])
        month = 0
    elif len(key) == 1:
        if layout == "bucket":
            year, month, bucket = 0, 0, int(key[0])
        elif layout == "year":
            year, month, bucket = int(key[0]), 0, 0
        else:
            # Fallback for unexpected layouts with single key component
            year, month, bucket = int(key[0]), 0, 0
    else:
        year, month, bucket = 0, 0, 0

    return _Merged(
        year=year,
        month=month,
        bucket=bucket,
        path=plan.target_relative_path,
        rows=rows,
        min_timestamp=min_timestamp,
        max_timestamp=max_timestamp,
        content_hash=content_hash,
        source_files=source_files,
        size_bytes=size_bytes,
    )


def _to_catalog_items(merged: list[_Merged]) -> list[CatalogFile]:
    ordered = sorted(merged, key=lambda m: (m.year, m.month, m.bucket))
    return [
        CatalogFile(
            path=m.path,
            year=m.year,
            month=m.month,
            bucket=m.bucket,
            rows=m.rows,
            min_timestamp=m.min_timestamp,
            max_timestamp=m.max_timestamp,
            content_hash=m.content_hash,
        )
        for m in ordered
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把数据湖的小 Parquet 压实为分区大文件")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--layout", default="year_month_bucket")
    parser.add_argument("--row-group-size", type=int, default=131072)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--memory-limit-mb", type=int, default=512)
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-fingerprint", action="store_true")
    parser.add_argument("--report", default=None)

    options = parser.parse_args(argv)

    source_root = Path(options.source_root).resolve()
    target_root = Path(options.target_root).resolve()

    if source_root == target_root:
        print("源目录与目标目录相同，拒绝执行")
        return 2

    records, generation = _load_records(source_root)
    plans = plan_partitions(records, options.layout)
    summary = plan_summary(plans)

    print(f"源文件 {summary['source_files']} 个 -> 分区 {summary['partitions']} 个, 行数 {summary['rows']}")

    if options.limit > 0:
        plans = plans[: options.limit]

    if options.dry_run:
        print(json.dumps(summary, ensure_ascii=False))
        return 0

    started = time.time()
    merged: list[_Merged] = []
    done = 0

    if options.workers <= 1:
        for plan in plans:
            item = _merge_one(plan, source_root, target_root, options)
            merged.append(item)
            done += 1
            print(f"[{done}/{len(plans)}] {plan.target_relative_path} rows={item.rows}")
    else:
        with ThreadPoolExecutor(max_workers=options.workers) as pool:
            futures = [pool.submit(_merge_one, plan, source_root, target_root, options) for plan in plans]
            for future in futures:
                item = future.result()
                merged.append(item)
                done += 1
                print(f"[{done}/{len(plans)}] {item.path} rows={item.rows}")

    catalog_items = _to_catalog_items(merged)
    copied = copy_catalog(source_root / "catalog.json", target_root)
    publish = publish_replacement(
        copied,
        catalog_items,
        expected_generation=generation,
        extra_metadata={
            "layout": options.layout,
            "compacted_from": str(source_root),
            "compacted_files": len(merged),
        },
    )

    verification = None
    if not options.skip_fingerprint:
        print("计算压实前指纹...")
        per_partition = options.layout == "year_month_bucket"
        before = compute_fingerprint(
            source_root,
            per_partition=per_partition,
            memory_limit_mb=options.memory_limit_mb,
        )
        print("计算压实后指纹...")
        after = compute_fingerprint(
            target_root,
            per_partition=per_partition,
            memory_limit_mb=options.memory_limit_mb,
        )
        verification = compare_fingerprints(before, after)
        print(f"等价性: {'PASS' if verification['equal'] else 'FAIL'}")

    report = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "layout": options.layout,
        "source_generation": generation,
        "source_files": summary["source_files"],
        "partitions": len(plans),
        "rows": summary["rows"],
        "published": publish,
        "verification": verification,
        "elapsed_seconds": round(time.time() - started, 3),
    }

    if options.report:
        Path(options.report).parent.mkdir(parents=True, exist_ok=True)
        Path(options.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "partitions": len(plans),
                "elapsed_seconds": report["elapsed_seconds"],
                "verified": None if verification is None else verification["equal"],
            },
            ensure_ascii=False,
        )
    )

    if verification is not None and not verification["equal"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

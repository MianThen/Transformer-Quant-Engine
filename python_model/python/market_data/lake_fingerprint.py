from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROW_TABLES = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


def _connect(memory_limit_mb: int):
    import duckdb

    connection = duckdb.connect(database=":memory:")
    connection.execute(f"SET memory_limit = '{int(memory_limit_mb)}MB'")
    connection.execute("SET threads = 4")
    return connection


GLOBAL_SQL = (
    "SELECT COUNT(*) AS rows, "
    "COUNT(DISTINCT symbol) AS symbols, "
    "CAST(COALESCE(MIN(timestamp), 0) AS BIGINT) AS min_timestamp, "
    "CAST(COALESCE(MAX(timestamp), 0) AS BIGINT) AS max_timestamp, "
    "CAST(COALESCE(SUM(CAST(hash(timestamp, symbol, open, high, low, close, volume) % 1000000007 AS BIGINT)), 0) AS BIGINT) AS row_hash "
    "FROM read_parquet(?, union_by_name = true)"
)


def compute_fingerprint(
    root: str | Path,
    *,
    per_partition: bool = False,
    memory_limit_mb: int = 512,
) -> dict:
    root = Path(root)
    manifest = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    generation = int(manifest.get("generation", 0))

    if not files:
        return {
            "root": str(root),
            "generation": generation,
            "files": 0,
            "rows": 0,
            "symbols": 0,
            "min_timestamp": 0,
            "max_timestamp": 0,
            "row_hash": 0,
            "partitions": [],
        }

    paths = [str(root / str(item["path"])) for item in files]
    connection = _connect(memory_limit_mb)
    try:
        row = connection.execute(GLOBAL_SQL, [paths]).fetchone()
        rows, symbols, min_ts, max_ts, row_hash = row
        rows = int(rows)
        symbols = int(symbols)
        min_ts = int(min_ts)
        max_ts = int(max_ts)
        row_hash = int(row_hash)

        partitions: list[dict] = []
        if per_partition:
            groups: dict[tuple[int, int, int], list[str]] = defaultdict(list)
            for item in files:
                key = (int(item["year"]), int(item["month"]), int(item["bucket"]))
                groups[key].append(str(root / item["path"]))

            for key, group_paths in sorted(groups.items()):
                part_row = connection.execute(GLOBAL_SQL, [group_paths]).fetchone()
                p_rows, p_symbols, p_min_ts, p_max_ts, p_row_hash = part_row
                partitions.append(
                    {
                        "year": key[0],
                        "month": key[1],
                        "bucket": key[2],
                        "files": len(group_paths),
                        "rows": int(p_rows),
                        "min_timestamp": int(p_min_ts),
                        "max_timestamp": int(p_max_ts),
                        "row_hash": int(p_row_hash),
                    }
                )

        return {
            "root": str(root),
            "generation": generation,
            "files": len(files),
            "rows": rows,
            "symbols": symbols,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
            "row_hash": row_hash,
            "partitions": partitions,
        }
    finally:
        connection.close()


def compare_fingerprints(before: dict, after: dict) -> dict:
    differences: list[dict] = []

    fields = ("rows", "symbols", "min_timestamp", "max_timestamp", "row_hash")
    for field in fields:
        b_val = before.get(field)
        a_val = after.get(field)
        if b_val != a_val:
            differences.append({"field": field, "before": b_val, "after": a_val})

    before_partitions = before.get("partitions", [])
    after_partitions = after.get("partitions", [])

    if before_partitions and after_partitions:
        before_map: dict[tuple[int, int, int], dict] = {}
        for p in before_partitions:
            key = (p["year"], p["month"], p["bucket"])
            before_map[key] = p

        for p in after_partitions:
            key = (p["year"], p["month"], p["bucket"])
            if key not in before_map:
                differences.append(
                    {
                        "field": f"partition:{key[0]}-{key[1]}-{key[2]}",
                        "before": None,
                        "after": "present",
                    }
                )
                continue

            b_p = before_map[key]
            for field in ("rows", "min_timestamp", "max_timestamp", "row_hash"):
                b_val = b_p.get(field)
                a_val = p.get(field)
                if b_val != a_val:
                    differences.append(
                        {
                            "field": f"partition:{key[0]}-{key[1]}-{key[2]}:{field}",
                            "before": b_val,
                            "after": a_val,
                        }
                    )

    return {"equal": not differences, "differences": differences}


def fingerprint_summary(fingerprint: dict) -> dict:
    return {
        "generation": int(fingerprint.get("generation", 0)),
        "files": int(fingerprint.get("files", 0)),
        "rows": int(fingerprint.get("rows", 0)),
        "symbols": int(fingerprint.get("symbols", 0)),
        "min_timestamp": int(fingerprint.get("min_timestamp", 0)),
        "max_timestamp": int(fingerprint.get("max_timestamp", 0)),
        "row_hash": int(fingerprint.get("row_hash", 0)),
    }

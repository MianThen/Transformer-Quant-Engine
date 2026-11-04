from __future__ import annotations

import hashlib
from pathlib import Path

BAR_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def merge_partition(
    *,
    source_paths: list[str] | tuple[str, ...],
    output_path: str | Path,
    row_group_size: int = 131_072,
    compression: str = "zstd",
    memory_limit_mb: int = 512,
    temp_dir: str | Path | None = None,
) -> dict:
    if not source_paths:
        raise ValueError("分区没有源文件")
    if row_group_size <= 0:
        raise ValueError("row_group_size 必须为正数")
    if memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb 必须为正数")

    import duckdb

    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(f"SET memory_limit = '{int(memory_limit_mb)}MB'")
        if temp_dir is not None:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            connection.execute(
                "SET temp_directory = '" + str(temp_dir).replace("'", "''") + "'"
            )

        files = [str(Path(item)) for item in source_paths]

        duplicate = connection.execute(
            "SELECT (SELECT COUNT(*) FROM (SELECT timestamp, symbol FROM read_parquet(?)))"
            " - (SELECT COUNT(*) FROM (SELECT DISTINCT timestamp, symbol FROM read_parquet(?)))",
            [files, files],
        ).fetchone()[0]
        if duplicate > 0:
            raise ValueError(f"分区内存在重复 (timestamp, symbol): {int(duplicate)} 行")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        escaped_output = str(output).replace("'", "''")
        copy_sql = (
            "COPY (SELECT timestamp, symbol, open, high, low, close, volume "
            "FROM read_parquet(?) ORDER BY timestamp, symbol) "
            "TO '" + escaped_output + "' (FORMAT PARQUET, COMPRESSION " + compression +
            ", ROW_GROUP_SIZE " + str(int(row_group_size)) + ")"
        )
        connection.execute(copy_sql, [files])

        import pyarrow.parquet as parquet

        table = parquet.read_table(str(output), columns=["timestamp"])
        rows = table.num_rows
        minimum = int(table.column("timestamp")[0].as_py()) if rows else 0
        maximum = int(table.column("timestamp")[-1].as_py()) if rows else 0

        return {
            "path": str(output),
            "rows": int(rows),
            "min_timestamp": int(minimum),
            "max_timestamp": int(maximum),
            "size_bytes": int(output.stat().st_size),
            "content_hash": file_sha256(output),
            "source_files": len(files),
        }
    finally:
        connection.close()

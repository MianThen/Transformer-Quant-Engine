from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 1
BAR_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


def symbol_bucket(symbol: str, bucket_count: int) -> int:
    """跨进程、跨 Python 版本稳定的标的分桶。"""
    if bucket_count <= 0 or bucket_count & (bucket_count - 1):
        raise ValueError("bucket_count 必须是 2 的幂")
    return zlib.crc32(symbol.encode("utf-8")) & (bucket_count - 1)


@dataclass(frozen=True)
class DataLakeConfig:
    root: Path
    bucket_count: int = 64
    compression: str = "zstd"
    max_rows_per_file: int = 2_000_000
    row_group_size: int = 128_000
    query_cache_bytes: int = 512 * 1024 * 1024
    ingest_batch_rows: int = 131_072
    ingest_csv_block_size_bytes: int = 8 * 1024 * 1024
    ingest_memory_limit_mb: int = 256
    ingest_max_open_files: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        if self.bucket_count <= 0 or self.bucket_count & (self.bucket_count - 1):
            raise ValueError("bucket_count 必须是 2 的幂")
        if self.max_rows_per_file <= 0 or self.row_group_size <= 0:
            raise ValueError("Parquet 文件和 row group 行数必须为正数")
        if self.row_group_size > self.max_rows_per_file:
            raise ValueError("row_group_size 不能大于 max_rows_per_file")
        if self.query_cache_bytes < 0:
            raise ValueError("query_cache_bytes 不能为负数")
        if (
            self.ingest_batch_rows <= 0
            or self.ingest_csv_block_size_bytes <= 0
            or self.ingest_memory_limit_mb <= 0
            or self.ingest_max_open_files <= 0
        ):
            raise ValueError("导入批次和内存限制必须为正数")

from __future__ import annotations

import hashlib
import shutil
import threading
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

from .catalog import CatalogFile, CatalogSnapshot, DataCatalog
from .config import BAR_COLUMNS, DataLakeConfig, symbol_bucket
from .lineage import DataLineage, dataset_lineage

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


@dataclass(frozen=True)
class IngestResult:
    ingestion_id: str
    rows: int
    files: int
    skipped: bool = False


@dataclass(frozen=True)
class IngestProgress:
    batches: int
    rows: int


@dataclass(frozen=True)
class BatchIngestProgress:
    files_completed: int
    files_total: int
    rows: int
    batches: int
    current_path: str


@dataclass(frozen=True)
class BatchIngestResult:
    files_total: int
    files_imported: int
    files_skipped: int
    rows: int
    fragments: int


@dataclass(frozen=True)
class DatasetSummary:
    generation: int
    files: int
    rows: int
    min_timestamp: int | None
    max_timestamp: int | None


@dataclass
class _IngestStats:
    batches: int = 0
    rows: int = 0
    min_timestamp: int | None = None
    max_timestamp: int | None = None
    buckets: set[int] | None = None

    def __post_init__(self) -> None:
        if self.buckets is None:
            self.buckets = set()


class _ArrowCache:
    def __init__(self, capacity_bytes: int) -> None:
        self.capacity_bytes = capacity_bytes
        self._bytes = 0
        self._items: OrderedDict[tuple, object] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: tuple):
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
            return value

    def put(self, key: tuple, table) -> None:
        size = int(table.nbytes)
        if self.capacity_bytes == 0 or size > self.capacity_bytes:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= int(old.nbytes)
            self._items[key] = table
            self._bytes += size
            while self._bytes > self.capacity_bytes:
                _, removed = self._items.popitem(last=False)
                self._bytes -= int(removed.nbytes)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0


class MinuteBarDataLake:
    """A 股分钟线导入与查询入口。

    物理布局为 ``year/month/bucket``，catalog 是查询可见性的唯一来源。
    大查询应使用 ``iter_batches``，避免把结果整体放入内存。
    """

    def __init__(self, config: DataLakeConfig | str | Path) -> None:
        self.config = config if isinstance(config, DataLakeConfig) else DataLakeConfig(Path(config))
        self.catalog = DataCatalog(self.config)
        self._cache = _ArrowCache(self.config.query_cache_bytes)

    def ingest(
        self,
        source: str | Path,
        *,
        progress: Callable[[IngestProgress], None] | None = None,
    ) -> IngestResult:
        """幂等导入一个 CSV/Parquet 文件，并提交为不可变 Parquet fragments。"""
        with _exclusive_file_lock(self.config.root / ".ingest.lock"):
            return self._ingest_locked(source, progress)

    def ingest_many(
        self,
        sources: Iterable[str | Path],
        *,
        progress: Callable[[BatchIngestProgress], None] | None = None,
        file_batch_size: int = 128,
        total_files: int | None = None,
    ) -> BatchIngestResult:
        """Stream many source files and commit each bounded batch once."""
        if file_batch_size <= 0:
            raise ValueError("file_batch_size 必须为正数")
        source_iter = iter(sources)
        declared_total = total_files
        files_total = total_files if total_files is not None else 0
        files_completed = 0
        files_imported = files_skipped = rows = fragments = 0
        while True:
            batch = list(islice(source_iter, file_batch_size))
            if not batch:
                break
            with _exclusive_file_lock(self.config.root / ".ingest.lock"):
                result = self._ingest_files_locked(
                    batch,
                    progress=progress,
                    files_completed=files_completed,
                    files_total=files_total,
                )
            files_completed += len(batch)
            files_imported += result["imported"]
            files_skipped += result["skipped"]
            rows += result["rows"]
            fragments += result["fragments"]
            if progress is not None:
                progress(BatchIngestProgress(
                    files_completed, files_total, rows, 0, ""
                ))
        return BatchIngestResult(
            files_total=files_completed if declared_total is None else declared_total,
            files_imported=files_imported,
            files_skipped=files_skipped,
            rows=rows,
            fragments=fragments,
        )

    def _ingest_files_locked(
        self,
        sources: list[str | Path],
        *,
        progress: Callable[[BatchIngestProgress], None] | None,
        files_completed: int,
        files_total: int,
    ) -> dict[str, int]:
        pa, pc, csv_module, dataset, parquet = _arrow_modules()
        pending: list[tuple[Path, str]] = []
        skipped = 0
        known_sources = set(self.catalog.source_fingerprints())
        for source in sources:
            path = Path(source).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"待导入文件不存在: {path}")
            if path.suffix.lower() not in {".csv", ".parquet", ".pq"}:
                raise ValueError(f"不支持的导入格式: {path.suffix or '<无扩展名>'}")
            fingerprint = _source_fingerprint(path)
            if fingerprint in known_sources:
                skipped += 1
            else:
                known_sources.add(fingerprint)
                pending.append((path, fingerprint))
        if not pending:
            return {"imported": 0, "skipped": skipped, "rows": 0, "fragments": 0}

        ingestion_id = uuid.uuid4().hex
        staging = self.config.root / ".staging" / ingestion_id
        staging.mkdir(parents=True, exist_ok=False)
        stats = _IngestStats()
        source_records: list[dict[str, object]] = []

        def source_batches():
            for path, fingerprint in pending:
                file_stats = _IngestStats()
                for batch in self._normalized_source_batches(
                    path, pa, pc, csv_module, parquet, file_stats, None
                ):
                    _merge_batch_stats(stats, batch, pc)
                    yield batch
                if file_stats.rows == 0:
                    raise ValueError(f"不能导入空行情文件: {path}")
                source_records.append({
                    "fingerprint": fingerprint,
                    "source_path": str(path),
                    "files": 0,
                    "rows": file_stats.rows,
                })
                if progress is not None:
                    progress(BatchIngestProgress(
                        files_completed + skipped + len(source_records),
                        files_total,
                        stats.rows,
                        stats.batches,
                        str(path),
                    ))

        try:
            partition_schema = pa.schema([
                ("year", pa.int16()), ("month", pa.int8()), ("bucket", pa.int16())
            ])
            reader = pa.RecordBatchReader.from_batches(
                _ingest_schema(pa), source_batches()
            )
            dataset.write_dataset(
                reader,
                base_dir=staging,
                format="parquet",
                partitioning=dataset.partitioning(partition_schema, flavor="hive"),
                basename_template=f"part-{ingestion_id}-{{i}}.parquet",
                existing_data_behavior="overwrite_or_ignore",
                max_open_files=self.config.ingest_max_open_files,
                max_rows_per_file=self.config.max_rows_per_file,
                max_rows_per_group=self.config.row_group_size,
                file_options=dataset.ParquetFileFormat().make_write_options(
                    compression=self.config.compression
                ),
            )
            staged_files = sorted(str(item) for item in staging.rglob("*.parquet"))
            if stats.rows == 0:
                raise ValueError("不能导入空行情文件")
            if not staged_files:
                raise RuntimeError("导入未生成 Parquet fragment")
            self._ensure_unique_staged_keys(staged_files, staging)
            self._ensure_no_existing_staged_keys(staged_files, stats, staging)
            committed_files = self._publish_staged(staging, parquet, pc)
            committed = self.catalog.commit_many(
                ingestion_id, source_records, committed_files
            )
            if not committed:
                return {
                    "imported": 0,
                    "skipped": skipped + len(pending),
                    "rows": 0,
                    "fragments": 0,
                }
            self._cache.clear()
            return {
                "imported": len(pending),
                "skipped": skipped,
                "rows": sum(item.rows for item in committed_files),
                "fragments": len(committed_files),
            }
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def lineage(self, snapshot: CatalogSnapshot | None = None) -> DataLineage:
        pinned = snapshot or self.catalog.snapshot()
        return dataset_lineage(pinned, self.config.root)

    def _ingest_locked(self, source, progress) -> IngestResult:
        pa, pc, csv_module, dataset, parquet = _arrow_modules()
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"待导入文件不存在: {path}")
        fingerprint = _source_fingerprint(path)
        if self.catalog.contains_source(fingerprint):
            return IngestResult("", 0, 0, skipped=True)

        ingestion_id = uuid.uuid4().hex
        staging = self.config.root / ".staging" / ingestion_id
        staging.mkdir(parents=True, exist_ok=False)

        try:
            partition_schema = pa.schema([
                ("year", pa.int16()), ("month", pa.int8()), ("bucket", pa.int16())
            ])
            stats = _IngestStats()
            batches = self._normalized_source_batches(
                path, pa, pc, csv_module, parquet, stats, progress
            )
            reader = pa.RecordBatchReader.from_batches(
                _ingest_schema(pa), batches
            )
            dataset.write_dataset(
                reader,
                base_dir=staging,
                format="parquet",
                partitioning=dataset.partitioning(partition_schema, flavor="hive"),
                basename_template=f"part-{ingestion_id}-{{i}}.parquet",
                existing_data_behavior="overwrite_or_ignore",
                max_open_files=self.config.ingest_max_open_files,
                max_rows_per_file=self.config.max_rows_per_file,
                max_rows_per_group=self.config.row_group_size,
                file_options=dataset.ParquetFileFormat().make_write_options(
                    compression=self.config.compression
                ),
            )
            if stats.rows == 0:
                raise ValueError("不能导入空行情文件")
            staged_files = sorted(str(item) for item in staging.rglob("*.parquet"))
            if not staged_files:
                raise RuntimeError("导入未生成 Parquet fragment")
            self._ensure_unique_staged_keys(staged_files, staging)
            self._ensure_no_existing_staged_keys(staged_files, stats, staging)
            committed_files = self._publish_staged(staging, parquet, pc)
            committed = self.catalog.commit(
                ingestion_id, fingerprint, str(path), committed_files
            )
            if not committed:
                return IngestResult("", 0, 0, skipped=True)
            self._cache.clear()
            return IngestResult(
                ingestion_id, sum(item.rows for item in committed_files), len(committed_files)
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def preview(self, source: str | Path, limit: int = 20):
        """只读取第一个批次用于界面预览，不物化整个源文件。"""
        if limit <= 0:
            raise ValueError("limit 必须为正数")
        pa, pc, csv_module, _, parquet = _arrow_modules()
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        source_batches = self._source_batches(path, pa, csv_module, parquet)
        try:
            batch = next(iter(source_batches))
        except StopIteration:
            return pa.table({name: pa.array([], type=_arrow_type(pa, name))
                             for name in BAR_COLUMNS})
        return self._normalize_arrow(batch, pa, pc).select(BAR_COLUMNS).slice(0, limit)

    def summary(self, snapshot: CatalogSnapshot | None = None) -> DatasetSummary:
        pinned = snapshot or self.catalog.snapshot()
        if not pinned.files:
            return DatasetSummary(pinned.generation, 0, 0, None, None)
        return DatasetSummary(
            pinned.generation,
            len(pinned.files),
            sum(item.rows for item in pinned.files),
            min(item.min_timestamp for item in pinned.files),
            max(item.max_timestamp for item in pinned.files),
        )

    def list_symbols(self, snapshot: CatalogSnapshot | None = None) -> tuple[str, ...]:
        pinned = snapshot or self.catalog.snapshot()
        files = [str(self.config.root / item.path) for item in pinned.files]
        if not files:
            return ()
        connection = _duckdb_module().connect(database=":memory:")
        try:
            rows = connection.execute(
                "SELECT DISTINCT symbol FROM read_parquet(?, "
                "hive_partitioning=false, union_by_name=true) ORDER BY symbol",
                [files],
            ).fetchall()
        finally:
            connection.close()
        return tuple(str(row[0]) for row in rows)

    def _source_batches(self, path, pa, csv_module, parquet):
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return csv_module.open_csv(
                path,
                read_options=csv_module.ReadOptions(
                    block_size=self.config.ingest_csv_block_size_bytes,
                    use_threads=True,
                ),
                convert_options=csv_module.ConvertOptions(column_types={
                    "timestamp": pa.int64(), "symbol": pa.string(),
                    "open": pa.float64(), "high": pa.float64(),
                    "low": pa.float64(), "close": pa.float64(),
                    "volume": pa.int64(),
                }),
            )
        if suffix in {".parquet", ".pq"}:
            return parquet.ParquetFile(path).iter_batches(
                batch_size=self.config.ingest_batch_rows,
                columns=list(BAR_COLUMNS),
            )
        raise ValueError(f"不支持的导入格式: {suffix}")

    def _normalized_source_batches(
        self, path, pa, pc, csv_module, parquet, stats, progress
    ):
        for source_batch in self._source_batches(path, pa, csv_module, parquet):
            table = self._normalize_arrow(source_batch, pa, pc)
            if table.num_rows == 0:
                continue
            limits = pc.min_max(table["timestamp"]).as_py()
            stats.batches += 1
            stats.rows += table.num_rows
            batch_min = int(limits["min"])
            batch_max = int(limits["max"])
            stats.min_timestamp = (
                batch_min if stats.min_timestamp is None
                else min(stats.min_timestamp, batch_min)
            )
            stats.max_timestamp = (
                batch_max if stats.max_timestamp is None
                else max(stats.max_timestamp, batch_max)
            )
            stats.buckets.update(table["bucket"].combine_chunks().unique().to_pylist())
            if progress is not None:
                progress(IngestProgress(stats.batches, stats.rows))
            for batch in table.to_batches(max_chunksize=table.num_rows):
                yield batch

    def query(
        self,
        *,
        symbols: str | Iterable[str] | None = None,
        start: int | None = None,
        end: int | None = None,
        columns: Sequence[str] = BAR_COLUMNS,
        use_cache: bool = True,
        snapshot: CatalogSnapshot | None = None,
    ):
        """物化查询结果为 Arrow Table；大结果请改用 iter_batches。"""
        normalized = _symbols_tuple(symbols)
        selected_columns = _validate_columns(columns)
        pinned = snapshot or self.catalog.snapshot()
        key = (pinned.generation, normalized, start, end, selected_columns)
        cached = self._cache.get(key) if use_cache else None
        if cached is not None:
            return cached
        connection, sql, parameters = self._prepare_query(
            normalized, start, end, selected_columns, pinned
        )
        try:
            table = connection.execute(sql, parameters).fetch_arrow_table()
        finally:
            connection.close()
        if use_cache:
            self._cache.put(key, table)
        return table

    def iter_batches(
        self,
        *,
        symbols: str | Iterable[str] | None = None,
        start: int | None = None,
        end: int | None = None,
        columns: Sequence[str] = BAR_COLUMNS,
        batch_size: int = 65_536,
        snapshot: CatalogSnapshot | None = None,
    ) -> Iterator[object]:
        """流式返回 Arrow RecordBatch；大范围排序可能由 DuckDB 使用临时磁盘。"""
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正数")
        pinned = snapshot or self.catalog.snapshot()
        connection, sql, parameters = self._prepare_query(
            _symbols_tuple(symbols), start, end, _validate_columns(columns), pinned
        )
        try:
            reader = connection.execute(sql, parameters).fetch_record_batch(batch_size)
            yield from reader
        finally:
            connection.close()

    def cross_section(self, timestamp: int, symbols: Iterable[str] | None = None):
        return self.query(symbols=symbols, start=timestamp, end=timestamp)

    def history(self, symbol: str, start: int | None = None, end: int | None = None):
        return self.query(symbols=symbol, start=start, end=end)

    def catalog_files_for_query(
        self, *, symbols=None, start=None, end=None,
        snapshot: CatalogSnapshot | None = None,
    ):
        normalized = _symbols_tuple(symbols)
        pinned = snapshot or self.catalog.snapshot()
        buckets = None
        if normalized:
            buckets = {symbol_bucket(symbol, self.config.bucket_count)
                       for symbol in normalized}
        return tuple(
            item for item in pinned.files
            if (buckets is None or item.bucket in buckets)
            and (start is None or item.max_timestamp >= start)
            and (end is None or item.min_timestamp <= end)
        )

    def files_for_query(
        self, *, symbols=None, start=None, end=None,
        snapshot: CatalogSnapshot | None = None,
    ) -> list[str]:
        return [str(self.config.root / item.path) for item in
                self.catalog_files_for_query(
                    symbols=symbols, start=start, end=end, snapshot=snapshot
                )]

    def _prepare_query(self, symbols, start, end, columns, snapshot):
        if start is not None and end is not None and start > end:
            raise ValueError("start 不能晚于 end")
        files = self._select_files(symbols, start, end, snapshot)
        duckdb = _duckdb_module()
        connection = duckdb.connect(database=":memory:")
        if not files:
            pa, _, _, _, _ = _arrow_modules()
            empty = pa.table({name: pa.array([], type=_arrow_type(pa, name)) for name in columns})
            connection.register("selected_bars", empty)
            source_sql = "selected_bars"
            parameters: list = []
        else:
            source_sql = "read_parquet(?, hive_partitioning=false, union_by_name=true)"
            parameters = [files]
        predicates = []
        if symbols:
            predicates.append("symbol IN (" + ",".join("?" for _ in symbols) + ")")
            parameters.extend(symbols)
        if start is not None:
            predicates.append("timestamp >= ?")
            parameters.append(start)
        if end is not None:
            predicates.append("timestamp <= ?")
            parameters.append(end)
        where = " WHERE " + " AND ".join(predicates) if predicates else ""
        sql = (
            f"SELECT {','.join(columns)} FROM {source_sql}{where} "
            "ORDER BY timestamp, symbol"
        )
        return connection, sql, parameters

    def _select_files(self, symbols, start, end, snapshot) -> list[str]:
        return self.files_for_query(
            symbols=symbols, start=start, end=end, snapshot=snapshot
        )

    def _normalize_arrow(self, table, pa, pc):
        missing = [name for name in BAR_COLUMNS if name not in table.column_names]
        if missing:
            raise ValueError("缺少行情字段: " + ", ".join(missing))
        table = pa.table({
            "timestamp": pc.cast(table["timestamp"], pa.int64(), safe=True),
            "symbol": pc.utf8_trim_whitespace(pc.cast(table["symbol"], pa.string())),
            "open": pc.cast(table["open"], pa.float64(), safe=True),
            "high": pc.cast(table["high"], pa.float64(), safe=True),
            "low": pc.cast(table["low"], pa.float64(), safe=True),
            "close": pc.cast(table["close"], pa.float64(), safe=True),
            "volume": pc.cast(table["volume"], pa.int64(), safe=True),
        })
        _validate_arrow(table, pc)

        timestamps = pc.cast(table["timestamp"], pa.timestamp("ns", tz="Asia/Shanghai"))
        years = pc.cast(pc.year(timestamps), pa.int16())
        months = pc.cast(pc.month(timestamps), pa.int8())
        encoded = pc.dictionary_encode(table["symbol"]).combine_chunks()
        bucket_lookup = pa.array([
            symbol_bucket(value.as_py(), self.config.bucket_count)
            for value in encoded.dictionary
        ], type=pa.int16())
        buckets = pc.take(bucket_lookup, encoded.indices)
        table = table.append_column("year", years).append_column("month", months)
        table = table.append_column("bucket", buckets)
        indices = pc.sort_indices(table, sort_keys=[
            ("year", "ascending"), ("month", "ascending"),
            ("bucket", "ascending"), ("timestamp", "ascending"),
            ("symbol", "ascending"),
        ])
        table = pc.take(table, indices)
        if table.num_rows > 1:
            timestamps = table["timestamp"].combine_chunks()
            symbols = table["symbol"].combine_chunks()
            duplicate = pc.and_(
                pc.equal(timestamps.slice(1), timestamps.slice(0, table.num_rows - 1)),
                pc.equal(symbols.slice(1), symbols.slice(0, table.num_rows - 1)),
            )
            if pc.any(duplicate).as_py():
                raise ValueError("导入批次包含重复的 (timestamp, symbol)")
        return table

    def _publish_staged(self, staging, parquet, pc) -> list[CatalogFile]:
        records = []
        for staged_file in staging.rglob("*.parquet"):
            relative = staged_file.relative_to(staging)
            parts = {part.split("=", 1)[0]: part.split("=", 1)[1]
                     for part in relative.parts[:-1]}
            metadata = parquet.read_metadata(staged_file)
            minimum, maximum = _parquet_timestamp_limits(
                staged_file, parquet, pc, self.config.ingest_batch_rows
            )
            destination = self.config.root / "bars" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged_file.replace(destination)
            records.append(CatalogFile(
                path=str(destination.relative_to(self.config.root)),
                year=int(parts["year"]), month=int(parts["month"]),
                bucket=int(parts["bucket"]), rows=metadata.num_rows,
                min_timestamp=minimum, max_timestamp=maximum,
                content_hash=_file_sha256(destination),
            ))
        return records

    def _ensure_unique_staged_keys(self, staged_files, staging) -> None:
        connection = self._ingest_connection(staging)
        try:
            duplicate = connection.execute(
                "SELECT 1 FROM read_parquet(?, hive_partitioning=false, "
                "union_by_name=true) GROUP BY timestamp, symbol "
                "HAVING COUNT(*) > 1 LIMIT 1",
                [staged_files],
            ).fetchone()
        finally:
            connection.close()
        if duplicate is not None:
            raise ValueError("导入批次包含重复的 (timestamp, symbol)")

    def _ensure_no_existing_staged_keys(self, staged_files, stats, staging) -> None:
        candidates = [
            str(self.config.root / item.path)
            for item in self.catalog.files()
            if item.bucket in stats.buckets
            and item.max_timestamp >= stats.min_timestamp
            and item.min_timestamp <= stats.max_timestamp
        ]
        if not candidates:
            return
        connection = self._ingest_connection(staging)
        try:
            duplicate = connection.execute(
                "SELECT 1 FROM read_parquet(?, hive_partitioning=false, "
                "union_by_name=true) i JOIN read_parquet(?, "
                "hive_partitioning=false, union_by_name=true) e "
                "USING (timestamp, symbol) LIMIT 1",
                [staged_files, candidates],
            ).fetchone()
        finally:
            connection.close()
        if duplicate is not None:
            raise ValueError("导入数据与数据湖现有 (timestamp, symbol) 主键冲突")

    def _ingest_connection(self, staging):
        temporary = staging / ".duckdb-temp"
        temporary.mkdir(exist_ok=True)
        connection = _duckdb_module().connect(database=":memory:")
        connection.execute(
            f"SET memory_limit = '{int(self.config.ingest_memory_limit_mb)}MB'"
        )
        escaped = str(temporary).replace("'", "''")
        connection.execute(f"SET temp_directory = '{escaped}'")
        return connection


def _validate_arrow(table, pc) -> None:
    if any(table[name].null_count for name in BAR_COLUMNS):
        raise ValueError("行情字段不能包含 null")
    invalid = pc.or_(pc.less_equal(table["open"], 0), pc.invert(pc.is_finite(table["open"])))
    for name in ("high", "low", "close"):
        invalid = pc.or_(invalid, pc.less_equal(table[name], 0))
        invalid = pc.or_(invalid, pc.invert(pc.is_finite(table[name])))
    invalid = pc.or_(invalid, pc.less(table["high"], table["open"]))
    invalid = pc.or_(invalid, pc.less(table["high"], table["close"]))
    invalid = pc.or_(invalid, pc.greater(table["low"], table["open"]))
    invalid = pc.or_(invalid, pc.greater(table["low"], table["close"]))
    invalid = pc.or_(invalid, pc.less(table["volume"], 0))
    invalid = pc.or_(invalid, pc.less_equal(table["timestamp"], 0))
    invalid = pc.or_(invalid, pc.equal(pc.utf8_length(table["symbol"]), 0))
    if pc.any(invalid).as_py():
        raise ValueError("行情包含非法 timestamp/symbol/OHLCV")


def _merge_batch_stats(stats: _IngestStats, batch, pc) -> None:
    limits = pc.min_max(batch["timestamp"]).as_py()
    batch_min = int(limits["min"])
    batch_max = int(limits["max"])
    stats.batches += 1
    stats.rows += batch.num_rows
    stats.min_timestamp = (
        batch_min if stats.min_timestamp is None
        else min(stats.min_timestamp, batch_min)
    )
    stats.max_timestamp = (
        batch_max if stats.max_timestamp is None
        else max(stats.max_timestamp, batch_max)
    )
    stats.buckets.update(batch["bucket"].unique().to_pylist())


def _ingest_schema(pa):
    return pa.schema([
        ("timestamp", pa.int64()),
        ("symbol", pa.string()),
        ("open", pa.float64()),
        ("high", pa.float64()),
        ("low", pa.float64()),
        ("close", pa.float64()),
        ("volume", pa.int64()),
        ("year", pa.int16()),
        ("month", pa.int8()),
        ("bucket", pa.int16()),
    ])


def _parquet_timestamp_limits(path, parquet, pc, batch_size) -> tuple[int, int]:
    minimum = None
    maximum = None
    for batch in parquet.ParquetFile(path).iter_batches(
        batch_size=batch_size, columns=["timestamp"]
    ):
        limits = pc.min_max(batch["timestamp"]).as_py()
        batch_min = int(limits["min"])
        batch_max = int(limits["max"])
        minimum = batch_min if minimum is None else min(minimum, batch_min)
        maximum = batch_max if maximum is None else max(maximum, batch_max)
    if minimum is None or maximum is None:
        raise ValueError("Parquet fragment 不能为空")
    return minimum, maximum


def _source_fingerprint(path: Path) -> str:
    return _file_sha256(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _symbols_tuple(symbols: str | Iterable[str] | None) -> tuple[str, ...] | None:
    if symbols is None:
        return None
    values = (symbols,) if isinstance(symbols, str) else tuple(symbols)
    return tuple(sorted({str(value) for value in values}))


def _validate_columns(columns: Sequence[str]) -> tuple[str, ...]:
    result = tuple(columns)
    invalid = set(result) - set(BAR_COLUMNS)
    if invalid or not result:
        raise ValueError("非法查询字段: " + ", ".join(sorted(invalid)))
    return result


def _arrow_type(pa, name: str):
    if name == "timestamp" or name == "volume":
        return pa.int64()
    if name == "symbol":
        return pa.string()
    return pa.float64()


def _arrow_modules():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.csv as csv_module
        import pyarrow.dataset as dataset
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("分钟线数据湖需要 pyarrow") from exc
    return pa, pc, csv_module, dataset, parquet


def _duckdb_module():
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("分钟线数据湖查询需要 duckdb") from exc
    return duckdb


@contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

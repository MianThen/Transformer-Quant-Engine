"""Arrow Scanner 快速路径与分区感知的确定性历史回放。"""

from __future__ import annotations

import calendar as month_calendar
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, Sequence
from zoneinfo import ZoneInfo

from .config import BAR_COLUMNS


SHANGHAI = ZoneInfo("Asia/Shanghai")
NANOSECONDS_PER_MINUTE = 60_000_000_000


@dataclass(frozen=True)
class ScanRequest:
    symbols: tuple[str, ...] | None
    start: int | None
    end: int | None
    columns: tuple[str, ...]
    batch_size: int


@dataclass(frozen=True)
class ReplayStats:
    target_bytes: int
    chunk_mode: str
    chunks: int
    batches: int
    rows: int
    peak_batch_bytes: int


class ArrowDatasetScanner:
    """直接扫描 catalog 可见的 Parquet fragment，不执行全局排序。"""

    def __init__(self, lake, snapshot=None) -> None:
        self.lake = lake
        self.snapshot = snapshot or lake.catalog.snapshot()

    def scanner(
        self,
        *,
        symbols: str | Iterable[str] | None = None,
        start: int | None = None,
        end: int | None = None,
        columns: Sequence[str] = BAR_COLUMNS,
        batch_size: int = 65_536,
        use_threads: bool = True,
    ):
        pa, dataset, _ = _arrow_modules()
        request = _normalize_request(symbols, start, end, columns, batch_size)
        files = self.lake.files_for_query(
            symbols=request.symbols, start=start, end=end,
            snapshot=self.snapshot,
        )
        if not files:
            table = pa.table({
                name: pa.array([], type=_arrow_type(pa, name))
                for name in request.columns
            })
            return dataset.dataset(table).scanner(
                columns=list(request.columns), batch_size=request.batch_size
            )
        expression = None
        if request.symbols:
            expression = dataset.field("symbol").isin(request.symbols)
        if start is not None:
            predicate = dataset.field("timestamp") >= start
            expression = predicate if expression is None else expression & predicate
        if end is not None:
            predicate = dataset.field("timestamp") <= end
            expression = predicate if expression is None else expression & predicate
        source = dataset.dataset(files, format="parquet")
        return source.scanner(
            columns=list(request.columns),
            filter=expression,
            batch_size=request.batch_size,
            use_threads=use_threads,
            batch_readahead=4,
            fragment_readahead=2,
        )

    def iter_batches(self, **query) -> Iterator[object]:
        yield from self.scanner(**query).to_batches()

    def reader(self, **query):
        return self.scanner(**query).to_reader()

    def lineage(self, **query):
        columns = tuple(query.get("columns", BAR_COLUMNS))
        return self.lake.lineage(self.snapshot).with_query(
            symbols=query.get("symbols"),
            start=query.get("start"),
            end=query.get("end"),
            columns=columns,
        )


class PartitionAwareIterator:
    """按目标内存切片，并输出严格 ``timestamp,symbol`` 有序的完整截面。"""

    def __init__(
        self,
        scanner: ArrowDatasetScanner,
        *,
        target_bytes: int = 256 * 1024 * 1024,
        scanner_batch_size: int = 131_072,
        decoded_size_multiplier: float = 4.0,
    ) -> None:
        if (target_bytes <= 0 or scanner_batch_size <= 0
                or decoded_size_multiplier < 1.0):
            raise ValueError("target_bytes/scanner_batch_size 必须为正数")
        self.scanner = scanner
        self.target_bytes = target_bytes
        self.scanner_batch_size = scanner_batch_size
        self.decoded_size_multiplier = decoded_size_multiplier
        self.stats = ReplayStats(target_bytes, "empty", 0, 0, 0, 0)

    def iter_batches(
        self,
        *,
        symbols: str | Iterable[str] | None = None,
        start: int | None = None,
        end: int | None = None,
        columns: Sequence[str] = BAR_COLUMNS,
    ) -> Iterator[object]:
        pa, _, pc = _arrow_modules()
        selected = tuple(columns)
        if "timestamp" not in selected or "symbol" not in selected:
            raise ValueError("有序历史回放必须投影 timestamp 和 symbol")
        normalized_symbols = _normalize_symbols(symbols)
        limits = self._resolve_limits(normalized_symbols, start, end)
        if limits is None:
            self.stats = ReplayStats(self.target_bytes, "empty", 0, 0, 0, 0)
            return
        replay_start, replay_end = limits
        windows, mode = self._windows(
            normalized_symbols, replay_start, replay_end
        )
        chunks = batches = rows = peak = 0
        last_key = None
        for chunk_start, chunk_end in windows:
            raw_batches = list(self.scanner.iter_batches(
                symbols=normalized_symbols,
                start=chunk_start,
                end=chunk_end,
                columns=selected,
                batch_size=self.scanner_batch_size,
            ))
            if not raw_batches:
                continue
            table = pa.Table.from_batches(raw_batches)
            if table.num_rows == 0:
                continue
            chunks += 1
            table = table.take(pc.sort_indices(
                table, sort_keys=[("timestamp", "ascending"), ("symbol", "ascending")]
            ))
            for batch in _slice_atomic_batches(table, self.target_bytes):
                first_key = (
                    batch.column(batch.schema.get_field_index("timestamp"))[0].as_py(),
                    batch.column(batch.schema.get_field_index("symbol"))[0].as_py(),
                )
                if last_key is not None and first_key <= last_key:
                    raise ValueError("分区回放结果不是严格 (timestamp, symbol) 有序")
                last_key = (
                    batch.column(batch.schema.get_field_index("timestamp"))[-1].as_py(),
                    batch.column(batch.schema.get_field_index("symbol"))[-1].as_py(),
                )
                batches += 1
                rows += batch.num_rows
                peak = max(peak, int(batch.nbytes))
                yield batch
        self.stats = ReplayStats(
            self.target_bytes, mode, chunks, batches, rows, peak
        )

    def reader(self, **query):
        pa, _, _ = _arrow_modules()
        columns = tuple(query.get("columns", BAR_COLUMNS))
        schema = pa.schema([(name, _arrow_type(pa, name)) for name in columns])
        return pa.RecordBatchReader.from_batches(schema, self.iter_batches(**query))

    def lineage(self, **query):
        return self.scanner.lineage(**query)

    def _resolve_limits(self, symbols, start, end):
        files = self.scanner.lake.catalog_files_for_query(
            symbols=symbols, start=start, end=end,
            snapshot=self.scanner.snapshot,
        )
        if not files:
            return None
        lower = max(
            min(item.min_timestamp for item in files),
            start if start is not None else -math.inf,
        )
        upper = min(
            max(item.max_timestamp for item in files),
            end if end is not None else math.inf,
        )
        return int(lower), int(upper)

    def _windows(self, symbols, start, end):
        files = self.scanner.lake.catalog_files_for_query(
            symbols=symbols, start=start, end=end,
            snapshot=self.scanner.snapshot,
        )
        month_bytes: dict[tuple[int, int], int] = {}
        for item in files:
            size = (self.scanner.lake.config.root / item.path).stat().st_size
            month_bytes[(item.year, item.month)] = (
                month_bytes.get((item.year, item.month), 0) + size
            )
        mode = (
            "month"
            if month_bytes
            and max(month_bytes.values()) * self.decoded_size_multiplier <= self.target_bytes
            else "day"
        )
        windows = []
        for period_start, period_end in _calendar_windows(start, end, mode):
            candidate_bytes = sum(
                (size / month_calendar.monthrange(year, month)[1]
                 if mode == "day" else size)
                for (year, month), size in month_bytes.items()
                if _overlaps_month(period_start, period_end, year, month)
            )
            candidate_bytes *= self.decoded_size_multiplier
            period_span = period_end - period_start + 1
            pieces = max(1, math.ceil(candidate_bytes / self.target_bytes))
            duration = max(NANOSECONDS_PER_MINUTE, math.ceil(period_span / pieces))
            cursor = period_start
            while cursor <= period_end:
                boundary = min(period_end, cursor + duration - 1)
                windows.append((cursor, boundary))
                cursor = boundary + 1
        return windows, mode


def _slice_atomic_batches(table, target_bytes):
    if table.num_rows == 0:
        return
    bytes_per_row = max(table.nbytes / table.num_rows, 1.0)
    target_rows = max(1, int(target_bytes / bytes_per_row))
    timestamps = table["timestamp"].combine_chunks()
    offset = 0
    while offset < table.num_rows:
        boundary = min(table.num_rows, offset + target_rows)
        while (
            boundary < table.num_rows
            and timestamps[boundary].as_py() == timestamps[boundary - 1].as_py()
        ):
            boundary += 1
        yield table.slice(offset, boundary - offset).combine_chunks().to_batches()[0]
        offset = boundary


def _calendar_windows(start, end, mode):
    start_local = datetime.fromtimestamp(start / 1e9, tz=timezone.utc).astimezone(SHANGHAI)
    end_local = datetime.fromtimestamp(end / 1e9, tz=timezone.utc).astimezone(SHANGHAI)
    cursor = start_local.date()
    while cursor <= end_local.date():
        if mode == "month":
            last_day = month_calendar.monthrange(cursor.year, cursor.month)[1]
            next_cursor = cursor.replace(day=last_day) + timedelta(days=1)
        else:
            next_cursor = cursor + timedelta(days=1)
        local_start = datetime.combine(cursor, datetime.min.time(), tzinfo=SHANGHAI)
        local_end = datetime.combine(next_cursor, datetime.min.time(), tzinfo=SHANGHAI)
        lower = max(start, int(local_start.timestamp() * 1e9))
        upper = min(end, int(local_end.timestamp() * 1e9) - 1)
        if lower <= upper:
            yield lower, upper
        cursor = next_cursor


def _overlaps_month(start, end, year, month):
    start_local = datetime.fromtimestamp(start / 1e9, tz=timezone.utc).astimezone(SHANGHAI)
    end_local = datetime.fromtimestamp(end / 1e9, tz=timezone.utc).astimezone(SHANGHAI)
    return (start_local.year, start_local.month) <= (year, month) <= (
        end_local.year, end_local.month
    )


def _normalize_request(symbols, start, end, columns, batch_size):
    if start is not None and end is not None and start > end:
        raise ValueError("start 不能晚于 end")
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    selected = tuple(columns)
    invalid = set(selected) - set(BAR_COLUMNS)
    if not selected or invalid:
        raise ValueError("非法 Scanner 投影字段: " + ", ".join(sorted(invalid)))
    return ScanRequest(_normalize_symbols(symbols), start, end, selected, batch_size)


def _normalize_symbols(symbols):
    if symbols is None:
        return None
    values = (symbols,) if isinstance(symbols, str) else tuple(symbols)
    return tuple(sorted({str(value) for value in values}))


def _arrow_type(pa, name):
    if name in {"timestamp", "volume"}:
        return pa.int64()
    if name == "symbol":
        return pa.string()
    return pa.float64()


def _arrow_modules():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.dataset as dataset
    except ImportError as exc:
        raise RuntimeError("Arrow Scanner 需要 pyarrow") from exc
    return pa, dataset, pc

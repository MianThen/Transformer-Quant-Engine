"""统一的历史 Bar 数据读取、校验、缓存与引擎适配。"""

from __future__ import annotations

import csv
import math
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

if TYPE_CHECKING:
    from .engine_api import MarketSnapshot


BAR_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
MARKET_STATE_COLUMNS = ("upper_limit", "lower_limit", "is_suspended")


class DataError(ValueError):
    """行情文件、schema 或数据内容不合法。"""


@dataclass(frozen=True)
class Bar:
    """回测系统内唯一的 OHLCV 数据格式，timestamp 单位为 UTC 纳秒。"""

    timestamp: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    upper_limit: float = 0.0
    lower_limit: float = 0.0
    is_suspended: bool = False

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise DataError("timestamp 不能为负数")
        if not self.symbol or not self.symbol.strip():
            raise DataError("symbol 不能为空")
        prices = (self.open, self.high, self.low, self.close)
        if not all(math.isfinite(value) and value > 0.0 for value in prices):
            raise DataError("OHLC 必须是有限的正数")
        if self.high < max(self.open, self.low, self.close):
            raise DataError("high 不能低于 open/low/close")
        if self.low > min(self.open, self.high, self.close):
            raise DataError("low 不能高于 open/high/close")
        if self.volume < 0:
            raise DataError("volume 不能为负数")
        limits = (self.upper_limit, self.lower_limit)
        if not all(math.isfinite(value) and value >= 0.0 for value in limits):
            raise DataError("涨跌停价格必须是有限的非负数，0 表示未提供")
        if self.upper_limit > 0.0 and self.lower_limit > self.upper_limit:
            raise DataError("lower_limit 不能高于 upper_limit")


@dataclass(frozen=True)
class CacheInfo:
    hits: int
    misses: int
    entries: int
    capacity: int


class _BarCache:
    """以文件指纹为键的线程安全 LRU 缓存。"""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("cache_capacity 不能为负数")
        self.capacity = capacity
        self._items: OrderedDict[tuple, tuple[Bar, ...]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    def get(self, key: tuple) -> tuple[Bar, ...] | None:
        with self._lock:
            value = self._items.get(key)
            if value is None:
                self._misses += 1
                return None
            self._items.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: tuple, value: tuple[Bar, ...]) -> None:
        if self.capacity == 0:
            return
        with self._lock:
            # 同一路径的旧文件指纹不再可命中，主动清理以免占用容量。
            path = key[0]
            for old_key in [item for item in self._items if item[0] == path and item != key]:
                del self._items[old_key]
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def info(self) -> CacheInfo:
        with self._lock:
            return CacheInfo(self._hits, self._misses, len(self._items), self.capacity)


class MarketDataStore:
    """读取历史行情并提供可重复的查询接口。

    缓存保存解析、校验和排序后的完整文件；symbol 与时间过滤在缓存之后执行，
    因此同一数据集的不同回测区间可以复用一次磁盘读取。
    """

    def __init__(self, cache_capacity: int = 8, strict: bool = True) -> None:
        self.strict = strict
        self._cache = _BarCache(cache_capacity)

    def load(
        self,
        path: str | Path,
        *,
        symbols: str | Iterable[str] | None = None,
        start: int | datetime | str | None = None,
        end: int | datetime | str | None = None,
        symbol: str | None = None,
        use_cache: bool = True,
    ) -> tuple[Bar, ...]:
        """返回按 ``(timestamp, symbol)`` 排序的 Bar，不包含 ``end`` 时刻之后数据。

        ``symbol`` 用于覆盖单标的文件内的 symbol 列；``symbols`` 用于查询过滤。
        时间区间为闭区间 ``[start, end]``。
        """
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"行情文件不存在: {source}")
        suffix = source.suffix.lower()
        if suffix not in {".csv", ".parquet", ".pq"}:
            raise DataError(f"不支持的行情格式: {suffix or '<无扩展名>'}")

        stat = source.stat()
        key = (str(source), stat.st_mtime_ns, stat.st_size, symbol, self.strict)
        bars = self._cache.get(key) if use_cache else None
        if bars is None:
            rows = self._read_csv(source) if suffix == ".csv" else self._read_parquet(source)
            bars = self._normalize(rows, source, symbol)
            if use_cache:
                self._cache.put(key, bars)

        selected = _normalize_symbols(symbols)
        start_ns = _parse_timestamp(start) if start is not None else None
        end_ns = _parse_timestamp(end) if end is not None else None
        if start_ns is not None and end_ns is not None and start_ns > end_ns:
            raise ValueError("start 不能晚于 end")
        if selected is None and start_ns is None and end_ns is None:
            return bars
        return tuple(
            bar for bar in bars
            if (selected is None or bar.symbol in selected)
            and (start_ns is None or bar.timestamp >= start_ns)
            and (end_ns is None or bar.timestamp <= end_ns)
        )

    def stream(self, path: str | Path, **query) -> Iterator[Bar]:
        """流式返回查询结果。缓存开启时不会再次解析文件。"""
        yield from self.load(path, **query)

    def latest(
        self,
        path: str | Path,
        symbol: str,
        *,
        at: int | datetime | str | None = None,
    ) -> Bar | None:
        """返回某标的在指定时刻或之前的最新一根 Bar。"""
        bars = self.load(path, symbols=symbol, end=at)
        return bars[-1] if bars else None

    def cache_info(self) -> CacheInfo:
        return self._cache.info()

    def clear_cache(self) -> None:
        self._cache.clear()

    @staticmethod
    def _read_csv(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            _require_columns(reader.fieldnames, path)
            return list(reader)

    @staticmethod
    def _read_parquet(path: Path) -> list[dict]:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError(
                "读取 Parquet 需要 pyarrow，请执行 pip install -r requirements.txt"
            ) from exc
        table = parquet.read_table(path)
        _require_columns(table.column_names, path)
        selected = list(BAR_COLUMNS) + [
            name for name in MARKET_STATE_COLUMNS if name in table.column_names
        ]
        return table.select(selected).to_pylist()

    def _normalize(
        self, rows: Sequence[dict], source: Path, symbol_override: str | None
    ) -> tuple[Bar, ...]:
        bars: list[Bar] = []
        seen: set[tuple[int, str]] = set()
        for row_number, row in enumerate(rows, start=2):
            try:
                bar = Bar(
                    timestamp=_parse_timestamp(row["timestamp"]),
                    symbol=str(symbol_override or row["symbol"]).strip(),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=_parse_volume(row["volume"]),
                    upper_limit=_parse_optional_price(row.get("upper_limit")),
                    lower_limit=_parse_optional_price(row.get("lower_limit")),
                    is_suspended=_parse_bool(row.get("is_suspended")),
                )
            except (DataError, TypeError, ValueError) as exc:
                raise DataError(f"{source} 第 {row_number} 行无效: {exc}") from exc
            identity = (bar.timestamp, bar.symbol)
            if self.strict and identity in seen:
                raise DataError(
                    f"{source} 第 {row_number} 行重复: timestamp={bar.timestamp}, "
                    f"symbol={bar.symbol}"
                )
            seen.add(identity)
            bars.append(bar)
        bars.sort(key=lambda item: (item.timestamp, item.symbol))
        return tuple(bars)


class DataFeed(ABC):
    """回测引擎数据源抽象。"""

    @abstractmethod
    def stream(self) -> Iterator["MarketSnapshot"]:
        raise NotImplementedError


class FileDataFeed(DataFeed):
    """把 MarketDataStore 的 Bar 转为引擎 MarketSnapshot。"""

    def __init__(
        self,
        path: str | Path,
        symbol: str | None = None,
        *,
        symbols: str | Iterable[str] | None = None,
        start: int | datetime | str | None = None,
        end: int | datetime | str | None = None,
        store: MarketDataStore | None = None,
        reference_data=None,
    ) -> None:
        self.path = Path(path)
        self.symbol = symbol
        self.symbols = symbols
        self.start = start
        self.end = end
        self.store = store or MarketDataStore()
        self.reference_data = reference_data
        self._prepared_bars: tuple[Bar, ...] | None = None
        self._prepared_lineage = None

    def bars(self) -> tuple[Bar, ...]:
        if self._prepared_bars is not None:
            return self._prepared_bars
        return self._load_bars()

    def _load_bars(self) -> tuple[Bar, ...]:
        return self.store.load(
            self.path,
            symbol=self.symbol,
            symbols=self.symbols,
            start=self.start,
            end=self.end,
        )

    def prepare(self):
        """固定小文件输入及其 lineage，保证执行和审计读取同一份内容。"""
        if self._prepared_lineage is not None:
            return self._prepared_lineage

        source = self.path.expanduser().resolve()
        before = _file_identity(source)
        bars = self._load_bars()
        fingerprint = _file_sha256(source)
        after = _file_identity(source)
        if before != after:
            raise DataError(f"准备回测期间行情文件发生变化: {source}")

        self._prepared_bars = bars
        self._prepared_lineage = self._lineage_from_fingerprint(fingerprint)
        return self._prepared_lineage

    def stream(self) -> Iterator["MarketSnapshot"]:
        from .engine_api import MarketSnapshot

        for bar in self.bars():
            snapshot = MarketSnapshot()
            snapshot.timestamp = bar.timestamp
            snapshot.symbol = bar.symbol
            snapshot.open = bar.open
            snapshot.high = bar.high
            snapshot.low = bar.low
            snapshot.close = bar.close
            snapshot.volume = bar.volume
            snapshot.upper_limit = bar.upper_limit
            snapshot.lower_limit = bar.lower_limit
            snapshot.is_suspended = bar.is_suspended
            if self.reference_data is not None:
                self.reference_data.enrich(snapshot)
            yield snapshot

    def lineage(self):
        return self.prepare()

    def _lineage_from_fingerprint(self, fingerprint: str):
        from .market_data.lineage import DataLineage, schema_hash

        selected = self.symbols
        if self.symbol is not None:
            selected = tuple(_normalize_symbols(selected) or ()) + (
                f"__symbol_override__={self.symbol}",
            )
        return DataLineage(
            0, schema_hash(), fingerprint, (fingerprint,)
        ).with_query(
            symbols=selected,
            start=_parse_timestamp(self.start) if self.start is not None else None,
            end=_parse_timestamp(self.end) if self.end is not None else None,
            columns=BAR_COLUMNS,
        )


class CSVDataFeed(FileDataFeed):
    """兼容原接口的 CSV 回测数据源。"""

    def __init__(self, csv_path: str | Path, symbol: str | None = None, **kwargs) -> None:
        if Path(csv_path).suffix.lower() != ".csv":
            raise DataError("CSVDataFeed 只接受 .csv 文件")
        super().__init__(csv_path, symbol, **kwargs)


class ParquetDataFeed(FileDataFeed):
    """Parquet 回测数据源。"""

    def __init__(self, parquet_path: str | Path, symbol: str | None = None, **kwargs) -> None:
        if Path(parquet_path).suffix.lower() not in {".parquet", ".pq"}:
            raise DataError("ParquetDataFeed 只接受 .parquet 或 .pq 文件")
        super().__init__(parquet_path, symbol, **kwargs)


class DataLakeFeed(DataFeed):
    """从分区分钟线数据湖向回测引擎提供行情。

    ``stream_batches`` 是大规模/截面策略首选接口；``stream`` 仅用于兼容当前
    逐 Bar 回调引擎，会产生 Python 对象和逐条跨语言调用开销。
    """

    def __init__(
        self,
        lake,
        *,
        symbols: str | Iterable[str] | None = None,
        start: int | None = None,
        end: int | None = None,
        batch_size: int = 65_536,
        reference_data=None,
        catalog_snapshot=None,
    ) -> None:
        self.lake = lake
        self.symbols = symbols
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.reference_data = reference_data
        self.catalog_snapshot = catalog_snapshot or lake.catalog.snapshot()

    def stream_batches(self):
        yield from self.lake.iter_batches(
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            batch_size=self.batch_size,
            snapshot=self.catalog_snapshot,
        )

    def stream(self) -> Iterator["MarketSnapshot"]:
        from .engine_api import MarketSnapshot

        for batch in self.stream_batches():
            available = set(batch.schema.names)
            selected = BAR_COLUMNS + tuple(
                name for name in MARKET_STATE_COLUMNS if name in available
            )
            columns = {
                name: batch.column(batch.schema.get_field_index(name)).to_pylist()
                for name in selected
            }
            upper_limits = columns.get("upper_limit")
            lower_limits = columns.get("lower_limit")
            suspension_flags = columns.get("is_suspended")
            for index in range(batch.num_rows):
                snapshot = MarketSnapshot()
                snapshot.timestamp = columns["timestamp"][index]
                snapshot.symbol = columns["symbol"][index]
                snapshot.open = columns["open"][index]
                snapshot.high = columns["high"][index]
                snapshot.low = columns["low"][index]
                snapshot.close = columns["close"][index]
                snapshot.volume = columns["volume"][index]
                snapshot.upper_limit = (
                    upper_limits[index] if upper_limits and upper_limits[index] is not None
                    else 0.0
                )
                snapshot.lower_limit = (
                    lower_limits[index] if lower_limits and lower_limits[index] is not None
                    else 0.0
                )
                snapshot.is_suspended = bool(
                    suspension_flags[index]
                    if suspension_flags and suspension_flags[index] is not None else False
                )
                if self.reference_data is not None:
                    self.reference_data.enrich(snapshot)
                yield snapshot

    def lineage(self):
        return self.lake.lineage(self.catalog_snapshot).with_query(
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            columns=BAR_COLUMNS,
        )


def _require_columns(columns: Sequence[str] | None, path: Path) -> None:
    available = set(columns or ())
    missing = [column for column in BAR_COLUMNS if column not in available]
    if missing:
        raise DataError(f"{path} 缺少字段: {', '.join(missing)}")


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_timestamp(value: int | float | datetime | str) -> int:
    if isinstance(value, bool):
        raise DataError("timestamp 不能是布尔值")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise DataError(f"非法 timestamp: {value}")
        return int(value)
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            raise DataError("timestamp 不能为空")
        try:
            return int(text)
        except ValueError:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as exc:
                raise DataError(f"无法解析 timestamp: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt - epoch
    return ((delta.days * 86_400 + delta.seconds) * 1_000_000_000
            + delta.microseconds * 1_000)


def _parse_volume(value: int | float | str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise DataError(f"volume 必须是整数: {value}")
    return int(number)


def _parse_optional_price(value) -> float:
    if value is None or str(value).strip() == "":
        return 0.0
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise DataError(f"非法涨跌停价格: {value}")
    return number


def _parse_bool(value) -> bool:
    if value is None or str(value).strip() == "":
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise DataError(f"非法布尔值: {value}")


def _normalize_symbols(symbols: str | Iterable[str] | None) -> frozenset[str] | None:
    if symbols is None:
        return None
    if isinstance(symbols, str):
        return frozenset((symbols,))
    return frozenset(str(symbol) for symbol in symbols)

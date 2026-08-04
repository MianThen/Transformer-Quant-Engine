"""显式 A 股交易日历与分钟交易时段校验。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo


SHANGHAI = ZoneInfo("Asia/Shanghai")


class BarTimestampConvention(str, Enum):
    START = "start"
    END = "end"


class Session(str, Enum):
    OPENING_AUCTION = "opening_auction"
    CONTINUOUS = "continuous"
    CLOSING_AUCTION = "closing_auction"
    CLOSED = "closed"


class CalendarValidationError(ValueError):
    """行情时间戳不属于配置的交易日或交易时段。"""


@dataclass(frozen=True)
class TradingDay:
    date: date
    is_open: bool = True


class ChinaAShareCalendar:
    """由显式开市日期驱动的 A 股交易日历。

    节假日不能可靠地从周末规则推导，因此构造时必须传入日期，或从带
    ``date,is_open`` 字段的 CSV/Parquet 读取。分钟时间戳按 UTC 纳秒输入。
    """

    def __init__(
        self,
        trading_days: Iterable[date | str | TradingDay],
        *,
        timestamp_convention: BarTimestampConvention | str = BarTimestampConvention.END,
        include_auction: bool = False,
    ) -> None:
        self.timestamp_convention = BarTimestampConvention(timestamp_convention)
        self.include_auction = include_auction
        days: set[date] = set()
        for item in trading_days:
            if isinstance(item, TradingDay):
                if item.is_open:
                    days.add(item.date)
            else:
                days.add(_parse_date(item))
        if not days:
            raise ValueError("交易日历不能为空")
        self._days = frozenset(days)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "ChinaAShareCalendar":
        source = Path(path).expanduser().resolve()
        rows = _read_rows(source)
        if not rows or "date" not in rows[0]:
            raise ValueError("交易日历必须包含 date 字段")
        days = [
            TradingDay(_parse_date(row["date"]), _parse_bool(row.get("is_open"), True))
            for row in rows
        ]
        return cls(days, **kwargs)

    def is_trading_day(self, value: date | datetime | str) -> bool:
        if isinstance(value, datetime):
            value = (
                value.date() if value.tzinfo is None
                else value.astimezone(SHANGHAI).date()
            )
        return _parse_date(value) in self._days

    def session(self, timestamp: int) -> Session:
        local = _to_shanghai(timestamp)
        if local.date() not in self._days:
            return Session.CLOSED
        wall = local.time().replace(tzinfo=None)
        if self.include_auction and time(9, 15) <= wall <= time(9, 25):
            return Session.OPENING_AUCTION
        if self.timestamp_convention == BarTimestampConvention.START:
            if time(9, 30) <= wall <= time(11, 29):
                return Session.CONTINUOUS
            if time(13, 0) <= wall <= time(14, 56):
                return Session.CONTINUOUS
            if time(14, 57) <= wall <= time(14, 59):
                return Session.CLOSING_AUCTION
        else:
            if time(9, 31) <= wall <= time(11, 30):
                return Session.CONTINUOUS
            if time(13, 1) <= wall <= time(14, 57):
                return Session.CONTINUOUS
            if time(14, 58) <= wall <= time(15, 0):
                return Session.CLOSING_AUCTION
        return Session.CLOSED

    def is_trading_minute(self, timestamp: int) -> bool:
        return self.session(timestamp) != Session.CLOSED

    def validate_timestamp(self, timestamp: int) -> None:
        if not self.is_trading_minute(timestamp):
            local = _to_shanghai(timestamp).isoformat()
            raise CalendarValidationError(f"非交易时间行情: {local}")

    @property
    def trading_days(self) -> tuple[date, ...]:
        return tuple(sorted(self._days))


def _to_shanghai(timestamp: int) -> datetime:
    if timestamp <= 0:
        raise CalendarValidationError("timestamp 必须是正的 UTC 纳秒")
    seconds, nanoseconds = divmod(timestamp, 1_000_000_000)
    return datetime.fromtimestamp(
        seconds + nanoseconds / 1_000_000_000, tz=timezone.utc
    ).astimezone(SHANGHAI)


def _parse_date(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _parse_bool(value, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"非法布尔值: {value}")


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("读取 Parquet 交易日历需要 pyarrow") from exc
        return parquet.read_table(path).to_pylist()
    raise ValueError(f"不支持的交易日历格式: {path.suffix}")

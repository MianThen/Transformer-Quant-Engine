"""A 股 point-in-time 证券状态、复权因子、公司行动与股票池。"""

from __future__ import annotations

import bisect
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


FACTOR_EXPOSURE_COLUMN_PREFIX = "factor_exposure__"


@dataclass(frozen=True)
class SecurityState:
    symbol: str
    effective_from: int
    effective_to: int | None = None
    is_listed: bool = True
    is_st: bool = False
    is_suspended: bool = False
    upper_limit: float = 0.0
    lower_limit: float = 0.0
    board: str = ""
    industry: str = ""
    lot_size: int = 100
    min_buy_quantity: int = 100
    factor_exposures: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol or self.effective_from < 0:
            raise ValueError("证券状态的 symbol/effective_from 非法")
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to 必须晚于 effective_from")
        if self.lot_size <= 0 or self.min_buy_quantity <= 0:
            raise ValueError("lot_size/min_buy_quantity 必须为正数")
        if self.upper_limit < 0.0 or self.lower_limit < 0.0:
            raise ValueError("涨跌停价格不能为负数")
        if self.upper_limit and self.lower_limit > self.upper_limit:
            raise ValueError("lower_limit 不能高于 upper_limit")
        if any(not math.isfinite(float(value))
               for value in self.factor_exposures.values()):
            raise ValueError("风格因子暴露必须是有限数")


class SecurityMaster:
    """按 ``[effective_from, effective_to)`` 查询历史证券状态。"""

    def __init__(self, states: Iterable[SecurityState]) -> None:
        grouped: dict[str, list[SecurityState]] = {}
        for state in states:
            grouped.setdefault(state.symbol, []).append(state)
        self._states: dict[str, tuple[SecurityState, ...]] = {}
        self._starts: dict[str, tuple[int, ...]] = {}
        factor_names: set[str] = set()
        for symbol, values in grouped.items():
            values.sort(key=lambda item: item.effective_from)
            previous = None
            for value in values:
                if previous is not None:
                    previous_end = previous.effective_to
                    if previous_end is None or previous_end > value.effective_from:
                        raise ValueError(f"{symbol} 的证券状态区间重叠")
                previous = value
            self._states[symbol] = tuple(values)
            self._starts[symbol] = tuple(item.effective_from for item in values)
            for value in values:
                factor_names.update(value.factor_exposures)
        self.factor_names = tuple(sorted(factor_names))

    @classmethod
    def from_file(cls, path: str | Path) -> "SecurityMaster":
        states = []
        for row in _read_rows(path):
            states.append(SecurityState(
                symbol=str(row["symbol"]).strip(),
                effective_from=_parse_timestamp(row["effective_from"]),
                effective_to=_parse_optional_timestamp(row.get("effective_to")),
                is_listed=_parse_bool(row.get("is_listed"), True),
                is_st=_parse_bool(row.get("is_st"), False),
                is_suspended=_parse_bool(row.get("is_suspended"), False),
                upper_limit=_parse_float(row.get("upper_limit"), 0.0),
                lower_limit=_parse_float(row.get("lower_limit"), 0.0),
                board=str(row.get("board") or "").strip(),
                industry=str(row.get("industry") or "").strip(),
                lot_size=_parse_int(row.get("lot_size"), 100),
                min_buy_quantity=_parse_int(row.get("min_buy_quantity"), 100),
                factor_exposures=_parse_factor_exposures(
                    row.get("factor_exposures_json")
                ),
            ))
        return cls(states)

    def at(self, symbol: str, timestamp: int) -> SecurityState | None:
        values = self._states.get(symbol)
        if not values:
            return None
        index = bisect.bisect_right(self._starts[symbol], timestamp) - 1
        if index < 0:
            return None
        state = values[index]
        if state.effective_to is not None and timestamp >= state.effective_to:
            return None
        return state

    def universe(
        self,
        timestamp: int,
        *,
        include_st: bool = False,
        boards: Iterable[str] | None = None,
    ) -> tuple[str, ...]:
        selected_boards = set(boards) if boards is not None else None
        symbols = []
        for symbol in self._states:
            state = self.at(symbol, timestamp)
            if state is None or not state.is_listed or (state.is_st and not include_st):
                continue
            if selected_boards is not None and state.board not in selected_boards:
                continue
            symbols.append(symbol)
        return tuple(sorted(symbols))


@dataclass(frozen=True)
class AdjustmentFactor:
    symbol: str
    effective_from: int
    factor: float
    effective_to: int | None = None

    def __post_init__(self) -> None:
        if not self.symbol or self.effective_from < 0:
            raise ValueError("复权因子的 symbol/effective_from 非法")
        if not math.isfinite(self.factor) or self.factor <= 0.0:
            raise ValueError("复权因子必须是有限正数")
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to 必须晚于 effective_from")


class AdjustmentFactorStore:
    """信号价格因子；约定 ``signal_price = raw_price * factor``。"""

    def __init__(self, factors: Iterable[AdjustmentFactor]) -> None:
        grouped: dict[str, list[AdjustmentFactor]] = {}
        for factor in factors:
            grouped.setdefault(factor.symbol, []).append(factor)
        self._factors = {}
        self._starts = {}
        for symbol, values in grouped.items():
            values.sort(key=lambda item: item.effective_from)
            for previous, current in zip(values, values[1:]):
                if (previous.effective_to is None
                        or previous.effective_to > current.effective_from):
                    raise ValueError(f"{symbol} 的复权因子区间重叠")
            self._factors[symbol] = tuple(values)
            self._starts[symbol] = tuple(item.effective_from for item in values)

    @classmethod
    def from_file(cls, path: str | Path) -> "AdjustmentFactorStore":
        return cls(AdjustmentFactor(
            symbol=str(row["symbol"]).strip(),
            effective_from=_parse_timestamp(row["effective_from"]),
            effective_to=_parse_optional_timestamp(row.get("effective_to")),
            factor=_parse_float(row["factor"]),
        ) for row in _read_rows(path))

    def factor_at(self, symbol: str, timestamp: int) -> float:
        values = self._factors.get(symbol)
        if not values:
            return 1.0
        index = bisect.bisect_right(self._starts[symbol], timestamp) - 1
        if index < 0:
            return 1.0
        value = values[index]
        if value.effective_to is not None and timestamp >= value.effective_to:
            return 1.0
        return value.factor


@dataclass(frozen=True)
class CorporateAction:
    symbol: str
    timestamp: int
    cash_dividend_per_share: float = 0.0
    share_multiplier: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.symbol or self.timestamp < 0:
            raise ValueError("公司行动的 symbol/timestamp 非法")
        if not math.isfinite(self.cash_dividend_per_share) or self.cash_dividend_per_share < 0:
            raise ValueError("每股现金分红不能为负数")
        if not math.isfinite(self.share_multiplier) or self.share_multiplier <= 0:
            raise ValueError("share_multiplier 必须是有限正数")


class CorporateActionStore:
    def __init__(self, actions: Iterable[CorporateAction]) -> None:
        self._actions = tuple(sorted(actions, key=lambda item: (item.timestamp, item.symbol)))
        identities = [(item.timestamp, item.symbol) for item in self._actions]
        if len(identities) != len(set(identities)):
            raise ValueError("同一 symbol/timestamp 只能有一条公司行动")
        self._timestamps = tuple(item.timestamp for item in self._actions)

    @classmethod
    def from_file(cls, path: str | Path) -> "CorporateActionStore":
        return cls(CorporateAction(
            symbol=str(row["symbol"]).strip(),
            timestamp=_parse_timestamp(row["timestamp"]),
            cash_dividend_per_share=_parse_float(
                row.get("cash_dividend_per_share"), 0.0
            ),
            share_multiplier=_parse_float(row.get("share_multiplier"), 1.0),
            description=str(row.get("description") or ""),
        ) for row in _read_rows(path))

    def between(self, start_exclusive: int, end_inclusive: int) -> tuple[CorporateAction, ...]:
        if end_inclusive < start_exclusive:
            raise ValueError("公司行动查询区间非法")
        left = bisect.bisect_right(self._timestamps, start_exclusive)
        right = bisect.bisect_right(self._timestamps, end_inclusive)
        return self._actions[left:right]

    def at(self, timestamp: int) -> tuple[CorporateAction, ...]:
        left = bisect.bisect_left(self._timestamps, timestamp)
        right = bisect.bisect_right(self._timestamps, timestamp)
        return self._actions[left:right]


class MarketReferenceData:
    """组合独立时态表，并把 point-in-time 状态富化到执行快照。"""

    def __init__(
        self,
        *,
        security_master: SecurityMaster | None = None,
        adjustment_factors: AdjustmentFactorStore | None = None,
        corporate_actions: CorporateActionStore | None = None,
    ) -> None:
        self.security_master = security_master
        self.adjustment_factors = adjustment_factors
        self.corporate_actions = corporate_actions

    def enrich(self, snapshot):
        if self.security_master is not None:
            state = self.security_master.at(snapshot.symbol, snapshot.timestamp)
            snapshot.is_listed = state is not None and state.is_listed
            if state is not None:
                snapshot.is_st = state.is_st
                snapshot.is_suspended = state.is_suspended
                snapshot.upper_limit = state.upper_limit
                snapshot.lower_limit = state.lower_limit
                snapshot.board = state.board
                snapshot.industry = state.industry
                snapshot.lot_size = state.lot_size
                snapshot.min_buy_quantity = state.min_buy_quantity
                snapshot.factor_exposures = dict(state.factor_exposures)
        factor = (
            self.adjustment_factors.factor_at(snapshot.symbol, snapshot.timestamp)
            if self.adjustment_factors is not None else 1.0
        )
        snapshot.adjustment_factor = factor
        snapshot.signal_open = snapshot.open * factor
        snapshot.signal_high = snapshot.high * factor
        snapshot.signal_low = snapshot.low * factor
        snapshot.signal_close = snapshot.close * factor
        return snapshot

    def enrich_batch(self, batch):
        """在 Arrow 列上完成时态富化，避免构造逐 Bar Python 对象。"""
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
        except ImportError as exc:
            raise RuntimeError("批量参考数据富化需要 pyarrow") from exc

        required = ("timestamp", "symbol", "open", "high", "low", "close")
        missing = [name for name in required if name not in batch.schema.names]
        if missing:
            raise ValueError(f"Arrow batch 缺少富化字段: {', '.join(missing)}")
        timestamps = [int(value) for value in batch["timestamp"].to_pylist()]
        symbols = [str(value) for value in batch["symbol"].to_pylist()]
        result = batch

        if self.security_master is not None:
            states = [
                self.security_master.at(symbol, timestamp)
                for symbol, timestamp in zip(symbols, timestamps)
            ]
            state_columns = {
                "is_listed": (pa.bool_(), False),
                "is_st": (pa.bool_(), False),
                "is_suspended": (pa.bool_(), False),
                "upper_limit": (pa.float64(), 0.0),
                "lower_limit": (pa.float64(), 0.0),
                "board": (pa.string(), ""),
                "industry": (pa.string(), ""),
                "lot_size": (pa.int64(), 1),
                "min_buy_quantity": (pa.int64(), 1),
            }
            values = {
                name: _column_values(result, name, default)
                for name, (_, default) in state_columns.items()
            }
            for index, state in enumerate(states):
                values["is_listed"][index] = state is not None and state.is_listed
                if state is None:
                    continue
                for name in state_columns:
                    values[name][index] = getattr(state, name)
            for name, (data_type, _) in state_columns.items():
                result = _set_column(result, name, pa.array(values[name], type=data_type))
            for factor_name in self.security_master.factor_names:
                column_name = FACTOR_EXPOSURE_COLUMN_PREFIX + factor_name
                factor_values = [
                    None if state is None else state.factor_exposures.get(factor_name)
                    for state in states
                ]
                result = _set_column(
                    result, column_name, pa.array(factor_values, type=pa.float64())
                )

        factors = [
            self.adjustment_factors.factor_at(symbol, timestamp)
            if self.adjustment_factors is not None else 1.0
            for symbol, timestamp in zip(symbols, timestamps)
        ]
        factor_array = pa.array(factors, type=pa.float64())
        result = _set_column(result, "adjustment_factor", factor_array)
        for name in ("open", "high", "low", "close"):
            result = _set_column(
                result, f"signal_{name}", pc.multiply(result[name], factor_array)
            )
        return result

    def universe(self, timestamp: int, **filters) -> tuple[str, ...]:
        if self.security_master is None:
            raise RuntimeError("未配置 SecurityMaster")
        return self.security_master.universe(timestamp, **filters)

    def actions_between(self, start_exclusive: int, end_inclusive: int):
        if self.corporate_actions is None:
            return ()
        return self.corporate_actions.between(start_exclusive, end_inclusive)

    def actions_at(self, timestamp: int):
        if self.corporate_actions is None:
            return ()
        return self.corporate_actions.at(timestamp)


def _column_values(batch, name: str, default) -> list:
    if name in batch.schema.names:
        return [default if value is None else value for value in batch[name].to_pylist()]
    return [default for _ in range(batch.num_rows)]


def _set_column(batch, name: str, values):
    index = batch.schema.get_field_index(name)
    if index >= 0:
        return batch.set_column(index, name, values)
    return batch.append_column(name, values)


def _read_rows(path: str | Path) -> list[dict]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))
    if source.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("读取 Parquet 时态数据需要 pyarrow") from exc
        return parquet.read_table(source).to_pylist()
    raise ValueError(f"不支持的时态数据格式: {source.suffix}")


def _parse_timestamp(value) -> int:
    number = int(value)
    if number < 0:
        raise ValueError("timestamp 不能为负数")
    return number


def _parse_optional_timestamp(value) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return _parse_timestamp(value)


def _parse_bool(value, default=False) -> bool:
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


def _parse_float(value, default=None) -> float:
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError("缺少浮点字段")
        return float(default)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"非法浮点值: {value}")
    return result


def _parse_int(value, default=None) -> int:
    if value is None or str(value).strip() == "":
        if default is None:
            raise ValueError("缺少整数字段")
        return int(default)
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"非法整数值: {value}")
    return int(number)


def _parse_factor_exposures(value) -> dict[str, float]:
    if value is None or str(value).strip() == "":
        return {}
    parsed = json.loads(str(value)) if isinstance(value, str) else dict(value)
    result = {str(name): float(exposure) for name, exposure in parsed.items()}
    if any(not math.isfinite(exposure) for exposure in result.values()):
        raise ValueError("风格因子暴露必须是有限数")
    return result

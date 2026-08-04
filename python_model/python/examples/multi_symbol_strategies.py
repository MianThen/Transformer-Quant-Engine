"""Multi-symbol portfolio variants of the built-in example strategies."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from typing import Iterable

from ..strategy import ColumnarStrategy


class _EqualWeightPortfolioStrategy(ColumnarStrategy):
    def __init__(
        self,
        symbols: Iterable[str],
        *,
        initial_cash: float,
        capital_utilization: float = 0.95,
        lot_size: int = 100,
    ) -> None:
        normalized = tuple(dict.fromkeys(str(symbol).strip() for symbol in symbols))
        if not normalized or any(not symbol for symbol in normalized):
            raise ValueError("symbols 不能为空")
        if not math.isfinite(initial_cash) or initial_cash <= 0.0:
            raise ValueError("initial_cash 必须是有限正数")
        if not math.isfinite(capital_utilization) or not 0.0 < capital_utilization < 1.0:
            raise ValueError("capital_utilization 必须在 (0, 1) 内")
        if isinstance(lot_size, bool) or not isinstance(lot_size, int) or lot_size <= 0:
            raise ValueError("lot_size 必须是正整数")
        super().__init__(symbols=list(normalized))
        self.initial_cash = float(initial_cash)
        self.capital_utilization = float(capital_utilization)
        self.lot_size = lot_size
        self._budget_per_symbol = (
            self.initial_cash * self.capital_utilization / len(normalized)
        )
        self._open_order_ids: dict[str, set[int]] = defaultdict(set)

    def _target_for_price(self, price: float) -> int:
        lots = int(self._budget_per_symbol / price) // self.lot_size
        return lots * self.lot_size

    def _target_columns(self, index: int, symbol: str, target: int):
        return self.target_position_columns(index, symbol, target)

    @staticmethod
    def _append_order(columns: dict[str, list[int]], order) -> None:
        if order is None:
            return
        for name in ("symbol_index", "side", "quantity"):
            columns[name].extend(order[name])

    def on_order_update(self, record) -> None:
        symbol = record.order.symbol
        if symbol not in self.symbols:
            return
        if record.status.name in {"ACCEPTED", "PARTIALLY_FILLED"}:
            self._open_order_ids[symbol].add(record.order.id)
        else:
            self._open_order_ids[symbol].discard(record.order.id)

    def _cancel_open_orders(self, symbol: str, timestamp: int) -> None:
        for order_id in tuple(self._open_order_ids[symbol]):
            self.cancel_order(order_id, timestamp)

    def on_fill(self, _fill) -> None:
        return None


class MultiSymbolMACrossStrategy(_EqualWeightPortfolioStrategy):
    """Run an independent MA state machine per symbol with shared portfolio cash."""

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        initial_cash: float,
        short_window: int = 5,
        long_window: int = 20,
        capital_utilization: float = 0.95,
        lot_size: int = 100,
    ) -> None:
        if not 1 <= short_window < long_window:
            raise ValueError("双均线参数必须满足 1 <= short_window < long_window")
        super().__init__(
            symbols,
            initial_cash=initial_cash,
            capital_utilization=capital_utilization,
            lot_size=lot_size,
        )
        self.short_window = short_window
        self.long_window = long_window
        self._prices = {
            symbol: deque(maxlen=long_window) for symbol in self.symbols
        }
        self._prev_short_above: dict[str, bool | None] = {
            symbol: None for symbol in self.symbols
        }

    def on_cross_section_view(self, batch):
        columns = {"symbol_index": [], "side": [], "quantity": []}
        for index in range(len(batch)):
            symbol = batch.symbol(index)
            prices = self._prices.get(symbol)
            if prices is None:
                continue
            price = batch.close(index)
            prices.append(price)
            if len(prices) < self.long_window:
                continue
            values = list(prices)
            short_above = (
                sum(values[-self.short_window:]) / self.short_window
                > sum(values) / self.long_window
            )
            previous = self._prev_short_above[symbol]
            self._prev_short_above[symbol] = short_above
            if previous is None or short_above == previous:
                continue
            self._cancel_open_orders(symbol, batch.timestamp(index))
            target = self._target_for_price(price) if short_above else 0
            self._append_order(
                columns, self._target_columns(index, symbol, target)
            )
        return columns if columns["quantity"] else None


class MultiSymbolMeanReversionStrategy(_EqualWeightPortfolioStrategy):
    """Run an independent mean-reversion state per symbol with shared cash."""

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        initial_cash: float,
        window: int = 20,
        num_std: float = 2.0,
        capital_utilization: float = 0.95,
        lot_size: int = 100,
    ) -> None:
        if window < 2 or not math.isfinite(num_std) or num_std <= 0.0:
            raise ValueError("均值回归窗口至少为 2，标准差倍数必须为正数")
        super().__init__(
            symbols,
            initial_cash=initial_cash,
            capital_utilization=capital_utilization,
            lot_size=lot_size,
        )
        self.window = window
        self.num_std = float(num_std)
        self._prices = {symbol: deque(maxlen=window) for symbol in self.symbols}

    def on_cross_section_view(self, batch):
        columns = {"symbol_index": [], "side": [], "quantity": []}
        for index in range(len(batch)):
            symbol = batch.symbol(index)
            prices = self._prices.get(symbol)
            if prices is None:
                continue
            price = batch.close(index)
            prices.append(price)
            if len(prices) < self.window:
                continue
            mean = statistics.fmean(prices)
            std = statistics.pstdev(prices)
            if price < mean - self.num_std * std:
                target = self._target_for_price(price)
            elif price > mean + self.num_std * std:
                target = 0
            else:
                continue
            self._cancel_open_orders(symbol, batch.timestamp(index))
            self._append_order(
                columns, self._target_columns(index, symbol, target)
            )
        return columns if columns["quantity"] else None

"""策略基类。用户继承此类编写自己的策略。

策略通过三个回调与引擎交互:
  - on_market_data: 收到行情时调用,返回要下的订单列表
  - on_fill:        成交回报回调
  - on_timer:       定时器回调(如每日收盘前平仓)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    # 类型标注用;运行期由 engine_api 决定 C++ 还是纯 Python 后端。
    from .engine_api import Fill, MarketSnapshot, Order, OrderRecord


class Strategy(ABC):
    """策略基类。"""

    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.engine = None  # 由 BacktestRunner 注入,用于查询持仓/下单上下文

    def on_market_data(self, snapshot: "MarketSnapshot") -> list["Order"]:
        """逐 Bar 兼容回调；纯截面策略无需覆盖。"""
        return []

    def on_cross_section(self, snapshots: list["MarketSnapshot"]) -> list["Order"]:
        """收到同一分钟的完整截面；默认适配现有逐 Bar 策略。"""
        orders = []
        for snapshot in snapshots:
            orders.extend(self.on_market_data(snapshot))
        return orders

    def on_fill(self, fill: "Fill") -> None:
        """成交回报回调。默认不处理,子类按需覆盖。"""
        return None

    def on_order_update(self, record: "OrderRecord") -> None:
        """订单接受、部分成交及终态回调。默认不处理。"""
        return None

    def get_cash(self) -> float:
        if self.engine is None:
            raise RuntimeError("策略尚未绑定回测引擎")
        return self.engine.get_cash()

    def get_position(self, symbol: str):
        return self._require_engine().get_position(symbol)

    def get_positions(self):
        return self._require_engine().get_positions()

    def get_portfolio(self):
        return self._require_engine().get_portfolio_snapshot()

    def cancel_order(self, order_id: int, timestamp: int = 0) -> bool:
        if isinstance(order_id, bool) or int(order_id) <= 0:
            raise ValueError("order_id 必须是正整数")
        _validate_timestamp(timestamp)
        return bool(self._require_engine().cancel_order(int(order_id), timestamp))

    def market_order(
        self, symbol: str, side, quantity: int, timestamp: int = 0
    ) -> "Order":
        from .engine_api import Order, OrderType

        order = Order()
        order.symbol = _validate_symbol(symbol)
        order.side = side
        order.type = OrderType.MARKET
        order.quantity = _validate_quantity(quantity)
        order.timestamp = _validate_timestamp(timestamp)
        return order

    def limit_order(
        self, symbol: str, side, quantity: int, limit_price: float,
        timestamp: int = 0,
    ) -> "Order":
        from .engine_api import OrderType

        price = float(limit_price)
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError("limit_price 必须是有限正数")
        order = self.market_order(symbol, side, quantity, timestamp)
        order.type = OrderType.LIMIT
        order.limit_price = price
        return order

    def target_position_order(
        self, symbol: str, target_quantity: int, timestamp: int = 0
    ) -> "Order | None":
        """按引擎实际持仓生成一张调仓市价单；已达目标时返回 None。"""
        from .engine_api import Side

        symbol = _validate_symbol(symbol)
        if isinstance(target_quantity, bool) or not isinstance(target_quantity, int):
            raise ValueError("target_quantity 必须是整数")
        current = int(self.get_position(symbol).quantity)
        delta = target_quantity - current
        if delta == 0:
            return None
        side = Side.BUY if delta > 0 else Side.SELL
        return self.market_order(symbol, side, abs(delta), timestamp)

    def close_position_order(
        self, symbol: str, timestamp: int = 0
    ) -> "Order | None":
        return self.target_position_order(symbol, 0, timestamp)

    def on_timer(self, timestamp: int) -> list["Order"]:
        """定时器回调(如每天收盘前平仓)。默认不操作。"""
        return []

    def _require_engine(self):
        if self.engine is None:
            raise RuntimeError("策略尚未绑定回测引擎")
        return self.engine


class ColumnarStrategy(Strategy):
    """直接消费 C++ ``MarketBatchView`` 的高吞吐截面策略。"""

    @abstractmethod
    def on_cross_section_view(self, batch):
        """返回列式订单 dict、Order 列表或 None。"""
        raise NotImplementedError

    def on_cross_section(self, snapshots: list["MarketSnapshot"]) -> list["Order"]:
        view = _SnapshotBatchView(snapshots)
        return _materialize_columnar_orders(
            self.on_cross_section_view(view), snapshots
        )

    def target_position_columns(
        self, symbol_index: int, symbol: str, target_quantity: int
    ) -> dict[str, list[int]] | None:
        """按真实持仓生成 C++ 可直接解析的列式目标仓位订单。"""
        from .engine_api import Side

        if isinstance(symbol_index, bool) or not isinstance(symbol_index, int):
            raise ValueError("symbol_index 必须是非负整数")
        if symbol_index < 0:
            raise ValueError("symbol_index 必须是非负整数")
        if isinstance(target_quantity, bool) or not isinstance(target_quantity, int):
            raise ValueError("target_quantity 必须是整数")
        current = int(self.get_position(_validate_symbol(symbol)).quantity)
        delta = target_quantity - current
        if delta == 0:
            return None
        return {
            "symbol_index": [symbol_index],
            "side": [int(Side.BUY if delta > 0 else Side.SELL)],
            "quantity": [abs(delta)],
        }


class _SnapshotBatchView:
    def __init__(self, snapshots: Sequence["MarketSnapshot"]) -> None:
        self._snapshots = snapshots

    def __len__(self) -> int:
        return len(self._snapshots)

    def symbol(self, index: int) -> str:
        return self._snapshots[index].symbol

    def timestamp(self, index: int) -> int:
        return self._snapshots[index].timestamp

    def open(self, index: int) -> float:
        return self._snapshots[index].open

    def high(self, index: int) -> float:
        return self._snapshots[index].high

    def low(self, index: int) -> float:
        return self._snapshots[index].low

    def close(self, index: int) -> float:
        return self._snapshots[index].close

    def volume(self, index: int) -> int:
        return self._snapshots[index].volume


def _materialize_columnar_orders(value, snapshots) -> list["Order"]:
    if value is None:
        return []
    if not isinstance(value, dict):
        return list(value)
    if "quantity" not in value:
        raise ValueError("列式订单缺少 quantity")
    quantities = list(value["quantity"])
    count = len(quantities)
    symbols = _optional_order_column(value, "symbol", count)
    symbol_indices = _optional_order_column(value, "symbol_index", count)
    sides = _optional_order_column(value, "side", count)
    types = _optional_order_column(value, "type", count)
    prices = _optional_order_column(value, "price", count)
    if symbols is None and symbol_indices is None and len(snapshots) != 1:
        raise ValueError("多标的列式订单必须提供 symbol 或 symbol_index")

    from .engine_api import Order, OrderType, Side

    orders = []
    for index, quantity in enumerate(quantities):
        order = Order()
        order.quantity = _validate_quantity(quantity)
        if symbols is not None:
            order.symbol = _validate_symbol(symbols[index])
        elif symbol_indices is not None:
            market_index = symbol_indices[index]
            if (isinstance(market_index, bool) or not isinstance(market_index, int)
                    or not 0 <= market_index < len(snapshots)):
                raise IndexError("symbol_index 超出行情截面")
            order.symbol = snapshots[market_index].symbol
        if sides is not None:
            order.side = Side(int(sides[index]))
        if types is not None:
            order.type = OrderType(int(types[index]))
        if prices is not None:
            price = float(prices[index])
            if not math.isfinite(price) or price <= 0.0:
                raise ValueError("price 必须是有限正数")
            order.limit_price = price
        orders.append(order)
    return orders


def _optional_order_column(value: dict, name: str, count: int):
    if name not in value:
        return None
    column = list(value[name])
    if len(column) != count:
        raise ValueError(f"{name} 长度与 quantity 不一致")
    return column


def _validate_symbol(symbol: str) -> str:
    value = str(symbol).strip()
    if not value:
        raise ValueError("symbol 不能为空")
    return value


def _validate_quantity(quantity: int) -> int:
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("quantity 必须是正整数")
    return quantity


def _validate_timestamp(timestamp: int) -> int:
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ValueError("timestamp 必须是非负整数")
    return timestamp

from __future__ import annotations

from types import SimpleNamespace

import pytest

from python.engine_api import OrderType, Side
from python.strategy import ColumnarStrategy, Strategy


class FakeEngine:
    def __init__(self, quantity=0):
        self.quantity = quantity
        self.canceled = []

    def get_cash(self):
        return 1_000.0

    def get_position(self, symbol):
        return SimpleNamespace(symbol=symbol, quantity=self.quantity)

    def get_positions(self):
        return [self.get_position("X")]

    def get_portfolio_snapshot(self):
        return SimpleNamespace(equity=1_000.0)

    def cancel_order(self, order_id, timestamp):
        self.canceled.append((order_id, timestamp))
        return True


def test_strategy_order_helpers_use_actual_engine_position():
    strategy = Strategy(["X"])
    strategy.engine = FakeEngine(quantity=40)

    buy = strategy.target_position_order("X", 100, 7)
    sell = strategy.target_position_order("X", -20, 8)
    close = strategy.close_position_order("X", 9)
    limit = strategy.limit_order("X", Side.BUY, 10, 12.5, 10)

    assert (buy.side, buy.quantity, buy.timestamp) == (Side.BUY, 60, 7)
    assert (sell.side, sell.quantity, sell.timestamp) == (Side.SELL, 60, 8)
    assert (close.side, close.quantity) == (Side.SELL, 40)
    assert limit.type == OrderType.LIMIT
    assert limit.limit_price == 12.5
    assert strategy.get_positions()[0].quantity == 40
    assert strategy.cancel_order(3, 11)
    assert strategy.engine.canceled == [(3, 11)]


def test_strategy_order_helpers_validate_business_inputs():
    strategy = Strategy(["X"])
    with pytest.raises(RuntimeError, match="尚未绑定"):
        strategy.get_cash()

    strategy.engine = FakeEngine()
    with pytest.raises(ValueError, match="quantity"):
        strategy.market_order("X", Side.BUY, 0)
    with pytest.raises(ValueError, match="limit_price"):
        strategy.limit_order("X", Side.BUY, 1, float("nan"))
    with pytest.raises(ValueError, match="target_quantity"):
        strategy.target_position_order("X", 1.5)
    with pytest.raises(ValueError, match="order_id"):
        strategy.cancel_order(0)


def test_columnar_target_position_uses_symbol_index_without_copying_symbol():
    class Example(ColumnarStrategy):
        def on_cross_section_view(self, batch):
            return self.target_position_columns(0, batch.symbol(0), 100)

    strategy = Example(["X"])
    strategy.engine = FakeEngine(quantity=40)
    result = strategy.on_cross_section_view(
        SimpleNamespace(symbol=lambda index: "X")
    )

    assert result == {
        "symbol_index": [0], "side": [int(Side.BUY)], "quantity": [60]
    }

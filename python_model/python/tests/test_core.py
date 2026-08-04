"""纯 Python 核心引擎单元测试。

用已知输入验证撮合、持仓、盈亏计算,与手算结果对照。
运行: pytest python/tests/ -v
"""

from __future__ import annotations

import math
import pytest

from python.core import (
    BacktestEngine,
    CorporateAction,
    ExecutionConfig,
    FeeSchedule,
    Fill,
    MarketSnapshot,
    Order,
    OrderStatus,
    OrderBook,
    OrderType,
    PnLTracker,
    PositionTracker,
    RejectReason,
    Side,
)


def _snap(symbol, ts, close):
    return MarketSnapshot(symbol=symbol, timestamp=ts, open=close, high=close,
                          low=close, close=close, volume=1000)


# ---------- 订单簿 ----------

def test_market_order_fills_at_last_price():
    book = OrderBook("X")
    book.update_market_data(_snap("X", 1, 10.0))
    order = Order(id=1, symbol="X", side=Side.BUY, type=OrderType.MARKET, quantity=100)
    fills = book.submit_order(order)
    assert len(fills) == 1
    assert fills[0].price == 10.0
    assert fills[0].quantity == 100


def test_market_order_no_price_no_fill():
    # 未收到行情(last_price=0)时不成交
    book = OrderBook("X")
    order = Order(id=1, symbol="X", side=Side.BUY, type=OrderType.MARKET, quantity=100)
    assert book.submit_order(order) == []


def test_limit_buy_rests_until_price_drops():
    book = OrderBook("X")
    book.update_market_data(_snap("X", 1, 10.0))
    # 限价 9.5 买,现价 10.0 不成交,挂单
    order = Order(id=1, symbol="X", side=Side.BUY, type=OrderType.LIMIT,
                  quantity=100, limit_price=9.5)
    assert book.submit_order(order) == []
    # 价格跌到 9.4，开盘价优于限价，按 9.4 成交。
    fills = book.update_market_data(_snap("X", 2, 9.4))
    assert len(fills) == 1
    assert fills[0].price == 9.4


# ---------- 持仓 ----------

def test_position_avg_cost_and_realized_pnl():
    pt = PositionTracker()
    # 买 100 @ 10  → 均价 10
    pt.apply_fill(Fill(symbol="X", side=Side.BUY, quantity=100, price=10.0))
    # 再买 100 @ 12 → 均价 11
    pt.apply_fill(Fill(symbol="X", side=Side.BUY, quantity=100, price=12.0))
    pos = pt.get_position("X")
    assert pos.quantity == 200
    assert abs(pos.avg_cost - 11.0) < 1e-9
    # 卖 100 @ 15 → 已实现 (15-11)*100 = 400
    pt.apply_fill(Fill(symbol="X", side=Side.SELL, quantity=100, price=15.0))
    pos = pt.get_position("X")
    assert pos.quantity == 100
    assert abs(pos.realized_pnl - 400.0) < 1e-9
    assert pt.closed_pnls == [400.0]


def test_position_market_value():
    pt = PositionTracker()
    pt.apply_fill(Fill(symbol="X", side=Side.BUY, quantity=100, price=10.0))
    assert abs(pt.market_value({"X": 12.0}) - 1200.0) < 1e-9


# ---------- 盈亏指标 ----------

def test_total_return():
    pnl = PnLTracker()
    pnl.record_snapshot(1, 1000.0, 1000.0)
    pnl.record_snapshot(2, 1100.0, 1100.0)
    assert abs(pnl.total_return() - 0.1) < 1e-9


def test_max_drawdown():
    pnl = PnLTracker()
    for eq in [1000, 1200, 900, 1100]:  # 峰值 1200 → 谷 900,回撤 25%
        pnl.record_snapshot(0, float(eq), float(eq))
    assert abs(pnl.max_drawdown() - 0.25) < 1e-9


def test_sharpe_zero_when_flat():
    pnl = PnLTracker()
    for _ in range(10):
        pnl.record_snapshot(0, 1000.0, 1000.0)  # 完全不动 → std=0 → Sharpe 0
    assert pnl.sharpe_ratio() == 0.0


# ---------- 引擎全链路 ----------

def test_engine_end_to_end_buy_and_hold():
    """买入持有(close 模式):第一根买 100 股 @10,价格涨到 12,权益应涨 200。

    close 模式=当根收盘价即时成交(会用到本 bar 信息),用于对照手算。
    默认的 next_open 无前视模型见 test_engine_next_open_fill。
    """
    engine = BacktestEngine(initial_cash=10_000.0, fill_timing="close")

    bought = {"done": False}

    def on_md(snap):
        if not bought["done"]:
            bought["done"] = True
            return [Order(symbol=snap.symbol, side=Side.BUY,
                          type=OrderType.MARKET, quantity=100)]
        return []

    engine.set_on_market_data(on_md)
    engine.push_market_data(_snap("X", 1, 10.0))  # 买入 100@10,花 1000
    engine.push_market_data(_snap("X", 2, 12.0))  # 持仓市值 1200

    # 现金 9000 + 持仓 1200 = 10200
    assert abs(engine.get_equity() - 10_200.0) < 1e-9
    assert abs(engine.get_total_return() - 0.02) < 1e-9
    assert len(engine.get_trade_history()) == 1


def test_engine_commission_deducted():
    """带手续费(close 模式):买 100@10,手续费按 notional*0.001 = 1 元,现金应少扣。"""
    engine = BacktestEngine(initial_cash=10_000.0, fill_timing="close")
    engine.set_commission_fn(lambda notional, is_sell: notional * 0.001)

    def on_md(snap):
        if snap.timestamp == 1:
            return [Order(symbol=snap.symbol, side=Side.BUY,
                          type=OrderType.MARKET, quantity=100)]
        return []

    engine.set_on_market_data(on_md)
    engine.push_market_data(_snap("X", 1, 10.0))
    # 现金 = 10000 - 1000(成交) - 1(手续费) = 8999
    assert abs(engine.get_cash() - 8999.0) < 1e-9


def test_engine_next_open_fill():
    """默认 next_open:bar1 close 出的市价买单,在 bar2 的 open 价成交(无前视)。

    bar1 open=close=10 → 策略下市价买 100,入 pending,本 bar 不成交(权益仍 10000)。
    bar2 open=11 close=12 → 挂单以 open=11 成交,现金 10000-1100=8900,
                            持仓市值按 close=12 → 100*12=1200,权益 10100。
    """
    engine = BacktestEngine(initial_cash=10_000.0)  # 默认 next_open

    def on_md(snap):
        if snap.timestamp == 1:
            return [Order(symbol=snap.symbol, side=Side.BUY,
                          type=OrderType.MARKET, quantity=100)]
        return []

    engine.set_on_market_data(on_md)
    engine.push_market_data(_snap("X", 1, 10.0))
    assert engine.get_cash() == 10_000.0             # bar1 未成交
    assert len(engine.get_trade_history()) == 0

    engine.push_market_data(
        MarketSnapshot(symbol="X", timestamp=2, open=11.0, high=12.0,
                       low=11.0, close=12.0, volume=1000)
    )
    assert abs(engine.get_cash() - 8_900.0) < 1e-9   # 以 open=11 成交
    assert abs(engine.get_equity() - 10_100.0) < 1e-9
    assert len(engine.get_trade_history()) == 1
    assert abs(engine.get_trade_history()[0].price - 11.0) < 1e-9
    assert engine.get_trade_history()[0].timestamp == 2


def test_cross_section_updates_atomically_once():
    engine = BacktestEngine(initial_cash=10_000.0, fill_timing="close")
    observed = []

    def on_cross_section(batch):
        observed.append([snapshot.symbol for snapshot in batch])
        return []

    engine.set_on_cross_section(on_cross_section)
    engine.process_market_data_batch([
        _snap("BBB", 1, 20.0),
        _snap("AAA", 1, 10.0),
    ])
    assert observed == [["AAA", "BBB"]]
    assert len(engine.get_equity_curve()) == 1

    with pytest.raises(ValueError, match="严格递增"):
        engine.process_market_data(_snap("AAA", 0, 10.0))


def test_total_return_includes_first_bar_commission():
    engine = BacktestEngine(initial_cash=1_000.0, fill_timing="close")
    engine.set_commission_fn(lambda notional, is_sell: notional * 0.001)
    engine.set_on_market_data(lambda snap: [
        Order(symbol=snap.symbol, side=Side.BUY,
              type=OrderType.MARKET, quantity=10)
    ])
    engine.process_market_data(_snap("X", 1, 10.0))
    assert abs(engine.get_equity() - 999.9) < 1e-9
    assert abs(engine.get_total_return() + 0.0001) < 1e-12


def test_a_share_execution_constraints_and_partial_fill():
    config = ExecutionConfig(max_volume_participation=0.1, slippage_bps=10.0)
    engine = BacktestEngine(10_000.0, "next_open", config)
    ordered = False

    def on_md(snapshot):
        nonlocal ordered
        if ordered:
            return []
        ordered = True
        return [Order(symbol=snapshot.symbol, quantity=15)]

    engine.set_on_market_data(on_md)
    for timestamp, price in ((1, 10.0), (2, 11.0), (3, 12.0)):
        engine.process_market_data(MarketSnapshot(
            symbol="X", timestamp=timestamp, open=price, high=price,
            low=price, close=price, volume=100,
        ))

    trades = engine.get_trade_history()
    assert [trade.quantity for trade in trades] == [10, 5]
    assert [trade.price for trade in trades] == pytest.approx([11.011, 12.012])


@pytest.mark.parametrize("state", ["suspended", "limit_up", "zero_volume"])
def test_untradeable_bar_rejects_buy(state):
    engine = BacktestEngine(10_000.0, "close")
    engine.set_on_market_data(
        lambda snapshot: [Order(symbol=snapshot.symbol, quantity=10)]
    )
    snapshot = _snap("X", 1, 10.0)
    if state == "suspended":
        snapshot.is_suspended = True
    elif state == "limit_up":
        snapshot.upper_limit = 10.0
    else:
        snapshot.volume = 0
    engine.process_market_data(snapshot)
    assert engine.get_trade_history() == []


def test_t_plus_one_uses_shanghai_trading_day():
    day1 = 1_752_714_000_000_000_000
    same_day = day1 + 3_600_000_000_000
    day2 = day1 + 86_400_000_000_000
    engine = BacktestEngine(10_000.0, "close")

    def on_md(snapshot):
        side = Side.BUY if snapshot.timestamp == day1 else Side.SELL
        return [Order(symbol=snapshot.symbol, side=side, quantity=100)]

    engine.set_on_market_data(on_md)
    engine.process_market_data(_snap("X", day1, 10.0))
    assert engine.get_position("X").sellable_quantity == 0
    engine.process_market_data(_snap("X", same_day, 10.0))
    assert len(engine.get_trade_history()) == 1
    engine.process_market_data(_snap("X", day2, 11.0))
    assert len(engine.get_trade_history()) == 2
    assert engine.get_position("X").quantity == 0


def test_t_zero_allows_same_day_sell_but_rejects_naked_short():
    config = ExecutionConfig(enforce_t_plus_one=False)
    engine = BacktestEngine(10_000.0, "close", config)
    engine.set_on_market_data(
        lambda snapshot: [Order(
            symbol=snapshot.symbol,
            side=Side.BUY if snapshot.timestamp == 1 else Side.SELL,
            quantity=100,
        )]
    )
    engine.process_market_data(_snap("X", 1, 10.0))
    engine.process_market_data(_snap("X", 2, 10.0))
    assert len(engine.get_trade_history()) == 2
    assert engine.get_position("X").quantity == 0

    naked = BacktestEngine(10_000.0, "close", config)
    naked.set_on_market_data(lambda snapshot: [Order(
        symbol=snapshot.symbol, side=Side.SELL, quantity=100,
    )])
    naked.process_market_data(_snap("X", 1, 10.0))
    assert naked.get_trade_history() == []
    assert naked.get_order_history()[0].reject_reason == RejectReason.INSUFFICIENT_POSITION


def test_allow_short_is_explicit_and_independent_from_t_plus_one():
    config = ExecutionConfig(enforce_t_plus_one=False, allow_short=True)
    engine = BacktestEngine(10_000.0, "close", config)
    engine.set_on_market_data(lambda snapshot: [Order(
        symbol=snapshot.symbol, side=Side.SELL, quantity=100,
    )])
    engine.process_market_data(_snap("X", 1, 10.0))
    assert len(engine.get_trade_history()) == 1
    assert engine.get_position("X").quantity == -100


def test_native_fee_schedule_uses_fill_timestamp():
    engine = BacktestEngine(10_000.0, "close")
    engine.set_fee_schedules([
        FeeSchedule(0, 2, 0.001, 0.0, 0.0),
        FeeSchedule(2, None, 0.002, 0.0, 0.0),
    ])
    engine.set_on_market_data(lambda snapshot: [Order(
        symbol=snapshot.symbol, side=Side.BUY, quantity=10,
    )])
    engine.process_market_data(_snap("X", 1, 10.0))
    engine.process_market_data(_snap("X", 2, 10.0))
    assert [trade.commission for trade in engine.get_trade_history()] == pytest.approx(
        [0.1, 0.2]
    )
    assert engine.get_cash() == pytest.approx(10_000.0 - 200.3)

    with pytest.raises(RuntimeError, match="不能同时配置"):
        engine.set_commission_fn(lambda _notional, _is_sell: 0.0)


def test_invalid_execution_config_is_rejected():
    with pytest.raises(ValueError, match="max_volume_participation"):
        BacktestEngine(execution_config=ExecutionConfig(max_volume_participation=math.nan))
    with pytest.raises(ValueError, match="initial_cash"):
        BacktestEngine(initial_cash=-1.0)


def test_invalid_market_data_and_commission_are_rejected():
    engine = BacktestEngine(10_000.0, "close")
    invalid = _snap("X", 1, 10.0)
    invalid.high = 9.0
    with pytest.raises(ValueError, match="MarketSnapshot"):
        engine.process_market_data(invalid)

    fee_engine = BacktestEngine(10_000.0, "close")
    fee_engine.set_commission_fn(lambda notional, is_sell: -1.0)
    fee_engine.set_on_market_data(
        lambda snapshot: [Order(symbol=snapshot.symbol, quantity=100)]
    )
    with pytest.raises(ValueError, match="commission"):
        fee_engine.process_market_data(_snap("X", 1, 10.0))


def test_order_rejections_are_auditable():
    lot_engine = BacktestEngine(10_000.0, "close")
    lot_engine.set_on_market_data(
        lambda snapshot: [Order(symbol=snapshot.symbol, quantity=10)]
    )
    lot_bar = _snap("X", 1, 10.0)
    lot_bar.lot_size = lot_bar.min_buy_quantity = 100
    lot_engine.process_market_data(lot_bar)
    record = lot_engine.get_order_history()[0]
    assert record.status == OrderStatus.REJECTED
    assert record.reject_reason == RejectReason.INVALID_LOT_SIZE

    cash_engine = BacktestEngine(500.0, "close")
    cash_engine.set_on_market_data(
        lambda snapshot: [Order(symbol=snapshot.symbol, quantity=100)]
    )
    cash_engine.process_market_data(_snap("X", 1, 10.0))
    assert cash_engine.get_order_history()[0].reject_reason == RejectReason.INSUFFICIENT_CASH
    assert cash_engine.get_cash() == 500.0


def test_partial_order_expiration_and_cancellation():
    engine = BacktestEngine(10_000.0, "close")
    engine.set_on_market_data(
        lambda snapshot: [Order(symbol=snapshot.symbol, quantity=15)]
    )
    snapshot = _snap("X", 1, 10.0)
    snapshot.volume = 100
    engine.process_market_data(snapshot)
    assert engine.get_order_history()[0].status == OrderStatus.PARTIALLY_FILLED
    engine.finalize(2)
    assert engine.get_order_history()[0].status == OrderStatus.EXPIRED

    cancel_engine = BacktestEngine(10_000.0, "close")
    cancel_engine.set_on_market_data(lambda snapshot: [Order(
        symbol=snapshot.symbol, type=OrderType.LIMIT, quantity=100, limit_price=9.0,
    )])
    cancel_engine.process_market_data(_snap("X", 1, 10.0))
    assert cancel_engine.cancel_order(1, 2)
    assert cancel_engine.get_order_history()[0].status == OrderStatus.CANCELED


def test_corporate_action_and_portfolio_risk():
    engine = BacktestEngine(10_000.0, "close")
    ordered = False

    def on_md(snapshot):
        nonlocal ordered
        if ordered:
            return []
        ordered = True
        return [Order(symbol=snapshot.symbol, quantity=100)]

    engine.set_on_market_data(on_md)
    snapshot = _snap("X", 1, 10.0)
    snapshot.industry = "Bank"
    snapshot.factor_exposures = {"size": 0.8}
    engine.process_market_data(snapshot)
    result = engine.apply_corporate_action(CorporateAction(
        symbol="X", timestamp=2, cash_dividend_per_share=0.2,
        share_multiplier=1.5,
    ))
    assert result.cash_dividend == pytest.approx(20.0)
    assert engine.get_position("X").quantity == 150
    assert engine.get_position("X").avg_cost == pytest.approx(10.0 / 1.5)
    portfolio = engine.get_portfolio_snapshot()
    assert portfolio.position_count == 1
    assert portfolio.industry_exposure["Bank"] > 0.0
    assert portfolio.factor_exposure["size"] > 0.0


def test_corporate_action_tolerates_exact_integer_float_roundoff():
    engine = BacktestEngine(10_000.0, "close")
    engine.set_on_market_data(
        lambda snapshot: [Order(symbol=snapshot.symbol, quantity=100)]
    )
    engine.process_market_data(_snap("X", 1, 10.0))
    result = engine.apply_corporate_action(CorporateAction(
        symbol="X", timestamp=2, share_multiplier=1.1,
    ))
    assert result.new_quantity == 110


def test_query_results_cannot_mutate_engine_state():
    engine = BacktestEngine(10_000.0, "close")
    engine.set_on_market_data(
        lambda snapshot: [Order(symbol=snapshot.symbol, quantity=100)]
    )
    engine.process_market_data(_snap("X", 1, 10.0))

    position = engine.get_position("X")
    position.quantity = 0
    trade = engine.get_trade_history()[0]
    trade.quantity = 0
    order = engine.get_order_history()[0]
    order.filled_quantity = 0
    order.order.quantity = 0

    assert engine.get_position("X").quantity == 100
    assert engine.get_trade_history()[0].quantity == 100
    assert engine.get_order_history()[0].filled_quantity == 100
    assert engine.get_order_history()[0].order.quantity == 100

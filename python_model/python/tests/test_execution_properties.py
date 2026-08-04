from __future__ import annotations

import random

import pytest

from python import engine_api as api


def _bar(timestamp: int, price: float, volume: int) -> object:
    snapshot = api.MarketSnapshot()
    snapshot.symbol = "FUZZ"
    snapshot.timestamp = timestamp
    snapshot.open = price
    snapshot.high = price
    snapshot.low = price
    snapshot.close = price
    snapshot.volume = volume
    return snapshot


@pytest.mark.parametrize("seed", range(24))
def test_seeded_partial_fill_accounting_invariants(seed: int) -> None:
    rng = random.Random(0x51A7E + seed)
    initial_cash = 1_000_000.0
    quantity = rng.randint(20, 80)
    volume = rng.randint(20, 50)
    participation = rng.choice((0.10, 0.20, 0.25, 0.50))
    slippage_bps = rng.uniform(0.0, 50.0)
    commission_rate = rng.uniform(0.0, 0.002)

    config = api.ExecutionConfig()
    config.max_volume_participation = participation
    config.slippage_bps = slippage_bps
    config.enforce_t_plus_one = False
    config.enforce_board_lot = False
    engine = api.BacktestEngine(initial_cash)
    engine.set_execution_config(config)
    engine.set_commission_fn(
        lambda notional, _is_sell: notional * commission_rate
    )

    buy_sent = False
    sell_sent = False

    def on_market_data(snapshot):
        nonlocal buy_sent, sell_sent
        if not buy_sent:
            buy_sent = True
            side = api.Side.BUY
        elif (
            not sell_sent
            and engine.get_position(snapshot.symbol).quantity == quantity
        ):
            sell_sent = True
            side = api.Side.SELL
        else:
            return []
        order = api.Order()
        order.symbol = snapshot.symbol
        order.side = side
        order.type = api.OrderType.MARKET
        order.quantity = quantity
        return [order]

    engine.set_on_market_data(on_market_data)
    price = rng.uniform(5.0, 50.0)
    for timestamp in range(1, 130):
        price = max(1.0, price * (1.0 + rng.uniform(-0.015, 0.015)))
        engine.process_market_data(_bar(timestamp, price, volume))
        if sell_sent and engine.get_position("FUZZ").quantity == 0:
            break
    engine.finalize(timestamp)

    assert sell_sent, "generated path did not reach the closing order"
    assert engine.get_position("FUZZ").quantity == 0
    trades = engine.get_trade_history()
    assert sum(
        trade.quantity if trade.side == api.Side.BUY else -trade.quantity
        for trade in trades
    ) == 0

    orders = engine.get_order_history()
    assert len(orders) == 2
    assert all(
        0 <= record.filled_quantity <= record.order.quantity for record in orders
    )
    assert all(record.filled_quantity == quantity for record in orders)
    assert all(record.status == api.OrderStatus.FILLED for record in orders)

    cash_from_ledger = initial_cash
    for trade in trades:
        direction = -1.0 if trade.side == api.Side.BUY else 1.0
        cash_from_ledger += direction * trade.price * trade.quantity
        cash_from_ledger -= trade.commission
    assert engine.get_cash() == pytest.approx(cash_from_ledger)
    assert engine.get_equity() == pytest.approx(engine.get_cash())

    round_trips = engine.get_round_trip_history()
    assert sum(item.quantity for item in round_trips) == quantity
    assert sum(item.commission for item in round_trips) == pytest.approx(
        sum(trade.commission for trade in trades)
    )
    assert sum(item.net_pnl for item in round_trips) == pytest.approx(
        engine.get_cash() - initial_cash,
        abs=max(1e-4 * len(trades), 1e-6),
    )
    assert all(
        item.net_pnl == pytest.approx(item.gross_pnl - item.commission)
        for item in round_trips
    )

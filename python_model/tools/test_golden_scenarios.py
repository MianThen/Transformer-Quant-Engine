from __future__ import annotations

import json
from pathlib import Path

import pytest

from python import engine_api as api


FIXTURE = Path(__file__).with_name("fixtures") / "golden_round_trip.json"


def _bar(source: dict) -> object:
    snapshot = api.MarketSnapshot()
    snapshot.symbol = "GOLDEN"
    snapshot.timestamp = source["timestamp"]
    snapshot.open = source["price"]
    snapshot.high = source["price"]
    snapshot.low = source["price"]
    snapshot.close = source["price"]
    snapshot.volume = source["volume"]
    return snapshot


def test_golden_partial_fill_round_trip() -> None:
    scenario = json.loads(FIXTURE.read_text(encoding="utf-8"))
    execution = scenario["execution"]
    config = api.ExecutionConfig()
    config.max_volume_participation = execution["max_volume_participation"]
    config.slippage_bps = execution["slippage_bps"]
    config.enforce_t_plus_one = False
    config.enforce_board_lot = False

    engine = api.BacktestEngine(scenario["initial_cash"])
    engine.set_execution_config(config)
    engine.set_commission_fn(
        lambda notional, _is_sell: notional * execution["commission_rate"]
    )
    orders_by_timestamp = {
        item["timestamp"]: item for item in scenario["orders"]
    }

    def on_market_data(snapshot):
        source = orders_by_timestamp.get(snapshot.timestamp)
        if source is None:
            return []
        order = api.Order()
        order.symbol = snapshot.symbol
        order.side = getattr(api.Side, source["side"])
        order.type = api.OrderType.MARKET
        order.quantity = source["quantity"]
        return [order]

    engine.set_on_market_data(on_market_data)
    for source in scenario["bars"]:
        engine.process_market_data(_bar(source))
    engine.finalize(scenario["bars"][-1]["timestamp"])

    expected = scenario["expected"]
    trades = engine.get_trade_history()
    assert [item.quantity for item in trades] == expected["trade_quantities"]
    assert [item.price for item in trades] == pytest.approx(
        expected["trade_prices"]
    )
    assert engine.get_cash() == pytest.approx(expected["cash"])
    assert engine.get_equity() == pytest.approx(expected["equity"])
    assert engine.get_total_return() == pytest.approx(expected["total_return"])
    assert engine.get_max_drawdown() == pytest.approx(
        expected["max_drawdown"], abs=1e-8
    )
    assert engine.get_win_rate() == pytest.approx(expected["win_rate"])
    assert engine.get_position("GOLDEN").quantity == 0

    records = engine.get_order_history()
    assert len(records) == 2
    assert all(record.status == api.OrderStatus.FILLED for record in records)
    assert all(
        record.filled_quantity == record.order.quantity for record in records
    )

    round_trips = engine.get_round_trip_history()
    assert len(round_trips) == len(expected["round_trips"])
    for actual, wanted in zip(round_trips, expected["round_trips"]):
        assert actual.quantity == wanted["quantity"]
        assert actual.entry_price == pytest.approx(wanted["entry_price"])
        assert actual.exit_price == pytest.approx(wanted["exit_price"])
        assert actual.gross_pnl == pytest.approx(wanted["gross_pnl"])
        assert actual.commission == pytest.approx(
            wanted["commission"], abs=1e-4
        )
        assert actual.net_pnl == pytest.approx(wanted["net_pnl"], abs=1e-4)

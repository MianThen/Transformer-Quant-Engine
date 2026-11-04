from __future__ import annotations

import pytest

cpp = pytest.importorskip("cpp_engine")

from python import core


def test_cpp_and_python_backends_match_on_cross_section_and_costs():
    cpp_engine = cpp.BacktestEngine(1_000.0, cpp.FillTiming.CLOSE)
    python_engine = core.BacktestEngine(1_000.0, "close")
    cpp_engine.set_commission_fn(lambda notional, is_sell: notional * 0.001)
    python_engine.set_commission_fn(lambda notional, is_sell: notional * 0.001)

    cpp_engine.set_on_cross_section(lambda batch: _cpp_orders(batch) if batch[0].timestamp == 1 else [])
    python_engine.set_on_cross_section(
        lambda batch: _python_orders(batch) if batch[0].timestamp == 1 else []
    )
    cpp_engine.process_market_data_batch([
        _cpp_bar("BBB", 1, 20.0), _cpp_bar("AAA", 1, 10.0)
    ])
    python_engine.process_market_data_batch([
        _python_bar("BBB", 1, 20.0), _python_bar("AAA", 1, 10.0)
    ])
    cpp_engine.process_market_data_batch([
        _cpp_bar("AAA", 2, 11.0), _cpp_bar("BBB", 2, 19.0)
    ])
    python_engine.process_market_data_batch([
        _python_bar("AAA", 2, 11.0), _python_bar("BBB", 2, 19.0)
    ])

    assert cpp_engine.get_cash() == pytest.approx(python_engine.get_cash())
    assert cpp_engine.get_equity() == pytest.approx(python_engine.get_equity())
    assert cpp_engine.get_total_return() == pytest.approx(python_engine.get_total_return())
    assert len(cpp_engine.get_trade_history()) == len(python_engine.get_trade_history()) == 2
    assert len(cpp_engine.get_equity_curve()) == len(python_engine.get_equity_curve()) == 2


def test_cpp_and_python_match_on_slippage_and_partial_fills():
    cpp_config = cpp.ExecutionConfig()
    cpp_config.max_volume_participation = 0.10
    cpp_config.slippage_bps = 10.0
    python_config = core.ExecutionConfig(
        max_volume_participation=0.10, slippage_bps=10.0
    )
    cpp_engine = cpp.BacktestEngine(10_000.0, cpp.FillTiming.NEXT_OPEN, cpp_config)
    python_engine = core.BacktestEngine(10_000.0, "next_open", python_config)

    cpp_engine.set_on_market_data(_single_buy_callback(cpp))
    python_engine.set_on_market_data(_single_buy_callback(core))
    for timestamp, price in ((1, 10.0), (2, 11.0), (3, 12.0)):
        cpp_engine.process_market_data(_cpp_bar("X", timestamp, price, volume=100))
        python_engine.process_market_data(_python_bar("X", timestamp, price, volume=100))

    cpp_trades = cpp_engine.get_trade_history()
    python_trades = python_engine.get_trade_history()
    assert [trade.quantity for trade in cpp_trades] == [10, 5]
    assert [trade.quantity for trade in python_trades] == [10, 5]
    assert [trade.price for trade in cpp_trades] == pytest.approx(
        [trade.price for trade in python_trades]
    )
    assert cpp_engine.get_cash() == pytest.approx(python_engine.get_cash())
    assert cpp_engine.get_position("X").quantity == python_engine.get_position("X").quantity


@pytest.mark.parametrize("state", ["suspended", "limit_up", "zero_volume"])
def test_cpp_and_python_match_on_untradeable_bars(state):
    cpp_engine = cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE)
    python_engine = core.BacktestEngine(10_000.0, "close")
    cpp_engine.set_on_market_data(lambda snapshot: [_order(cpp, snapshot, cpp.Side.BUY, 10)])
    python_engine.set_on_market_data(
        lambda snapshot: [_order(core, snapshot, core.Side.BUY, 10)]
    )
    kwargs = {
        "is_suspended": state == "suspended",
        "upper_limit": 10.0 if state == "limit_up" else 0.0,
        "volume": 0 if state == "zero_volume" else 1_000,
    }
    cpp_engine.process_market_data(_cpp_bar("X", 1, 10.0, **kwargs))
    python_engine.process_market_data(_python_bar("X", 1, 10.0, **kwargs))
    assert cpp_engine.get_trade_history() == []
    assert python_engine.get_trade_history() == []


def test_cpp_and_python_match_on_t_plus_one():
    day1 = 1_752_714_000_000_000_000
    same_day = day1 + 3_600_000_000_000
    day2 = day1 + 86_400_000_000_000
    cpp_engine = cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE)
    python_engine = core.BacktestEngine(10_000.0, "close")
    cpp_engine.set_on_market_data(_buy_then_sell_callback(cpp, day1))
    python_engine.set_on_market_data(_buy_then_sell_callback(core, day1))
    for timestamp, price in ((day1, 10.0), (same_day, 10.0), (day2, 11.0)):
        cpp_engine.process_market_data(_cpp_bar("X", timestamp, price, volume=1_000))
        python_engine.process_market_data(_python_bar("X", timestamp, price, volume=1_000))

    assert len(cpp_engine.get_trade_history()) == len(python_engine.get_trade_history()) == 2
    assert cpp_engine.get_cash() == pytest.approx(python_engine.get_cash())
    assert cpp_engine.get_position("X").quantity == python_engine.get_position("X").quantity == 0
    assert (
        cpp_engine.get_position("X").sellable_quantity
        == python_engine.get_position("X").sellable_quantity
        == 0
    )


def test_t_zero_does_not_implicitly_enable_short_selling():
    cpp_config = cpp.ExecutionConfig()
    cpp_config.enforce_t_plus_one = False
    python_config = core.ExecutionConfig(enforce_t_plus_one=False)
    for module, engine, make_bar in (
        (cpp, cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE, cpp_config), _cpp_bar),
        (core, core.BacktestEngine(10_000.0, "close", python_config), _python_bar),
    ):
        engine.set_on_market_data(
            lambda snapshot, module=module: [
                _order(module, snapshot, module.Side.SELL, 10)
            ]
        )
        engine.process_market_data(make_bar("X", 1, 10.0, volume=1_000))
        assert engine.get_trade_history() == []
        assert (
            engine.get_order_history()[0].reject_reason
            == module.RejectReason.INSUFFICIENT_POSITION
        )


def test_explicit_short_selling_and_native_fee_schedules_match():
    cpp_config = cpp.ExecutionConfig()
    cpp_config.enforce_t_plus_one = False
    cpp_config.allow_short = True
    python_config = core.ExecutionConfig(enforce_t_plus_one=False, allow_short=True)
    engines = (
        (cpp, cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE, cpp_config), _cpp_bar),
        (core, core.BacktestEngine(10_000.0, "close", python_config), _python_bar),
    )
    for module, engine, make_bar in engines:
        engine.set_fee_schedules([
            module.FeeSchedule(0, 2, 0.001, 0.0, 0.001, 0.0),
            module.FeeSchedule(2, None, 0.002, 0.0, 0.0005, 0.0),
        ])
        engine.set_on_market_data(
            lambda snapshot, module=module: [
                _order(module, snapshot, module.Side.SELL, 10)
            ]
        )
        engine.process_market_data(make_bar("X", 1, 10.0, volume=1_000))
        engine.process_market_data(make_bar("X", 2, 10.0, volume=1_000))

    cpp_trades = engines[0][1].get_trade_history()
    python_trades = engines[1][1].get_trade_history()
    assert [trade.commission for trade in cpp_trades] == pytest.approx([0.2, 0.25])
    assert [trade.commission for trade in cpp_trades] == pytest.approx(
        [trade.commission for trade in python_trades]
    )
    assert engines[0][1].get_cash() == pytest.approx(engines[1][1].get_cash())


def test_cpp_and_python_match_on_order_audit_cash_and_corporate_actions():
    engines = (
        (cpp, cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE)),
        (core, core.BacktestEngine(10_000.0, "close")),
    )
    for module, engine in engines:
        sent = False

        def callback(snapshot):
            nonlocal sent
            if sent:
                return []
            sent = True
            return [_order(module, snapshot, module.Side.BUY, 100)]

        engine.set_on_market_data(callback)
        snapshot = (
            _cpp_bar("X", 1, 10.0, volume=1_000)
            if module is cpp else _python_bar("X", 1, 10.0, volume=1_000)
        )
        snapshot.lot_size = snapshot.min_buy_quantity = 100
        snapshot.industry = "Bank"
        snapshot.factor_exposures = {"size": 0.8}
        engine.process_market_data(snapshot)
        assert engine.get_order_history()[0].status == module.OrderStatus.FILLED

        action = module.CorporateAction()
        action.symbol = "X"
        action.timestamp = 2
        action.cash_dividend_per_share = 0.2
        action.share_multiplier = 1.5
        result = engine.apply_corporate_action(action)
        assert result.cash_dividend == pytest.approx(20.0)
        assert engine.get_position("X").quantity == 150
        assert engine.get_portfolio_snapshot().industry_exposure["Bank"] > 0.0
        assert engine.get_portfolio_snapshot().factor_exposure["size"] > 0.0

    assert engines[0][1].get_cash() == pytest.approx(engines[1][1].get_cash())
    assert engines[0][1].get_portfolio_snapshot().gross_exposure == pytest.approx(
        engines[1][1].get_portfolio_snapshot().gross_exposure
    )


def test_cpp_and_python_match_on_rejection_and_expiration():
    for module, engine in (
        (cpp, cpp.BacktestEngine(500.0, cpp.FillTiming.CLOSE)),
        (core, core.BacktestEngine(500.0, "close")),
    ):
        engine.set_on_market_data(
            lambda snapshot, module=module: [
                _order(module, snapshot, module.Side.BUY, 100)
            ]
        )
        snapshot = (
            _cpp_bar("X", 1, 10.0, volume=1_000)
            if module is cpp else _python_bar("X", 1, 10.0, volume=1_000)
        )
        engine.process_market_data(snapshot)
        record = engine.get_order_history()[0]
        assert record.status == module.OrderStatus.REJECTED
        assert record.reject_reason == module.RejectReason.INSUFFICIENT_CASH

    for module, engine in (
        (cpp, cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE)),
        (core, core.BacktestEngine(10_000.0, "close")),
    ):
        engine.set_on_market_data(
            lambda snapshot, module=module: [
                _order(module, snapshot, module.Side.BUY, 15)
            ]
        )
        snapshot = (
            _cpp_bar("X", 1, 10.0, volume=100)
            if module is cpp else _python_bar("X", 1, 10.0, volume=100)
        )
        engine.process_market_data(snapshot)
        assert engine.get_order_history()[0].status == module.OrderStatus.PARTIALLY_FILLED
        engine.finalize(2)
        assert engine.get_order_history()[0].status == module.OrderStatus.EXPIRED


def test_next_open_delays_limit_orders_until_the_following_bar():
    for module, engine, make_bar in (
        (cpp, cpp.BacktestEngine(10_000.0, cpp.FillTiming.NEXT_OPEN), _cpp_bar),
        (core, core.BacktestEngine(10_000.0, "next_open"), _python_bar),
    ):
        sent = False

        def callback(snapshot):
            nonlocal sent
            if sent:
                return []
            sent = True
            order = _order(module, snapshot, module.Side.BUY, 10)
            order.type = module.OrderType.LIMIT
            order.limit_price = 10.0
            return [order]

        engine.set_on_market_data(callback)
        engine.process_market_data(make_bar("X", 1, 10.0, volume=1_000))
        assert engine.get_trade_history() == []
        engine.process_market_data(make_bar("X", 2, 9.0, volume=1_000))
        trades = engine.get_trade_history()
        assert len(trades) == 1
        assert trades[0].timestamp == 2
        assert trades[0].price == pytest.approx(9.0)


def test_next_open_partial_limit_remainder_is_not_duplicated():
    configs = []
    cpp_config = cpp.ExecutionConfig()
    cpp_config.max_volume_participation = 0.10
    configs.append(
        (cpp, cpp.BacktestEngine(10_000.0, cpp.FillTiming.NEXT_OPEN, cpp_config), _cpp_bar)
    )
    configs.append(
        (
            core,
            core.BacktestEngine(
                10_000.0,
                "next_open",
                core.ExecutionConfig(max_volume_participation=0.10),
            ),
            _python_bar,
        )
    )
    for module, engine, make_bar in configs:
        sent = False

        def callback(snapshot):
            nonlocal sent
            if sent:
                return []
            sent = True
            order = _order(module, snapshot, module.Side.BUY, 15)
            order.type = module.OrderType.LIMIT
            order.limit_price = 10.0
            return [order]

        engine.set_on_market_data(callback)
        for timestamp in (1, 2, 3):
            engine.process_market_data(
                make_bar("X", timestamp, 10.0, volume=100)
            )
        assert [trade.quantity for trade in engine.get_trade_history()] == [10, 5]
        record = engine.get_order_history()[0]
        assert record.status == module.OrderStatus.FILLED
        assert record.filled_quantity == 15


def test_cross_section_orders_cannot_reuse_stale_market_data():
    for module, engine, make_bar in (
        (cpp, cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE), _cpp_bar),
        (core, core.BacktestEngine(10_000.0, "close"), _python_bar),
    ):
        def callback(batch):
            if batch[0].timestamp != 2:
                return []
            order = _order(module, batch[0], module.Side.BUY, 10)
            order.symbol = "STALE"
            return [order]

        engine.set_on_cross_section(callback)
        engine.process_market_data_batch([
            make_bar("LIVE", 1, 10.0, volume=1_000),
            make_bar("STALE", 1, 10.0, volume=1_000),
        ])
        engine.process_market_data_batch([
            make_bar("LIVE", 2, 10.0, volume=1_000),
        ])
        record = engine.get_order_history()[0]
        assert record.status == module.OrderStatus.REJECTED
        assert record.reject_reason == module.RejectReason.STALE_MARKET_DATA
        assert engine.get_trade_history() == []


def test_strategy_orders_only_match_bar_liquidity_not_each_other():
    cpp_config = cpp.ExecutionConfig()
    cpp_config.enforce_t_plus_one = False
    cpp_config.allow_short = True
    python_config = core.ExecutionConfig(enforce_t_plus_one=False, allow_short=True)
    engines = (
        (cpp, cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE, cpp_config), _cpp_bar),
        (core, core.BacktestEngine(10_000.0, "close", python_config), _python_bar),
    )
    for module, engine, make_bar in engines:
        def callback(snapshot):
            buy = _order(module, snapshot, module.Side.BUY, 10)
            buy.type = module.OrderType.LIMIT
            buy.limit_price = 9.0
            sell = _order(module, snapshot, module.Side.SELL, 10)
            sell.type = module.OrderType.LIMIT
            sell.limit_price = 8.0
            return [buy, sell]

        engine.set_on_market_data(callback)
        engine.process_market_data(make_bar("X", 1, 10.0, volume=1_000))
        trades = engine.get_trade_history()
        assert len(trades) == 1
        assert trades[0].side == module.Side.SELL
        assert trades[0].price == pytest.approx(10.0)
        records = engine.get_order_history()
        assert records[0].status == module.OrderStatus.ACCEPTED
        assert records[0].filled_quantity == 0
        assert records[1].status == module.OrderStatus.FILLED


def test_pending_orders_do_not_fill_on_unlisted_bars():
    for module, engine, make_bar in (
        (cpp, cpp.BacktestEngine(10_000.0, cpp.FillTiming.NEXT_OPEN), _cpp_bar),
        (core, core.BacktestEngine(10_000.0, "next_open"), _python_bar),
    ):
        engine.set_on_market_data(_single_buy_callback(module))
        engine.process_market_data(make_bar("X", 1, 10.0, volume=1_000))
        engine.process_market_data(
            make_bar("X", 2, 10.0, volume=1_000, is_listed=False)
        )
        assert engine.get_trade_history() == []
        record = engine.get_order_history()[0]
        assert record.status == module.OrderStatus.CANCELED
        assert record.reject_reason == module.RejectReason.NOT_LISTED


def test_callback_failure_poisoned_engine_cannot_be_reused():
    for module, engine, make_bar in (
        (cpp, cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE), _cpp_bar),
        (core, core.BacktestEngine(10_000.0, "close"), _python_bar),
    ):
        engine.set_on_market_data(
            lambda snapshot, module=module: [
                _order(module, snapshot, module.Side.BUY, 10)
            ]
        )

        def fail_on_accept(_record):
            raise RuntimeError("order callback failed")

        engine.set_on_order_update(fail_on_accept)
        with pytest.raises(RuntimeError, match="order callback failed"):
            engine.process_market_data(make_bar("X", 1, 10.0, volume=1_000))
        with pytest.raises(RuntimeError, match="poison"):
            engine.process_market_data(make_bar("X", 1, 10.0, volume=1_000))


def test_corporate_action_keeps_current_equity_queries_consistent():
    for module, engine, make_bar in (
        (cpp, cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE), _cpp_bar),
        (core, core.BacktestEngine(10_000.0, "close"), _python_bar),
    ):
        sent = False

        def callback(snapshot):
            nonlocal sent
            if sent:
                return []
            sent = True
            return [_order(module, snapshot, module.Side.BUY, 10)]

        engine.set_on_market_data(callback)
        engine.process_market_data(make_bar("X", 1, 10.0, volume=1_000))
        action = module.CorporateAction()
        action.symbol = "X"
        action.timestamp = 2
        action.cash_dividend_per_share = 0.2
        action.share_multiplier = 1.5
        engine.apply_corporate_action(action)
        assert engine.get_equity() == pytest.approx(
            engine.get_portfolio_snapshot().equity
        )


def test_slippage_cannot_produce_non_positive_sell_prices():
    cpp_config = cpp.ExecutionConfig()
    cpp_config.slippage_bps = 10_000.0
    with pytest.raises(ValueError, match="execution config"):
        cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE, cpp_config)
    with pytest.raises(ValueError, match="slippage_bps"):
        core.BacktestEngine(
            10_000.0, "close", core.ExecutionConfig(slippage_bps=10_000.0)
        )


def test_resting_partial_fills_cannot_overdraw_cash_with_minimum_fees():
    for module, engine, make_bar in (
        (cpp, cpp.BacktestEngine(20.0, cpp.FillTiming.CLOSE), _cpp_bar),
        (core, core.BacktestEngine(20.0, "close"), _python_bar),
    ):
        sent = False

        def callback(snapshot):
            nonlocal sent
            if sent:
                return []
            sent = True
            order = _order(module, snapshot, module.Side.BUY, 10)
            order.type = module.OrderType.LIMIT
            order.limit_price = 1.0
            return [order]

        engine.set_on_market_data(callback)
        engine.set_commission_fn(lambda _notional, _is_sell: 5.0)
        engine.process_market_data(make_bar("X", 1, 2.0, volume=10))
        for timestamp in range(2, 8):
            engine.process_market_data(
                make_bar("X", timestamp, 1.0, volume=10)
            )
        record = engine.get_order_history()[0]
        assert engine.get_cash() == pytest.approx(2.0)
        assert engine.get_cash() >= 0.0
        assert len(engine.get_trade_history()) == 3
        assert record.status == module.OrderStatus.CANCELED
        assert record.reject_reason == module.RejectReason.INSUFFICIENT_CASH


def test_round_trip_win_rate_includes_entry_and_exit_commissions():
    cpp_config = cpp.ExecutionConfig()
    cpp_config.enforce_t_plus_one = False
    python_config = core.ExecutionConfig(enforce_t_plus_one=False)
    for module, engine, make_bar in (
        (cpp, cpp.BacktestEngine(10_000.0, cpp.FillTiming.CLOSE, cpp_config), _cpp_bar),
        (core, core.BacktestEngine(10_000.0, "close", python_config), _python_bar),
    ):
        def callback(snapshot):
            side = (
                module.Side.BUY
                if snapshot.timestamp in (1, 3)
                else module.Side.SELL
            )
            return [_order(module, snapshot, side, 10)]

        engine.set_commission_fn(lambda _notional, _is_sell: 1.0)
        engine.set_on_market_data(callback)
        for timestamp, price in ((1, 10.0), (2, 10.15), (3, 10.0), (4, 10.3)):
            engine.process_market_data(
                make_bar("X", timestamp, price, volume=1_000)
            )

        round_trips = engine.get_round_trip_history()
        assert len(round_trips) == 2
        assert [item.net_pnl for item in round_trips] == pytest.approx([-0.5, 1.0])
        assert [item.commission for item in round_trips] == pytest.approx([2.0, 2.0])
        assert engine.get_win_rate() == pytest.approx(0.5)


def test_initial_equity_is_the_baseline_for_drawdown():
    for module, engine, make_bar in (
        (cpp, cpp.BacktestEngine(1_000.0, cpp.FillTiming.CLOSE), _cpp_bar),
        (core, core.BacktestEngine(1_000.0, "close"), _python_bar),
    ):
        engine.set_commission_fn(lambda _notional, _is_sell: 10.0)
        engine.set_on_market_data(
            lambda snapshot: [_order(module, snapshot, module.Side.BUY, 10)]
        )
        engine.process_market_data(make_bar("X", 1, 10.0, volume=1_000))
        assert engine.get_equity() == pytest.approx(990.0)
        assert engine.get_max_drawdown() == pytest.approx(0.01)


def _cpp_bar(symbol, timestamp, close, *, volume=100, upper_limit=0.0,
             lower_limit=0.0, is_suspended=False, is_listed=True):
    bar = cpp.MarketSnapshot()
    bar.symbol, bar.timestamp = symbol, timestamp
    bar.open = bar.high = bar.low = bar.close = close
    bar.volume = volume
    bar.upper_limit = upper_limit
    bar.lower_limit = lower_limit
    bar.is_suspended = is_suspended
    bar.is_listed = is_listed
    return bar


def _python_bar(symbol, timestamp, close, *, volume=100, upper_limit=0.0,
                lower_limit=0.0, is_suspended=False, is_listed=True):
    return core.MarketSnapshot(
        symbol, timestamp, close, close, close, close, volume,
        upper_limit, lower_limit, is_suspended, is_listed,
    )


def _order(module, snapshot, side, quantity):
    order = module.Order()
    order.symbol = snapshot.symbol
    order.side = side
    order.type = module.OrderType.MARKET
    order.quantity = quantity
    return order


def _single_buy_callback(module):
    ordered = False

    def callback(snapshot):
        nonlocal ordered
        if ordered:
            return []
        ordered = True
        return [_order(module, snapshot, module.Side.BUY, 15)]

    return callback


def _buy_then_sell_callback(module, buy_timestamp):
    def callback(snapshot):
        side = module.Side.BUY if snapshot.timestamp == buy_timestamp else module.Side.SELL
        return [_order(module, snapshot, side, 100)]

    return callback


def _cpp_orders(batch):
    orders = []
    for snapshot in batch:
        order = cpp.Order()
        order.symbol = snapshot.symbol
        order.side = cpp.Side.BUY
        order.type = cpp.OrderType.MARKET
        order.quantity = 10
        orders.append(order)
    return orders


def _python_orders(batch):
    return [
        core.Order(symbol=snapshot.symbol, side=core.Side.BUY,
                   type=core.OrderType.MARKET, quantity=10)
        for snapshot in batch
    ]

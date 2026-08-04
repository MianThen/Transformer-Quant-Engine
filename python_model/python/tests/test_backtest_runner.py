from __future__ import annotations

import random

import numpy as np
import pytest

from python.backtest_runner import BacktestRunner
from python.broker import Broker
from python.data_feed import DataFeed
from python.engine_api import ExecutionConfig, MarketSnapshot
from python.engine_api import Order, Side
from python.market_data import (
    AdjustmentFactor,
    AdjustmentFactorStore,
    ChinaAShareCalendar,
    CorporateAction,
    CorporateActionStore,
    MarketReferenceData,
    SecurityMaster,
    SecurityState,
)
from python.strategy import ColumnarStrategy, Strategy


class ListFeed(DataFeed):
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def stream(self):
        yield from self.snapshots


class CrossSectionStrategy(Strategy):
    def __init__(self):
        super().__init__(["AAA", "BBB"])
        self.observed = []

    def on_cross_section(self, snapshots):
        self.observed.append((snapshots[0].timestamp,
                              tuple(snapshot.symbol for snapshot in snapshots)))
        return []


def _bar(symbol, timestamp, close):
    snapshot = MarketSnapshot()
    snapshot.symbol = symbol
    snapshot.timestamp = timestamp
    snapshot.open = snapshot.high = snapshot.low = snapshot.close = close
    snapshot.volume = 1000
    return snapshot


def test_runner_streams_one_complete_timestamp_at_a_time():
    strategy = CrossSectionStrategy()
    feed = ListFeed([
        _bar("BBB", 1, 20.0), _bar("AAA", 1, 10.0),
        _bar("AAA", 2, 11.0), _bar("BBB", 2, 21.0),
    ])
    runner = BacktestRunner(strategy, feed, initial_cash=10_000.0)
    result = runner.run()

    assert strategy.observed == [
        (1, ("AAA", "BBB")),
        (2, ("AAA", "BBB")),
    ]
    assert len(result.equity_curve) == 2
    assert strategy.engine is None
    with pytest.raises(RuntimeError, match="只能运行一次"):
        runner.run()


def test_runner_forwards_order_lifecycle_to_strategy():
    class AuditedStrategy(Strategy):
        def __init__(self):
            super().__init__(["AAA"])
            self.sent = False
            self.statuses = []

        def on_market_data(self, snapshot):
            if self.sent:
                return []
            self.sent = True
            return [self.market_order("AAA", Side.BUY, 100, snapshot.timestamp)]

        def on_order_update(self, record):
            self.statuses.append(record.status.name)

    strategy = AuditedStrategy()
    BacktestRunner(
        strategy,
        ListFeed([_bar("AAA", 1, 10.0), _bar("AAA", 2, 11.0)]),
        initial_cash=10_000.0,
    ).run()

    assert strategy.statuses == ["ACCEPTED", "FILLED"]


def test_runner_supports_backend_neutral_columnar_strategy():
    class ColumnarBuyOnce(ColumnarStrategy):
        def __init__(self):
            super().__init__(["AAA", "BBB"])
            self.calls = []
            self.sent = False

        def on_cross_section_view(self, batch):
            self.calls.append((len(batch), batch.symbol(0), batch.close(1)))
            if self.sent:
                return None
            self.sent = True
            return self.target_position_columns(0, batch.symbol(0), 100)

    strategy = ColumnarBuyOnce()
    result = BacktestRunner(
        strategy,
        ListFeed([
            _bar("AAA", 1, 10.0), _bar("BBB", 1, 20.0),
            _bar("AAA", 2, 11.0), _bar("BBB", 2, 21.0),
        ]),
        initial_cash=10_000.0,
    ).run()

    assert strategy.calls == [(2, "AAA", 20.0), (2, "AAA", 21.0)]
    assert len(result.trades) == 1
    assert result.trades[0].symbol == "AAA"


def test_runner_uses_enriched_arrow_batch_path_when_available():
    pa = pytest.importorskip("pyarrow")
    reference = MarketReferenceData(
        security_master=SecurityMaster([
            SecurityState("AAA", 0, None, industry="Bank"),
            SecurityState("BBB", 0, None, industry="Tech"),
        ]),
        adjustment_factors=AdjustmentFactorStore([
            AdjustmentFactor("AAA", 0, 0.5),
            AdjustmentFactor("BBB", 0, 2.0),
        ]),
    )

    class BatchOnlyFeed(DataFeed):
        reference_data = reference

        def stream(self):
            raise AssertionError("批量快路径不应调用 stream()")

        def stream_batches(self):
            yield pa.record_batch({
                "timestamp": pa.array([1, 1, 2, 2], type=pa.int64()),
                "symbol": ["AAA", "BBB", "AAA", "BBB"],
                "open": [10.0, 20.0, 11.0, 21.0],
                "high": [10.0, 20.0, 11.0, 21.0],
                "low": [10.0, 20.0, 11.0, 21.0],
                "close": [10.0, 20.0, 11.0, 21.0],
                "volume": pa.array([100, 100, 100, 100], type=pa.int64()),
            })

        def lineage(self):
            return None

    strategy = CrossSectionStrategy()
    result = BacktestRunner(strategy, BatchOnlyFeed()).run()

    assert strategy.observed == [(1, ("AAA", "BBB")), (2, ("AAA", "BBB"))]
    assert len(result.equity_curve) == 2


def test_runner_enforces_calendar_reference_data_and_corporate_actions():
    first = 1_768_786_260_000_000_000  # 2026-01-19 09:31 Asia/Shanghai
    second = first + 60_000_000_000
    third = first + 86_400_000_000_000
    reference = MarketReferenceData(
        security_master=SecurityMaster([
            SecurityState("AAA", 0, None, industry="Bank", upper_limit=11.0,
                          lower_limit=9.0, lot_size=100, min_buy_quantity=100),
        ]),
        adjustment_factors=AdjustmentFactorStore([
            AdjustmentFactor("AAA", 0, 0.5),
        ]),
        corporate_actions=CorporateActionStore([
            CorporateAction("AAA", third, cash_dividend_per_share=0.2,
                            share_multiplier=1.5),
        ]),
    )
    calendar = ChinaAShareCalendar(["2026-01-19", "2026-01-20"])

    class BuyOnce(Strategy):
        def __init__(self):
            super().__init__(["AAA"])
            self.sent = False
            self.signal_prices = []

        def on_market_data(self, snapshot):
            self.signal_prices.append(snapshot.signal_ref_price())
            if self.sent:
                return []
            self.sent = True
            order = Order()
            order.symbol = snapshot.symbol
            order.side = Side.BUY
            order.quantity = 100
            return [order]

    strategy = BuyOnce()
    result = BacktestRunner(
        strategy,
        ListFeed([_bar("AAA", first, 10.0), _bar("AAA", second, 10.0),
                  _bar("AAA", third, 7.0)]),
        initial_cash=10_000.0,
        calendar=calendar,
        reference_data=reference,
    ).run()

    assert strategy.signal_prices == [5.0, 5.0, 3.5]
    assert len(result.trades) == 1
    assert result.orders[0].filled_quantity == 100
    assert result.corporate_actions[0].cash_dividend == 20.0
    assert result.portfolio.industry_exposure["Bank"] > 0.0


def test_runner_rejects_out_of_session_bar():
    calendar = ChinaAShareCalendar(["2026-01-19"])
    noon = 1_768_795_200_000_000_000  # 2026-01-19 12:00 Asia/Shanghai
    with pytest.raises(ValueError, match="非交易时间"):
        BacktestRunner(
            CrossSectionStrategy(), ListFeed([_bar("AAA", noon, 10.0)]),
            calendar=calendar,
        ).run()


def test_runner_validates_calendar_once_per_cross_section():
    class CountingCalendar:
        def __init__(self):
            self.timestamps = []

        def validate_timestamp(self, timestamp):
            self.timestamps.append(timestamp)

    calendar = CountingCalendar()
    BacktestRunner(
        CrossSectionStrategy(),
        ListFeed([
            _bar("AAA", 1, 10.0), _bar("BBB", 1, 20.0),
            _bar("AAA", 2, 11.0), _bar("BBB", 2, 21.0),
        ]),
        calendar=calendar,
    ).run()

    assert calendar.timestamps == [1, 2]


def test_runner_uses_feed_reference_once_and_rejects_conflicts():
    class CountingReference:
        def __init__(self):
            self.enriched = []

        def enrich(self, snapshot):
            self.enriched.append((snapshot.timestamp, snapshot.symbol))
            return snapshot

        def actions_at(self, _timestamp):
            return ()

        def actions_between(self, _start, _end):
            return ()

    class EnrichingFeed(ListFeed):
        def __init__(self, snapshots, reference_data):
            super().__init__(snapshots)
            self.reference_data = reference_data

        def stream(self):
            for snapshot in self.snapshots:
                self.reference_data.enrich(snapshot)
                yield snapshot

    reference = CountingReference()
    feed = EnrichingFeed([_bar("AAA", 1, 10.0), _bar("AAA", 2, 11.0)], reference)
    BacktestRunner(CrossSectionStrategy(), feed).run()

    assert reference.enriched == [(1, "AAA"), (2, "AAA")]
    with pytest.raises(ValueError, match="不同的 reference_data"):
        BacktestRunner(CrossSectionStrategy(), feed, reference_data=CountingReference())


def test_random_seed_is_reproducible_and_restores_process_state():
    class RandomStrategy(Strategy):
        def __init__(self):
            super().__init__(["AAA"])
            self.observed = []

        def on_cross_section(self, _snapshots):
            self.observed.append((random.random(), float(np.random.random())))
            return []

    random.seed(1234)
    np.random.seed(5678)
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    first = RandomStrategy()
    second = RandomStrategy()
    BacktestRunner(first, ListFeed([_bar("AAA", 1, 10.0)]), random_seed=7).run()
    assert random.getstate() == python_state
    restored_numpy_state = np.random.get_state()
    assert restored_numpy_state[0] == numpy_state[0]
    assert np.array_equal(restored_numpy_state[1], numpy_state[1])
    assert restored_numpy_state[2:] == numpy_state[2:]

    BacktestRunner(second, ListFeed([_bar("AAA", 1, 10.0)]), random_seed=7).run()
    assert second.observed == first.observed


def test_lineage_is_captured_before_market_data_is_consumed():
    class AuditedFeed(ListFeed):
        def __init__(self, snapshots):
            super().__init__(snapshots)
            self.calls = []

        def lineage(self):
            self.calls.append("lineage")
            return None

        def stream(self):
            self.calls.append("stream")
            assert self.calls == ["lineage", "stream"]
            yield from self.snapshots

    feed = AuditedFeed([_bar("AAA", 1, 10.0)])
    BacktestRunner(CrossSectionStrategy(), feed).run()
    assert feed.calls == ["lineage", "stream"]


def test_broker_slippage_is_applied_and_conflicts_are_rejected():
    class BuyOnce(Strategy):
        def __init__(self):
            super().__init__(["AAA"])
            self.sent = False

        def on_market_data(self, snapshot):
            if self.sent:
                return []
            self.sent = True
            order = Order()
            order.symbol = snapshot.symbol
            order.side = Side.BUY
            order.quantity = 1
            return [order]

    broker = Broker(
        commission_rate=0.0,
        min_commission=0.0,
        stamp_tax_rate=0.0,
        slippage=0.001,
    )
    result = BacktestRunner(
        BuyOnce(),
        ListFeed([_bar("AAA", 1, 10.0), _bar("AAA", 2, 10.0)]),
        broker=broker,
    ).run()
    assert result.trades[0].price == pytest.approx(10.01)
    assert result.run_spec.execution_config["slippage_bps"] == pytest.approx(10.0)

    config = ExecutionConfig()
    config.slippage_bps = 12.5
    with pytest.raises(ValueError, match="配置冲突"):
        BacktestRunner(
            BuyOnce(),
            ListFeed([_bar("AAA", 1, 10.0)]),
            broker=broker,
            execution_config=config,
        )

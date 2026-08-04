"""双均线交叉策略示例。

金叉(短均线上穿长均线)买入,死叉(短均线下穿长均线)卖出。
这是最经典的趋势跟踪策略,适合作为第一个跑通全流程的示例。
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from ..strategy import ColumnarStrategy

if TYPE_CHECKING:
    from ..engine_api import Fill, MarketSnapshot, Order


class MACrossStrategy(ColumnarStrategy):
    """双均线交叉策略。"""

    def __init__(self, symbol: str, short_window: int = 5, long_window: int = 20,
                 order_size: int = 100):
        super().__init__(symbols=[symbol])
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.order_size = order_size

        self.prices: deque[float] = deque(maxlen=long_window)
        self.prev_short_above = None  # 上一根短均线是否在长均线之上,用于识别穿越
        self.open_order_ids: set[int] = set()

    def on_cross_section_view(self, batch):
        price = batch.close(0)
        timestamp = batch.timestamp(0)
        self.prices.append(price)
        if len(self.prices) < self.long_window:
            return None

        prices = list(self.prices)
        short_ma = sum(prices[-self.short_window:]) / self.short_window
        long_ma = sum(prices) / self.long_window
        short_above = short_ma > long_ma

        # 首次凑齐窗口时只记录状态,不触发(避免开盘即下单)
        if self.prev_short_above is None:
            self.prev_short_above = short_above
            return None

        golden_cross = short_above and not self.prev_short_above  # 短上穿长
        death_cross = not short_above and self.prev_short_above   # 短下穿长

        self.prev_short_above = short_above
        if not golden_cross and not death_cross:
            return None
        self._cancel_open_orders(timestamp)
        target = self.order_size if golden_cross else 0
        return self.target_position_columns(0, batch.symbol(0), target)

    def on_order_update(self, record) -> None:
        if record.order.symbol != self.symbol:
            return
        if record.status.name in {"ACCEPTED", "PARTIALLY_FILLED"}:
            self.open_order_ids.add(record.order.id)
        else:
            self.open_order_ids.discard(record.order.id)

    def _cancel_open_orders(self, timestamp: int) -> None:
        for order_id in tuple(self.open_order_ids):
            self.cancel_order(order_id, timestamp)

    def on_fill(self, fill: "Fill") -> None:
        pass


def main():
    """跑通双均线示例。"""
    import os

    from ..backtest_runner import BacktestRunner
    from ..data_feed import CSVDataFeed
    from ..engine_api import BACKEND
    from storage.trade_store import TradeStore

    feed = CSVDataFeed("data/sample/sample_ohlcv.csv", symbol="000001")
    strategy = MACrossStrategy(symbol="000001", short_window=5, long_window=20,
                               order_size=20_000)
    store = TradeStore(os.getenv("QBT_DB_PATH", "backtest.db"))
    try:
        runner = BacktestRunner(strategy, feed, initial_cash=1_000_000.0,
                                store=store)
        result = runner.run()
    finally:
        store.close()

    print(f"引擎后端: {result.backend}")
    print(f"成交笔数: {len(result.trades)}")
    print(f"期末权益: {result.final_equity:,.2f}")
    print(f"总收益率: {result.total_return:.2%}")
    print(f"Sharpe:   {result.sharpe_ratio:.2f}")
    print(f"最大回撤: {result.max_drawdown:.2%}")
    print(f"已保存回测: run_id={result.run_id}")


if __name__ == "__main__":
    main()

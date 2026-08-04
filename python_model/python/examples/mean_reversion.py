"""均值回归策略示例。

价格偏离均值超过阈值时反向操作:跌破下轨买入,升破上轨卖出。
用布林带思路(均值 ± k 倍标准差)。第二个示例策略,展示策略基类可扩展性。
"""

from __future__ import annotations

import statistics
from collections import deque
from typing import TYPE_CHECKING

from ..strategy import ColumnarStrategy

if TYPE_CHECKING:
    from ..engine_api import Fill, MarketSnapshot, Order


class MeanReversionStrategy(ColumnarStrategy):
    """基于布林带的均值回归策略。"""

    def __init__(self, symbol: str, window: int = 20, num_std: float = 2.0,
                 order_size: int = 20_000):
        super().__init__(symbols=[symbol])
        self.symbol = symbol
        self.window = window
        self.num_std = num_std
        self.order_size = order_size

        self.prices: deque[float] = deque(maxlen=window)
        self.open_order_ids: set[int] = set()

    def on_cross_section_view(self, batch):
        price = batch.close(0)
        timestamp = batch.timestamp(0)
        self.prices.append(price)
        if len(self.prices) < self.window:
            return None

        prices = list(self.prices)
        mean = statistics.fmean(prices)
        std = statistics.pstdev(prices)
        upper = mean + self.num_std * std
        lower = mean - self.num_std * std

        if price < lower:
            target = self.order_size
        elif price > upper:
            target = 0
        else:
            return None
        self._cancel_open_orders(timestamp)
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
    """跑通均值回归示例。"""
    import os

    from ..backtest_runner import BacktestRunner
    from ..data_feed import CSVDataFeed
    from storage.trade_store import TradeStore

    feed = CSVDataFeed("data/sample/sample_ohlcv.csv", symbol="000001")
    strategy = MeanReversionStrategy(symbol="000001", window=20, num_std=2.0,
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

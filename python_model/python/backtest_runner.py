"""回测运行器:Python 策略层 ↔ 引擎的桥接。

职责:
  1. 从 DataFeed 拉行情,灌入引擎
  2. 把策略回调注册到引擎(引擎收到行情时回调策略)
  3. 运行引擎主循环
  4. 从引擎收集结果(落库由存储层单独负责)

引擎后端由 engine_api 决定:优先 C++(cpp_engine),否则纯 Python(core)。
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .broker import Broker
from .data_feed import DataFeed
from .engine_api import (
    BACKEND,
    BacktestEngine,
    CorporateAction,
    ExecutionConfig,
    FeeSchedule as EngineFeeSchedule,
)
from .events import BatchEventRunner, HistoricalReplaySource
from .run_spec import RunSpec, capture_run_spec
from .strategy import ColumnarStrategy, Strategy

if TYPE_CHECKING:
    from storage.trade_store import TradeStore


_EXECUTION_CONFIG_FIELDS = (
    "max_volume_participation",
    "slippage_bps",
    "enforce_price_limits",
    "enforce_t_plus_one",
    "allow_short",
    "enforce_board_lot",
    "enforce_cash",
    "market_order_price_buffer_bps",
)


@dataclass
class BacktestResult:
    """回测结果汇总。"""

    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    annual_return: float = 0.0
    win_rate: float = 0.0
    final_equity: float = 0.0
    initial_cash: float = 0.0
    backend: str = ""
    run_id: int | None = None
    trades: list = field(default_factory=list)         # list[TradeRecord]
    round_trips: list = field(default_factory=list)    # list[RoundTripRecord]
    equity_curve: list = field(default_factory=list)   # list[EquityPoint]
    orders: list = field(default_factory=list)         # list[OrderRecord]
    corporate_actions: list = field(default_factory=list)
    portfolio: object | None = None
    data_lineage: object | None = None
    run_spec: RunSpec | None = None


class BacktestRunner:
    """驱动一次完整回测。"""

    def __init__(
        self,
        strategy: Strategy,
        data_feed: DataFeed,
        initial_cash: float = 1_000_000.0,
        broker: Broker | None = None,
        store: "TradeStore | None" = None,
        execution_config: ExecutionConfig | None = None,
        calendar=None,
        reference_data=None,
        strategy_parameters: dict[str, object] | None = None,
        random_seed: int | None = None,
    ):
        self.strategy = strategy
        self.data_feed = data_feed
        self.initial_cash = initial_cash
        self.broker = broker or Broker()
        self.store = store
        self.calendar = calendar
        feed_reference = getattr(data_feed, "reference_data", None)
        if (
            reference_data is not None
            and feed_reference is not None
            and reference_data is not feed_reference
        ):
            raise ValueError("DataFeed 与 BacktestRunner 不能配置不同的 reference_data")
        self.reference_data = reference_data or feed_reference
        self._feed_enriches_reference = feed_reference is not None
        self.strategy_parameters = dict(strategy_parameters or {})
        if random_seed is not None and (
            isinstance(random_seed, bool) or not isinstance(random_seed, int)
        ):
            raise ValueError("random_seed 必须是整数或 None")
        self.random_seed = random_seed
        self._last_action_timestamp: int | None = None
        self._data_lineage = None
        self._run_spec: RunSpec | None = None
        self._has_run = False

        self.execution_config = _resolved_execution_config(
            execution_config, self.broker
        )
        self.engine = BacktestEngine(initial_cash)
        self.engine.set_execution_config(self.execution_config)

    def run(self) -> BacktestResult:
        """主流程:注册回调 → 灌数据 → 收集结果。"""
        if self._has_run:
            raise RuntimeError("BacktestRunner 实例只能运行一次")
        self._has_run = True
        run_id = None
        try:
            self._prepare_run()
            if self.store is not None:
                run_id = self.store.begin_run(
                    type(self.strategy).__name__,
                    self.strategy.symbols,
                    self.initial_cash,
                    BACKEND,
                    self._run_spec,
                )
            self.strategy.engine = self.engine
            if (
                isinstance(self.strategy, ColumnarStrategy)
                and hasattr(self.engine, "set_on_cross_section_view")
            ):
                self.engine.set_on_cross_section_view(
                    self.strategy.on_cross_section_view
                )
            elif hasattr(self.engine, "set_on_cross_section"):
                self.engine.set_on_cross_section(self.strategy.on_cross_section)
            else:
                self.engine.set_on_market_data(self.strategy.on_market_data)
            self.engine.set_on_fill(self.strategy.on_fill)
            if hasattr(self.engine, "set_on_order_update"):
                self.engine.set_on_order_update(self.strategy.on_order_update)

            if (
                type(self.broker) is Broker
                and hasattr(self.engine, "set_fee_schedules")
            ):
                self.engine.set_fee_schedules([
                    EngineFeeSchedule(
                        effective_from=item.effective_from,
                        effective_to=item.effective_to,
                        commission_rate=item.commission_rate,
                        min_commission=item.min_commission,
                        stamp_tax_rate=item.stamp_tax_rate,
                        transfer_fee_rate=item.transfer_fee_rate,
                    )
                    for item in self.broker.fee_schedules
                ])
            elif hasattr(self.engine, "set_commission_fn"):
                self.engine.set_commission_fn(self.broker.commission)

            with _seeded_random_state(self.random_seed):
                current_timestamp = self._replay()
                self.engine.run()  # 兼容事件队列；流式主路径下队列为空。
                if hasattr(self.engine, "finalize"):
                    self.engine.finalize(current_timestamp or 0)
            result = self._collect_results()
            if self.store is not None:
                result.run_id = self.store.persist_result(
                    type(self.strategy).__name__,
                    self.strategy.symbols,
                    result,
                    run_id=run_id,
                )
            return result
        except BaseException as error:
            if self.store is not None and run_id is not None:
                try:
                    self.store.mark_run_failed(run_id, error)
                except Exception as audit_error:
                    raise RuntimeError(
                        f"回测失败，且 run_id={run_id} 的失败状态保存失败: "
                        f"{audit_error}"
                    ) from error
            raise
        finally:
            # 打破 strategy -> engine -> Python callback -> strategy 的跨语言引用环。
            self.strategy.engine = None

    def _prepare_run(self) -> None:
        prepare = getattr(self.data_feed, "prepare", None)
        prepared_lineage = prepare() if callable(prepare) else None
        self._data_lineage = prepared_lineage or (
            self.data_feed.lineage()
            if hasattr(self.data_feed, "lineage") else None
        )
        self._run_spec = capture_run_spec(
            backend=BACKEND,
            engine_type=BacktestEngine,
            strategy=self.strategy,
            strategy_parameters=self.strategy_parameters,
            initial_cash=self.initial_cash,
            fill_timing="NEXT_OPEN",
            execution_config=self.execution_config,
            broker=self.broker,
            random_seed=self.random_seed,
            calendar=self.calendar,
            reference_data=self.reference_data,
            data_lineage=self._data_lineage,
        )

    def _replay(self) -> int | None:
        if self._supports_batch_replay():
            source = HistoricalReplaySource(
                self.data_feed,
                calendar=self.calendar,
                reference_data=self.reference_data,
            )
            stats = BatchEventRunner(self.engine, finalize=False).run(source)
            return stats.last_timestamp or None

        # 只缓存一个 timestamp 的完整截面；不会把整段历史塞进事件队列。
        batch = []
        current_timestamp = None
        for snap in self.data_feed.stream():
            if self.reference_data is not None and not self._feed_enriches_reference:
                self.reference_data.enrich(snap)
            if current_timestamp is None:
                current_timestamp = snap.timestamp
            if snap.timestamp < current_timestamp:
                raise ValueError("DataFeed 必须按 timestamp 非递减输出")
            if snap.timestamp != current_timestamp:
                self._process_batch(batch)
                batch = []
                current_timestamp = snap.timestamp
            batch.append(snap)
        if batch:
            self._process_batch(batch)
        return current_timestamp

    def _supports_batch_replay(self) -> bool:
        if not callable(getattr(self.data_feed, "stream_batches", None)):
            return False
        # 自定义 Broker 的回调依赖逐 timestamp 更新其内部时钟。
        if type(self.broker) is not Broker:
            return False
        return (
            self.reference_data is None
            or callable(getattr(self.reference_data, "enrich_batch", None))
        )

    def _process_batch(self, batch: list) -> None:
        timestamp = batch[0].timestamp
        if self.calendar is not None:
            self.calendar.validate_timestamp(timestamp)
        if hasattr(self.broker, "set_timestamp"):
            self.broker.set_timestamp(timestamp)
        if self.reference_data is not None:
            actions = (
                self.reference_data.actions_at(timestamp)
                if self._last_action_timestamp is None
                else self.reference_data.actions_between(
                    self._last_action_timestamp, timestamp
                )
            )
            for source in actions:
                action = CorporateAction()
                action.symbol = source.symbol
                action.timestamp = source.timestamp
                action.cash_dividend_per_share = source.cash_dividend_per_share
                action.share_multiplier = source.share_multiplier
                action.description = source.description
                self.engine.apply_corporate_action(action)
            self._last_action_timestamp = timestamp
        if hasattr(self.engine, "process_market_data_batch"):
            self.engine.process_market_data_batch(batch)
            return
        for snapshot in batch:
            self.engine.push_market_data(snapshot)

    def _collect_results(self) -> BacktestResult:
        result = BacktestResult(
            total_return=self.engine.get_total_return(),
            sharpe_ratio=self.engine.get_sharpe_ratio(),
            max_drawdown=self.engine.get_max_drawdown(),
            annual_return=self.engine.get_annual_return(),
            win_rate=self.engine.get_win_rate(),
            final_equity=self.engine.get_equity(),
            initial_cash=self.initial_cash,
            backend=BACKEND,
            trades=self.engine.get_trade_history(),
            round_trips=self.engine.get_round_trip_history(),
            equity_curve=self.engine.get_equity_curve(),
            orders=(self.engine.get_order_history()
                    if hasattr(self.engine, "get_order_history") else []),
            corporate_actions=(self.engine.get_corporate_action_history()
                               if hasattr(self.engine, "get_corporate_action_history") else []),
            portfolio=(self.engine.get_portfolio_snapshot()
                       if hasattr(self.engine, "get_portfolio_snapshot") else None),
            data_lineage=self._data_lineage,
        )
        result.run_spec = self._run_spec
        return result


def _resolved_execution_config(source, broker: Broker):
    config = ExecutionConfig()
    if source is not None:
        for name in _EXECUTION_CONFIG_FIELDS:
            setattr(config, name, getattr(source, name))

    broker_slippage_bps = float(broker.slippage) * 10_000.0
    if broker_slippage_bps:
        configured = float(config.slippage_bps)
        if configured and not math.isclose(
            configured, broker_slippage_bps, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "Broker.slippage 与 ExecutionConfig.slippage_bps 配置冲突"
            )
        config.slippage_bps = broker_slippage_bps
    return config


@contextmanager
def _seeded_random_state(seed: int | None):
    if seed is None:
        yield
        return

    python_state = random.getstate()
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy 是正式依赖
        np = None
        numpy_state = None
    else:
        numpy_state = np.random.get_state()

    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (2 ** 32))
    try:
        yield
    finally:
        random.setstate(python_state)
        if np is not None and numpy_state is not None:
            np.random.set_state(numpy_state)

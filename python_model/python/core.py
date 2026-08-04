"""纯 Python 参考引擎。

这是 C++ 引擎(cpp_engine)的功能等价实现,接口与 pybind11 绑定保持一致,
用于在 C++ 未编译时跑通并验证整条回测链路。逻辑正确后,C++ 版可作对照。

对齐 bindings.cpp 暴露的类型:
  Side / OrderType / Order / Fill / MarketSnapshot / TradeRecord / EquityPoint
  / BacktestEngine(构造、set_on_market_data/on_fill、run、结果查询)

设计取舍:回测数据是 OHLCV bar,没有真实盘口深度。
  - next_open:策略在 bar t 产生的市价单和限价单从 bar t+1 开始执行
  - close:策略订单在当前 bar 的参考价执行
  - 未成交限价单进入 resting 列表,后续 bar 满足条件再成交
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Callable, Optional

from .broker import FeeSchedule


class Side(IntEnum):
    BUY = 0
    SELL = 1


class OrderType(IntEnum):
    MARKET = 0
    LIMIT = 1


class OrderStatus(IntEnum):
    ACCEPTED = 0
    PARTIALLY_FILLED = 1
    FILLED = 2
    CANCELED = 3
    REJECTED = 4
    EXPIRED = 5


class RejectReason(IntEnum):
    NONE = 0
    INVALID_ORDER = 1
    UNKNOWN_SYMBOL = 2
    NOT_LISTED = 3
    INVALID_LOT_SIZE = 4
    INSUFFICIENT_CASH = 5
    INSUFFICIENT_POSITION = 6
    STALE_MARKET_DATA = 7


@dataclass
class ExecutionConfig:
    max_volume_participation: float = 0.10
    slippage_bps: float = 0.0
    enforce_price_limits: bool = True
    enforce_t_plus_one: bool = True
    allow_short: bool = False
    enforce_board_lot: bool = True
    enforce_cash: bool = True
    market_order_price_buffer_bps: float = 0.0

    def validate(self) -> None:
        if (
            not math.isfinite(self.max_volume_participation)
            or not 0.0 <= self.max_volume_participation <= 1.0
        ):
            raise ValueError("max_volume_participation 必须在 [0, 1] 内")
        if (
            not math.isfinite(self.slippage_bps)
            or not 0.0 <= self.slippage_bps < 10_000.0
        ):
            raise ValueError("slippage_bps 必须在 [0, 10000) 内")
        if (
            not math.isfinite(self.market_order_price_buffer_bps)
            or self.market_order_price_buffer_bps < 0.0
        ):
            raise ValueError("market_order_price_buffer_bps 不能为负数")


@dataclass
class Order:
    id: int = 0
    symbol: str = ""
    side: Side = Side.BUY
    type: OrderType = OrderType.MARKET
    quantity: int = 0
    limit_price: float = 0.0
    timestamp: int = 0


@dataclass
class OrderRecord:
    order: Order = field(default_factory=Order)
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.ACCEPTED
    reject_reason: RejectReason = RejectReason.NONE
    updated_timestamp: int = 0
    message: str = ""


@dataclass
class Fill:
    order_id: int = 0
    symbol: str = ""
    side: Side = Side.BUY
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    timestamp: int = 0


@dataclass
class MarketSnapshot:
    symbol: str = ""
    timestamp: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    upper_limit: float = 0.0
    lower_limit: float = 0.0
    is_suspended: bool = False
    is_listed: bool = True
    is_st: bool = False
    lot_size: int = 1
    min_buy_quantity: int = 1
    board: str = ""
    industry: str = ""
    factor_exposures: dict[str, float] = field(default_factory=dict)
    adjustment_factor: float = 1.0
    signal_open: float = 0.0
    signal_high: float = 0.0
    signal_low: float = 0.0
    signal_close: float = 0.0

    def ref_price(self) -> float:
        return self.close

    def signal_ref_price(self) -> float:
        return self.signal_close if self.signal_close > 0.0 else self.close


@dataclass
class TradeRecord:
    order_id: int = 0
    symbol: str = ""
    side: Side = Side.BUY
    quantity: int = 0
    price: float = 0.0
    commission: float = 0.0
    timestamp: int = 0


@dataclass
class RoundTripRecord:
    symbol: str = ""
    entry_side: Side = Side.BUY
    quantity: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    opened_at: int = 0
    closed_at: int = 0
    gross_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0


@dataclass
class _OpenRoundTripPosition:
    quantity: int = 0
    average_price: float = 0.0
    open_commission: float = 0.0
    opened_at: int = 0


@dataclass
class EquityPoint:
    timestamp: int = 0
    equity: float = 0.0
    cash: float = 0.0


# 每笔成交手续费的回调:(notional, is_sell) -> commission
CommissionFn = Callable[[float, bool], float]


def _validate_market_snapshot(md: MarketSnapshot) -> None:
    prices = (md.open, md.high, md.low, md.close)
    if (
        md.timestamp <= 0
        or not md.symbol
        or not all(math.isfinite(price) and price > 0.0 for price in prices)
        or md.high < max(prices)
        or md.low > min(prices)
        or md.volume < 0
        or md.lot_size <= 0
        or md.min_buy_quantity <= 0
        or not math.isfinite(md.upper_limit)
        or md.upper_limit < 0.0
        or not math.isfinite(md.lower_limit)
        or md.lower_limit < 0.0
        or (md.upper_limit > 0.0 and md.lower_limit > md.upper_limit)
        or not math.isfinite(md.adjustment_factor)
        or md.adjustment_factor <= 0.0
    ):
        raise ValueError("非法 MarketSnapshot")


class OrderBook:
    """与 C++ 后端一致的 Bar 级单标的撮合器。"""

    def __init__(self, symbol: str, config: ExecutionConfig | None = None):
        self.symbol = symbol
        self.last_price = 0.0
        self._config = config or ExecutionConfig()
        self._resting: list[Order] = []
        self._remaining_volume = 0
        self._upper_limit = 0.0
        self._lower_limit = 0.0
        self._is_suspended = False
        self._is_listed = True
        self._has_market_data = False

    def set_reference_price(self, price: float) -> None:
        """设定参考价(如用某根 bar 的 open 价成交延迟到本 bar 的市价单)。"""
        if price > 0.0:
            self.last_price = price

    def begin_market_data(self, md: MarketSnapshot) -> None:
        self.last_price = md.ref_price()
        self._upper_limit = md.upper_limit
        self._lower_limit = md.lower_limit
        self._is_suspended = md.is_suspended
        self._is_listed = md.is_listed
        self._has_market_data = True
        if not md.is_listed or md.is_suspended or md.volume <= 0:
            self._remaining_volume = 0
        else:
            self._remaining_volume = math.floor(
                md.volume * self._config.max_volume_participation
            )

    def update_market_data(self, md: MarketSnapshot) -> list[Fill]:
        self.begin_market_data(md)
        return self.match_resting_orders(md)

    def match_resting_orders(self, md: MarketSnapshot) -> list[Fill]:
        fills: list[Fill] = []
        if not self._is_listed or self._is_suspended or self._remaining_volume <= 0:
            return fills
        still_resting: list[Order] = []
        ordered = sorted(
            enumerate(self._resting),
            key=lambda item: (
                0 if item[1].side == Side.BUY else 1,
                -item[1].limit_price if item[1].side == Side.BUY
                else item[1].limit_price,
                item[0],
            ),
        )
        for _, order in ordered:
            triggered = (
                md.low <= order.limit_price if order.side == Side.BUY
                else md.high >= order.limit_price
            )
            if not triggered or self._remaining_volume <= 0:
                still_resting.append(order)
                continue
            price = (
                min(md.open, order.limit_price) if order.side == Side.BUY
                else max(md.open, order.limit_price)
            )
            fill = self._make_fill(order, price, md.timestamp)
            if fill is None:
                still_resting.append(order)
                continue
            fills.append(fill)
            if fill.quantity < order.quantity:
                order.quantity -= fill.quantity
                still_resting.append(order)
        self._resting = still_resting
        return fills

    def cancel_order(self, order_id: int) -> int:
        canceled = sum(
            order.quantity for order in self._resting if order.id == order_id
        )
        self._resting = [
            order for order in self._resting if order.id != order_id
        ]
        return canceled

    def submit_order(self, order: Order) -> list[Fill]:
        if (
            order.quantity <= 0
            or order.side not in (Side.BUY, Side.SELL)
            or order.type not in (OrderType.MARKET, OrderType.LIMIT)
            or order.symbol != self.symbol
            or (order.type == OrderType.LIMIT and order.limit_price <= 0.0)
        ):
            return []
        if order.type == OrderType.MARKET:
            direction = 1.0 if order.side == Side.BUY else -1.0
            price = self.last_price * (
                1.0 + direction * self._config.slippage_bps / 10_000.0
            )
            fill = self._make_fill(order, price, order.timestamp)
            return [fill] if fill is not None else []

        fill = self._try_fill_limit(order, order.timestamp)
        if fill is not None:
            if fill.quantity == order.quantity:
                return [fill]
            remaining = Order(**vars(order))
            remaining.quantity -= fill.quantity
            self._resting.append(remaining)
            return [fill]
        self._resting.append(Order(**vars(order)))
        return []

    def _try_fill_limit(self, order: Order, ts: int) -> Optional[Fill]:
        """限价单可成交判定:买单要求现价 <= 限价,卖单要求现价 >= 限价。"""
        if self.last_price <= 0.0:
            return None
        can_fill = (
            (order.side == Side.BUY and self.last_price <= order.limit_price)
            or (order.side == Side.SELL and self.last_price >= order.limit_price)
        )
        if not can_fill:
            return None
        return self._make_fill(order, self.last_price, ts)

    def _make_fill(self, order: Order, price: float, ts: int) -> Optional[Fill]:
        if (
            price <= 0.0
            or not math.isfinite(price)
            or order.quantity <= 0
            or not self._is_listed
            or self._is_suspended
            or (self._has_market_data and self._remaining_volume <= 0)
        ):
            return None
        blocked_at_limit = self._config.enforce_price_limits and (
            (order.side == Side.BUY and self._upper_limit > 0.0
             and price >= self._upper_limit)
            or (order.side == Side.SELL and self._lower_limit > 0.0
                and price <= self._lower_limit)
        )
        if blocked_at_limit:
            return None
        quantity = order.quantity
        if self._has_market_data:
            quantity = min(quantity, self._remaining_volume)
        if quantity <= 0:
            return None
        if self._has_market_data:
            self._remaining_volume -= quantity
        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=0.0,
            timestamp=ts,
        )


@dataclass
class Position:
    symbol: str = ""
    quantity: int = 0        # 正多头,负空头
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    sellable_quantity: int = 0


@dataclass
class CorporateAction:
    symbol: str = ""
    timestamp: int = 0
    cash_dividend_per_share: float = 0.0
    share_multiplier: float = 1.0
    description: str = ""


@dataclass
class CorporateActionResult:
    symbol: str = ""
    timestamp: int = 0
    cash_dividend: float = 0.0
    old_quantity: int = 0
    new_quantity: int = 0


@dataclass
class PortfolioSnapshot:
    cash: float = 0.0
    equity: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    largest_position_weight: float = 0.0
    position_count: int = 0
    industry_exposure: dict[str, float] = field(default_factory=dict)
    factor_exposure: dict[str, float] = field(default_factory=dict)


class PositionTracker:
    """持仓管理:按成交更新持仓、结算已实现盈亏(平均成本法)。"""

    def __init__(self):
        self._positions: dict[str, Position] = {}
        self._current_trading_day: int | None = None
        # 每次平仓的已实现盈亏(不含手续费),供胜率统计
        self.closed_pnls: list[float] = []

    def apply_fill(self, fill: Fill) -> None:
        pos = self._positions.setdefault(fill.symbol, Position(symbol=fill.symbol))
        signed = fill.quantity if fill.side == Side.BUY else -fill.quantity
        old_qty = pos.quantity
        if fill.side == Side.SELL:
            pos.sellable_quantity = max(0, pos.sellable_quantity - fill.quantity)

        if old_qty == 0 or (old_qty > 0) == (signed > 0):
            # 建仓或同向加仓:重算加权平均成本
            total_cost = pos.avg_cost * abs(old_qty) + fill.price * abs(signed)
            pos.quantity = old_qty + signed
            pos.avg_cost = total_cost / abs(pos.quantity) if pos.quantity != 0 else 0.0
        else:
            # 反向:先平仓结算盈亏
            close_qty = min(abs(signed), abs(old_qty))
            direction = 1 if old_qty > 0 else -1
            pnl = (fill.price - pos.avg_cost) * close_qty * direction
            pos.realized_pnl += pnl
            self.closed_pnls.append(pnl)

            remaining = abs(signed) - close_qty
            new_qty = old_qty + signed
            pos.quantity = new_qty
            if new_qty == 0:
                pos.avg_cost = 0.0
            elif remaining > 0:
                # 反向超量:剩余部分开反向新仓,成本为本次成交价
                pos.avg_cost = fill.price

    def roll_trading_day(self, timestamp: int) -> None:
        nanoseconds_per_day = 86_400_000_000_000
        shanghai_offset = 8 * 3_600_000_000_000
        trading_day = (timestamp + shanghai_offset) // nanoseconds_per_day
        if self._current_trading_day is None:
            self._current_trading_day = trading_day
            return
        if trading_day == self._current_trading_day:
            return
        self._current_trading_day = trading_day
        for position in self._positions.values():
            position.sellable_quantity = max(position.quantity, 0)

    def available_to_sell(self, symbol: str) -> int:
        return self.get_position(symbol).sellable_quantity

    def apply_corporate_action(self, action: CorporateAction) -> CorporateActionResult:
        if (
            not action.symbol
            or action.timestamp < 0
            or not math.isfinite(action.cash_dividend_per_share)
            or action.cash_dividend_per_share < 0.0
            or not math.isfinite(action.share_multiplier)
            or action.share_multiplier <= 0.0
        ):
            raise ValueError("非法公司行动")
        result = CorporateActionResult(symbol=action.symbol, timestamp=action.timestamp)
        position = self._positions.get(action.symbol)
        if position is None or position.quantity <= 0:
            return result
        result.old_quantity = position.quantity
        result.cash_dividend = position.quantity * action.cash_dividend_per_share
        quantity = position.quantity * action.share_multiplier
        sellable = position.sellable_quantity * action.share_multiplier
        rounded_quantity = round(quantity)
        rounded_sellable = round(sellable)
        if (
            abs(quantity - rounded_quantity) > 1e-9
            or abs(sellable - rounded_sellable) > 1e-9
        ):
            raise ValueError("公司行动产生了非整数持仓")
        if action.share_multiplier != 1.0:
            position.quantity = int(rounded_quantity)
            position.sellable_quantity = int(rounded_sellable)
            position.avg_cost /= action.share_multiplier
        result.new_quantity = position.quantity
        return result

    def get_position(self, symbol: str) -> Position:
        return replace(self._positions.get(symbol, Position(symbol=symbol)))

    def all_positions(self) -> list[Position]:
        return [
            replace(self._positions[symbol]) for symbol in sorted(self._positions)
        ]

    def market_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, pos in self._positions.items():
            total += pos.quantity * prices.get(symbol, pos.avg_cost)
        return total

    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self._positions.values())


class PnLTracker:
    """盈亏跟踪:记录权益曲线和成交,计算绩效指标。"""

    def __init__(self):
        self.equity_curve: list[EquityPoint] = []
        self.trades: list[TradeRecord] = []
        self.round_trips: list[RoundTripRecord] = []
        self._open_round_trips: dict[str, _OpenRoundTripPosition] = {}

    def record_snapshot(self, time: int, equity: float, cash: float) -> None:
        self.equity_curve.append(EquityPoint(timestamp=time, equity=equity, cash=cash))

    def record_trade(self, fill: Fill) -> None:
        self.trades.append(
            TradeRecord(
                order_id=fill.order_id,
                symbol=fill.symbol,
                side=fill.side,
                quantity=fill.quantity,
                price=fill.price,
                commission=fill.commission,
                timestamp=fill.timestamp,
            )
        )
        self._match_round_trip(fill)

    def _match_round_trip(self, fill: Fill) -> None:
        position = self._open_round_trips.setdefault(
            fill.symbol, _OpenRoundTripPosition()
        )
        delta = fill.quantity if fill.side == Side.BUY else -fill.quantity
        if position.quantity == 0 or (position.quantity > 0) == (delta > 0):
            old_quantity = abs(position.quantity)
            if old_quantity == 0:
                position.opened_at = fill.timestamp
            position.average_price = (
                position.average_price * old_quantity + fill.price * abs(delta)
            ) / (old_quantity + abs(delta))
            position.quantity += delta
            position.open_commission += fill.commission
            return

        position_quantity = abs(position.quantity)
        closed = min(position_quantity, abs(delta))
        direction = 1.0 if position.quantity > 0 else -1.0
        opening_fee = position.open_commission * closed / position_quantity
        closing_fee = fill.commission * closed / fill.quantity
        gross_pnl = (fill.price - position.average_price) * closed * direction
        total_commission = opening_fee + closing_fee
        self.round_trips.append(RoundTripRecord(
            symbol=fill.symbol,
            entry_side=Side.BUY if position.quantity > 0 else Side.SELL,
            quantity=closed,
            entry_price=position.average_price,
            exit_price=fill.price,
            opened_at=position.opened_at,
            closed_at=fill.timestamp,
            gross_pnl=gross_pnl,
            commission=total_commission,
            net_pnl=gross_pnl - total_commission,
        ))
        position.open_commission = max(position.open_commission - opening_fee, 0.0)
        old_quantity = position.quantity
        position.quantity += delta
        if position.quantity == 0:
            self._open_round_trips.pop(fill.symbol, None)
        elif (position.quantity > 0) != (old_quantity > 0):
            position.average_price = fill.price
            position.open_commission = max(fill.commission - closing_fee, 0.0)
            position.opened_at = fill.timestamp

    def apply_corporate_action(self, action: CorporateAction) -> None:
        position = self._open_round_trips.get(action.symbol)
        if position is None or action.share_multiplier == 1.0:
            return
        adjusted = position.quantity * action.share_multiplier
        if abs(adjusted - round(adjusted)) > 1e-9:
            raise ValueError("公司行动产生了非整数 round-trip 持仓")
        position.quantity = int(round(adjusted))
        position.average_price /= action.share_multiplier

    def _returns(self, initial_equity: float | None = None) -> list[float]:
        # 分钟权益先收敛为每日最后一个点，避免把 1 分钟收益误按 252 日年化。
        daily = []
        nanoseconds_per_day = 86_400_000_000_000
        for point in self.equity_curve:
            day = point.timestamp // nanoseconds_per_day
            if daily and daily[-1][0] == day:
                daily[-1] = (day, point.equity)
            else:
                daily.append((day, point.equity))
        eq = [equity for _, equity in daily]
        if initial_equity is not None:
            eq.insert(0, initial_equity)
        out = []
        for prev, cur in zip(eq, eq[1:]):
            out.append((cur - prev) / prev if prev > 0 else 0.0)
        return out

    def total_return(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        first = self.equity_curve[0].equity
        last = self.equity_curve[-1].equity
        return (last - first) / first if first > 0 else 0.0

    def sharpe_ratio(
        self, risk_free_rate: float = 0.02, periods_per_year: int = 252,
        initial_equity: float | None = None,
    ) -> float:
        rets = self._returns(initial_equity)
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        std = math.sqrt(var)
        if std == 0:
            return 0.0
        rf_per_period = risk_free_rate / periods_per_year
        return (mean - rf_per_period) / std * math.sqrt(periods_per_year)

    def max_drawdown(self, initial_equity: float | None = None) -> float:
        peak = initial_equity if initial_equity is not None else float("-inf")
        mdd = 0.0
        for p in self.equity_curve:
            peak = max(peak, p.equity)
            if peak > 0:
                dd = (peak - p.equity) / peak
                mdd = max(mdd, dd)
        return mdd

    def annual_return(self, periods_per_year: int = 252) -> float:
        del periods_per_year  # 保留参数兼容旧调用，实际按时间戳计算。
        if len(self.equity_curve) < 2:
            return 0.0
        elapsed = self.equity_curve[-1].timestamp - self.equity_curve[0].timestamp
        if elapsed <= 0 or self.equity_curve[0].equity <= 0:
            return 0.0
        days = elapsed / 86_400_000_000_000
        ratio = self.equity_curve[-1].equity / self.equity_curve[0].equity
        return ratio ** (365.0 / days) - 1 if ratio >= 0 else 0.0

    def win_rate(self, closed_pnls: list[float] | None = None) -> float:
        del closed_pnls  # 兼容旧接口；统一以 matched round-trip 的净收益计算。
        if not self.round_trips:
            return 0.0
        wins = sum(1 for item in self.round_trips if item.net_pnl > 0.0)
        return wins / len(self.round_trips)


class BacktestEngine:
    """回测引擎:驱动行情 → 撮合 → 持仓/盈亏。

    接口对齐 cpp_engine.BacktestEngine。C++ 版内部是事件优先级队列;
    Python 版在 push_market_data 里直接顺序驱动(bar 数据本就有序)。
    """

    def __init__(self, initial_cash: float = 1_000_000.0,
                 fill_timing: str = "next_open",
                 execution_config: ExecutionConfig | None = None):
        if not math.isfinite(initial_cash) or initial_cash < 0.0:
            raise ValueError("initial_cash 必须是有限非负数")
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._books: dict[str, OrderBook] = {}
        self._last_prices: dict[str, float] = {}
        self._latest_market_data: dict[str, MarketSnapshot] = {}
        self._positions = PositionTracker()
        self._pnl = PnLTracker()
        self._current_time = 0
        self._next_order_id = 1
        self._on_market_data: Optional[Callable[[MarketSnapshot], list[Order]]] = None
        self._on_cross_section: Optional[
            Callable[[list[MarketSnapshot]], list[Order]]
        ] = None
        self._on_fill: Optional[Callable[[Fill], None]] = None
        self._on_order_update: Optional[Callable[[OrderRecord], None]] = None
        self._commission_fn: Optional[CommissionFn] = None
        self._fee_schedules: tuple[FeeSchedule, ...] = ()
        self._execution_config = execution_config or ExecutionConfig()
        self._execution_config.validate()
        self._reserved_sell_quantity: dict[str, int] = {}
        self._reserved_buy_cash: dict[int, float] = {}
        self._order_records: dict[int, OrderRecord] = {}
        self._order_sequence: list[int] = []
        self._corporate_action_history: list[CorporateActionResult] = []
        self._poisoned = False
        # 成交时点模型:
        #   "next_open" —— bar t 出的信号在 bar t+1 的 open 成交(默认,无前视偏差)
        #   "close"     —— bar t 出的信号在 bar t 的 close 立即成交(会用到未来信息)
        if fill_timing not in ("next_open", "close"):
            raise ValueError(f"未知 fill_timing: {fill_timing}")
        self._fill_timing = fill_timing
        # next_open 模式下,本 bar 产生、待下一 bar 激活的订单(按 symbol)
        self._pending: dict[str, list[Order]] = {}
        self._last_processed_time: int | None = None

    # ---- 注册回调 ----
    def set_on_market_data(self, cb: Callable[[MarketSnapshot], list[Order]]) -> None:
        self._on_market_data = cb

    def set_on_cross_section(
        self, cb: Callable[[list[MarketSnapshot]], list[Order]]
    ) -> None:
        self._on_cross_section = cb

    def set_on_fill(self, cb: Callable[[Fill], None]) -> None:
        self._on_fill = cb

    def set_on_order_update(self, cb: Callable[[OrderRecord], None]) -> None:
        self._on_order_update = cb

    def set_commission_fn(self, fn: CommissionFn) -> None:
        """注入手续费模型。"""
        if self._fee_schedules:
            raise RuntimeError("commission_fn 与 fee_schedules 不能同时配置")
        self._commission_fn = fn

    def set_fee_schedules(self, schedules: list[FeeSchedule]) -> None:
        if self._books:
            raise RuntimeError("fee_schedules 必须在处理行情前设置")
        if self._commission_fn is not None:
            raise RuntimeError("commission_fn 与 fee_schedules 不能同时配置")
        ordered = tuple(sorted(schedules, key=lambda item: item.effective_from))
        if not ordered:
            raise ValueError("fee_schedules 不能为空")
        for previous, current in zip(ordered, ordered[1:]):
            if (
                previous.effective_to is None
                or previous.effective_to > current.effective_from
            ):
                raise ValueError("费率生效区间重叠")
        self._fee_schedules = ordered

    def set_execution_config(self, config: ExecutionConfig) -> None:
        if self._books:
            raise RuntimeError("execution config 必须在处理行情前设置")
        config.validate()
        self._execution_config = config

    # ---- 灌数据 ----
    def push_market_data(self, md: MarketSnapshot) -> None:
        """兼容入口：即时处理一根行情。"""
        self.process_market_data(md)

    def process_market_data(self, md: MarketSnapshot) -> None:
        self.process_market_data_batch([md])

    def process_market_data_batch(self, batch: list[MarketSnapshot]) -> None:
        """处理同一时间戳的完整截面，并只记录一个权益点。

        处理次序(next_open 模式,贴近实盘):
          1. 完成截面内所有标的上一 bar 的待成交订单
          2. 更新整个截面的 close 和限价单
          3. 回调一次截面策略
          4. 统一 mark-to-market
        close 模式则退化为原来的"当根 close 即时成交"。
        """
        self._ensure_usable()
        if not batch:
            return
        ordered = sorted(batch, key=lambda snapshot: snapshot.symbol)
        timestamp = ordered[0].timestamp
        if self._last_processed_time is not None and timestamp <= self._last_processed_time:
            raise ValueError("行情批次 timestamp 必须严格递增")
        for snapshot in ordered:
            _validate_market_snapshot(snapshot)
        if any(snapshot.timestamp != timestamp for snapshot in ordered):
            raise ValueError("同一批行情必须具有相同 timestamp")
        symbols = [snapshot.symbol for snapshot in ordered]
        if any(not symbol for symbol in symbols) or len(set(symbols)) != len(symbols):
            raise ValueError("同一截面的 symbol 必须非空且唯一")
        try:
            self._process_validated_market_batch(ordered, timestamp)
        except Exception:
            self._poisoned = True
            raise

    def _process_validated_market_batch(
        self, ordered: list[MarketSnapshot], timestamp: int
    ) -> None:
        self._current_time = timestamp
        self._positions.roll_trading_day(timestamp)

        for md in ordered:
            self._latest_market_data[md.symbol] = md
            book = self._book_for(md.symbol)
            book.begin_market_data(md)
            if not md.is_listed:
                open_order_ids = [
                    order_id
                    for order_id in self._order_sequence
                    if self._order_records[order_id].order.symbol == md.symbol
                    and self._is_order_open(order_id)
                ]
                for order_id in open_order_ids:
                    self._cancel_open_order(
                        order_id,
                        timestamp,
                        RejectReason.NOT_LISTED,
                        "order canceled because the symbol is no longer listed",
                    )
            pending = self._pending.pop(md.symbol, [])
            remaining_orders: list[Order] = []
            if pending:
                book.set_reference_price(md.open)
                for order in pending:
                    order.timestamp = timestamp
                    executed = self._execute_order(order)
                    order.quantity -= executed
                    if (
                        order.type == OrderType.MARKET
                        and order.quantity > 0
                        and self._is_order_open(order.id)
                    ):
                        remaining_orders.append(order)
            if remaining_orders:
                self._pending[md.symbol] = remaining_orders
            book.set_reference_price(md.ref_price())
            for fill in book.match_resting_orders(md):
                self._apply_fill(fill)
            self._last_prices[md.symbol] = md.ref_price()

        if self._on_cross_section is not None:
            default_symbol = ordered[0].symbol if len(ordered) == 1 else ""
            self._submit_strategy_orders(
                self._on_cross_section(ordered), default_symbol, timestamp
            )
        elif self._on_market_data is not None:
            for md in ordered:
                self._submit_strategy_orders(
                    self._on_market_data(md), md.symbol, timestamp
                )

        self._mark_to_market(timestamp)
        self._last_processed_time = timestamp

    def run(self) -> None:
        """C++ 版是事件循环;Python 版数据已在 push 时即时处理,这里为空实现。"""
        # 保留以对齐接口。若改为先 add_event 后统一 run,可在此消费事件队列。
        pass

    def cancel_order(self, order_id: int, timestamp: int = 0) -> bool:
        self._ensure_usable()
        try:
            return self._cancel_order(order_id, timestamp)
        except Exception:
            self._poisoned = True
            raise

    def _cancel_order(self, order_id: int, timestamp: int = 0) -> bool:
        return self._cancel_open_order(
            order_id, timestamp or self._current_time,
            RejectReason.NONE, "canceled by user",
        )

    def _cancel_open_order(
        self,
        order_id: int,
        timestamp: int,
        reason: RejectReason,
        message: str,
    ) -> bool:
        if not self._is_order_open(order_id):
            return False
        record = self._order_records[order_id]
        for symbol, orders in self._pending.items():
            self._pending[symbol] = [order for order in orders if order.id != order_id]
        book = self._books.get(record.order.symbol)
        if book is not None:
            book.cancel_order(order_id)
        self._release_order_reservations(order_id, 0, release_all=True)
        record.status = OrderStatus.CANCELED
        record.reject_reason = reason
        record.updated_timestamp = timestamp
        record.message = message
        self._notify_order(record)
        return True

    def finalize(self, timestamp: int = 0) -> None:
        self._ensure_usable()
        try:
            self._finalize(timestamp)
        except Exception:
            self._poisoned = True
            raise

    def _finalize(self, timestamp: int = 0) -> None:
        final_time = timestamp or self._current_time
        for order_id in self._order_sequence:
            if not self._is_order_open(order_id):
                continue
            record = self._order_records[order_id]
            for symbol, orders in self._pending.items():
                self._pending[symbol] = [
                    order for order in orders if order.id != order_id
                ]
            book = self._books.get(record.order.symbol)
            if book is not None:
                book.cancel_order(order_id)
            self._release_order_reservations(order_id, 0, release_all=True)
            record.status = OrderStatus.EXPIRED
            record.updated_timestamp = final_time
            record.message = "expired at end of backtest"
            self._notify_order(record)

    def apply_corporate_action(self, action: CorporateAction) -> CorporateActionResult:
        self._ensure_usable()
        try:
            return self._apply_corporate_action(action)
        except Exception:
            self._poisoned = True
            raise

    def _apply_corporate_action(
        self, action: CorporateAction
    ) -> CorporateActionResult:
        for order_id in list(self._order_sequence):
            record = self._order_records[order_id]
            if record.order.symbol == action.symbol and self._is_order_open(order_id):
                self.cancel_order(order_id, action.timestamp)
        result = self._positions.apply_corporate_action(action)
        self._pnl.apply_corporate_action(action)
        self._cash += result.cash_dividend
        self._corporate_action_history.append(result)
        return result

    # ---- 内部 ----
    def _book_for(self, symbol: str) -> OrderBook:
        book = self._books.get(symbol)
        if book is None:
            book = OrderBook(symbol, self._execution_config)
            self._books[symbol] = book
        return book

    def _submit_strategy_orders(
        self, orders: list[Order], default_symbol: str, timestamp: int
    ) -> None:
        for order in orders:
            order.id = self._next_order_id
            self._next_order_id += 1
            if not order.symbol:
                order.symbol = default_symbol
            if not order.symbol:
                raise ValueError("截面策略订单必须指定 symbol")
            if order.timestamp == 0:
                order.timestamp = timestamp
            elif order.timestamp != timestamp:
                self._reject_order(
                    order,
                    RejectReason.INVALID_ORDER,
                    "order timestamp must match the strategy callback timestamp",
                )
                continue
            market = self._latest_market_data.get(order.symbol)
            if market is None:
                self._reject_order(
                    order, RejectReason.UNKNOWN_SYMBOL, "symbol has no market data"
                )
                continue
            if market.timestamp != timestamp:
                self._reject_order(
                    order,
                    RejectReason.STALE_MARKET_DATA,
                    "symbol has no market data in the current cross-section",
                )
                continue
            reason = self._validate_order(order, market)
            if reason != RejectReason.NONE:
                self._reject_order(order, reason, "order validation failed")
                continue
            if not self._execution_config.allow_short and order.side == Side.SELL:
                available = (
                    self._positions.available_to_sell(order.symbol)
                    if self._execution_config.enforce_t_plus_one
                    else max(self._positions.get_position(order.symbol).quantity, 0)
                )
                reserved = self._reserved_sell_quantity.get(order.symbol, 0)
                if order.quantity > max(available - reserved, 0):
                    self._reject_order(
                        order, RejectReason.INSUFFICIENT_POSITION,
                        "insufficient sellable quantity",
                    )
                    continue
            reserved_cash = 0.0
            if self._execution_config.enforce_cash and order.side == Side.BUY:
                reserved_cash = self._estimate_required_cash(order, market)
                if reserved_cash > self._cash - self._total_reserved_cash() + 1e-9:
                    self._reject_order(
                        order, RejectReason.INSUFFICIENT_CASH, "insufficient cash"
                    )
                    continue
            self._accept_order(order, reserved_cash)
            if not self._is_order_open(order.id):
                continue
            if self._fill_timing == "next_open":
                self._pending.setdefault(order.symbol, []).append(order)
            else:
                executed = self._execute_order(order)
                if order.type == OrderType.MARKET and executed < order.quantity:
                    remaining = Order(**vars(order))
                    remaining.quantity -= executed
                    self._pending.setdefault(order.symbol, []).append(remaining)

    def _validate_order(
        self, order: Order, market: MarketSnapshot
    ) -> RejectReason:
        if (
            order.quantity <= 0
            or order.side not in (Side.BUY, Side.SELL)
            or order.type not in (OrderType.MARKET, OrderType.LIMIT)
            or (
                order.type == OrderType.LIMIT
                and (not math.isfinite(order.limit_price) or order.limit_price <= 0.0)
            )
        ):
            return RejectReason.INVALID_ORDER
        if not market.is_listed:
            return RejectReason.NOT_LISTED
        if self._execution_config.enforce_board_lot:
            lot = max(market.lot_size, 1)
            minimum = max(market.min_buy_quantity, 1)
            if order.side == Side.BUY and (
                order.quantity < minimum or order.quantity % lot != 0
            ):
                return RejectReason.INVALID_LOT_SIZE
            if order.side == Side.SELL and order.quantity % lot != 0:
                position = self._positions.get_position(order.symbol)
                if order.quantity != max(position.quantity, 0):
                    return RejectReason.INVALID_LOT_SIZE
        return RejectReason.NONE

    def _estimate_required_cash(self, order: Order, market: MarketSnapshot) -> float:
        if order.side != Side.BUY:
            return 0.0
        price = order.limit_price
        if order.type == OrderType.MARKET:
            price = market.ref_price() * (
                1.0
                + (
                    self._execution_config.slippage_bps
                    + self._execution_config.market_order_price_buffer_bps
                ) / 10_000.0
            )
            if market.upper_limit > 0.0:
                price = max(price, market.upper_limit)
        if not math.isfinite(price) or price <= 0.0:
            return math.inf
        notional = order.quantity * price
        commission = self._calculate_commission(
            order.timestamp, notional, is_sell=False
        )
        return notional + commission

    def _calculate_commission(
        self, timestamp: int, notional: float, *, is_sell: bool
    ) -> float:
        if self._commission_fn is not None:
            commission = self._commission_fn(notional, is_sell)
        elif self._fee_schedules:
            schedule = next((
                item for item in reversed(self._fee_schedules)
                if timestamp >= item.effective_from
                and (item.effective_to is None or timestamp < item.effective_to)
            ), None)
            if schedule is None:
                raise ValueError(f"timestamp={timestamp} 没有可用费率")
            commission = max(
                notional * schedule.commission_rate, schedule.min_commission
            )
            commission += notional * schedule.transfer_fee_rate
            if is_sell:
                commission += notional * schedule.stamp_tax_rate
        else:
            commission = 0.0
        if not math.isfinite(commission) or commission < 0.0:
            raise ValueError("commission 必须是有限非负数")
        return commission

    def _total_reserved_cash(self) -> float:
        return sum(self._reserved_buy_cash.values())

    def _accept_order(self, order: Order, reserved_cash: float) -> None:
        record = OrderRecord(
            order=Order(**vars(order)),
            status=OrderStatus.ACCEPTED,
            updated_timestamp=order.timestamp,
        )
        self._order_records[order.id] = record
        self._order_sequence.append(order.id)
        if reserved_cash > 0.0:
            self._reserved_buy_cash[order.id] = reserved_cash
        if not self._execution_config.allow_short and order.side == Side.SELL:
            self._reserved_sell_quantity[order.symbol] = (
                self._reserved_sell_quantity.get(order.symbol, 0) + order.quantity
            )
        self._notify_order(record)

    def _reject_order(
        self, order: Order, reason: RejectReason, message: str
    ) -> None:
        record = OrderRecord(
            order=Order(**vars(order)),
            status=OrderStatus.REJECTED,
            reject_reason=reason,
            updated_timestamp=order.timestamp,
            message=message,
        )
        self._order_records[order.id] = record
        self._order_sequence.append(order.id)
        self._notify_order(record)

    def _update_order_fill(self, fill: Fill) -> None:
        record = self._order_records.get(fill.order_id)
        if record is None:
            return
        old_notional = record.avg_fill_price * record.filled_quantity
        record.filled_quantity += fill.quantity
        record.avg_fill_price = (
            old_notional + fill.price * fill.quantity
        ) / record.filled_quantity
        record.status = (
            OrderStatus.FILLED
            if record.filled_quantity >= record.order.quantity
            else OrderStatus.PARTIALLY_FILLED
        )
        record.updated_timestamp = fill.timestamp
        self._notify_order(record)

    def _release_order_reservations(
        self, order_id: int, filled_quantity: int, *, release_all: bool = False
    ) -> None:
        record = self._order_records.get(order_id)
        if record is None:
            return
        remaining = max(record.order.quantity - record.filled_quantity, 0)
        cash = self._reserved_buy_cash.get(order_id)
        if cash is not None:
            if release_all or filled_quantity >= remaining:
                self._reserved_buy_cash.pop(order_id, None)
            elif remaining > 0 and filled_quantity > 0:
                self._reserved_buy_cash[order_id] = (
                    cash * (remaining - filled_quantity) / remaining
                )
        if not self._execution_config.allow_short and record.order.side == Side.SELL:
            released = remaining if release_all else filled_quantity
            reserved = max(
                self._reserved_sell_quantity.get(record.order.symbol, 0) - released, 0
            )
            if reserved:
                self._reserved_sell_quantity[record.order.symbol] = reserved
            else:
                self._reserved_sell_quantity.pop(record.order.symbol, None)

    def _is_order_open(self, order_id: int) -> bool:
        record = self._order_records.get(order_id)
        return record is not None and record.status in {
            OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED,
        }

    def _notify_order(self, record: OrderRecord) -> None:
        if self._on_order_update is not None:
            self._on_order_update(record)

    def _ensure_usable(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "backtest engine is poisoned after a failed state mutation"
            )

    def _execute_order(self, order: Order) -> int:
        if not self._is_order_open(order.id):
            return 0
        if self._execution_config.enforce_cash and order.side == Side.BUY:
            market = MarketSnapshot(**vars(self._latest_market_data[order.symbol]))
            market.close = self._book_for(order.symbol).last_price
            market.upper_limit = 0.0
            required = self._estimate_required_cash(order, market)
            own_reserve = self._reserved_buy_cash.get(order.id, 0.0)
            available = self._cash - self._total_reserved_cash() + own_reserve
            if required > available + 1e-9:
                record = self._order_records[order.id]
                self._release_order_reservations(order.id, 0, release_all=True)
                record.status = (
                    OrderStatus.REJECTED if record.filled_quantity == 0
                    else OrderStatus.CANCELED
                )
                record.reject_reason = RejectReason.INSUFFICIENT_CASH
                record.updated_timestamp = order.timestamp
                record.message = "insufficient cash at execution price"
                self._notify_order(record)
                return 0
        executed = 0
        for fill in self._book_for(order.symbol).submit_order(order):
            executed += fill.quantity
            self._apply_fill(fill)
        return executed

    def _apply_fill(self, fill: Fill) -> None:
        self._validate_fill(fill)
        fill.commission = self._calculate_commission(
            fill.timestamp,
            fill.price * fill.quantity,
            is_sell=fill.side == Side.SELL,
        )
        # 扣现金:买入付出 notional + 手续费,卖出收入 notional - 手续费
        notional = fill.price * fill.quantity
        if self._execution_config.enforce_cash:
            own_reserve = (
                self._reserved_buy_cash.get(fill.order_id, 0.0)
                if fill.side == Side.BUY
                else 0.0
            )
            available = self._cash - self._total_reserved_cash() + own_reserve
            required_cash = (
                notional + fill.commission
                if fill.side == Side.BUY
                else max(fill.commission - notional, 0.0)
            )
            if required_cash > available + 1e-9:
                self._book_for(fill.symbol).cancel_order(fill.order_id)
                self._release_order_reservations(
                    fill.order_id, 0, release_all=True
                )
                record = self._order_records[fill.order_id]
                record.status = (
                    OrderStatus.REJECTED
                    if record.filled_quantity == 0
                    else OrderStatus.CANCELED
                )
                record.reject_reason = RejectReason.INSUFFICIENT_CASH
                record.updated_timestamp = fill.timestamp
                record.message = "insufficient cash for fill including commission"
                self._notify_order(record)
                return
        self._release_order_reservations(fill.order_id, fill.quantity)
        if fill.side == Side.BUY:
            self._cash -= notional + fill.commission
        else:
            self._cash += notional - fill.commission

        self._positions.apply_fill(fill)
        self._pnl.record_trade(fill)
        self._update_order_fill(fill)
        if self._on_fill is not None:
            self._on_fill(fill)

    def _validate_fill(self, fill: Fill) -> None:
        record = self._order_records.get(fill.order_id)
        if record is None:
            raise RuntimeError("fill references an unknown order")
        remaining = record.order.quantity - record.filled_quantity
        if (
            not self._is_order_open(fill.order_id)
            or fill.symbol != record.order.symbol
            or fill.side != record.order.side
            or fill.quantity <= 0
            or fill.quantity > remaining
            or not math.isfinite(fill.price)
            or fill.price <= 0.0
            or not math.isfinite(fill.commission)
            or fill.commission < 0.0
            or fill.timestamp != self._current_time
        ):
            raise RuntimeError("fill violates its originating order")

    def _mark_to_market(self, time: int) -> None:
        position_value = self._positions.market_value(self._last_prices)
        equity = self._cash + position_value
        self._pnl.record_snapshot(time, equity, self._cash)

    # ---- 结果查询(对齐 C++ 接口)----
    def get_equity(self) -> float:
        return self._cash + self._positions.market_value(self._last_prices)

    def get_cash(self) -> float:
        return self._cash

    def get_total_return(self) -> float:
        if self._initial_cash == 0:
            return 0.0
        return self.get_equity() / self._initial_cash - 1.0

    def get_sharpe_ratio(self) -> float:
        return self._pnl.sharpe_ratio(initial_equity=self._initial_cash)

    def get_max_drawdown(self) -> float:
        return self._pnl.max_drawdown(self._initial_cash)

    def get_annual_return(self) -> float:
        curve = self._pnl.equity_curve
        if len(curve) < 2 or self._initial_cash <= 0 or self.get_equity() < 0:
            return 0.0
        elapsed = curve[-1].timestamp - curve[0].timestamp
        if elapsed <= 0:
            return 0.0
        days = elapsed / 86_400_000_000_000
        return (self.get_equity() / self._initial_cash) ** (365.0 / days) - 1

    def get_win_rate(self) -> float:
        return self._pnl.win_rate()

    def get_trade_history(self) -> list[TradeRecord]:
        return [replace(trade) for trade in self._pnl.trades]

    def get_round_trip_history(self) -> list[RoundTripRecord]:
        return [replace(item) for item in self._pnl.round_trips]

    def get_equity_curve(self) -> list[EquityPoint]:
        return [replace(point) for point in self._pnl.equity_curve]

    def get_position(self, symbol: str) -> Position:
        return self._positions.get_position(symbol)

    def get_positions(self) -> list[Position]:
        return list(self._positions.all_positions())

    def get_order_history(self) -> list[OrderRecord]:
        return [
            replace(
                self._order_records[order_id],
                order=replace(self._order_records[order_id].order),
            )
            for order_id in self._order_sequence
        ]

    def get_corporate_action_history(self) -> list[CorporateActionResult]:
        return [replace(action) for action in self._corporate_action_history]

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        equity = self._cash + self._positions.market_value(self._last_prices)
        gross_value = 0.0
        net_value = 0.0
        largest_value = 0.0
        position_count = 0
        industry_values: dict[str, float] = {}
        factor_values: dict[str, float] = {}
        for position in self._positions.all_positions():
            if position.quantity == 0 or position.symbol not in self._last_prices:
                continue
            value = position.quantity * self._last_prices[position.symbol]
            gross_value += abs(value)
            net_value += value
            largest_value = max(largest_value, abs(value))
            position_count += 1
            market = self._latest_market_data.get(position.symbol)
            industry = market.industry if market and market.industry else "UNKNOWN"
            industry_values[industry] = industry_values.get(industry, 0.0) + value
            if market is not None:
                for factor, exposure in market.factor_exposures.items():
                    factor_values[factor] = (
                        factor_values.get(factor, 0.0) + value * exposure
                    )
        if equity == 0.0:
            return PortfolioSnapshot(
                cash=self._cash, equity=equity, position_count=position_count
            )
        return PortfolioSnapshot(
            cash=self._cash,
            equity=equity,
            gross_exposure=gross_value / equity,
            net_exposure=net_value / equity,
            largest_position_weight=largest_value / abs(equity),
            position_count=position_count,
            industry_exposure={
                industry: value / equity for industry, value in industry_values.items()
            },
            factor_exposure={
                factor: value / equity for factor, value in factor_values.items()
            },
        )

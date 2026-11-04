"""C++ 引擎驱动:加载 ML 构建的 cpp_engine,按黄金口径逐截面回放并抽取全量历史。

黄金口径复用 qbt_ml.cli._backtest_artifact 的映射:signal_* 与成交价同值、
lot_size=100、min_buy_quantity=100、无涨跌停约束、process_market_data_batch 驱动;
会话进行中不调用 finalize(挂单跨日保留,与冠军整段回放行为一致)。
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

from .config import ENGINE_MODULE_DIR


def load_engine_module():
    """导入 ML 构建的 cpp_engine。

    venv 里可能存在同名命名空间包(editable 安装指向 C++ 源码目录,无编译产物),
    因此先导入再校验 __ml_enabled__;不合格时把构建目录插到 sys.path 首位重导。
    """
    module = None
    try:
        module = importlib.import_module("cpp_engine")
    except ImportError:
        pass
    if not getattr(module, "__ml_enabled__", False):
        sys.path.insert(0, str(ENGINE_MODULE_DIR))
        sys.modules.pop("cpp_engine", None)
        module = importlib.import_module("cpp_engine")
    if not getattr(module, "__ml_enabled__", False):
        raise RuntimeError("cpp_engine 未启用 ML;请使用 QBT_ENABLE_ML=ON 构建")
    if getattr(module, "__ml_backend__", None) != "onnxruntime":
        raise RuntimeError("cpp_engine 后端不是 onnxruntime,无法加载 ONNX 制品")
    return module


def _enum_name(value) -> str:
    return str(value).split(".")[-1]


def _snapshot(module, row) -> object:
    market = module.MarketSnapshot()
    market.timestamp = int(row.timestamp)
    market.symbol = str(row.symbol)
    for name in ("open", "high", "low", "close"):
        price = float(getattr(row, name))
        setattr(market, name, price)
        setattr(market, f"signal_{name}", price)
    market.volume = int(row.volume)
    market.is_listed = bool(row.is_listed)
    market.is_suspended = bool(row.is_suspended)
    market.is_st = bool(row.is_st)
    market.upper_limit = 0.0
    market.lower_limit = 0.0
    market.lot_size = 100
    market.min_buy_quantity = 100
    return market


def replay(
    module,
    *,
    artifact_path: Path,
    policy: dict,
    initial_cash: float,
    frame: pd.DataFrame,
    cost_profile=None,
) -> dict:
    """回放整段行情,返回全量订单/成交/权益/持仓/往返历史。

    cost_profile 为 None 时保持历史零成本口径;传入 CostProfile 则登记
    佣金/印花税/过户费/滑点——引擎要求这些必须在处理行情前设置。
    """
    if cost_profile is None:
        execution = module.ExecutionConfig()
    else:
        from .costs import build_execution_config

        execution = build_execution_config(module, cost_profile)
    engine = module.BacktestEngine(
        initial_cash, module.FillTiming.NEXT_OPEN, execution
    )
    if cost_profile is not None:
        from .costs import build_fee_schedules

        engine.set_fee_schedules(build_fee_schedules(module, cost_profile))
    engine.set_model_strategy(str(artifact_path), dict(policy), {}, {})

    for _, section in frame.groupby("timestamp", sort=True):
        batch = [_snapshot(module, row) for row in section.itertuples(index=False)]
        engine.process_market_data_batch(batch)

    orders = []
    for record in engine.get_order_history():
        order = record.order
        orders.append({
            "session_id": None,  # runner 回填
            "order_id": int(order.id),
            "decision_id": str(getattr(order, "decision_id", "") or ""),
            "ts": int(order.timestamp),
            "symbol": order.symbol,
            "side": _enum_name(order.side),
            "type": _enum_name(order.type),
            "quantity": int(order.quantity),
            "status": _enum_name(record.status),
            "reject_reason": _enum_name(record.reject_reason),
            "filled_quantity": int(record.filled_quantity),
            "avg_fill_price": float(record.avg_fill_price),
        })

    fills = []
    for index, trade in enumerate(engine.get_trade_history()):
        fills.append({
            "session_id": None,
            "fill_index": index,
            "order_id": int(trade.order_id),
            "ts": int(trade.timestamp),
            "symbol": trade.symbol,
            "side": _enum_name(trade.side),
            "quantity": int(trade.quantity),
            "price": float(trade.price),
            "commission": float(trade.commission),
        })

    equity = [
        {
            "session_id": None,
            "ts": int(point.timestamp),
            "equity": float(point.equity),
            "cash": float(point.cash),
        }
        for point in engine.get_equity_curve()
    ]

    round_trips = []
    for index, trip in enumerate(engine.get_round_trip_history()):
        round_trips.append({
            "session_id": None,
            "rt_index": index,
            "symbol": trip.symbol,
            "entry_side": _enum_name(trip.entry_side),
            "quantity": int(trip.quantity),
            "entry_price": float(trip.entry_price),
            "exit_price": float(trip.exit_price),
            "opened_ns": int(trip.opened_at),
            "closed_ns": int(trip.closed_at),
            "gross_pnl": float(trip.gross_pnl),
            "commission": float(trip.commission),
            "net_pnl": float(trip.net_pnl),
        })

    last_ts = int(equity[-1]["ts"]) if equity else 0
    positions = []
    for position in engine.get_positions():
        if position.quantity == 0:
            continue
        positions.append({
            "session_id": None,
            "asof_ns": last_ts,
            "symbol": position.symbol,
            "quantity": int(position.quantity),
            "sellable": int(position.sellable_quantity),
            "avg_cost": float(position.avg_cost),
        })

    return {
        "orders": orders,
        "fills": fills,
        "equity": equity,
        "positions": positions,
        "round_trips": round_trips,
        "final_equity": float(engine.get_equity()),
        "final_cash": float(engine.get_cash()),
        "total_return": float(engine.get_total_return()),
        "cost_profile": None if cost_profile is None else cost_profile.as_dict(),
        "total_commission": sum(fill["commission"] for fill in fills),
    }

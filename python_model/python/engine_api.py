"""显式选择 C++ 或 Python 回测后端。

``QBT_BACKEND`` 支持 ``auto``（默认）、``cpp`` 和 ``python``。auto 仅在
cpp_engine 确实未安装时回退；二进制损坏或依赖加载失败会直接报错，避免静默换后端。
"""

from __future__ import annotations

import os


REQUESTED_BACKEND = os.getenv("QBT_BACKEND", "auto").strip().lower()
if REQUESTED_BACKEND not in {"auto", "cpp", "python"}:
    raise ValueError("QBT_BACKEND 必须是 auto、cpp 或 python")


def _load_python_backend():
    from . import core
    return core


def _load_cpp_backend():
    import cpp_engine
    return cpp_engine


if REQUESTED_BACKEND == "python":
    _backend = _load_python_backend()
    BACKEND = "python"
else:
    try:
        _backend = _load_cpp_backend()
        BACKEND = "cpp"
    except ModuleNotFoundError as exc:
        if exc.name != "cpp_engine" or REQUESTED_BACKEND == "cpp":
            raise
        _backend = _load_python_backend()
        BACKEND = "python"

BacktestEngine = _backend.BacktestEngine
CorporateAction = _backend.CorporateAction
CorporateActionResult = _backend.CorporateActionResult
EquityPoint = _backend.EquityPoint
ExecutionConfig = _backend.ExecutionConfig
FeeSchedule = _backend.FeeSchedule
Fill = _backend.Fill
MarketSnapshot = _backend.MarketSnapshot
Order = _backend.Order
OrderRecord = _backend.OrderRecord
OrderStatus = _backend.OrderStatus
OrderType = _backend.OrderType
PortfolioSnapshot = _backend.PortfolioSnapshot
Position = _backend.Position
RejectReason = _backend.RejectReason
RoundTripRecord = _backend.RoundTripRecord
Side = _backend.Side
TradeRecord = _backend.TradeRecord

__all__ = [
    "BacktestEngine",
    "CorporateAction",
    "CorporateActionResult",
    "EquityPoint",
    "ExecutionConfig",
    "FeeSchedule",
    "Fill",
    "MarketSnapshot",
    "Order",
    "OrderRecord",
    "OrderStatus",
    "OrderType",
    "PortfolioSnapshot",
    "Position",
    "RejectReason",
    "RoundTripRecord",
    "Side",
    "TradeRecord",
    "BACKEND",
    "REQUESTED_BACKEND",
]

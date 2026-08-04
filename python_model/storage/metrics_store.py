"""绩效指标落库/查询。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.backtest_runner import BacktestResult


class MetricsStore:
    """绩效指标读写。复用同一个 SQLite 连接。"""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def save_metrics(self, run_id: int, result: "BacktestResult") -> None:
        """写入一次回测的绩效指标。"""
        self.conn.execute(
            "INSERT OR REPLACE INTO performance_metrics "
            "(run_id, total_return, annual_return, sharpe_ratio, max_drawdown, "
            " win_rate, total_trades, total_round_trips, gross_pnl, net_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                result.total_return,
                result.annual_return,
                result.sharpe_ratio,
                result.max_drawdown,
                result.win_rate,
                len(result.trades),
                len(result.round_trips),
                sum(item.gross_pnl for item in result.round_trips),
                sum(item.net_pnl for item in result.round_trips),
            ),
        )
        self.conn.commit()

    def get_metrics(self, run_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM performance_metrics WHERE run_id = ?", (run_id,)
        ).fetchone()

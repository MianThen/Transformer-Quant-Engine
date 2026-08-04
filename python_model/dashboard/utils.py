"""Dashboard 共用的数据读取与风险计算。"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def database_path() -> str:
    return str(os.getenv("QBT_DB_PATH", "backtest.db"))


def resolve_run_id(store, session_state) -> int | None:
    run_id = session_state.get("run_id")
    if run_id is not None and store.get_run(run_id) is not None:
        return int(run_id)
    runs = store.list_runs()
    if not runs:
        return None
    run_id = int(runs[0]["id"])
    session_state["run_id"] = run_id
    return run_id


def rows_frame(rows) -> pd.DataFrame:
    return pd.DataFrame([dict(row) for row in rows])


def equity_frame(rows, initial_equity: float | None = None) -> pd.DataFrame:
    frame = rows_frame(rows)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    previous = frame["equity"].shift(1)
    if initial_equity is not None:
        previous.iloc[0] = initial_equity
    frame["return"] = frame["equity"].divide(previous).subtract(1.0).fillna(0.0)
    frame["peak"] = frame["equity"].cummax()
    if initial_equity is not None:
        frame["peak"] = frame["peak"].clip(lower=initial_equity)
    frame["drawdown"] = frame["equity"] / frame["peak"] - 1.0
    return frame


def rolling_sharpe(
    returns: pd.Series, window: int = 20, risk_free_rate: float = 0.02,
) -> pd.Series:
    mean = returns.rolling(window).mean()
    volatility = returns.rolling(window).std(ddof=1)
    excess = mean - risk_free_rate / 252.0
    return excess.divide(volatility.replace(0.0, np.nan)) * np.sqrt(252.0)


def page_window(total: int, page: int, page_size: int) -> tuple[int, int, int]:
    if total < 0 or page_size <= 0:
        raise ValueError("total/page_size 非法")
    pages = max(1, (total + page_size - 1) // page_size)
    current = min(max(int(page), 1), pages)
    return current, page_size, (current - 1) * page_size


def max_drawdown_duration(drawdown: pd.Series) -> int:
    longest = current = 0
    for value in drawdown:
        if value < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest

from __future__ import annotations

import numpy as np
import pandas as pd


def build_next_open_labels(source: pd.DataFrame, horizon_bars: int = 5) -> pd.DataFrame:
    if horizon_bars <= 0:
        raise ValueError("horizon_bars 必须为正数")
    required = {"timestamp", "symbol", "open", "close"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError("NEXT_OPEN 标签缺少字段: " + ", ".join(sorted(missing)))
    table = source.copy().sort_values(["symbol", "timestamp"], kind="stable")
    if table.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("NEXT_OPEN 标签不允许重复的 timestamp/symbol")
    grouped = table.groupby("symbol", sort=False)
    entry = grouped["open"].shift(-1)
    exit_price = grouped["close"].shift(-(horizon_bars + 1))
    y_return = np.log(exit_price / entry)
    result = table[["timestamp", "symbol"]].copy()
    result["entry_open"] = entry
    result["exit_close"] = exit_price
    result["expected_return"] = y_return
    result["direction"] = (y_return > 0).astype(np.float32)
    result["realized_volatility"] = y_return.abs()
    result["label_valid"] = np.isfinite(y_return) & (entry > 0) & (exit_price > 0)
    return result.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)

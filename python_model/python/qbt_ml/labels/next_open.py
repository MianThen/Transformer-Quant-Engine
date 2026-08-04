from __future__ import annotations

import numpy as np
import pandas as pd

from .spec import LabelSpec


def build_next_open_labels(
    source: pd.DataFrame,
    horizon_bars: int = 5,
    *,
    label_spec: LabelSpec | None = None,
) -> pd.DataFrame:
    spec = label_spec or LabelSpec.next_open(horizon_bars)
    if horizon_bars != spec.horizon_bars:
        raise ValueError("horizon_bars 与 LabelSpec 不一致")
    required = {"timestamp", "symbol", "open", "close"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError("NEXT_OPEN 标签缺少字段: " + ", ".join(sorted(missing)))
    table = source.copy().sort_values(["symbol", "timestamp"], kind="stable")
    if table.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("NEXT_OPEN 标签不允许重复的 timestamp/symbol")
    for name in ("open", "close"):
        signal_name = f"signal_{name}"
        if signal_name in table:
            candidate = pd.to_numeric(table[signal_name], errors="coerce")
            table[signal_name] = candidate.where(candidate > 0, table[name])
        else:
            table[signal_name] = table[name]
    grouped = table.groupby("symbol", sort=False)
    entry = grouped["signal_open"].shift(-1)
    exit_price = grouped["signal_close"].shift(-spec.exit_offset_bars)
    entry_timestamp = grouped["timestamp"].shift(-spec.entry_offset_bars)
    exit_timestamp = grouped["timestamp"].shift(-spec.exit_offset_bars)
    y_return = np.log(exit_price / entry)

    subreturns = [np.log(grouped["signal_close"].shift(-1) / entry)]
    for offset in range(2, spec.horizon_bars + 1):
        current_close = grouped["signal_close"].shift(-offset)
        previous_close = grouped["signal_close"].shift(-(offset - 1))
        subreturns.append(np.log(current_close / previous_close))
    return_matrix = np.column_stack([value.to_numpy(np.float64) for value in subreturns])
    subreturns_valid = np.isfinite(return_matrix).all(axis=1)
    realized_volatility = np.full(len(table), np.nan, dtype=np.float64)
    realized_volatility[subreturns_valid] = return_matrix[subreturns_valid].std(
        axis=1, ddof=spec.volatility_ddof
    )

    result = table[["timestamp", "symbol"]].copy()
    result["signal_asof"] = table["timestamp"].to_numpy()
    result["label_entry_timestamp"] = entry_timestamp
    result["label_exit_timestamp"] = exit_timestamp
    result["entry_open"] = entry
    result["exit_close"] = exit_price
    result["expected_return"] = y_return
    result["direction"] = (y_return > 0).astype(np.float32)
    result["realized_volatility"] = realized_volatility
    result["label_valid"] = (
        np.isfinite(y_return) & np.isfinite(realized_volatility)
        & (entry > 0) & (exit_price > 0)
        & (entry_timestamp > table["timestamp"])
        & (exit_timestamp >= entry_timestamp)
    )
    return result.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)

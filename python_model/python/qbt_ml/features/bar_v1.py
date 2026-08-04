from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.schemas import BAR_V1, FeatureSchema


@dataclass(frozen=True)
class BarFeatureFrame:
    table: pd.DataFrame
    values: np.ndarray
    valid_mask: np.ndarray
    schema: FeatureSchema


def build_bar_v1(source: pd.DataFrame, schema: FeatureSchema = BAR_V1) -> BarFeatureFrame:
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError("BAR_V1 缺少字段: " + ", ".join(sorted(missing)))
    table = source.copy().sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    if table.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("BAR_V1 不允许重复的 timestamp/symbol")
    table["timestamp"] = pd.to_numeric(table["timestamp"], errors="coerce")
    timestamp_values = table["timestamp"].to_numpy(dtype=np.float64)
    if (not np.isfinite(timestamp_values).all() or
            (timestamp_values <= 0).any() or
            (timestamp_values != np.floor(timestamp_values)).any()):
        raise ValueError("BAR_V1 timestamp 必须是正整数")
    table["timestamp"] = table["timestamp"].astype(np.int64)
    symbols = table["symbol"].astype("string")
    if symbols.isna().any() or symbols.str.strip().eq("").any():
        raise ValueError("BAR_V1 symbol 不能为空")
    table["symbol"] = symbols.astype(str)
    numeric = ("open", "high", "low", "close", "volume")
    for name in numeric:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    if not np.isfinite(table[list(numeric)].to_numpy(dtype=np.float64)).all():
        raise ValueError("BAR_V1 OHLCV 必须是有限数值")
    if (table[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("BAR_V1 价格必须为正数")
    if (table["volume"] < 0).any():
        raise ValueError("BAR_V1 成交量不能为负数")
    if ((table["high"] < table[["open", "close", "low"]].max(axis=1)).any() or
            (table["low"] > table[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("BAR_V1 OHLC 关系无效")

    signal = {}
    for name in ("open", "high", "low", "close"):
        signal_name = f"signal_{name}"
        if signal_name in table:
            candidate = pd.to_numeric(table[signal_name], errors="coerce")
            signal[name] = candidate.where(candidate > 0, table[name])
        else:
            signal[name] = table[name]

    features = pd.DataFrame(index=table.index, columns=schema.feature_names, dtype=np.float64)
    for _, positions in table.groupby("symbol", sort=False).indices.items():
        index = np.asarray(positions, dtype=np.int64)
        close = np.log(signal["close"].iloc[index]).reset_index(drop=True)
        open_price = np.log(signal["open"].iloc[index]).reset_index(drop=True)
        high = np.log(signal["high"].iloc[index]).reset_index(drop=True)
        low = np.log(signal["low"].iloc[index]).reset_index(drop=True)
        log_volume = np.log1p(table["volume"].iloc[index].reset_index(drop=True))
        return_1 = close.diff(1)
        local = {
            "log_return_1": return_1,
            "log_return_5": close.diff(5),
            "log_return_10": close.diff(10),
            "log_return_20": close.diff(20),
            "intraday_range": high - low,
            "close_open_return": close - open_price,
            "overnight_gap": open_price - close.shift(1),
            "log_volume": log_volume,
            "volume_zscore_20": _rolling_zscore(log_volume, 20),
            "volatility_5": return_1.rolling(5, min_periods=5).std(ddof=0),
            "volatility_10": return_1.rolling(10, min_periods=10).std(ddof=0),
            "volatility_20": return_1.rolling(20, min_periods=20).std(ddof=0),
            "volatility_60": return_1.rolling(60, min_periods=60).std(ddof=0),
            "ma_deviation_5": close - _log_mean_price(close, 5),
            "ma_deviation_10": close - _log_mean_price(close, 10),
            "ma_deviation_20": close - _log_mean_price(close, 20),
            "price_position_20": _price_position(close, 20),
            "breakout_20": close - close.rolling(20, min_periods=20).max(),
        }
        for name, value in local.items():
            features.loc[index, name] = value.to_numpy()

    features["cross_section_return_rank"] = features["log_return_1"].groupby(
        table["timestamp"], sort=False
    ).rank(method="average", pct=True)
    suspended = _boolean_column(table, "is_suspended", False)
    listed = _boolean_column(table, "is_listed", True)
    is_st = _boolean_column(table, "is_st", False)
    features["is_suspended"] = suspended.astype(np.float64)
    features["is_listed"] = listed.astype(np.float64)
    features["is_st"] = is_st.astype(np.float64)
    features["is_tradable"] = (listed & ~suspended).astype(np.float64)

    finite = np.isfinite(features.to_numpy(dtype=np.float64)).all(axis=1)
    valid = finite & listed.to_numpy() & ~suspended.to_numpy()
    values = features.fillna(0.0).to_numpy(dtype=np.float32, copy=True)
    result_table = table[["timestamp", "symbol"]].copy()
    return BarFeatureFrame(result_table, values, valid.astype(np.uint8), schema)


def _boolean_column(table: pd.DataFrame, name: str, default: bool) -> pd.Series:
    if name not in table:
        return pd.Series(default, index=table.index, dtype=bool)
    return table[name].fillna(default).astype(bool)


def _rolling_zscore(value: pd.Series, window: int) -> pd.Series:
    rolling = value.rolling(window, min_periods=window)
    deviation = rolling.std(ddof=0)
    return (value - rolling.mean()) / deviation.replace(0.0, np.nan)


def _log_mean_price(log_price: pd.Series, window: int) -> pd.Series:
    return np.log(np.exp(log_price).rolling(window, min_periods=window).mean())


def _price_position(log_price: pd.Series, window: int) -> pd.Series:
    price = np.exp(log_price)
    rolling = price.rolling(window, min_periods=window)
    low, high = rolling.min(), rolling.max()
    return (price - low) / (high - low).replace(0.0, np.nan)

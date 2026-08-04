from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowedDataset:
    features: np.ndarray
    valid_mask: np.ndarray
    timestamps: np.ndarray
    symbols: np.ndarray
    row_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.features.ndim != 3 or self.valid_mask.shape != self.features.shape[:2]:
            raise ValueError("窗口张量必须是 [N,T,F]，mask 必须是 [N,T]")


def build_windows(feature_frame, lookback: int) -> WindowedDataset:
    if lookback <= 0:
        raise ValueError("lookback 必须为正数")
    table = feature_frame.table
    values = feature_frame.values
    row_valid = feature_frame.valid_mask
    windows, masks, timestamps, symbols, row_indices = [], [], [], [], []
    for symbol, positions in table.groupby("symbol", sort=False).indices.items():
        ordered = np.asarray(positions, dtype=np.int64)
        for end_offset, row_index in enumerate(ordered):
            start_offset = max(0, end_offset + 1 - lookback)
            selected = ordered[start_offset:end_offset + 1]
            padding = lookback - selected.size
            window = np.zeros((lookback, values.shape[1]), dtype=np.float32)
            mask = np.zeros(lookback, dtype=np.uint8)
            window[padding:] = values[selected]
            mask[padding:] = row_valid[selected].astype(np.uint8)
            windows.append(window)
            masks.append(mask)
            timestamps.append(table.iloc[row_index]["timestamp"])
            symbols.append(symbol)
            row_indices.append(row_index)
    return WindowedDataset(
        np.asarray(windows, dtype=np.float32),
        np.asarray(masks, dtype=np.uint8),
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(symbols, dtype=object),
        np.asarray(row_indices, dtype=np.int64),
    )

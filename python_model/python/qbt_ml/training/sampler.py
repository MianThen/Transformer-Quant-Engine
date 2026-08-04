from __future__ import annotations

import math

import numpy as np


class CrossSectionBatchSampler:
    """按完整 timestamp 截面组 batch，永不拆分同一截面。"""

    def __init__(
        self,
        timestamps,
        *,
        timestamps_per_batch: int = 1,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        values = np.asarray(timestamps)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("timestamps 必须是一维非空数组")
        if timestamps_per_batch <= 0:
            raise ValueError("timestamps_per_batch 必须为正数")
        self.timestamps = values
        self.timestamps_per_batch = int(timestamps_per_batch)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        unique = np.unique(values)
        self._groups = tuple(np.flatnonzero(values == timestamp).tolist() for timestamp in unique)

    def __iter__(self):
        order = np.arange(len(self._groups))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        for start in range(0, order.size, self.timestamps_per_batch):
            batch = []
            for group_index in order[start:start + self.timestamps_per_batch]:
                batch.extend(self._groups[int(group_index)])
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self._groups) / self.timestamps_per_batch)

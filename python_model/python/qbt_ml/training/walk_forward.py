from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WalkForwardSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class TimestampSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_timestamps: np.ndarray
    validation_timestamps: np.ndarray
    test_timestamps: np.ndarray


def chronological_timestamp_split(
    timestamps,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    purge_timestamps: int = 0,
    embargo_timestamps: int = 0,
) -> TimestampSplit:
    """按唯一事件时间切分，保证同一截面不会跨数据集。"""
    values = np.asarray(timestamps)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("timestamps 必须是一维非空数组")
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=np.float64
    )
    if not np.isfinite(fractions).all() or (fractions <= 0).any():
        raise ValueError("train/validation/test 比例必须为有限正数")
    if not np.isclose(fractions.sum(), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("train/validation/test 比例之和必须为 1")
    if purge_timestamps < 0 or embargo_timestamps < 0:
        raise ValueError("purge/embargo 不能为负数")

    unique = np.unique(values)
    gap_after_train = purge_timestamps
    gap_after_validation = purge_timestamps + embargo_timestamps
    usable = unique.size - gap_after_train - gap_after_validation
    if usable < 3:
        raise ValueError("去除 purge/embargo 后至少需要三个唯一时间点")
    train_count = max(1, int(np.floor(usable * train_fraction)))
    validation_count = max(1, int(np.floor(usable * validation_fraction)))
    test_count = usable - train_count - validation_count
    if test_count < 1:
        if train_count >= validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        test_count = usable - train_count - validation_count
    if test_count < 1:
        raise ValueError("时间切分后测试区间为空")

    validation_start = train_count + gap_after_train
    test_start = validation_start + validation_count + gap_after_validation
    train_times = unique[:train_count]
    validation_times = unique[validation_start:validation_start + validation_count]
    test_times = unique[test_start:test_start + test_count]
    return TimestampSplit(
        train=np.flatnonzero(np.isin(values, train_times)),
        validation=np.flatnonzero(np.isin(values, validation_times)),
        test=np.flatnonzero(np.isin(values, test_times)),
        train_timestamps=train_times,
        validation_timestamps=validation_times,
        test_timestamps=test_times,
    )


def walk_forward_splits(
    sample_count: int,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    step: int | None = None,
    purge: int = 0,
    embargo: int = 0,
):
    sizes = (sample_count, train_size, validation_size, test_size)
    if any(value <= 0 for value in sizes) or purge < 0 or embargo < 0:
        raise ValueError("样本数和窗口大小必须为正，purge/embargo 不能为负")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError("step 必须为正数")
    start = 0
    while True:
        train_end = start + train_size
        validation_start = train_end + purge
        validation_end = validation_start + validation_size
        test_start = validation_end + purge + embargo
        test_end = test_start + test_size
        if test_end > sample_count:
            return
        yield WalkForwardSplit(
            np.arange(start, train_end, dtype=np.int64),
            np.arange(validation_start, validation_end, dtype=np.int64),
            np.arange(test_start, test_end, dtype=np.int64),
        )
        start += step


def timestamp_walk_forward_splits(
    timestamps,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    step: int | None = None,
    purge_timestamps: int = 0,
    embargo_timestamps: int = 0,
):
    """Map complete timestamp windows back to row indices."""
    values = np.asarray(timestamps)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("timestamps 必须是一维非空数组")
    unique = np.unique(values)
    for split in walk_forward_splits(
        unique.size,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        step=step,
        purge=purge_timestamps,
        embargo=embargo_timestamps,
    ):
        train_times = unique[split.train]
        validation_times = unique[split.validation]
        test_times = unique[split.test]
        yield TimestampSplit(
            train=np.flatnonzero(np.isin(values, train_times)),
            validation=np.flatnonzero(np.isin(values, validation_times)),
            test=np.flatnonzero(np.isin(values, test_times)),
            train_timestamps=train_times,
            validation_timestamps=validation_times,
            test_timestamps=test_times,
        )

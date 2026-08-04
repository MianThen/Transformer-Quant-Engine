from __future__ import annotations

import numpy as np


_MASK64 = (1 << 64) - 1


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return state, (value ^ (value >> 31)) & _MASK64


def adjusted_p_values(p_values: np.ndarray, multiplier: float = 1.0) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be a non-empty probability vector")
    order = np.argsort(values, kind="stable")
    ranked = values[order] * values.size * multiplier / np.arange(1, values.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(values)
    result[order] = np.minimum(ranked, 1.0)
    return result


def paired_block_bootstrap_mean(
    differences: np.ndarray,
    *,
    block_length: int,
    resamples: int,
    seed: int,
) -> tuple[float, int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not 0 < block_length <= values.size:
        raise ValueError("invalid paired block-bootstrap input")
    observed = float(values.mean())
    centered = values - observed
    state = seed
    exceedances = 0
    for _ in range(resamples):
        sample: list[float] = []
        while len(sample) < values.size:
            state, random_value = _splitmix64(state)
            start = random_value % values.size
            for offset in range(block_length):
                if len(sample) == values.size:
                    break
                sample.append(float(centered[(start + offset) % values.size]))
        if float(np.mean(sample)) >= observed:
            exceedances += 1
    return (1 + exceedances) / (1 + resamples), exceedances

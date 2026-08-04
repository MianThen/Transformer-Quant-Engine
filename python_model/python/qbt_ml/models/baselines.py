from __future__ import annotations

import numpy as np


class RidgeBaseline:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError("alpha 不能为负数")
        self.alpha = float(alpha)
        self.coefficients_: np.ndarray | None = None

    def fit(self, features, target) -> "RidgeBaseline":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        if x.ndim != 2 or y.shape != (x.shape[0],):
            raise ValueError("Ridge 输入 shape 不匹配")
        design = np.column_stack([np.ones(x.shape[0]), x])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coefficients_ = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, features) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("RidgeBaseline 尚未 fit")
        x = np.asarray(features, dtype=np.float64)
        return self.coefficients_[0] + x @ self.coefficients_[1:]

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


class LogisticBaseline:
    def __init__(self, alpha: float = 1.0, max_iterations: int = 100) -> None:
        if alpha < 0 or max_iterations <= 0:
            raise ValueError("alpha 不能为负数，max_iterations 必须为正数")
        self.alpha = float(alpha)
        self.max_iterations = int(max_iterations)
        self.coefficients_: np.ndarray | None = None

    def fit(self, features, target) -> "LogisticBaseline":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        if x.ndim != 2 or y.shape != (x.shape[0],):
            raise ValueError("Logistic 输入 shape 不匹配")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("Logistic target 必须为 0/1")
        design = np.column_stack([np.ones(x.shape[0]), x])
        coefficients = np.zeros(design.shape[1], dtype=np.float64)
        penalty = np.eye(design.shape[1], dtype=np.float64) * self.alpha
        penalty[0, 0] = 0.0
        for _ in range(self.max_iterations):
            logits = np.clip(design @ coefficients, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            weights = np.clip(probability * (1.0 - probability), 1e-8, None)
            adjusted = logits + (y - probability) / weights
            hessian = design.T @ (weights[:, None] * design) + penalty
            right = design.T @ (weights * adjusted)
            updated = np.linalg.solve(hessian, right)
            if np.max(np.abs(updated - coefficients)) < 1e-10:
                coefficients = updated
                break
            coefficients = updated
        self.coefficients_ = coefficients
        return self

    def predict_proba(self, features) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("LogisticBaseline 尚未 fit")
        x = np.asarray(features, dtype=np.float64)
        logits = np.clip(self.coefficients_[0] + x @ self.coefficients_[1:], -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

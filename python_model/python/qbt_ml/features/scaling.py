"""Train-fold-only feature standardization for robust training candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


PROTECTED_FEATURE_NAMES = frozenset({
    "is_suspended", "is_listed", "is_st", "is_tradable",
})


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


@dataclass(frozen=True)
class FeatureStandardizerV1:
    """Per-feature mean/std fitted only from valid train-fold observations."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    protected: tuple[bool, ...]
    scale_floor: float = 1e-6
    schema_version: int = 1

    def __post_init__(self) -> None:
        count = len(self.feature_names)
        if count == 0 or len(set(self.feature_names)) != count:
            raise ValueError("feature_names 必须非空且不能重复")
        if len(self.mean) != count or len(self.scale) != count:
            raise ValueError("scaler 统计量长度必须与 feature_names 一致")
        if len(self.protected) != count:
            raise ValueError("protected 长度必须与 feature_names 一致")
        if not np.isfinite(np.asarray(self.mean, dtype=np.float64)).all():
            raise ValueError("scaler mean 必须有限")
        scales = np.asarray(self.scale, dtype=np.float64)
        if not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("scaler scale 必须为有限正数")
        if not np.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("scale_floor 必须为有限正数")

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        valid_mask: np.ndarray,
        feature_names: Sequence[str],
        *,
        scale_floor: float = 1e-6,
    ) -> "FeatureStandardizerV1":
        values = np.asarray(features, dtype=np.float64)
        mask = np.asarray(valid_mask, dtype=bool)
        names = tuple(feature_names)
        if values.ndim != 3 or mask.shape != values.shape[:2]:
            raise ValueError("features 必须为 [N,T,F]，valid_mask 必须为 [N,T]")
        if len(names) != values.shape[-1]:
            raise ValueError("feature_names 长度必须等于 F")
        if not np.isfinite(values[mask]).all():
            raise ValueError("有效 train-fold 特征必须有限")
        if not np.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError("scale_floor 必须为有限正数")
        selected = values[mask]
        if selected.shape[0] == 0:
            raise ValueError("train-fold 没有有效观测，无法拟合 scaler")
        means = selected.mean(axis=0)
        scales = selected.std(axis=0, ddof=0)
        scales = np.where(scales < scale_floor, 1.0, scales)
        protected = tuple(name in PROTECTED_FEATURE_NAMES for name in names)
        means = np.where(np.asarray(protected), 0.0, means)
        scales = np.where(np.asarray(protected), 1.0, scales)
        return cls(
            feature_names=names,
            mean=tuple(float(value) for value in means),
            scale=tuple(float(value) for value in scales),
            protected=protected,
            scale_floor=float(scale_floor),
        )

    @property
    def canonical_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "protected": list(self.protected),
            "scale_floor": self.scale_floor,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.canonical_payload).encode("utf-8")
        ).hexdigest()

    def transform(
        self,
        features: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.shape[-1] != len(self.feature_names):
            raise ValueError("features 的 F 与 scaler 不一致")
        if not np.isfinite(values).all():
            raise ValueError("transform 输入必须有限；缺失值应先显式填补")
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        transformed = (values - mean) / scale
        transformed[..., np.asarray(self.protected, dtype=bool)] = values[
            ..., np.asarray(self.protected, dtype=bool)
        ]
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape != values.shape[:2]:
                raise ValueError("valid_mask 形状与 features 不一致")
            transformed = np.where(mask[..., None], transformed, 0.0)
        return transformed.astype(np.float32, copy=False)

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.shape[-1] != len(self.feature_names):
            raise ValueError("features 的 F 与 scaler 不一致")
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        restored = values * scale + mean
        protected = np.asarray(self.protected, dtype=bool)
        restored[..., protected] = values[..., protected]
        return restored.astype(np.float32, copy=False)

    def center_impute(
        self,
        features: np.ndarray,
        missing_mask: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32).copy()
        missing = np.asarray(missing_mask, dtype=bool)
        if values.ndim != 3 or missing.shape != values.shape:
            raise ValueError("missing_mask 必须与 features 同形状")
        if not np.isfinite(values[~missing]).all():
            raise ValueError("非缺失输入必须有限")
        continuous = ~np.asarray(self.protected, dtype=bool)
        values[..., continuous] = np.where(
            missing[..., continuous],
            np.asarray(self.mean, dtype=np.float32)[continuous],
            values[..., continuous],
        )
        return values

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            _canonical_json({**self.canonical_payload, "sha256": self.sha256}) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: str | Path) -> "FeatureStandardizerV1":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        supplied_hash = payload.pop("sha256", None)
        instance = cls(
            feature_names=tuple(payload["feature_names"]),
            mean=tuple(payload["mean"]),
            scale=tuple(payload["scale"]),
            protected=tuple(payload["protected"]),
            scale_floor=float(payload["scale_floor"]),
            schema_version=int(payload.get("schema_version", 1)),
        )
        if supplied_hash != instance.sha256:
            raise ValueError("scaler sha256 校验失败")
        return instance


__all__ = ["FeatureStandardizerV1", "PROTECTED_FEATURE_NAMES"]

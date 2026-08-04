from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


BAR_V1_FEATURE_NAMES = (
    "log_return_1", "log_return_5", "log_return_10", "log_return_20",
    "intraday_range", "close_open_return", "overnight_gap", "log_volume",
    "volume_zscore_20", "volatility_5", "volatility_10", "volatility_20",
    "volatility_60", "ma_deviation_5", "ma_deviation_10",
    "ma_deviation_20", "price_position_20", "breakout_20",
    "cross_section_return_rank", "is_suspended", "is_listed", "is_st",
    "is_tradable",
)


@dataclass(frozen=True)
class FeatureSchema:
    schema_version: int
    profile: str
    feature_names: tuple[str, ...]
    static_feature_names: tuple[str, ...] = ()
    value_dtype: str = "float32"
    mask_dtype: str = "uint8"
    layout: str = "NTF"

    def __post_init__(self) -> None:
        if self.schema_version <= 0 or not self.profile or not self.feature_names:
            raise ValueError("FeatureSchema 的版本、profile 和特征不能为空")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("FeatureSchema 特征名不能重复")
        if self.layout != "NTF" or self.value_dtype != "float32":
            raise ValueError("第一版仅支持 float32 NTF 布局")

    @property
    def canonical_json(self) -> str:
        value = asdict(self)
        value["feature_names"] = list(self.feature_names)
        value["static_feature_names"] = list(self.static_feature_names)
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def hash64(self) -> int:
        return int(self.sha256[:16], 16)

    def write(self, path: str | Path) -> None:
        # 文件内容就是参与 schema 身份计算的 canonical JSON，避免内存 hash 与制品 hash 分叉。
        Path(path).write_text(self.canonical_json, encoding="utf-8")


BAR_V1 = FeatureSchema(1, "BAR_V1", BAR_V1_FEATURE_NAMES)

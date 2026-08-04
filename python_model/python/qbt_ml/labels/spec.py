from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LabelSpec:
    schema_version: int = 1
    name: str = "NEXT_OPEN_HOLD_CLOSE"
    horizon_bars: int = 5
    signal_offset_bars: int = 0
    entry_offset_bars: int = 1
    exit_offset_bars: int = 5
    return_type: str = "log_return"
    volatility_type: str = "holding_period_subreturn_std"
    volatility_ddof: int = 0
    direction_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.horizon_bars <= 0:
            raise ValueError("LabelSpec 版本无效或 horizon_bars 不是正数")
        if self.signal_offset_bars != 0 or self.entry_offset_bars != 1:
            raise ValueError("V1.1 只支持信号日收盘后、NEXT_OPEN 入场")
        if self.exit_offset_bars != self.horizon_bars:
            raise ValueError("exit_offset_bars 必须等于 horizon_bars")
        if self.return_type != "log_return" or self.volatility_ddof != 0:
            raise ValueError("V1.1 只支持 log return 和总体标准差")

    @classmethod
    def next_open(cls, horizon_bars: int = 5) -> "LabelSpec":
        return cls(horizon_bars=horizon_bars, exit_offset_bars=horizon_bars)

    @property
    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

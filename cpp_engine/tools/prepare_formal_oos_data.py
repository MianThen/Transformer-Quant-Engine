"""Phase 4A/4B 正式 OOS 数据预备（合同 docs/phase4_ab_formal_oos_contract.md）。

从 PIT 源表构造情景/前向收益矩阵与 tradable 掩码，输出二进制 + prep_manifest（SHA-256 全链）。
仅使用回看数据；前向收益只作 OOS 度量。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("/Users/Zhuanz/PycharmProjects/PythonProject/data/research/phase1e_pit_120_2020plus.parquet")
OUTPUT = Path("/Users/Zhuanz/CLionProjects/quant-backtester-cpp/work/formal-oos-data")
FORWARD_HORIZON = 5
CONTRACT_PATH = Path("/Users/Zhuanz/CLionProjects/quant-backtester-cpp/docs/phase4_ab_formal_oos_contract.md")
CONFIDENCE_MAPPING = "identity@fixed-0.30"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    frame = pd.read_parquet(SOURCE)
    frame["timestamp"] = frame["timestamp"].astype("int64")
    symbols = sorted(frame["symbol"].astype(str).unique())
    timestamps = np.sort(frame["timestamp"].unique())

    close = np.full((timestamps.size, len(symbols)), np.nan, dtype=np.float64)
    tradable = np.zeros((timestamps.size, len(symbols)), dtype=np.uint8)
    row_index = {value: position for position, value in enumerate(timestamps)}
    column_index = {value: position for position, value in enumerate(symbols)}
    for record in frame.itertuples(index=False):
        close[row_index[int(record.timestamp)], column_index[str(record.symbol)]] = float(record.close)
        tradable[row_index[int(record.timestamp)], column_index[str(record.symbol)]] = 1 if bool(record.is_tradable) else 0

    scenarios = np.zeros_like(close)
    scenarios[1:] = close[1:] / close[:-1] - 1.0
    forward = np.zeros_like(close)
    forward[:-FORWARD_HORIZON] = close[FORWARD_HORIZON:] / close[:-FORWARD_HORIZON] - 1.0
    valid = np.isfinite(close)
    # 前向收益要求两端点均有效：基点或 5 期后终点无效都置 0
    endpoint_valid = valid & np.roll(valid, -FORWARD_HORIZON, axis=0)
    scenarios[~valid] = 0.0
    forward[~endpoint_valid] = 0.0
    forward[-FORWARD_HORIZON:] = 0.0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "scenarios.f64": scenarios.astype("<f8").tobytes(),
        "forward.f64": forward.astype("<f8").tobytes(),
        "timestamps.i64": timestamps.astype("<i8").tobytes(),
        "tradable.u8": tradable.tobytes(),
        "valid.u8": valid.astype(np.uint8).tobytes(),
    }
    for name, payload in outputs.items():
        (OUTPUT / name).write_bytes(payload)
    (OUTPUT / "symbols.txt").write_text("\n".join(symbols) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_parquet_sha256": sha256_file(SOURCE),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "timestamp_count": int(timestamps.size),
        "timestamp_first": int(timestamps[0]),
        "timestamp_last": int(timestamps[-1]),
        "forward_horizon": FORWARD_HORIZON,
        "confidence_mapping": CONFIDENCE_MAPPING,
        "confidence_mapping_sha256": sha256_bytes(CONFIDENCE_MAPPING.encode("utf-8")),
        "files": {name: sha256_file(OUTPUT / name) for name in list(outputs) + ["symbols.txt"]},
        "pit_note": "scenarios 为回看收益；forward 仅 OOS 度量，不进入任何估计",
    }
    (OUTPUT / "prep_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(OUTPUT),
        "timestamps": int(timestamps.size),
        "symbols": len(symbols),
        "first": int(timestamps[0]),
        "last": int(timestamps[-1]),
        "tradable_ratio": float(tradable.mean()),
        "manifest_sha256": sha256_file(OUTPUT / "prep_manifest.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

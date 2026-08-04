"""生成确定性的合成 OHLCV 样本数据,供回测示例与测试使用。

用固定种子的正弦趋势 + 伪随机噪声,保证结果可复现(无外部依赖)。
生成的价格有明显的上下波动,能触发双均线的金叉/死叉。

用法: python -m python.gen_sample_data [行数] [输出路径]
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

SYMBOL = "000001"
DAY_NS = 86_400_000_000_000  # 一天的纳秒数
START_TS = 1_704_067_200_000_000_000  # 2024-01-01 附近


def _lcg(seed: int):
    """线性同余伪随机,避免依赖 random 模块保证跨环境可复现。"""
    state = seed
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF  # [0,1)


def generate(n: int = 250) -> list[dict]:
    rng = _lcg(42)
    rows = []
    base = 10.0
    for i in range(n):
        # 低频正弦趋势(约 60 根一个周期)+ 噪声
        trend = 2.0 * math.sin(i / 60.0 * 2 * math.pi)
        noise = (next(rng) - 0.5) * 0.4
        close = base + trend + noise
        open_ = close - (next(rng) - 0.5) * 0.2
        high = max(open_, close) + next(rng) * 0.15
        low = min(open_, close) - next(rng) * 0.15
        volume = int(1_000_000 + next(rng) * 500_000)
        rows.append(
            {
                "timestamp": START_TS + i * DAY_NS,
                "symbol": SYMBOL,
                "open": round(open_, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(close, 4),
                "volume": volume,
            }
        )
    return rows


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    out = sys.argv[2] if len(sys.argv) > 2 else "data/sample/sample_ohlcv.csv"
    rows = generate(n)
    write_csv(rows, out)
    print(f"已生成 {len(rows)} 行 → {out}")


if __name__ == "__main__":
    main()

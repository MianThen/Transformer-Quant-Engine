"""Aggregate Phase 5 supervised ablation across seeds; test infots gain direction consistency.

用法: python3 aggregate_seeds.py
数据源（缺哪个跳过哪个，至少 1 个 seed 才输出）:
  seed 20260911: outputs/phase5-supervised/runs/phase5-supervised
  seed 20260912/13: outputs/phase5-supervised-seeds/runs/phase5-supervised-seed<seed>
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/Users/Zhuanz/CLionProjects/quant-backtester-cpp/work/kaggle-phase5-supervised/outputs")
SEEDS = {
    20260911: BASE / "phase5-supervised/runs/phase5-supervised",
    20260912: BASE / "phase5-supervised-seeds/runs/phase5-supervised-seed20260912",
    20260913: BASE / "phase5-supervised-seeds/runs/phase5-supervised-seed20260913",
}
GROUPS = ("from_scratch", "fixed_augmentation", "infots")
FOLDS = (1, 2, 3)
METRICS = ("composite_error", "return_mae", "direction_brier", "volatility_mae", "ndcg_at_20", "rank_ic")
FOCUS = ("rank_ic", "ndcg_at_20", "volatility_mae")


def load_seed(root: Path) -> dict | None:
    table: dict[tuple[str, int], dict[str, float]] = {}
    for fold in FOLDS:
        for group in GROUPS:
            path = root / f"fold-{fold}" / group / "artifact.json"
            if not path.is_file():
                return None
            artifact = json.loads(path.read_text(encoding="utf-8"))
            table[(group, fold)] = artifact["metrics"]
    return table


def main() -> int:
    data: dict[int, dict] = {}
    for seed, root in SEEDS.items():
        table = load_seed(root)
        if table is None:
            print(f"[skip] seed {seed}: 结果不完整 ({root})")
        else:
            data[seed] = table
    if not data:
        raise SystemExit("没有任何完整 seed 结果")

    print(f"\n=== 每 seed 三组均值（{len(data)} seeds: {sorted(data)}）===")
    print(f"{'seed':>10s} {'group':>20s}" + "".join(f"{m:>16s}" for m in METRICS))
    for seed in sorted(data):
        for group in GROUPS:
            means = {m: sum(data[seed][(group, f)][m] for f in FOLDS) / 3 for m in METRICS}
            print(f"{seed:>10d} {group:>20s}" + "".join(f"{means[m]:16.6f}" for m in METRICS))

    print("\n=== infots - from_scratch（逐 seed，焦点指标）===")
    print(f"{'seed':>10s}" + "".join(f"{m:>16s}" for m in FOCUS))
    positive: dict[str, list[bool]] = {m: [] for m in FOCUS}
    for seed in sorted(data):
        deltas = {}
        for m in FOCUS:
            delta = sum(data[seed][("infots", f)][m] - data[seed][("from_scratch", f)][m] for f in FOLDS) / 3
            deltas[m] = delta
            better = delta > 0 if m != "volatility_mae" else delta < 0
            positive[m].append(better)
        print(f"{seed:>10d}" + "".join(f"{deltas[m]:+16.6f}" for m in FOCUS))

    print("\n=== fold 级方向一致性（infots 优于 from_scratch 的 fold 数 / 总）===")
    for m in FOCUS:
        wins = 0
        total = 0
        for seed in sorted(data):
            for f in FOLDS:
                delta = data[seed][("infots", f)][m] - data[seed][("from_scratch", f)][m]
                better = delta > 0 if m != "volatility_mae" else delta < 0
                wins += int(better)
                total += 1
        print(f"  {m:>16s}: {wins}/{total}")

    print("\n=== 稳定性判定 ===")
    n = len(data)
    for m in FOCUS:
        count = sum(positive[m])
        note = "一致" if count == n else ("不一致" if count == 0 else "混合")
        print(f"  {m:>16s}: {count}/{n} seed 方向正确 -> {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

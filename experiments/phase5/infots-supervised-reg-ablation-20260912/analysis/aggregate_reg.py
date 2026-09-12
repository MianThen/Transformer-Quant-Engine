"""Phase 5 正则化复训配对评估（预注册决策规则）。

规则：
  R1 正控制：reg run 中 infots−from_scratch 的 vol MAE 须 3/3 seed < 0；否则复训设计破坏原效应，终止。
  R2 翻转检验：reg run 中 infots−from_scratch 的 rank_ic / ndcg 仅当 3/3 seed 同向且 |Δ|>0.005 才算翻转证据。
  R3 动态检验：reg run 的 selected_epoch 中位数应显著 > 原 run（验证正则化确实改变了训练动态）。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

BASE = Path("/Users/Zhuanz/CLionProjects/quant-backtester-cpp/work/kaggle-phase5-supervised/outputs")
ORIG = {
    20260911: BASE / "phase5-supervised/runs/phase5-supervised",
    20260912: BASE / "phase5-supervised-seeds/runs/phase5-supervised-seed20260912",
    20260913: BASE / "phase5-supervised-seeds/runs/phase5-supervised-seed20260913",
}
REG = {seed: BASE / f"phase5-supervised-reg/runs/phase5-supervised-reg-seed{seed}" for seed in ORIG}
GROUPS = ("from_scratch", "fixed_augmentation", "infots")
FOLDS = (1, 2, 3)


def load(root: Path) -> tuple[dict, list[int]] | None:
    table, epochs = {}, []
    for fold in FOLDS:
        for group in GROUPS:
            path = root / f"fold-{fold}" / group / "artifact.json"
            if not path.is_file():
                return None
            artifact = json.loads(path.read_text(encoding="utf-8"))
            table[(group, fold)] = artifact["metrics"]
            epochs.append(artifact["selected_epoch"])
    return table, epochs


def deltas(table: dict, metric: str) -> float:
    return sum(table[("infots", f)][metric] - table[("from_scratch", f)][metric] for f in FOLDS) / 3


def means(table: dict, group: str, metric: str) -> float:
    return sum(table[(group, f)][metric] for f in FOLDS) / 3


def main() -> int:
    orig = {s: load(r) for s, r in ORIG.items()}
    reg = {s: load(r) for s, r in REG.items()}
    for label, data in (("orig", orig), ("reg", reg)):
        missing = [s for s, v in data.items() if v is None]
        if missing:
            print(f"[{label}] 缺失 seed: {missing}")
    if any(v is None for v in reg.values()):
        print("reg 结果不完整，等待")
        return 1

    print("\n=== R3 训练动态（selected_epoch）===")
    for label, data in (("orig", orig), ("reg", reg)):
        ready = {s: v for s, v in data.items() if v is not None}
        if not ready:
            continue
        all_eps = [e for _, (_, eps) in ready.items() for e in eps]
        print(f"  {label}: median={statistics.median(all_eps):.1f} max={max(all_eps)} "
              f"mean={statistics.mean(all_eps):.1f} (n={len(all_eps)})")

    print("\n=== 每 seed：infots−from_scratch（reg vs orig）===")
    print(f"{'seed':>10} {'metric':>14} {'orig':>10} {'reg':>10}")
    rules = {"rank_ic": [], "ndcg_at_20": [], "volatility_mae": []}
    for seed in sorted(reg):
        for metric in ("rank_ic", "ndcg_at_20", "volatility_mae"):
            o = deltas(orig[seed][0], metric) if orig.get(seed) else float("nan")
            r = deltas(reg[seed][0], metric)
            print(f"{seed:>10} {metric:>14} {o:10.6f} {r:10.6f}")
            if metric == "volatility_mae":
                rules[metric].append(r < 0)
            else:
                rules[metric].append((r > 0, abs(r) > 0.005))

    print("\n=== 预注册决策规则评估 ===")
    vol_ok = all(rules["volatility_mae"])
    print(f"R1 正控制 (vol<0 3/3): {'PASS' if vol_ok else 'FAIL'} -> {'原效应保持' if vol_ok else '复训破坏原效应，按规则终止'}")
    for metric in ("rank_ic", "ndcg_at_20"):
        same_dir = all(d for d, _ in rules[metric])
        big_enough = all(m for _, m in rules[metric])
        flip = same_dir and big_enough
        print(f"R2 {metric}: 同向 3/3={'Y' if same_dir else 'N'} |Δ|>0.005={'Y' if big_enough else 'N'} "
              f"-> {'构成翻转证据(REFERENCE_ONLY)' if flip else '不构成翻转证据，维持无排序增益判定'}")

    print("\n=== reg run 三组均值（参考）===")
    print(f"{'group':>20} {'rank_ic':>10} {'ndcg':>10} {'vol_mae':>10} {'ret_mae':>10}")
    for group in GROUPS:
        vals = {m: sum(means(reg[s][0], group, m) for s in reg) / len(reg)
                for m in ("rank_ic", "ndcg_at_20", "volatility_mae", "return_mae")}
        print(f"{group:>20} {vals['rank_ic']:10.6f} {vals['ndcg_at_20']:10.6f} "
              f"{vals['volatility_mae']:10.6f} {vals['return_mae']:10.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

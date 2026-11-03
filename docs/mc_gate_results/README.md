# MC-GATE-VALIDATION 运行产出归档

本目录保存能在仓库里长期留存的运行产出。**原始决策记录（`decisions.jsonl.gz`，520 MB /
3,594,000 条）体积过大，未入仓库**——它们留在目标机 `/tmp/qgate/runs/<runid>/`，
可用 `--out <dir>` 重跑复现（约 28 分钟单核）。

## 运行一览

| runid | 日期 | 说明 |
|---|---|---|
| `full_20261102`（旧） | 11-02 | 首次全网格。**ABL 块无效**（驱动未分派 variant，全跑 `GateMode::Full`） |
| `full_20261103_041848`（**现行**） | 11-03 | 修好 `--out`、`mode` 分派、AC-29 弃权规则后的重跑。**以这一次为准** |

## 文件

| 文件 | 内容 | 行数 | SHA-256 |
|---|---|---:|---|
| `aggregates_full_grid_20261103_run2.jsonl` | **现行**全网格逐格聚合 | 1,614 | `b1b64292…f7eb9686` |
| `aggregates_full_grid_20261102.jsonl` | 首次运行（仅存档，ABL 块无效） | 1,614 | `d5910e49…fafe6d6b5` |
| `aggregates_abl_20261103.jsonl` | ABL 单独补跑（`--skip 199 --limit 40`） | 240 | `08fd6e04…7e6ecc70` |

每行格式：`{"cell_id": N, "rule": "GATE_Full", "reps": R, "selected": S}`；
`cell_id` → 块/参数的映射见 `configs/mc/mc_gate_core_v1.json` 的 `cells`。

## 两次运行的差异（已核对，可解释）

同一二进制连跑两遍**逐字节一致**（已实测 aggregates 与 decisions 均相同），
所以两次 run 的差异全部来自**代码变更**：

| 变更 | 影响面 | 实测 |
|---|---|---|
| `mode` 分派（ABL 四变体真正执行） | 仅 ABL 块 | 16.01% → **32.27%**（+16.26pp） |
| `make_tau` → `make_tau_k`（κ 来自 assign 子流，`mixed`/`mixed_signed` 按 κ 旋转） | MIX / DEC | −0.16 / +0.06pp（可忽略） |
| AC-29（收益 ≤−1.0 记 ABSTAIN） | 全部 | 未观察到可测影响 |
| 其余 10 个块 | — | **逐位相同** |

**结论：v1.5 的判定不作废**，唯一实质变化是 ABL 的消融数字变成真实值。

## 已闭合的核对

- **AC-15**：从 3,594,000 条原始记录重算 → 1,614 行聚合，**零不符**
  （`tools/mc_gate_recompute_check.py`，目标机上 10.4 秒）。

## 已知缺口

1. **`D=202` 与实际 `193` 不符**：差额 9 = `RET` 的 `K=10` 辅助面板（`ds=1..9`，`calls_per_rep=0`）。
   驱动未实现 `calls_per_rep`，`K=10` 每 rep 只生成 1 个面板。详见
   `docs/mc_gate_ac08_truth_recovery.md`。
2. 原始 `decisions.jsonl.gz` 只存在于目标机，**未入仓库**；重新生成需 28 分钟。

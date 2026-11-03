# MC-GATE-VALIDATION 运行产出归档

本目录保存**能在仓库里长期留存**的运行产出。原始决策记录（`decisions.jsonl.gz`，全网格 520 MB /
3,594,000 条）体积过大且曾被后续重跑覆盖，**未归档**——因此当前无法从原始记录重算表格（AC-15 未闭合）。

| 文件 | 内容 | 行数 | SHA-256 |
|---|---|---:|---|
| `aggregates_full_grid_20261102.jsonl` | 全网格首次运行（269 cell × 6 规则）的逐格聚合 | 1,614 | `d5910e49…fafe6d6b5` |
| `aggregates_abl_20261103.jsonl` | ABL 消融补跑（40 cell × 6 规则） | 240 | `08fd6e04…7e6ecc70` |

## 每行格式

```json
{"cell_id": 0, "rule": "GATE_Full", "reps": 8000, "selected": 20}
```

`cell_id` 到块/参数（`block`/`T`/`N`/`rho`/`S`/`composition`/`variant`/`ds`）的映射见
`configs/mc/mc_gate_core_v1.json` 的 `cells` 数组。

## 已知缺口

1. **原始 `decisions.jsonl.gz` 已丢失**：全网格那份被 ABL 补跑覆盖（驱动把产物写死在同一路径）。
   要恢复需重跑全网格（约 28 分钟单核），并把输出改到按 runid 分目录。
2. 因此 §`docs/mc_gate_full_run_results.md` 里的表格只能从 `aggregates_*.jsonl` 复算，
   无法与原始逐条记录交叉核对。

# Phase 1D 状态

日期：2026-08-03
状态：`REAL_BASELINE_RESEARCH_PROXY_EXIT_COMPLETE / PROMOTION_BLOCKED`

## 已完成

- `DriftMonitorSpecV1`、training-static/reference window、fast/confirm 语义、available-at 检查、
  raw/feature/prediction/label drift、PSI/KS/moments、joint drift、stationary bootstrap、BH/BY、
  alert persistence 和 deterministic report hash 已实现。
- C++ `DriftSnapshotContractV0`、Python `FrozenDriftSnapshot` 只读校验桥已存在。
- `tools/run_phase1d_real_baseline.py` 使用 PIT OHLCV、真实 Phase 1E ranking score、真实成熟 label
  衍生的 forward return 和因果派生特征生成可复用真实 baseline report。
- 当前 baseline 会显式报告 universe/raw/feature/ranking/label/embedding drift；长期不可得执行字段只
  输出 `UNAVAILABLE/PROXY` availability，不会用合成值填充。
- Phase 1E e4 的 validation/test artifacts 已补齐六个 prediction heads、64 维 embedding snapshot、
  固定 anchor 和逐文件 SHA-256 manifest；reference/current manifest 的跨 split 兼容字段校验通过。
- embedding drift 为 `OK`，固定 anchor 的 linear CKA 为 `1.0`；embedding 仅作 observation-only 诊断。

## 外部执行字段不再阻塞；研究退出已完成

当前报告复用真实 `BAR_V1` model-ready feature pipeline（23 个特征，schema hash 已冻结），六输出和
embedding 层均可用；`formal_exit=true`、`phase_exit_eligible=true`、`promotion_eligible=false`。
Phase 1C 的
reference/cost/limit/lot/corporate-action 缺失只作为 `RESEARCH_PROXY` limitation，不形成研究阶段
blocker；monitor 是 observation-only，不改变 checkpoint、policy、risk estimator 或 C++ order/fill/equity。

本轮基线报告：`runs/phase1d-real-baseline/phase1d_real_baseline_report.json`，报告 SHA-256：
`0d9a516e8dd2f77914b5741144da349def9097dddd44ffd177685e7a04d4ec6f`。

## 运行

```text
python3 tools/run_phase1d_real_baseline.py
```

输出：`runs/phase1d-real-baseline/phase1d_real_baseline_report.json`。

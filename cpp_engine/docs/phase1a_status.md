# Phase 1A 状态：C++ 风险预算纵向切片

更新时间：2026-08-08

## 当前判定

Phase 1A A1/A2 数学与研究 Replay closure 已完成：

```text
engineering_scaffold_complete = true
phase_exit_eligible = true
promotion_eligible = false
evidence_level = RESEARCH_REFERENCE
```

## 已完成

- `LW-LIN-CC`、sample covariance、风险贡献、long-only Risk Budget 和 Top-K equal-weight policy 已接入
  固定输入 closure runner。
- `LW-NLS-MV-QUEST` forward/inverse reference、完整 shrinkage diagnostics、`p<n` regular branch 和
  `p>n` singular/null-space branch 均通过固定 oracle。
- 固定输入配对报告同时记录三种 estimator 的 covariance hash、shrinkage/concentration diagnostics、
  Top-K/Risk-Budget 权重、权重和与风险预算误差。
- QuEST 低迭代/无效配置保持原 estimator identity 并失败关闭，不回退到 LW-LIN-CC。
- closure artifact 是确定性 JSON，带 numeric artifact hash 和 SHA-256 sidecar；`promotion_eligible` 固定为
  `false`。

## 证据

- runner：`portfolio_math/tests/phase1a_closure_report.cpp`
- CTest：`test_phase1a_closure`
- C++ Replay CTest：`test_phase1a_replay`（BacktestEngine → policy → reconciler → order/fill → fee → ReturnLedger → empirical CVaR）
- 全量 CTest：`46/46` 通过（`build/phase1c-python`，2026-08-08）
- report：`runs/phase1a-closure/phase1a_closure_report.json`
- report SHA-256：`1005e62cb3b36d478f0a43ceca248d91a62cf08fe5eb8ef55fc4e47769787567`

## 晋级边界

Phase 1A 研究退出不等于生产风险或交易经济晋级。历史费用/税费、涨跌停、公司行动/复权、lot、真实
滑点和正式 reference-price provenance 仍由 Phase 1C/生产 gate 管理；当前不得把 reference/research
结果改称 official risk 或 production net return。

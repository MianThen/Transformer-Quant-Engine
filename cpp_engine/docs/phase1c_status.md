# Phase 1C 状态

日期：2026-08-03
状态：`ENGINEERING_COMPLETE_RESEARCH_PROXY / ECONOMIC_PROMOTION_DEFERRED`

## 已完成

- C++ `ReturnLedger`：period return、毛/净收益、费用、公司行动现金项、现金利息、会计残差和
  deterministic ledger hash；非法 period、外部现金流和会计不一致均失败关闭。
- C++ implementation shortfall、period contribution、security/active attribution、return metrics、
  paired stationary bootstrap、Newey-West/HAC、effective trials 和 DSR 组件已实现并通过现有
  C++ 测试。
- C++ proxy Replay 已真实调用 `BacktestEngine`、`PeriodContributionReplaySink` 和 empirical CVaR，
  现有六个 PCGrad/GradNorm fold report 均有 ledger、return analysis 和 `report_sha256`。
- `tools/audit_phase1c_closure.py` 会验证六个 proxy report 的 C++ Replay/CVaR、研究退出/比较 gate、
  ledger/analysis promotion gate、period return 对齐、字段 availability、hash 和 limitation 完整性，
  并生成 closure manifest。

## 研究工程退出已通过；生产经济晋级延后

当前数据可以生成可复现的研究结果，但不能证明生产净收益或生产 CVaR：

- reference/fill 使用 bar-close proxy；
- 没有可冻结的佣金/印花税 schedule；
- 没有涨跌停字段；
- 没有公司行动/复权 provenance；
- 没有 board-lot provenance；
- 没有 slippage provenance；
- 没有正式 benchmark holdings/reference provenance。

因此 closure manifest 的 `formal_exit=true`、`phase_exit_eligible=true`、
`research_comparison_eligible=true`、`promotion_eligible=false`；proxy 回放只能作为
`RESEARCH_PROXY_ONLY`，不得静默升级为生产经济证据。

## 验证

```text
python3 tools/audit_phase1c_closure.py
```

输出：`runs/phase1c-closure/phase1c_closure_report.json`。

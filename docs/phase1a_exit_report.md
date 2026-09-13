# Phase 1A 研究工程退出报告

## Gate

| 门槛 | 状态 | 证据 |
|---|---|---|
| A1 sample/LW-LIN-CC covariance 与 diagnostics | PASS/REFERENCE | `phase1a_closure_report.json` |
| Risk contribution 与 long-only Risk Budget | PASS/REFERENCE | 三个 estimator 均输出可重放 Risk Budget 权重 |
| Top-K / Risk-Budget policy switch | PASS/REFERENCE | closure runner 对三种 estimator 均完成两种 policy |
| QuEST forward/inverse reference | PASS/REFERENCE | `test_portfolio_math` 与 closure runner |
| `p<n` regular branch | PASS/REFERENCE | closure `regular_p_lt_n=true` |
| `p>n` singular/null-space branch | PASS/REFERENCE | closure `singular_p_gt_n=true` |
| QuEST numerical failure fail-closed | PASS/REFERENCE | invalid inverse configuration 无 fallback |
| 固定输入三估计器配对报告 | PASS/REFERENCE | input fingerprint、covariance hash、artifact hash、SHA-256 sidecar |
| C++ research Replay closure | PASS/REFERENCE | `test_phase1a_closure` + `test_phase1a_replay`；全量 CTest `46/46` 通过 |
| 生产经济 gate | DEFERRED | reference-price/cost/execution provenance 仍不可用或为 proxy |

## 判定

```json
{
  "engineering_scaffold_complete": true,
  "phase_exit_eligible": true,
  "promotion_eligible": false,
  "evidence_level": "RESEARCH_REFERENCE",
  "blocking_reason": "production_economic_provenance_deferred"
}
```

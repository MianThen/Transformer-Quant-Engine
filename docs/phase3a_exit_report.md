# Phase 3A 退出报告（当前版本）

## Gate

| 门槛 | 状态 | 证据 |
|---|---|---|
| PIT exposure / timestamp future guard | PASS | `test_factor_model` |
| PIT exposure source manifest contract | PASS | source schema/snapshot/provenance hash、覆盖区间、available-at 与 PIT 标记校验；future/unavailable 关闭测试 |
| Factor-return WLS residual orthogonality | PASS | artifact diagnostics + oracle |
| Shrunk EWMA factor covariance PSD | PASS | eigenvalue/PSD test |
| Specific variance shrinkage and floor | PASS | floor diagnostics |
| Factor-form/dense variance and gradient parity | PASS | finite difference + materialized diagnostic |
| Factor-form optimizer | PASS | simplex/box, deterministic convergence |
| Attribution V2 accounting PnL reconciliation | PASS | success/failure/future tests |
| Factor payload provenance hashes | PASS/REFERENCE | `test_factor_model`；exposure/covariance/specific variance 独立 hash 已写入 artifact JSON |
| Real PIT factor source and provenance manifest | NOT RUN / SOURCE_UNAVAILABLE | 全量审计 `69,358` 个 Baostock parquet；唯一 `BAOSTOCK_DAILY_STATE_V1` schema 不含 industry/style 或显式 exposure availability，`source_schema_hash=24873c819c1c34ccc3ea01b1604ca13f9ea46bde4d78b22c478be6e4addb85d6`，未生成真实来源 artifact |
| Three purged OOS risk windows | NOT RUN | 尚无真实 OOS 报告 |
| Trading-cost-after-risk economic gate | NOT RUN | 历史执行字段仍为 proxy/unavailable |

## 判定

```json
{
  "engineering_scaffold_complete": true,
  "phase_exit_eligible": false,
  "promotion_eligible": false,
  "evidence_level": "RESEARCH_ONLY",
  "blocking_reason": "missing_pit_factor_oos_and_economic_evidence"
}
```

本报告只确认模型合同、会计对账和数学 parity，不声称已复制商业 Barra/USE4 规范，也不把 proxy execution 结果称为正式净收益。后续 `FACTOR-PIT-VRA`、dynamic loading 和 regime covariance 必须分别注册、分别验收，失败候选不得通过联合模型掩盖。

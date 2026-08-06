# Phase 4A 退出报告（当前版本）

## Gate

| 门槛 | 状态 | 证据 |
|---|---|---|
| ViewSpec V1 schema、hash、available_at guard | PASS | `portfolio_math/posterior.h`、`test_posterior` |
| PIT prior-scenario builder、PSD、future guard | PASS | `build_prior_scenario_artifact`、`test_posterior` |
| Gaussian BL no-view/zero/full-confidence/relative-mean | PASS | `test_posterior` |
| FFV equality mean-view no-view/zero/full-confidence | PASS/REFERENCE | `test_posterior`；只重加权既有 scenarios |
| FFV KL/ESS/max-weight/view residual | PASS/REFERENCE | `PosteriorScenarioArtifactV1` 与确定性 hash |
| FFV support/infeasible/repeated-view/mapping guard | PASS/REFERENCE | `test_posterior` fail-closed cases |
| posterior mean/covariance/quantile/ES 重算 parity | PASS/REFERENCE | 同一 posterior probability vector 独立重算 |
| FFV mean lower/upper bound | PASS/REFERENCE | 仅对被触发 bound 重加权；inactive bound 保持 prior parity，KKT 符号/近共线失败关闭 |
| FFV coupled active-set bounds | PASS/REFERENCE | `test_posterior`：双资产同时约束、active count、residual 和 artifact contract |
| Posterior Direct fixed downstream anchor | PASS/REFERENCE | 固定 capped-simplex projected-gradient；BL/FFV 只替换 posterior artifact |
| direction/ranking/quantile/volatility views | NOT IMPLEMENTED | 后续独立 solver/oracle |
| 完整 inequality active-set solver | NOT IMPLEMENTED | 当前冲突或负对偶 multiplier 一律失败关闭 |
| BL vs FFV Posterior Direct policy 对照 | NOT RUN | policy 已冻结，OOS/cost 尚未运行 |
| 三个新 purged OOS、成本与稳定性 gate | NOT RUN | 当前没有 Phase 4A formal OOS |

## 判定

```json
{
  "engineering_scaffold_complete": true,
  "phase_exit_eligible": false,
  "promotion_eligible": false,
  "evidence_level": "REFERENCE_ONLY",
  "blocking_reason": "rich_views_active_set_downstream_policy_and_formal_oos_missing"
}
```

当前实现证明 CPU 数学 reference 和失败关闭边界，不代表 FFV 已通过生产 gate，也没有接入未校准的
direction/volatility/ranking/quantile head。GPU 训练包状态不因 Phase 4A 改变。

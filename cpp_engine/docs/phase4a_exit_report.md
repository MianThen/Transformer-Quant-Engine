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
| Rich view calibration/fail-closed contract | PASS/REFERENCE | `test_posterior`：未校准 rich view 拒绝；已校准元数据可审计但暂不进入 mean solver |
| Direction FFV probability equality oracle | PASS/REFERENCE | `test_posterior`：threshold event probability、confidence target、support/KL/ESS 与 solver fail-closed |
| Volatility FFV second-moment oracle | PASS/REFERENCE | `test_posterior`：固定中心二阶矩、目标方差转换、posterior 重算与 solver fail-closed |
| Ranking FFV pairwise probability oracle | PASS/REFERENCE | `test_posterior`：relative loading、strict-margin tie policy、目标 outrank probability 与 solver fail-closed |
| Posterior Direct fixed downstream anchor | PASS/REFERENCE | 固定 capped-simplex projected-gradient；BL/FFV 只替换 posterior artifact |
| direction view | PASS/REFERENCE | 校准后的 threshold event probability equality；尚无正式 OOS |
| direction/volatility/ranking views | PASS/REFERENCE | 独立 scenario-function oracle；尚无正式 OOS |
| Quantile FFV CDF oracle | PASS/REFERENCE | `test_posterior`：target 下 CDF equality、`<=` tie policy、quantile 重算与浮点容差 |
| Explicit inequality active-set adapter | PASS/REFERENCE | `apply_ffv_active_set_views`：mixed active/inactive、KKT sign、support guard、冲突 fail-closed |
| BL/FFV fixed-downstream policy comparison | PASS/REFERENCE | `test_posterior_direct`：no-view parity、收益/方差/objective/L1 diagnostics、artifact hash 与 deterministic replay |
| 完整 inequality active-set solver | PASS/REFERENCE | lower/upper bound adapter 与多约束 KKT 参考路径；尚无正式 OOS |
| BL vs FFV Posterior Direct policy 对照 | PASS/REFERENCE | 固定 downstream reference 已完成；正式 OOS/cost 尚未运行 |
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

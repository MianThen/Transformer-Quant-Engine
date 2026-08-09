# Posterior Direct V1 合同

## 目的

`Posterior Direct` 是 Phase 4A 固定的下游 policy reference。BL 与 FFV 只允许改变
`PosteriorScenarioArtifactV1`，不得改变本 policy 的优化器、约束、迭代规则或 reconciler 配置。

## Anchor

给定 posterior mean `mu`、posterior covariance `Sigma`，求解 long-only、fully-invested、可选单资产上限的
确定性 capped-simplex 问题：

```text
maximize_w  mu' w - 0.5 * risk_aversion * w' Sigma w
subject to 0 <= w_i <= max_single_weight
           sum_i w_i = target_investment
```

求解使用固定步长 projected-gradient 和 capped-simplex projection；不使用交易成本、未来收益、detoned
covariance 或运行时 fallback。交易成本与硬约束只由既有 `SinglePeriodReconciler` 处理。

## 诊断与边界

- 保存 posterior artifact hash、iterations、KKT/fixed-point residual、weight sum、expected return、variance
  和 objective。
- posterior artifact 无效、单资产上限不可行、矩阵非有限或未收敛时失败关闭并不输出 anchor。
- 当前结果是 `REFERENCE_ONLY`，`eligible_for_official_risk=false`；尚未接入真实 OOS、成本 gate 或生产订单。

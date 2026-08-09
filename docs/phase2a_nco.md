# Phase 2A：NCO-MinVar / NCO-RiskBudget

## 当前状态

NCO 的第一版研究切片已完成，状态为 `RESEARCH_ONLY`。它消费 ONC/cluster artifact 提供的
`cluster_id_by_symbol` 结构，但优化过程只使用冻结的 `official_covariance`；cluster discovery
correlation 不会覆盖 official risk covariance。

## NCO-MinVar

NCO-MinVar 分两层求解 long-only simplex minimum variance：

1. 每个 cluster 内使用确定性的 projected-gradient simplex solver 求局部最小方差组合；
2. 将局部组合映射成 cluster covariance，再求 cluster-level 最小方差组合；
3. 最终资产权重等于 cluster weight 与局部 asset weight 的乘积。

求解器固定 `target_investment`、最大迭代次数和收敛容差；非有限、非对称、非 PSD 或空 cluster
输入失败关闭，不回退到 HRP、Top-K 或上一期权重。

## NCO-RiskBudget

NCO-RiskBudget 同样采用两层结构：cluster 内使用等风险预算，cluster 间使用调用方预先登记且和为
`1` 的 `cluster_risk_budgets`。两层都复用 `solve_long_only_risk_budget`，并记录最大风险预算误差、
迭代次数、预测风险和 cluster weights。

## 研究边界

- `eligible_for_official_risk=false`；当前不覆盖 `LW-LIN-CC`、`LW-NLS-MV-QUEST` 或 Risk Budget
  冻结 baseline。
- 当前不处理成本、换手、行业上限、成交量、lot、涨跌停、滑点或账户现金硬约束；这些交给后续
  `PortfolioPolicyArtifact V1` 和 single-period reconciler。
- 当前不接入正式 CVaR、Replay 或生产 policy fallback。

## 验证

`test_nco_policy` 覆盖：

- NCO-MinVar 的两层组合、simplex 和风险输出；
- NCO-RiskBudget 的 cluster budget 与误差诊断；
- 资产置换等变性；
- 空 cluster、预算和不为一、非 PSD covariance 的失败关闭。

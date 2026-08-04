# Phase 2A：RMT Constant-Residual

## 目的

在标准化收益相关矩阵上使用 Marcenko–Pastur 上界区分 signal 与 noise eigenvalues，将 noise
子空间替换为保持原始 noise trace 的常数残差，再恢复单位对角 correlation 并映射回原始波动率
协方差。

## 输入与合同

- 输入为有限、非空、至少两个观测的收益矩阵；每列代表一个资产。
- 收益先按列去均值，协方差使用 `1/T` 约定，与现有 covariance estimator 保持一致。
- `RmtDenoisingSpec` 的 `edge_tolerance`、`eigenvalue_floor` 和 `psd_tolerance` 必须有限且有效。
- noise residual 若低于 `eigenvalue_floor`，算法返回 `NUMERICAL_FAILURE`，不通过抬高 residual
  伪造 trace preservation；这覆盖精确低秩 `p>T` 输入。
- RMT 的 `RiskPreprocessorSpec` 必须有正的 concentration guard、正的 eigenvalue floor 和非零
  `rmt_spec_hash`；QuEST hash 必须为零，loss profile 必须为 `NOT_APPLICABLE`。

## 诊断

结果记录：

- `aspect_ratio` 与 MP noise boundary；
- raw/cleaned eigenvalues；
- retained signal rank 与 noise eigenvalue count；
- constant residual eigenvalue；
- 输入/输出 correlation trace 与 trace drift；
- 最大相对 covariance diagonal drift；
- condition number、PSD repair amount 和 `eligible_for_official_risk`。

恢复后的 correlation 被强制对称并恢复单位对角；协方差使用输入样本的原始波动率向量。RMT
诊断不写入 cluster artifact，也不允许 detoned correlation 进入 official risk、risk contribution、
CVaR 或 reconciler。

## 研究状态

该实现只完成数学施工和单元 oracle，不代表 RMT 已经晋级。路线图要求同时通过 synthetic spectrum、
独立 oracle、OOS variance forecast、组合稳定性和成本指标；在这些证据形成前，RMT 保持
`RESEARCH_ONLY`，冻结的 `LW-LIN-CC` 与 `LW-NLS-MV-QUEST` 不受影响。

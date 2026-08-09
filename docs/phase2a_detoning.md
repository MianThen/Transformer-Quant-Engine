# Phase 2A：Cluster-Only Detoning

## 目的与边界

`detone_correlation` 从输入 correlation 中移除预注册的最大主成分；V1 仅允许
`components in {0, 1}`。输出重新标准化为单位对角并验证 PSD，职责仅是 cluster discovery 输入。

detoned correlation 不得覆盖 `RiskModelArtifact`，不得进入 official predicted risk、risk contribution、
CVaR 或 reconciler。`eligible_for_official_risk` 固定为 `false`。

## 数值合同

- 输入必须为有限、非空、方阵且对称的 correlation-like matrix；正对角会先标准化为单位对角。
- 主成分移除后若任一对角线退化到 `eigenvalue_floor` 以下，返回 `DEGENERATE_OUTPUT`，不增加隐式
  jitter 或人工行业簇修复。
- 输出必须有限、对称、单位对角和 PSD；所有 tie-break 及 cluster assignment 由后续 linkage/ONC
  层负责，本 API 不生成平面 cluster。
- 诊断保留 raw/detoned eigenvalues、移除组件数、trace drift、最小输出 eigenvalue 和 PSD repair
  amount。

当前实现仅完成 detoning 数学切片；hierarchical linkage、ONC partition、`ClusterModelArtifact V1`
及其 rolling stability 尚未实现。

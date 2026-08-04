# Phase 2A：ClusterModelArtifact V1

## 目的

`ClusterModelArtifact` 统一保存 HRP linkage 和 ONC partition 的可重放结构，但明确区别两种 variant：

- `HRP_HIERARCHICAL_LINKAGE`：保存 linkage method、merge-tree hash 和 quasi-diagonal order；不保存平面
  cluster assignment；
- `ONC_PARTITION`：保存 ONC spec、K 搜索/seed/repeat、cluster assignment、silhouette/quality 和
  quasi-diagonal order；不能当作 HRP dendrogram。

## Provenance 合同

artifact 校验 `schema_version=1`、official/denoised risk hash、correlation source、detone components、
symbol 唯一有序性、fit/available 时间顺序、所有 hash 字段、variant 专属字段和完整 permutation order。
`finalize_cluster_model_artifact` 生成确定性的 artifact hash；修改已 finalize 的字段会被 tamper 检测
拒绝。序列化 manifest 保留 `symbols`、`onc_spec`、`cluster_id_by_symbol`、`quasi_diagonal_order`、
quality/silhouette、时间戳和 artifact hash。

artifact 只用于复现 HRP/NCO 的结构输入，不能被风险报告当作 covariance 来源；
`eligible_for_official_risk=false`，也不进入 CVaR 或 reconciler。

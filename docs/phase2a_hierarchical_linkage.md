# Phase 2A：Hierarchical Linkage

## 目的与合同

`hierarchical_linkage` 消费 raw、RMT-denoised 或 denoised-detoned correlation，使用冻结距离：

```text
d_ij = sqrt((1-rho_ij)/2)
```

V1 固定为 deterministic complete linkage。每次合并先比较 cluster distance，再用
`(min(cluster_id), max(cluster_id))` 做 tie-break；新 cluster id 按 merge 顺序递增。结果记录完整
merge tree、每个 merge 的 distance/size 和 quasi-diagonal order。

## 边界

- 输入必须有限、对称、正对角且归一化后为 PSD correlation；无效输入失败关闭。
- `detoned` 只改变 cluster discovery 输入；linkage 结果不能被解释为 official risk covariance。
- `eligible_for_official_risk=false`，linkage hash 和输入 correlation hash 用于后续
  `ClusterModelArtifact V1` provenance。
- 当前只生成 HRP 所需的 dendrogram/order；尚未生成 ONC 平面 partition，也未接入 HRP/NCO policy。

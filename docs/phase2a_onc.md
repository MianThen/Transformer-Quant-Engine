# Phase 2A：ONC Partition

## 目的与固定合同

`onc_partition` 在给定 correlation/distance 上独立搜索平面 cluster 数量，不把 hierarchical linkage
dendrogram 当作 ONC partition。V1 使用 deterministic k-medoids：

- 固定 `min_clusters`、`max_clusters` 和 `min_cluster_size`；
- 固定 seed 集、repeat 数、最大迭代次数和距离 tie-break；
- 用相关距离 silhouette 均值作为 quality statistic；
- quality 相同时优先较小 `K`，再按 canonical assignment 字典序决定。

输出包含 `cluster_id_by_symbol`、按 cluster/id 排序的 quasi-diagonal order、每个 K/seed 的 quality 与
silhouette 范围、最优/次优差距、输入 correlation hash 和 partition hash。

## 边界

- 输入归一化为 unit-diagonal、有限、对称、PSD correlation；无效输入和无可行 partition 失败关闭。
- 不能满足最小簇大小的 K 不进入有效候选；不临时改用人工行业簇。
- ONC partition 是 `ClusterModelArtifact V1` 的候选结构输入，不是 HRP merge tree，也不提供 official
  risk covariance。
- `eligible_for_official_risk=false`；ONC 结果尚未接入 NCO、HRP、CVaR 或 reconciler。

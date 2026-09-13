# ViewSpec V1 与 PriorScenarioArtifact V1 合同

## 目的

本合同只冻结 Phase 4A 的 CPU reference 输入边界、Gaussian equality mean-view，以及 FFV 的 equality/
单条 activated mean-bound 数学；不宣称完成 rich-view FFV 或生产 posterior policy。

## ViewSpec V1

必须提供：`view_id`、`kind=mean|mean_lower_bound|mean_upper_bound`、`available_at`、长度等于资产数的 `loading`、`target`、
`confidence∈[0,1]`、正的 `observation_variance` 和非零 `source_artifact_hash`。`available_at` 必须不晚于
`decision_at`；还必须提供非零 `confidence_mapping_hash`。未知、未校准或未来 view 直接
`INVALID_INPUT/FUTURE_DATA`；同一 posterior 内混用 mapping hash 直接拒绝。

有效 confidence 的噪声映射为 `Ω = observation_variance * (1-confidence)`；confidence=0 不激活该 view，
confidence=1 进入零噪声极限。没有任何 active view 时 posterior 的 mean/covariance 必须与 prior 完全相同。

Gaussian BL 只接受 `mean` equality view。FFV 对 `mean_lower_bound` 使用 `E[P·R] >= target_eff`，对
`mean_upper_bound` 使用 `E[P·R] <= target_eff`；若 prior 已满足 bound，则不重加权。当前是确定性
reference：所有被触发的不等式同时以 equality 解出，若任一对偶 multiplier 为负、views 近共线或数值奇异，
直接失败关闭，不冒充完整 active-set solver。

## PriorScenarioArtifact V1

输入为 `[M,N]` PIT scenario return matrix 和严格递增的 `M` 个 timestamp。所有 timestamp 必须不晚于
数据 `available_at` 与 `decision_at`，`M≥2`，只使用有限值。prior mean 为等权样本均值，covariance 为
`1/(M-1)` 样本协方差；artifact 保存完整 scenario timestamps、scenario payload、componentwise support min/max、
scenario hash、fit interval 和 effective sample size。

## PosteriorScenarioArtifact V1

artifact 保存完整 scenario timestamps、prior support min/max、prior/posterior mean/covariance、view set hash、
view residual、KL availability、support guard 字段和确定性 artifact hash。FFV 的 `min_probability` 是每条
scenario 的正概率下限；在该下限下无法达到的 support 边界 target 必须在迭代前标记为 `INFEASIBLE`，不会用
近零权重静默伪造全置信度。当前所有产物强制 `eligible_for_official_risk=false`。

固定 downstream 使用 `docs/phase4a_posterior_direct_contract.md` 的 `Posterior Direct V1`；BL 与 FFV 的 OOS
对照必须复用同一 capped-simplex solver、risk aversion、weight cap、target investment 和 reconciler spec。

## 边界

本合同不提供 reference-price provenance、费用/税费、涨跌停、lot 或真实滑点；也不把这些缺失字段填充为
零值。当前 FFV 只支持 equality mean 与有限的 mean-bound reference；生产晋级仍需独立的执行经济合同和真实数据。

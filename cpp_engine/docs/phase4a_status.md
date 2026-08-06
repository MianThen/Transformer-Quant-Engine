# Phase 4A 状态：Distributional Posterior

更新时间：2026-08-06

## 当前判定

Phase 4A 已开始 CPU 数学施工，目前完成第一阶梯合同、Gaussian equality mean-view，以及 FFV equality/
mean-bound reference：

```text
engineering_scaffold_complete = true
phase_exit_eligible = false
promotion_eligible = false
```

## 本轮交付

- `portfolio_math/include/portfolio_math/posterior.h` / `src/posterior.cpp`：`ViewSpecV1`、view hash、
  confidence-mapping provenance、PIT 时间门禁的 prior-scenario builder、`PosteriorScenarioArtifactV1`、
  Gaussian equality mean-view 更新，以及 FFV equality/mean-bound 对偶 Newton/线搜索 reference。
- Prior builder 要求 scenario timestamp 严格递增、全部不晚于 `available_at`/`decision_at`，有限值、至少两条
  scenario、PSD covariance 和确定性 hash；prior/posterior 均保留完整 scenario timestamps 与 support min/max，
  时间篡改或未来输入失败关闭。
- mean-view 采用
  `m_post = m_prior + ΣP'(PΣP' + Ω)^-1(q - Pm_prior)`，confidence=0 等价 no-view，confidence=1
  保留 full-confidence 极限（允许 posterior covariance 半正定），重复 view 不做运行时 fallback。
- FFV 只重加权既有 scenarios，固定使用 `target_eff = prior_view + confidence * (target - prior_view)`；保存
  posterior probabilities、`KL(p||q)`、ESS、最大场景权重、view residual、support guard、posterior
  quantile 和 lower-tail ES。FFV 可处理被触发的 mean lower/upper bound；prior 已满足的 bound 保持 no-view
  parity，Gaussian BL 明确拒绝不等式 view。
- `min_probability` 定义为每个 scenario 的正概率下限：无法满足该下限的 support boundary 在 Newton 前
  `INFEASIBLE`；重复、近共线/奇异 views、超出 prior support、future view、混用 confidence mapping hash 或
  不满足 inequality KKT 符号的对偶解均失败关闭。模型输出的
  confidence 没有独立 mapping hash 时不能进入后验。
- `test_posterior` 覆盖 prior/no-view parity、zero/full-confidence、FFV 重算 parity、future/support、
  positive-probability boundary、duplicate/near-collinear view、mean lower/upper bound、Gaussian inequality
  rejection 和 mapping provenance fail-closed。
- 新增 coupled active-set oracle：同一 FFV posterior 同时满足两个资产的 lower/upper mean bound，
  并验证 active constraint count、view residual 与 posterior artifact contract。
- 新增 rich view family contract：direction、volatility、ranking、quantile view 必须绑定非零
  calibration artifact hash；在对应 solver 尚未实现前，Gaussian/FFV mean solver 明确拒绝这些 family，
  防止未校准 head 进入后验。
- 新增 direction FFV equality oracle：以 loading 组合收益超过 statistic threshold 的事件概率作为
  scenario function，支持校准目标概率、confidence mapping、support/KL/ESS/hash 重算；Gaussian BL 和
  mean-only FFV wrapper 对 direction view 失败关闭。
- `portfolio_math/posterior_direct.h` / `src/posterior_direct.cpp`：固定 long-only、fully-invested、capped-simplex
  mean-variance `Posterior Direct` anchor；BL/FFV 只替换 posterior artifact，cost/hard-constraint 仍交给同一
  `SinglePeriodReconciler`。
- `test_posterior_direct` 覆盖 posterior artifact hash provenance、simplex/cap、higher posterior mean preference
  和 infeasible/invalid options fail-closed；本地全量 CTest 为 `29/29 passed`。

## 验证

使用本地已有 Eigen source cache 配置的独立构建目录：

```text
/tmp/qbt-phase4a-build-local/portfolio_math/test_posterior
=> test_posterior: all checks passed
```

## 尚未完成

全量 CTest 已使用 CLion bundled CMake 重建验证：29/29 tests passed（含 direction FFV oracle）。

- direction 与 volatility FFV 已有 reference：volatility 采用固定中心的二阶矩约束，目标波动率在
  solver 内平方为目标方差；ranking 采用 pairwise loading 超过预注册 margin 的 outrank probability，
  ties 使用严格大于策略；尚无 quantile view、完整 active-set adapter、BL-vs-FFV Posterior Direct
  OOS 对照或真实成本 gate；当前 mean inequality 和 Posterior Direct 都是严格失败关闭的 reference。
- 当前只允许已校准的 direction/volatility reference；未校准 direction/volatility/ranking/quantile head 不得接入 view。
- 当前 artifact 明确 `eligible_for_official_risk=false`，不构成 Phase 4A 退出或生产/实盘晋级。

下一步按路线图补 quantile view 的独立 oracle，完成 inequality active-set adapter，再用已固定的
Posterior Direct policy 做 BL/FFV 对照；真实 OOS 和成本 gate 之前不晋级。

# Phase 4B 状态：NCO-FFV 与 policy-family governance

更新时间：2026-08-08

## 当前判定

Phase 4B 已开始 CPU reference 施工，但尚未形成正式 OOS 或生产晋级证据：

engineering_scaffold_complete = true

phase_exit_eligible = false

promotion_eligible = false

## 本轮交付

- 新增 solve_nco_ffv_minvar，将已校验的 PosteriorScenarioArtifactV1.posterior_covariance
  接入既有 NCO-MinVar cluster/intra-cluster/inter-cluster solver。
- NCO-FFV 结果固定保存 posterior artifact hash、cluster specification hash、NCO diagnostics、
  policy artifact hash 和 eligible_for_official_risk=false。
- 新增 deterministic serialization 与 replay test；无 posterior、cluster shape、future/status
  或 NCO options 错误时失败关闭。
- 当前实现只证明 posterior covariance → NCO-MinVar 的 reference parity，不自动选择 policy winner，
  不修改现有 reconciler，也不把 proxy covariance 变成 official risk model。
- 新增 `PolicyFamilyGovernanceArtifactV1`：在同一 policy family 的 trial-major return 上生成相关矩阵、
  ONC policy clustering、effective trial count、BH/BY/Storey FDR 和逐策略 DSR；未经注册 formal OOS、
  经济 gate 与 final untouched freeze 的输入不能冻结 winner，且所有当前结果固定
  `promotion_eligible=false`。
- NCO 增加簇内/簇间 objective 单层替换消融：`FULL`、`INTRA_ONLY`、`INTER_ONLY`；
  消融模式与 intra/inter enabled flags 固定进入 FFV artifact hash 和 JSON provenance，
  非法枚举失败关闭。
- 新增固定 downstream 三策略对照：Posterior Direct、NCO-FFV、NCO risk-only 统一经过同一
  `SinglePeriodReconciler`；risk-only 明确消费 prior covariance，结果保存三路 anchor/target、
  posterior 评价矩、成本/约束诊断、provenance hash，且不自动选择 winner。
- 治理 artifact 序列化已补齐独立审计 payload：policy IDs、原始 p-values、policy-return correlation、
  ONC partition/candidates、effective-trial diagnostics、FDR adjusted p-values/rejections、逐策略 DSR
  和全部 winner gate inputs；policy ID 采用 JSON escape，重复序列化保持确定性。
- `phase4b_governance_reference` runner 已生成固定输入 reference artifact：
  `runs/phase4b-governance-reference/phase4b_governance_reference.json`，report body SHA-256 为
  `8f109988ddc8fa493c1afda3103a4397644bd766e80ee2ff0ef57452f50d5f64`，文件 sidecar SHA-256 为
  `db467d86b2ea566fae088825afa39981d5a121dd1af80413272755f88b5f9e4d`；该产物明确保持
  `phase_exit_eligible=false`、`promotion_eligible=false`。

实现位置：

- portfolio_math/include/portfolio_math/nco_policy.h
- portfolio_math/src/nco_policy.cpp
- portfolio_math/tests/test_nco_policy.cpp
- portfolio_math/include/portfolio_math/nco_policy_comparison.h
- portfolio_math/src/nco_policy_comparison.cpp
- portfolio_math/tests/test_nco_policy_comparison.cpp
- portfolio_math/include/portfolio_math/nco_ffv.h
- portfolio_math/src/nco_ffv.cpp
- portfolio_math/tests/test_nco_ffv.cpp
- performance_analytics/include/performance_analytics/policy_family_governance.h
- performance_analytics/src/policy_family_governance.cpp
- performance_analytics/tests/test_policy_family_governance.cpp

## 验证

CLion bundled CMake 全量 CTest：

新增治理定向链：`test_policy_family_governance`、`test_return_analysis`、`test_multiple_testing`、
`test_onc_partition` 共 4/4 通过；`test_policy_family_governance` 额外校验完整审计 payload；本轮
全量 CTest 共 46/46 通过（含 `test_nco_policy_comparison`、`test_phase1a_replay`、
`test_phase4b_governance_reference`）。

## 尚未完成

- policy-family 的真实注册后 return 输入。
- 成本/约束 reconciler 后的经济门槛和 final untouched winner freeze。
- 当前所有 NCO-FFV 结果仍为 REFERENCE_ONLY，不能进入生产或实盘。

## 正式 OOS 终局判定（2026-09-12 登记）

按 `docs/phase4_ab_formal_oos_contract.md`（预注册）完成 266 期 walk-forward 正式 OOS
（PIT-119 宇宙、估计窗 252、5 点调仓、10 bps 单边成本、视图降级阶梯 13.5%）：

```text
formal_oos_complete = true
winner              = incumbent (NCO risk-only, prior covariance)
posterior_family    = NOT_PROMOTED   # 三 challenger 0/3 窗口、CI 全负、FDR 全拒
phase_exit_eligible = true           # 对照闭环完成
promotion_eligible  = false
```

证据：`experiments/phase4ab/formal-oos-20260912/`（GitHub）与
`docs/phase4ab/formal_oos_report_20260912.json`（本仓库）。至此 Phase 4A/4B 的
"NOT RUN" 项全部闭合：三 policy 对照、governance 诊断（FDR/DSR/子窗口）与成本后
经济对照均已执行并按规则冻结判定。

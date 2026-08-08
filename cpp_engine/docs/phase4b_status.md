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

实现位置：

- portfolio_math/include/portfolio_math/nco_ffv.h
- portfolio_math/src/nco_ffv.cpp
- portfolio_math/tests/test_nco_ffv.cpp

## 验证

CLion bundled CMake 全量 CTest：

30/30 tests passed

## 尚未完成

- NCO-FFV 与 Posterior Direct、NCO risk-only 的固定 downstream OOS 对照。
- 簇内/簇间 objective 单层替换消融、policy-return correlation、有效试验数、FDR/DSR。
- 成本/约束 reconciler 后的经济门槛和 final untouched winner freeze。
- 当前所有 NCO-FFV 结果仍为 REFERENCE_ONLY，不能进入生产或实盘。

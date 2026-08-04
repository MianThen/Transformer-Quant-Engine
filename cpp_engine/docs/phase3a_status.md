# Phase 3A 状态：Factor Risk 与 Alpha 提纯

更新时间：2026-08-04

## 当前判定

Phase 3A 第一阶梯的工程实现已落地，阶段整体尚未退出：

```text
engineering_scaffold_complete = true
phase_exit_eligible = false
promotion_eligible = false
```

尚未满足的部分是多窗口风险预测、交易成本后 OOS 和完整 PIT universe 证据；当前实现不把历史缺失的公司行动、复权、真实滑点或 reference provenance 伪装成已知数据。

## 已交付：FACTOR-PIT-EWMA

- `FactorModelSpec` 冻结 EWMA decay、factor/specific shrinkage、variance floor、PSD floor、annualization 和 schema/WLS/config hash。
- `FactorModelInput` 要求 PIT exposure、可选逐日 exposure history、asset-return timestamps、fit window、`available_at <= decision_at`；超出 fit window 的 future row 返回 `FUTURE_DATA`。
- 每个 return date 使用相同 PIT exposure 做截面 WLS，输出 factor returns 和 specific returns，并记录 WLS residual orthogonality。
- factor covariance 使用 shrunk EWMA，并执行对称 PSD repair；specific risk 使用 EWMA、横截面均值 shrinkage 和 floor。
- artifact 自带 exposures、factor returns、specific returns、factor covariance、specific variance、metadata 和 hash，输入 exposure 后续变更不会改变已产出的 artifact。
- `factor_model_artifact_hash` 与 `serialize_factor_model_artifact` 提供可复算 hash 和 metadata manifest；完整矩阵 payload 仍由 artifact 内存/上层存储负责保存。
- `FactorExposureSourceManifest` 冻结 source schema、snapshot、provenance、覆盖区间、可用时间和 PIT 标记；未知或未来来源会在建模入口关闭，不会被当作真实 exposure。

实现位置：

- `portfolio_math/include/portfolio_math/factor_model.h`
- `portfolio_math/src/factor_model.cpp`
- `portfolio_math/tests/test_factor_model.cpp`

## Factor-form 优化与 Attribution V2

- factor-form variance/gradient 使用 `(B' w)' F (B' w) + w' D w`，生产接口不需要物化 `N x N` covariance。
- long-only simplex/box 最小方差 solver 使用 matrix-free factor gradient；`materialize_factor_covariance` 只用于 parity/oracle 测试。
- Attribution V2 输出 factor contribution、specific contribution、portfolio factor exposure，并以 accounting portfolio return 做唯一总额对账。
- Attribution V2 对 PIT availability、reconciliation failure 和 artifact hash 均有显式状态。

实现位置：

- `performance_analytics/include/performance_analytics/factor_attribution_v2.h`
- `performance_analytics/src/factor_attribution_v2.cpp`
- `performance_analytics/tests/test_factor_attribution_v2.cpp`

## 验证

使用 `build/phase1c-python` 配置编译并执行 CTest：

- `32/32` C++ tests passed。
- 新增 `test_factor_model` 和 `test_factor_attribution_v2` 均通过。

测试覆盖 PIT/future mutation、source manifest future/unavailable guard、WLS orthogonality、PSD、specific floor、factor/dense variance parity、gradient finite difference、Attribution V2 accounting reconciliation 和失败关闭。

## 后续顺序

1. 接入可审计的 PIT industry/style exposure source，使用已冻结的 `FactorExposureSourceManifest` 生成真实 `FACTOR-PIT-EWMA` artifact 和 availability manifest。
2. 增加因子协方差 QLIKE、realized/predicted variance、risk-contribution error 和至少三个 purged OOS 窗口。
3. 只有 baseline 通过后，才单独研究 `FACTOR-PIT-VRA`；动态 loading、regime covariance 和 Kalman family 不得提前组合。

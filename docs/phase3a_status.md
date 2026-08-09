# Phase 3A 状态：Factor Risk 与 Alpha 提纯

更新时间：2026-08-08

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
- `factor_model_artifact_hash` 与 `serialize_factor_model_artifact` 提供可复算 hash 和 metadata manifest；
  另保存独立的 `factor_exposure_payload_hash`、`factor_covariance_payload_hash` 和
  `specific_variance_payload_hash`，完整矩阵 payload 仍由 artifact 内存/上层存储负责保存。
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

- 当前全量 CTest `46/46` 通过；Phase 3A 定向测试与 full-A factor-form smoke 均包含在内。
- 新增 `test_factor_model`、`test_factor_risk_diagnostics`、
  `test_factor_risk_performance` 和 `test_factor_attribution_v2` 均通过。

## 新增：因子风险预测诊断脚手架

`evaluate_factor_risk_diagnostics` 已作为独立组件加入 `portfolio_math`，不改变
`FACTOR-PIT-EWMA` artifact 或 Replay 数值。入口绑定同一个 `FactorRiskModelView` 和非零
artifact hash，不再接受可与 factor covariance 脱节的独立 dense asset covariance。它在同一组冻结
输入上输出：

- factor covariance QLIKE：`log(det(F)) + trace(F^-1 R)`，其中 `R` 是已实现因子收益的中心化协方差；
- factor/specific/total predicted portfolio variance 及其加总残差；
- realized/predicted portfolio variance ratio；
- predicted/realized asset risk-contribution share 的 mean absolute error 和 maximum absolute error；
- 每个资产的 predicted/realized risk-contribution vectors。

预测 variance/marginal risk contribution 直接使用 `B/F/D` factor form；已实现 variance/risk contribution
通过资产收益与组合收益的流式协方差计算。生产诊断路径复杂度为 `O(NK + TN + K^3)`，内存为
`O(NK + TN + N)` 的既有输入与线性工作区，不物化 `N x N` covariance。小规模 dense covariance 只在
测试中作为 oracle。

入口统一拒绝零 artifact hash、非有限值、维度/时间戳不匹配、非正定 factor covariance、非正的
specific variance/组合方差和未来数据；遇到 `available_at > decision_at` 或
`timestamp > decision_at` 返回 `FUTURE_DATA`，不会静默裁剪或用 fallback 补齐。
实现和测试位于：

- `portfolio_math/include/portfolio_math/factor_risk_diagnostics.h`
- `portfolio_math/src/factor_risk_diagnostics.cpp`
- `portfolio_math/tests/test_factor_risk_diagnostics.cpp`

这项交付是数值诊断和失败关闭的工程脚手架，不等于真实 PIT factor source、三个 purged OOS
窗口或交易成本后经济证据已经取得。

测试覆盖 PIT/future mutation、source manifest future/unavailable guard、WLS orthogonality、PSD、specific
floor、factor/dense variance 与 risk-contribution parity、streaming realized/dense parity、gradient finite
difference、Attribution V2 accounting reconciliation、失败关闭和 `N=5000` deterministic smoke；规模测试
不创建 dense asset covariance。

## Factor-form 性能证据

`test_factor_risk_performance` 使用固定 `K=8`、`T=64` 的确定性 synthetic fixture，分别覆盖 `N=200`
和代表 full-A 数量级的 `N=5000`。延迟分位数只作为 Release snapshot 记录，不设置依赖机器速度的绝对
通过阈值；正式门槛检查以下机器无关性质：

- `N=200` 的 predicted/realized variance 和逐资产 risk-contribution 与 dense covariance oracle 一致；
- `N=5000` 从不调用 `materialize_factor_covariance`，重复运行输出逐项确定；
- 由实际输入 shape 和实现中的线性临时量推导 conservative workspace bound，并要求
  `input payload + workspace <= dense N x N bytes / 16`；
- 支持 `getrusage` 的平台同时记录首次至重复运行的 peak RSS 增量，并只使用相对 dense bytes 的宽松
  guard，不把某台机器的 RSS 或 latency 数值冻结成跨平台阈值。

当前本机 `Release -O3, LTO=false` 快照：

| N | dense parity | p50 | p95 | p99 | peak RSS delta | input payload | linear workspace bound | dense N x N |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 200 | PASS | 28.7 us | 43.4 us | 49.5 us | 278,528 B | 123,520 B | 115,712 B | 320,000 B |
| 5,000 | NOT_MATERIALIZED | 0.655 ms | 0.777 ms | 0.801 ms | 917,504 B | 2,965,120 B | 1,958,912 B | 200,000,000 B |

该快照证明当前诊断在 full-A 数量级保持线性存储路径，但不替代真实 PIT universe 的 p50/p95/p99、
峰值 RSS、三个 purged OOS 窗口或交易成本后经济证据。

## 后续顺序

1. 接入可审计的 PIT industry/style exposure source，使用已冻结的 `FactorExposureSourceManifest` 生成真实 `FACTOR-PIT-EWMA` artifact 和 availability manifest。
2. 将当前诊断接到真实滚动预测，冻结配对 bootstrap 规则并完成至少三个 purged OOS 窗口。
3. 只有 baseline 通过后，才单独研究 `FACTOR-PIT-VRA`；动态 loading、regime covariance 和 Kalman family 不得提前组合。

## Baostock source audit

已对 `/Users/Zhuanz/PycharmProjects/scrapy/data/security_state/provider=baostock` 的全部 `69,358`
个 parquet 文件进行 metadata-only 审计：

- 唯一 schema：`BAOSTOCK_DAILY_STATE_V1`，覆盖字段为 `timestamp`、`symbol`、上市/停牌/ST/可交易状态、
  `universe_asof` 和 `reference_data_known_at_max`。
- industry exposure 字段：`0`；style exposure 字段：`0`；显式 exposure `available_at/decision_at/asof`：`0`。
- 结论：`SOURCE_UNAVAILABLE`，不能从该 security-state 数据生成真实 `FACTOR-PIT-EWMA` exposure artifact。
- 该来源 schema 指纹为 `24873c819c1c34ccc3ea01b1604ca13f9ea46bde4d78b22c478be6e4addb85d6`；它只代表
  schema/metadata，不代表 exposure 已知。

审计工具：`tools/audit_phase3a_exposure_source.py`；报告：
`runs/phase3a-exposure-source-audit/phase3a_exposure_source_audit.json`；报告 SHA-256：
`d7af11fc154b05164889108ea0c7b7f827d3fd6d89b78856345f81d230121742`。

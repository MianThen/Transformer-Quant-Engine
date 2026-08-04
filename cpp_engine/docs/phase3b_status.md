# Phase 3B 状态：Conditional Tail Risk

更新时间：2026-08-04

## 当前判定

Phase 3B 已完成固定组合经验尾部风险第一阶梯，并完成单组合 GARCH-FHS 第二阶梯的工程候选；阶段整体尚未退出：

```text
engineering_scaffold_complete = true
phase_exit_eligible = false
promotion_eligible = false
```

## 已交付：TAIL-EMPIRICAL-ES

- `estimate_tail_risk` 保持固定组合、portfolio-return-series 和 Rockafellar-Uryasev 经验 VaR/ES 口径；带权离散尾部质量点分配、符号和 `return_cvar` 已有 parity 测试。
- 新增 `backtest_tail_risk`，对逐期 realized return、VaR loss 和 ES loss 做时间顺序、future-data、有限性和 `ES >= VaR` 校验。
- 联合回测输出异常次数/率、ES 违约次数/率、平均 VaR 超额损失、平均 ES 超额损失、Kupiec LR/p-value、Christoffersen transition/LR/p-value 和确定性 input/artifact hash。
- 新增 `serialize_tail_risk_backtest_artifact`；`reference_price_quality=PROXY` 或 `ARRIVAL_PROXY` 时强制 `promotion_eligible=false`。
- 当前接口只处理固定组合收益序列，不把缺失的费用、真实滑点、reference provenance 或 PIT 因子数据伪装成观测值。

## 已交付：TAIL-GARCH-FHS-ES（单组合研究候选）

- 新增确定性的 Gaussian-QMLE GARCH(1,1) 过滤：`omega>0`、`alpha>=0`、`beta>=0`、`alpha+beta<1`，固定一步 forecast 和 variance floor。
- 仅支持 `PORTFOLIO_RETURN_SERIES`，枚举全部有效 standardized residual 并使用同一组合历史行；资产向量/因子同步 FHS 仍 fail closed。
- 输出 GARCH 参数、stationarity margin、forecast variance、standardized residual 均值/方差、Ljung-Box、squared-residual Ljung-Box/ARCH-LM、最大残差和缺失比例。
- 非有限数据、未来时间戳、未冻结均值/波动规格、诊断越界和不支持的横截面场景均拒绝输出；同一输入重复运行产生相同 artifact hash。
- 合成 GARCH recovery、proxy promotion gate 与上述失败路径已加入 `test_portfolio_math`。

该候选只证明单变量条件尺度过滤和 residual replay 的可重放合同，不能证明资产间相关性、PIT factor-FHS 或真实交易成本后的生产 CVaR。

## 已交付：TAIL-GARCH-FHS-EVT-ES（单组合研究候选）

- 在同一单组合 GARCH-FHS 场景上按冻结的 `evt_threshold_quantile_max` 选择训练期 POT 阈值，并以确定性 method-of-moments 拟合 GPD 超额。
- 只有超额数量达到 `evt_minimum_exceedances`、`beta>0`、支持域有效、`shape < evt_shape_upper_guard < 1` 且目标尾部概率低于阈值尾部质量时，才输出有限 VaR/ES。
- 阈值、超额数量、GPD shape/scale、GARCH 诊断和 deterministic artifact hash 均写入结果；样本不足、非法形状、未来/缺失输入和不支持横截面路径 fail closed。
- 该实现是单变量 EVT oracle，不宣称已完成资产向量拼接、极端尾部真实 OOS 或生产晋级。

## 已交付：TAIL-EXPECTILE（直接单组合研究候选）

- 新增训练期加权非对称平方损失求解，使用固定 `expectile_level` 和确定性二分，不把 `tau` 静默当作 VaR/ES 的 `alpha`。
- 结果单独写入 `expectile_loss` 与校准 level；时间戳、缺失值、权重和矩阵输入均按同一 fail-closed 合同校验，并生成 deterministic artifact hash。
- `TAIL-EXPECTILE-TAYLOR-MAPPED-ES` 仍拒绝输出，直到 Taylor 映射参数/校准规则单独冻结；当前没有把直接 Expectile 冒充 ES。

实现位置：

- `portfolio_math/include/portfolio_math/tail_risk.h`
- `portfolio_math/src/tail_risk.cpp`
- `portfolio_math/tests/test_portfolio_math.cpp`

## 验证

- `test_portfolio_math`：通过经验 VaR/ES 手算、带权质量点、平移、future guard、联合异常和 proxy promotion gate。
- 全量 C++ CTest：`32/32` 通过。

## 未完成与边界

- Taylor-mapped Expectile-ES、FZ0/ESR 完整经济回测和至少三个 purged OOS 窗口仍未完成；本阶段目前只提供 Kupiec/Christoffersen 异常诊断。
- 经验 backtest 只提供研究证据；在 `RESEARCH_PROXY` 下不能解释为真实净收益 CVaR 或生产风险门禁。
- Feature-PGD 延期不阻塞本阶段；Phase 3A 真实 PIT industry/style exposure 仍是独立数据源缺口。

## 下一步

下一步单独冻结 Taylor→ES 映射和 FZ0/ESR 联合回测；随后再做 factor-specific synchronized FHS。不得把这些组件在同一实验中同时首次变化。

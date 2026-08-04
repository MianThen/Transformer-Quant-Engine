# Phase 3B 状态：Conditional Tail Risk

更新时间：2026-08-05

## 当前判定

Phase 3B 已完成固定组合经验尾部风险的第一阶梯，阶段整体尚未退出：

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

实现位置：

- `portfolio_math/include/portfolio_math/tail_risk.h`
- `portfolio_math/src/tail_risk.cpp`
- `portfolio_math/tests/test_portfolio_math.cpp`

## 验证

- `test_portfolio_math`：通过经验 VaR/ES 手算、带权质量点、平移、future guard、联合异常和 proxy promotion gate。
- 全量 C++ CTest：`32/32` 通过。

## 未完成与边界

- GARCH(1,1)-FHS、POT-GPD、Expectile、FZ0/ESR 完整经济回测和至少三个 purged OOS 窗口仍未完成；本阶梯只提供 Kupiec/Christoffersen 异常诊断。
- 经验 backtest 只提供研究证据；在 `RESEARCH_PROXY` 下不能解释为真实净收益 CVaR 或生产风险门禁。
- Feature-PGD 延期不阻塞本阶段；Phase 3A 真实 PIT industry/style exposure 仍是独立数据源缺口。

## 下一步

在经验 baseline 冻结后，单独实现 N<=200 的 `TAIL-GARCH-FHS-ES` synthetic recovery、stationarity/residual diagnostics 和同步 residual-row replay；不得与 EVT、Expectile 或 factor-specific streaming 同时变化。

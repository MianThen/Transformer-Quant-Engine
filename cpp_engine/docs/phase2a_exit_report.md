# Phase 2A 研究工程退出报告

日期：2026-08-04
状态：`phase_exit_eligible=true`、`promotion_eligible=false`

## 结论

Phase 2A 的研究工程施工完成。RMT、detoning、linkage、ONC、HRP、NCO、PortfolioPolicyArtifact、
single-period reconciler、policy-to-order 以及 C++ Replay/CVaR 集成均有独立实现和测试；失败不回退
到 Top-K、HRP 或上一期权重。

## 新增验收

- `SinglePeriodConstraintView` 支持分组上限和调用方提供的单资产最大交易权重。
- reconciler 输出线性成本、二次冲击和总预测成本，并检查三者一致。
- `build_policy_order_intents` 完成权重到 lot 对齐订单意图的事务性转换，检查数据可信、上市/停牌、
  可卖量、最大订单量和输出容量。
- `test_phase2a_policy_replay` 覆盖 `target → order → fill → fee → period ledger → empirical CVaR`；
  period accounting residual 为零，显式佣金进入账本，CVaR artifact 保持 `PROXY` 且不晋级。

## 验证

`build/phase1c-python`（`QBT_ENABLE_PORTFOLIO_MATH=ON`、`QBT_ENABLE_PERFORMANCE_ANALYTICS=ON`）完整构建
通过；Phase 2A 数学、artifact、reconciler、订单桥接和回放 fixture 全部通过。

## 证据边界

历史费用/税费、真实成交量参与率、涨跌停、lot provenance、公司行动/复权、真实滑点和正式
reference-price provenance 仍不可用或仅为 proxy。它们不再阻塞 Phase 2A 研究退出，但会阻止正式净
收益、真实 implementation shortfall 和生产 economic promotion；所有相关 artifact 必须保持
`reference_price_quality=PROXY` 或 `UNAVAILABLE`、`promotion_eligible=false`。

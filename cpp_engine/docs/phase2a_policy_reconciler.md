# Phase 2A：PortfolioPolicyArtifact V1 与单期 Reconciler

## PortfolioPolicyArtifact V1

`PortfolioPolicyArtifact` 统一保存 policy id/config hash、official risk provenance、可选 cluster/
posterior provenance、簇内/簇间 objective hash、reconciler hash、anchor/target 权重 hash、两层
status 和 anchor/reconciler diagnostics。artifact 使用确定性的 FNV-1a hash；修改任一字段后校验失败。

失败 artifact 可以保存失败的 `anchor_status`/`reconciler_status` 和空权重，用于证明系统没有静默
切换到 Top-K、HRP 或上一期 policy。当前 `eligible_for_official_risk` 固定为 `false`。

## Single-period Reconciler V1

`reconcile_single_period` 求解一次调仓的 long-only 目标权重：

```text
0.5 * anchor_penalty * (w - w_anchor)^2
+ linear_cost * |w - w_current|
+ 0.5 * quadratic_impact * (w - w_current)^2
```

当前约束为 long-only、隐式 cash/target-investment 上限、single-name cap、行业/分组上限、单资产
最大交易权重（由调用方提供的 participation proxy）和 one-way turnover cap；算法使用确定性的
proximal projected-gradient，并输出 anchor distance、turnover、predicted cost、active constraints、
KKT residual 和 constraint violation。`target_investment` 是投资资产权重总和的上限，未分配部分保留
为显式上层 cash，而不是由 reconciler 猜测成某个资产。

`SinglePeriodConstraintView.group_ids` 使用从 `0` 开始的稠密分组编号，`group_caps[g]` 是该组的
目标权重上限；分组可以由调用方映射到行业或其他预注册 bucket。`max_trade_weights[i]` 是资产
`i` 的绝对权重变动上限，约束为 `|w[i] - current[i]| <= max_trade_weights[i]`。它只表达调用方
已经计算好的参与率/可成交容量代理，不等同于真实成交量参与率；真实参与率需要 decision_at 可知的
历史成交量、lot、涨跌停和 fill 规则，并在 policy-to-order 层再次校验。

成本向量必须在 `decision_at` 可用且 `costs_available=true`；缺失成本直接 `INVALID_INPUT`，不把零
成本当作观测值。anchor 不可行时允许在上述约束内修复；current weights 不可行、参数非法或迭代不收敛
时失败关闭，返回空 target，不触发 policy fallback。

`build_policy_order_intents` 将成功的 reconciler 权重按 decision-time bar close 转成 lot 对齐的
`OrderIntent`，检查上市/停牌/数据可信、可卖数量、最大订单量和输出容量；它在失败时原子清空输出，
不会把失败结果替换成 anchor、Top-K 或上一期订单。该桥接明确标记 reference price 为 proxy，真实
订单层仍需接入可审计的 reference/slippage provenance。

行业/分组上限与调用方最大交易权重视图已经接入研究 reconciler，但仍没有真实成交量、参与率、费用
schedule、涨跌停、lot、滑点、公司行动/复权和账户级现金对账 provenance；因此这些约束不能被解释为
历史可回放的执行事实，必须由后续 policy-to-order/实盘层提供或失败关闭。

## 验证

- `test_reconciler`：zero-cost anchor parity、成本/turnover/single-name cap、分组上限、单资产最大
  交易权重、三者联合约束、非法约束视图、成本缺失和 infeasible current failure closure；
- `test_policy_to_order`：reconciler failure、数据不可信、lot quantization、T+1 可卖量和输出溢出
  的事务性失败关闭；
- `test_portfolio_policy_artifact`：manifest 字段、hash、序列化、篡改检测和失败状态记录。
- `test_phase2a_policy_replay`：policy → order → fill → fee → period return ledger → empirical CVaR
  的 C++ 端到端 fixture；账务 residual 必须为零，artifact 保持 `PROXY`/`promotion_eligible=false`。

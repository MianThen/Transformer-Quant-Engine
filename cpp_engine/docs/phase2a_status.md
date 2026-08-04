# Phase 2A 状态

日期：2026-08-04
状态：`RESEARCH_ENGINEERING_COMPLETE / PHASE2A_EXIT_ELIGIBLE`

## 当前结论

Phase 2A 已完成两个 RMT 单变量切片：`RMT_CONSTANT_RESIDUAL` 和
`RMT_TARGETED_SHRINKAGE`、仅供聚类发现使用的 `detone_correlation`、确定性的 complete
hierarchical linkage、独立的 ONC partition，以及 `ClusterModelArtifact V1` 的结构/校验/序列化层。
RMT 可以在固定收益窗口上构造去噪协方差，detoning
可以移除预注册主成分并恢复单位对角，linkage 可以生成 merge tree 和 quasi-diagonal order，ONC 可以
搜索 K 并输出 partition；HRP、NCO-MinVar 和 NCO-RiskBudget 现在可以生成研究 anchor；它们仍是
研究候选，不会自动替换 `LW-LIN-CC` 或 `LW-NLS-MV-QUEST` 的 official risk。

Phase 2A 研究工程退出条件已完成：结构风险、聚类 anchor、统一 reconciler、policy-to-order 桥接、
成本分解、联合约束失败关闭，以及 C++ `target → order → fill → fee → ReturnLedger → CVaR` fixture
均已通过。`phase_exit_eligible=true`，但这不是生产经济晋级。

路线图第 2.2 节将 RMT 谱去噪、聚类模型、HRP、Risk Budget 和 NCO 列为缺口；Phase 2A 才是这些
组件的施工阶段。当前工程已有两个 RMT 实现、独立 detoning API、complete linkage API、ONC
partition API、`ClusterModelArtifact V1` schema、可运行的 HRP anchor 和 NCO-MinVar/NCO-RiskBudget
研究切片、`PortfolioPolicyArtifact V1` 和 single-period reconciler V1；reconciler 已支持分组上限
与调用方提供的单资产最大交易权重；policy-to-order 已接入事务性失败关闭，端到端回放已覆盖固定佣金
proxy。正式成交量参与率、历史执行成本/reference provenance 和真实数据晋级仍保持关闭。

## 已交付文件

- `portfolio_math/include/portfolio_math/rmt_denoising.h`
- `portfolio_math/src/rmt_denoising.cpp`
- `portfolio_math/tests/test_rmt_denoising.cpp`
- `portfolio_math/include/portfolio_math/detoning.h`
- `portfolio_math/src/detoning.cpp`
- `portfolio_math/tests/test_detoning.cpp`
- `portfolio_math/include/portfolio_math/hierarchical_linkage.h`
- `portfolio_math/src/hierarchical_linkage.cpp`
- `portfolio_math/tests/test_hierarchical_linkage.cpp`
- `portfolio_math/include/portfolio_math/onc_partition.h`
- `portfolio_math/src/onc_partition.cpp`
- `portfolio_math/tests/test_onc_partition.cpp`
- `portfolio_math/include/portfolio_math/cluster_model_artifact.h`
- `portfolio_math/src/cluster_model_artifact.cpp`
- `portfolio_math/tests/test_cluster_model_artifact.cpp`
- `portfolio_math/include/portfolio_math/hrp_policy.h`
- `portfolio_math/src/hrp_policy.cpp`
- `portfolio_math/tests/test_hrp_policy.cpp`
- `portfolio_math/include/portfolio_math/nco_policy.h`
- `portfolio_math/src/nco_policy.cpp`
- `portfolio_math/tests/test_nco_policy.cpp`
- `portfolio_math/include/portfolio_math/portfolio_policy_artifact.h`
- `portfolio_math/src/portfolio_policy_artifact.cpp`
- `portfolio_math/tests/test_portfolio_policy_artifact.cpp`
- `portfolio_math/include/portfolio_math/reconciler.h`
- `portfolio_math/src/reconciler.cpp`
- `portfolio_math/tests/test_reconciler.cpp`
- `portfolio_math/include/portfolio_math/policy_to_order.h`
- `portfolio_math/src/policy_to_order.cpp`
- `portfolio_math/tests/test_policy_to_order.cpp`
- `portfolio_math/include/portfolio_math/covariance.h`
- `portfolio_math/include/portfolio_math/risk_model.h`
- `portfolio_math/src/risk_model.cpp`
- `portfolio_math/tests/test_portfolio_math.cpp`

## 验证

在 `build/phase2a-rmt` 配置下通过：

- `test_rmt_denoising`；
- `test_detoning`；
- `test_hierarchical_linkage`；
- `test_onc_partition`；
- `test_cluster_model_artifact`；
- `test_hrp_policy`；
- `test_nco_policy`；
- `test_reconciler`；
- `test_policy_to_order`；
- `test_portfolio_policy_artifact`；
- `test_portfolio_math`。

在启用 `QBT_ENABLE_PERFORMANCE_ANALYTICS=ON` 的完整配置下另通过：

- `test_phase2a_policy_replay`；

覆盖范围包括 MP 边界、signal/noise rank、constant residual 与 targeted shrinkage 独立 oracle、
signal 保持、强度边界、协方差对角线映射、PSD、置换等变性、detoning 市场主成分移除、单位对角
恢复、零成分 parity、退化输出、complete-linkage 距离 oracle、merge tree、quasi-diagonal order、
tie-break、ONC block recovery、K 搜索、最小簇大小、seed/repeats、silhouette、partition hash、置换
等价、ClusterModelArtifact HRP/ONC variant、schema/hash/time/source provenance、序列化字段、tamper
检测、非有限输入、零方差输入、RMT spec hash 合同，以及精确低秩 `p>T` 在 floor 会破坏 trace 时的
失败关闭。

## 晋级边界

`eligible_for_official_risk=false`。Phase 2A 研究回放允许使用显式登记的 fixed-cost/reference proxy，
但不生成 official-risk promotion 结论；真实执行数据到位前不进入生产 Replay/CVaR 或 policy fallback。

HRP 已完成独立的 recursive-bisection risk-only anchor 和失败关闭测试；它消费 linkage 的
quasi-diagonal order，但递归簇方差和 predicted risk 仍只使用 official risk covariance。HRP
不会进入正式 CVaR 或 reconciler，也没有替换 Risk Budget。详见 `docs/phase2a_hrp.md`。

NCO-MinVar 和 NCO-RiskBudget 已完成独立的两层组合与失败关闭测试；NCO 使用 official covariance
做层内/层间优化，cluster correlation 只提供结构，且 `eligible_for_official_risk=false`。详见
`docs/phase2a_nco.md`。

`PortfolioPolicyArtifact V1` 和 single-period reconciler V1 已完成结构、hash、zero-cost parity、
成本/换手/单名上限、分组上限、单资产最大交易权重、线性/二次成本分解和失败关闭测试；
`build_policy_order_intents` 与 C++ period replay/CVaR fixture 已接入。reconciler 缺失成本时拒绝运行，
不把零成本伪装成观测值；分组/最大交易权重和固定佣金仍是研究输入，不是已验证的历史执行 provenance。
详见 `docs/phase2a_policy_reconciler.md`。

Phase 2A 后续工作仅剩实盘经济晋级所需的真实费用/税费、成交量参与率、涨跌停、lot、公司行动/复权、
滑点/reference provenance 和 live shadow 对账；这些不阻塞研究工程退出。cluster correlation 不得
覆盖 official risk covariance；`ClusterModelArtifact V1` 仍只负责复现结构输入，不提供 risk covariance。

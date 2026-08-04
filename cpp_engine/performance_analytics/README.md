# Performance Analytics

该可选模块承载 C++ Replay 的权威收益核算合同。只有开启
`QBT_ENABLE_PERFORMANCE_ANALYTICS` 时才参与构建，不改变默认回测核心路径。

当前已实现：

- `PerformanceSpecV1`：频率、交易日历、年化、基准、费用容差和最小样本语义；
- `ReturnLedger`：逐期简单收益、毛净收益桥、外部资金流失败关闭和确定性哈希；
- `ImplementationShortfallLedger`：冻结每个决策的 paper target、reference/arrival/end mark、实际
  期初期末持仓与成交，计算 execution price cost、explicit fees、unexecuted opportunity cost，
  并验证 `paper_pnl - real_net_pnl = implementation_shortfall`。
- `ShortfallReplayAdapter`：显式管理 `open target -> fills -> close/next target -> finalize` 生命周期，
  防止多个 paper decision 混入同一 measurement interval。
- `ShortfallReplaySink`：消费通用 Replay decision/execution/end 事件并自动驱动 adapter；当前使用
  official bar close proxy，所有记录强制保持不可晋级。
- `SecurityContributionLedger`：逐期保存证券市值变化、成交现金流、公司行动现金、费用与净贡献，
  汇总到 PIT industry，并同时验证持仓、现金和权益三条会计恒等式。
- `PeriodContributionCoordinator`：消费外部交易日历确认的 session 边界，冻结期初/期末股票池、
  成交和公司行动事件，并生成已对账的贡献记录。
- `PeriodContributionReplaySink`：消费 C++ 引擎显式 period-open/close、成交和公司行动事件，自动驱动
  period coordinator；未关闭 period 时禁止 Replay finalize。
- `Brinson-Fachler + Menchero`：benchmark holdings/PIT provenance 齐全时生成单期 allocation、selection、
  interaction 与独立 other effects，再用 Menchero beta 精确链接到跨期几何累计收益差。

贡献账本只接受由 C++ Replay 冻结的 period snapshot，不自行按固定 UTC 日切推断 period，也不读取
Python 重建的持仓。当前 `period_return_input()` 可把已对账记录无损送入 `ReturnLedger`。period
coordinator 要求调用方提供冻结的 `calendar_id/session_id` 边界；不会把固定 UTC 纳秒日冒充上海
交易日。`BacktestEngine::open_performance_period/close_performance_period` 只能在已有行情的显式边界
调用，边界内任一标的 mark 陈旧都会失败关闭。期初与期末股票池允许变化，并以两侧并集保留已平仓
和新进入证券；period 未关闭时禁止 Replay finalize。

shortfall ledger 要求每个 measurement interval 不重叠，资产和成交标识唯一，期末持仓必须等于
期初持仓加有符号成交。存在未执行 target 时必须记录原因；使用 arrival/bar proxy 作为 reference
仍可生成研究记录，但 `promotion_eligible=false`。输入在账本内按稳定键排序，逻辑相同的 Replay
生成相同哈希。

adapter 不从订单反推 target。组合/策略层必须在决策时提交完整 `FrozenPaperTarget`；执行层只追加真实
成交，下一次 target 更新前或 Replay 结束时提交期末持仓和 mark。任何未知标的、重复成交、未关闭
target reset 或 finalize 后事件都会失败关闭。空 target 明确表示全现金 interval。策略运行时通过
`StrategyDecisionView` 暴露单调 `decision_id` 和仅在 target 改变时更新的只读快照，执行层不得从
订单反推该快照。启动时的隐式全现金状态不产生伪决策；已有 target 退出为空仓时仍产生正式退出
decision。`BacktestEngine` 通过通用 sink 自动转发新决策、带 provenance 的真实成交和 Replay end。
订单和成交均携带原始 `decision_id`；旧决策的迟到成交进入新 interval 时立即失败关闭，禁止错配归因。
Python 只读取冻结的 C++ ledger，不重新撮合或重建净值。

当同时启用 `QBT_ENABLE_PERFORMANCE_ANALYTICS`、`QBT_ENABLE_PORTFOLIO_MATH` 和
`QBT_BUILD_PYTHON` 时，`cpp_engine` 暴露 `PeriodContributionReplaySink`、显式
`open_performance_period`/`close_performance_period` 以及 `estimate_empirical_cvar`。
Python 可从同一 sink 读取 period returns 和 deterministic ledger hash，再将这些收益送入
C++ RU-ES/CVaR；查询结果同时包含可落盘的 TailRisk artifact JSON。缺少正式执行数据时，
ledger 和 TailRisk artifact 必须使用 `PROXY` 并保持 `promotion_eligible=false`。

`serialize_return_analysis_report` 会把 C++ ledger 的收益指标、ledger SHA、manifest 和报告 SHA
汇总为 `return_analysis_v1`。Python `FrozenReturnAnalysisReport` 只读校验报告 SHA、指标结构和
reference-price promotion gate，不重新计算或修改 C++ 结果。

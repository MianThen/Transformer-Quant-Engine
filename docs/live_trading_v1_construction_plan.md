# 实盘 V1 施工方案

更新时间：2026-08-09

状态：`LIVE_PHASE_0 / EXTERNAL_INPUTS_PENDING`

## 0. 本版目标和完成定义

本版不是“为以后接实盘预留接口”，也不以 Mock、回放、Paper 或 Shadow 作为最终交付。
本版的平台终点是：

```text
PLATFORM_LIVE_VERIFIED = true
```

本文后续的 `LIVE_VERIFIED` 是 `PLATFORM_LIVE_VERIFIED` 的简称，表示交易平台已通过真实生产闭环；它不自动
授权任意研究模型使用真实资金。具体模型还必须单独满足：

```text
MODEL_CANARY_ELIGIBLE(model_hash) = true
MODEL_LIVE_APPROVED(model_hash) = true
```

只有同时满足以下条件，才允许把本版标记为完成：

1. 一个已明确的真实券商/柜台及其生产 API 已接入；
2. 生产账户的真实行情、资金、持仓、活动委托和成交查询已同步；
3. 经签名批准的 `INFRA_CANARY` 策略或已晋级模型通过同一套实盘行情、账户、ExecutionPlanner、风控和
   OMS 链路运行，禁止人工旁路直调 submit；
4. 在生产账户完成受控真实报单、券商确认、撤单，并至少完成一笔真实成交；
5. 真实成交的手续费、现金、总持仓、可用持仓与券商日终及下一交易日结果对账无未知差异；
6. 断线、进程重启、重复/乱序回报、撤单与成交竞态不会造成重复经济记账或盲目重发；
7. 每一笔真实委托均可追溯到行情、模型、策略决策、风控判断、报单、券商回报和账户变化；
8. 运维人员在 Python/Dashboard 不可用时仍可执行禁止新单、撤活动单、只减仓、只读恢复和人工解除；
9. `LIVE_VERIFIED` 来自不可变验收证据，不是可由配置随意写入的布尔值。

`SHADOW_READY`、`PAPER_READY`、`PROD_READ_ONLY` 和 `LIVE_READY` 都只是中间状态，不能替代
`LIVE_VERIFIED`。

本施工不依赖学术论文。当前需要的是目标券商官方 SDK、接口、错误码、回调线程、重连、查询和
生产认证文档。任何官方接口没有承诺的唯一性、顺序或恢复语义都不得自行假设。

## 1. 审计结论

### 1.1 当前系统边界

| 系统 | 当前职责 | 本版定位 |
| --- | --- | --- |
| Python 工程 | 数据湖、PIT 数据、特征/训练、模型制品、回测编排、SQLite 与 Dashboard | 研究与控制面；负责模型晋级、部署清单、实盘只读投影和运维界面，不进入报单热路径 |
| `cpp_engine` | Bar 回测撮合、现金、持仓、T+1、费用、公司行动和 PnL | 实盘语义的参考实现和测试 oracle；不直接作为真实柜台 |
| `ml_runtime` / `strategy_runtime` | BAR_V1 特征、推理、组合目标和基础逐单校验 | 改造成可由 LiveRuntime 宿主的实盘策略运行时 |
| `trading_engine` | Mock TCP 行情、SPSC、Mock 报单、订单状态机、有限 WAL 和重连原型 | 保留网络与队列基础，重建生产级 LiveRuntime、OMS、账户、风控和真实 Adapter |
| `engine_common` | 若干跨模块 POD 类型和策略接口 | 逐步由版本化 `trading_domain` 契约替代；禁止继续增加有损类型转换 |

Python 工程位置：`/Users/Zhuanz/PycharmProjects/PythonProject`。

### 1.2 当前不能实盘的 P0 阻断项

| 阻断项 | 当前证据 | 本方案对应施工包 |
| --- | --- | --- |
| 主程序只统计行情并发送演示单，没有策略、执行规划、账户和风控闭环 | `trading_engine/src/main.cpp` | WP3-WP10 |
| 策略目标没有限价生成、撤换、部分成交收敛和重复 rebalance 抑制 | `strategy_runtime/src/model_strategy_runtime.cpp` | WP7 |
| 策略明确拒绝 `live=true`，执行回报为空 | `strategy_runtime/src/model_strategy_runtime.cpp` | WP8 |
| live 构建未链接 ML/策略运行时，生产组合未进 CI | `CMakePresets.json`、`trading_engine/CMakeLists.txt` | WP0、WP13 |
| 实时 BBO 无法满足 BAR_V1 完整截面 | `trading_engine/src/feed/*` 与 `ml_runtime/src/bar_v1_feature_pipeline.cpp` | WP5 |
| 网关发送恢复查询后立即 READY，没有等待一致性查询和对账 | `trading_engine/src/oms/order_gateway.cpp` | WP3、WP4、WP9 |
| 重复成交仍向下游回调，可能重复记账 | `trading_engine/src/oms/order_gateway.cpp` | WP2、WP3、WP4 |
| WAL 不能恢复完整委托、券商 ID、成交 ID、账户或决策 | `trading_engine/src/oms/order_gateway.cpp` | WP2、WP3 |
| 永久行情 gap 后重新被标为可信，所谓 resync 只清状态 | `trading_engine/src/feed/decoder.cpp`、`feed_handler.cpp` | WP5、WP10 |
| 订单 Adapter 没有登录、账户、持仓、委托、成交和能力查询 | `trading_engine/include/adapter/order_adapter.h` | WP1、WP9 |
| 没有 broker 权威 AccountMirror、独立前置风控或日内损失门禁 | 当前无对应生产模块 | WP4、WP6 |
| standalone live 进程没有原生模型制品加载器 | 加载逻辑仅在 `cpp_engine/src/bindings.cpp` | WP8 |
| SymbolRegistry 可在 feed 扩容时与核心线程并发读取 | `engine_common/include/engine_common/symbol_registry.h` | WP5 |
| 生产报单没有最后一跳物理门禁 | 当前只有策略层 shadow | WP6、WP10 |
| 没有同账户单实例 fencing，可能双进程报单 | 当前无对应模块 | WP10、WP12 |
| 无 secrets、独立紧急控制入口、告警、进程监督和生产 runbook | 当前无生产部署体系 | WP12 |
| 当前研究制品均未获生产晋级 | `docs/phase5_status.md` 等状态文档 | WP8；不阻断独立 `INFRA_CANARY` 平台验收 |

## 2. 首版范围冻结

### 2.1 建议的最小真实闭环

结合当前代码的 A 股 Bar 语义，建议首版冻结为：

```text
单券商 + 单现金账户 + A 股现货多头 + 单策略
+ 分钟级完整截面 + 限价 DAY + 报单/撤单 + 生产账户真实成交
```

这是待用户确认的工程范围，不是从现有代码推断出的业务决定。以下能力首版不默认包含：

- 融资融券、做空、期货今昨仓、期权；
- 多券商、多账户、主备双活；
- 算法单、条件单、改单；
- Tick/HFT 决策；
- 自动市价清仓。

若目标资产、账户或 SDK 不符合上述范围，必须在 S0 重新冻结字段、账本和风险语义，不能用 A 股现金账户
的假设硬套。

### 2.2 Live Phase 0 必须冻结的外部输入

以下信息不能从代码中获得，也是完成真实 Adapter 的硬前置：

| 输入 | 必填内容 |
| --- | --- |
| 券商/柜台 | 厂商、产品/API 名称、SDK 版本、生产与模拟环境名称 |
| 部署环境 | Windows/Linux/macOS、CPU 架构、编译器、是否只允许指定机房/IP |
| 账户 | 资产类别、现金/信用/期货等账户类型、单/多账户 |
| 账户隔离 | 专用账户/子账户、终端 ID、并发登录/会话抢占语义、人工和手机交易政策、初始持仓归属 |
| 行情 | 来源、L1/L2/Tick/分钟 Bar、是否有 snapshot、增量序列和重放能力 |
| 参考数据 | ReferenceDataProvider、版本、symbol/venue 映射、日历、会话、tick/lot、价格限制和盘中状态来源 |
| 交易能力 | 支持市场、订单类型、TIF、撤单/改单、价格和数量单位、流控 |
| 认证 | 官方开发权限、模拟账号、生产账号、证书/密钥引用、IP 白名单 |
| 合规授权 | 券商允许程序化交易的账户/应用授权、生产认证要求、行情数据使用许可 |
| 风险值 | 单笔数量/金额、单标的仓位、总敞口、订单速率、日亏损、回撤上限 |
| Canary | 白名单标的、方向、最小合法数量、价格上限、最大允许损失和授权窗口 |
| 验收周期 | Shadow、官方模拟和真实 Canary 各自需连续通过的交易日数量 |
| 运行阈值 | stale/超时/队列/磁盘/fsync/时钟/RTO/RPO/对账容差和连续通过天数 |

在这些信息未冻结前，可以进行只读审计、契约草案、测试框架和通用模拟器预研，但不能正式冻结
`trading_domain`/Adapter 合同，不能承诺真实 Adapter 工期，不能进入生产联调，也不能宣布实盘就绪。

### 2.3 Live Phase 0 退出条件

以下条件必须全部满足，Live Phase 0 才能标记为 `EXITED`：

- [ ] 已选择首个真实券商/柜台；
- [ ] 已确认 API/SDK 的正式名称、版本、获取渠道和授权条件；
- [ ] 已确认生产与官方模拟/认证环境是否可用，以及账号申请路径；
- [ ] 已确认券商允许该账户/API 用于程序化交易，并满足生产认证与行情许可要求；
- [ ] 已确认日志留存、必要报备、生产准入和自动化交易相关要求，并取得可归档依据；
- [ ] 已冻结部署 OS、CPU 架构、编译器和厂商 SDK 兼容矩阵；
- [ ] 已取得并归档对应版本的官方接口、错误码、回调线程、重连、查询和认证文档；
- [ ] 已确认首版资产、账户、市场、行情粒度、订单类型、TIF 和撤单能力；
- [ ] 已确认行情是券商同源还是独立供应商；若独立，已同样冻结供应商和接口版本；
- [ ] 已确认 SDK 的 client/broker/exchange order ID、成交唯一键和查询一致性能力；
- [ ] 已确认 client order ID 可在厂商保留窗口内查询或由厂商幂等拒重；无法解除 submit 歧义时 Phase 0
      不得退出；
- [ ] 已确认行情与交易查询具备官方 snapshot/sequence/watermark 和模拟认证能力，或已接受官方认可的
      等价恢复/认证方案；两者都没有时 Phase 0 不得退出；
- [ ] 已冻结 ReferenceDataProvider 及版本、盘中更新、跨供应商 symbol/venue 映射和失信处理；
- [ ] 已冻结专用账户/子账户和人工交易政策；若账户非空，已完成持仓 ownership/allocation 合同；
- [ ] 已冻结最小 V1 范围及明确不包含项；
- [ ] 已指定风险参数与真实 Canary 授权的确认人，但具体每日额度仍默认保持为零；
- [ ] 已指定 LiveVerificationReport 的签名人、审批人、证据留存期和自动失效条件；
- [ ] 已冻结所有 readiness、超时、容量、恢复和对账阈值，不能以“新鲜”“正常”“过久”等文字代替数值；
- [ ] 已冻结独立审计副本、轮转/留存、备份恢复和关键交易审计 `RPO=0` 合同；
- [ ] C++ 与 Python 两个工程均已建立可回滚 Git 基线和版本标签；
- [ ] 已形成并评审 `Live Scope`、`Vendor Capability Matrix`、`SDK Field Mapping` 和风险清单。

Phase 0 退出物至少包括：

```text
live_scope_v1
vendor_capability_matrix
sdk_field_mapping
official_document_inventory (含版本与 hash)
deployment_compatibility_record
production_authorization_record
account_isolation_and_position_ownership_contract
vendor_recovery_and_certification_record
reference_data_contract
live_risk_parameter_contract
operational_threshold_contract
verification_signing_and_approval_policy
audit_replication_and_retention_contract
phase0_exit_report
```

券商、API/SDK 版本和部署操作系统是其中的硬门槛之一，不设默认值，也不以 Mock/FIX/CTP/QMT/XTP 等
任一实现代替用户选择。

### 2.4 必须查阅的官方资料

不需要论文。收到券商信息后必须逐项查阅并把文档版本/hash 写入 Adapter conformance 清单：

1. SDK 支持的 OS/CPU/编译器、初始化、释放和链接方式；
2. 生产/模拟认证、证书、加密、IP 白名单和凭据轮换；
3. 回调线程模型、对象生命周期、阻塞限制和 SDK 线程安全保证；
4. 行情 snapshot/增量、序号、交易所时间、断线恢复和订阅限制；
5. client/broker/exchange order ID 的关系、保留窗口、按 ID 查询和幂等保证；
6. 同步返回值与异步确认的区别、状态枚举、错误码和拒单码；
7. fill-before-ack、cancel/fill race、撤单拒绝和迟到成交语义；
8. 活动委托、当日成交、资金、冻结资金、持仓和可卖数量的查询范围与一致性窗口；
9. 心跳、流控、重连、交易日切换、清算和生产认证流程；
10. 数据与交易接口的许可和生产使用限制；
11. 程序化交易准入、日志留存、必要报备、专用终端/账户和人工交易限制。

## 3. 目标架构

```text
                        Python 研究/控制面
             模型晋级 -> DeploymentManifest -> 运行审批
                         ^                  |
                         |                  v
ReferenceDataProvider -> Symbol/Session Contract
真实行情 -> VendorMarketAdapter -> MarketIntegrity -> BarAggregator
                                            |
                                            v
                                     C++ LiveRuntime
                              StrategyRuntime + Policy
                                            |
                                   TargetPositionBatch
                                            v
                                     ExecutionPlanner
                                            |
                                OrderIntent / ChildOrderPlan
                                            v
                                      LiveRiskGate
                                            |
                                         ModeRouter
                                            |
                                            v
真实柜台 <- VendorBrokerAdapter <- Durable OMS / Outbox
    |                                       ^
    +-> 回报/查询 -> Inbox/Dedup -> OrderReducer
    +-> 资金/持仓 -------------> AccountMirror/Reconciler
                                            |
                                            v
                 Append-only LiveJournal -> live.db 投影 -> Dashboard/告警
```

### 3.1 权威边界

| 数据 | 权威来源 | 本地职责 |
| --- | --- | --- |
| 委托与成交外部状态 | 券商/交易所查询和回报 | OMS 保留完整投影、未决状态和差异，不静默覆盖 |
| 资金、冻结资金、持仓、可卖数量 | 券商账户快照 | AccountMirror 事件化更新并持续对账 |
| 策略目标与决策 | C++ StrategyRuntime | 保存模型/config/hash 与产生决策时的完整输入引用 |
| 目标到委托的执行计划 | C++ ExecutionPlanner | 根据账户、活动订单和行情生成可恢复 child order，并保证向目标收敛 |
| 前置风险判断 | C++ LiveRiskGate | 每次批准/拒绝都持久化，不能由 Adapter 绕过 |
| 原始事件与恢复历史 | C++ LiveJournal | 只追加、可校验、可回放；SQLite 只是查询投影 |
| 模型制品和晋级状态 | Python 研究/控制面 | 生成不可变 DeploymentManifest，live 只读加载 |
| symbol/venue、日历和交易状态 | 版本化 ReferenceDataProvider 与券商/行情映射 | 生成冻结版本，盘中更新也必须事件化和可回放 |

Python 不直接调用券商报单 API，也不直接修改 C++ 的账户和订单真值。Dashboard 的控制命令必须经过独立、
可认证、可审计的 control channel，不能通过改数据库触发下单。

### 3.2 建议代码落点

以下是 Phase 0 退出后的建议目录，不在 Phase 0 前创建厂商专属实现：

| 落点 | 职责 |
| --- | --- |
| `trading_domain/` | V2 事件、ID、定点金额/价格、schema、显式 enum 转换和 capability contract |
| `trading_engine/live/market/` | `ReferenceDataProvider`、MarketIntegrity、snapshot/recovery、BarAggregator、SessionController |
| `trading_engine/live/account/` | AccountMirror、PositionOwnership、snapshot boundary、ReconciliationEngine |
| `trading_engine/live/oms/` | OrderReducer、Outbox/Inbox、dedup、DAY expiry、cancel-all、checkpoint |
| `trading_engine/live/risk/` | LiveRiskGate、KillSwitch、ReduceOnly、额度与 projected exposure |
| `trading_engine/live/execution/` | ExecutionPlanner、定价、child order、撤换和 target convergence |
| `trading_engine/live/runtime/` | LiveRuntime、readiness、ModeRouter、fencing、优雅停机 |
| `trading_engine/live/journal/` | hash-chain journal、轮转、独立副本、回放和 LiveVerificationReport |
| `trading_engine/adapters/<vendor>/` | 厂商 SDK 的 Market/Broker/Query Adapter；只依赖官方能力，不泄漏到核心契约 |
| `trading_engine/bridge/<vendor>/`（条件性） | Python/专用 OS SDK 的 IPC bridge、跨进程 outbox 和恢复 |
| `strategy_runtime/`、`ml_runtime/` | 原生 ArtifactLoader、模型策略、执行回报和特征 warmup |
| `configs/live/` | schema、DeploymentManifest、风险/阈值/账户白名单；生产配置不可变并带 hash |
| Python `python/live/`、`storage/live_schema.sql`、Dashboard | manifest 晋级、journal 投影、只读状态、告警和运维展示 |
| `trading_engine/tests/live/` 与 Python live tests | simulator、capture replay、故障注入、conformance 和跨语言 parity |

厂商 SDK 的头文件、密钥和 OS 绑定只能出现在 `adapters/<vendor>` 或条件性 bridge；C++ 核心不得通过
`#ifdef VENDOR_*` 扩散厂商语义。

## 4. 核心契约

### 4.1 统一领域事件

新增版本化的 `trading_domain`，至少包含：

- `MarketEventV2`、`MarketSnapshotBoundary`、`BarFrameV2`；
- `DecisionEvent`、`TargetPositionBatch`、`ExecutionPlan`、`OrderIntentV2`；
- `RiskDecisionEvent`、`OrderCommand`；
- `OrderEventV2`、`ExecutionEventV2`；
- `AccountSnapshot`、`PositionSnapshot`、`AccountDelta`；
- `ReconciliationEvent`、`RuntimeStateEvent`、`OperatorCommandEvent`。

订单链的必填关联字段：

```text
run_id / trading_day / account_id / strategy_id / model_hash / config_hash
decision_id / intent_id / client_order_id / broker_order_id / exchange_order_id / exec_id
venue / symbol_id / side / position_effect / order_type / tif
quantity / price_ticks / price_scale / currency
exchange_ts / broker_ts / receive_utc_ts / process_monotonic_ts
provider_sequence / session_id / raw_event_ref / schema_version
```

字段不适用于某资产时必须显式标记 `NOT_APPLICABLE`，不能以 `0` 同时表示“未知”和“无此字段”。

### 4.2 ID、时钟与幂等

- `client_order_id` 在账户和厂商保留窗口内跨进程、跨重启唯一，并在首次发送前持久化；其生成器状态也必须
  durable，不能因重启复用旧 ID；
- 官方存在唯一 `exec_id` 时，使用官方键去重；不存在时，必须按官方文档定义复合键并记录依据；
- 同一 execution 被重复、乱序或跨重启重放任意次数，只能产生一次资金和持仓效果；
- 使用交易所时间、UTC 接收时间、UTC 处理时间和单调时钟，不用单一时间字段混合语义；
- 交易日以券商/交易所会话为准，不按进程本地自然日推导；
- 超时只表示 `AMBIGUOUS`，绝不等价于报单失败，查询确认前禁止盲目重发。
- SDK 调用边界发生进程崩溃时，使用 `PREPARED -> DISPATCH_STARTED -> SDK_ACCEPTED/SDK_REJECTED/AMBIGUOUS`
  的持久状态；`PREPARED` 可按规则继续 dispatch，`DISPATCH_STARTED` 一律先 query/reconcile。

### 4.3 Broker Adapter 能力

替换当前过窄的 `IOrderAdapter`，最小能力为：

```text
connect / authenticate / disconnect / health / capabilities
query_trading_session
query_account / query_positions / query_open_orders / query_executions
submit / cancel / cancel_all
on_order_event / on_execution / on_account_event / on_connection_event
```

SDK 同步返回仅表示“本地调用是否受理”，正式订单状态只能由官方异步回报或查询确定。
生产 Adapter 必须输出 capability matrix；不支持的订单类型/TIF 在风险前置阶段拒绝。

官方 capability matrix 必须证明：行情可以在 gap 后恢复可信，交易/账户查询可以建立一致性边界，遗漏执行
回报可以从官方查询恢复。若官方接口不提供 snapshot/sequence/watermark，必须有经过官方确认且能稳定收敛的
全量重订阅/重复查询方案；两者都没有时，该厂商不满足 Phase 0 退出条件。

如果厂商只提供 Python SDK 或只支持特定 OS，则增加独立 `VendorBridge` 进程和版本化 IPC；核心 OMS、账户和
风险仍留在 C++。bridge 是条件性独立施工包，必须定义跨进程 outbox、fencing、IPC 背压、请求歧义和恢复，
并覆盖任一进程被 kill 后的查询对账与去重。是否采用 in-process C++ Adapter 或 bridge 必须由官方 SDK
约束决定。

### 4.4 金额、估值与会计口径

- 实盘热路径的价格、金额、费用和汇率使用带 scale 的定点整数，并对溢出和舍入方向做失败关闭；
- 日初权益、日内损益、最大回撤和 gross/net exposure 的公式在 S1 固定；
- mark price 的来源、stale mark 处理、预计费用与最终费用、结算差异和外部出入金必须分别建模；
- 初始持仓必须归属到策略、保留仓或外部持仓，目标仓位不得默认占用或卖出未认领资产；
- broker 结单仍是外部权威，本地预计费用不能被当成最终对账金额。

## 5. Readiness、模式与故障原则

### 5.1 Readiness 状态机

```text
BOOT
 -> AUTHENTICATED
 -> REFERENCE_SYNCED
 -> MARKET_SYNCED
 -> ACCOUNT_SYNCED
 -> ORDERS_RECONCILED
 -> STRATEGY_READY (INFRA_CANARY_READY 或 MODEL_READY)
 -> RISK_READY
 -> OPERATIONS_READY
 -> SHADOW_READY / PAPER_READY / LIVE_READY
 -> DEGRADED / KILLED / STOPPED
```

只有下列条件同时为真时才允许 `LIVE_READY`：

- 官方 Adapter 登录成功且会话未过期；
- symbol master、交易日历、tick/lot、涨跌停和交易状态已加载并冻结；
- 行情完成官方 snapshot + 增量衔接，序列连续且未过期；
- 资金、持仓、活动委托和当日成交已在厂商水位或稳定收敛边界内完成查询，查询期间的回报已重放；
- 本地 journal 与 broker 对账完成，未知差异为零；
- 当前选择的策略已就绪：`INFRA_CANARY` 的签名 manifest/白名单有效，或模型制品的 schema、hash、warmup 和
  对应晋级标志有效；
- 风控配置有效，账户快照新鲜，kill switch 可用；
- 生产 journal 可写、磁盘容量正常、单实例 fencing 有效；
- 独立紧急控制入口、告警、supervisor 和对应 runbook 已通过演练；
- `operational_threshold_contract` 已加载，所有 freshness、延迟、容量和对账判断都有冻结数值；
- ModeRouter 已通过当日、账户、额度和 operator arming 校验。

任一条件失效，默认行为是禁止增加风险，同时保留查询、撤单和经风控确认的减仓能力。

### 5.2 最后一跳物理门禁

ModeRouter 必须位于 `VendorBrokerAdapter::submit()` 的唯一上游，不能只靠策略的 `shadow` 标志：

| 模式 | 策略/风控 | 真实 submit |
| --- | --- | --- |
| `SHADOW` | 全部执行并记录 | 物理禁止，调用计数必须为 0 |
| `PAPER`（仅 `live-sim`） | 全部执行并由 simulator 撮合 | 物理禁止，调用计数必须为 0 |
| `INFRA_CANARY / MODEL_CANARY` | 全部执行 | 仅对应签名 manifest 的白名单账户/标的/数量/金额/时间窗 |
| `LIVE` | 全部执行 | 仅 DeploymentManifest 与当日 arming 额度内 |
| `REDUCE_ONLY` | 只允许降低风险 | 禁止任何增加绝对仓位的命令 |
| `KILLED` | 停止新决策 | 禁止新单；允许经过审计的撤单 |

拆分 `live-sim` 与 `live-prod` 两种产物：`live-sim` 可包含 simulator/PAPER，但不链接生产 Adapter；
`live-prod` 可进入 SHADOW/CANARY/LIVE，但不包含 Mock、simulator 或 PAPER 路由。进入 `CANARY/LIVE` 至少
需要两道独立门禁：不可变部署配置允许实盘，
以及运行时针对账户、交易日和限额的显式 arming。默认额度为零，重启或跨交易日后自动失效。
`INFRA_CANARY` 的 manifest 只能通过受限 `CANARY` 路径，ModeRouter 必须拒绝其进入普通 `LIVE`；普通 `LIVE`
只能加载已生成 `MODEL_LIVE_APPROVED(model_hash)` 的模型 manifest。

### 5.3 同账户单实例

V1 不做双活。每个 `environment + broker + account` 必须持有唯一 fencing lease/锁；第二实例只能只读退出。
锁丢失或 fencing token 不一致时立刻禁止新单。具体使用操作系统锁还是外部 lease，由部署环境决定。

本系统 fencing 不能阻止手机、券商终端或另一主机直接操作同一账户。V1 应使用专用账户/子账户和专用终端
标识，并冻结人工交易政策；若券商无法提供排他会话，则任何外部订单都触发 `DEGRADED + REDUCE_ONLY`。
非空账户必须先完成持仓 ownership/allocation，未认领持仓不得参与策略 target delta。

### 5.4 可测试运行阈值

Phase 0/S1 必须产生版本化 `operational_threshold_contract`，至少冻结：

```text
market_stale_ns / account_snapshot_max_age_ns / max_clock_offset_ns
callback_queue_high_watermark / callback_queue_hard_limit
journal_fsync_warn_ns / journal_fsync_fail_ns / minimum_disk_free_bytes
order_ack_timeout_ns / cancel_timeout_ns / max_pending_age_ns
reconnect_attempt_budget / reconciliation_timeout_ns
cash_position_fee_tolerance / RTO / RPO
shadow_days / certification_days / canary_days
```

数值由目标市场、SDK、策略频率和用户风险限额决定，在 Phase 0 未冻结前不填默认值。Readiness、告警、测试和
Gate F 必须读取同一合同，禁止仅使用“新鲜”“正常”“足够”“过久”等不可验收描述。

## 6. 施工包

### WP0：基线、构建与生产组合

施工内容：

- C++ 仓库当前无提交，Python 工程未发现 Git 基线；先建立可回滚提交、版本号和 release tag；
- 增加隔离的 `live-sim` 与 `live-prod` 构建组合；`live-prod` 包含 live + ML + strategy + journal +
  account + risk + 指定 Vendor Adapter，但不链接 Mock/simulator；
- 生产构建记录 compiler、SDK、代码 commit、schema、配置和模型 hash；
- 修正 CI 组合遗漏，加入 ASan/UBSan/TSan、故障注入和完整 live integration suite；
- Mock 与 Production Adapter 产物分离，生产包拒绝 Mock 配置。

完成标准：可从干净环境重复构建，产物可证明来源，并可回滚到上一个已验收版本。

### WP1：统一交易领域契约

施工内容：

- 建立独立 `trading_domain` 库，冻结 V2 schema 和 enum 映射；
- 消除 `engine_common`、`cpp_engine`、`trading_engine` 三套订单枚举之间的 ordinal 转换；
- 增加 schema version、向后兼容读取和拒绝未知必填字段的规则；
- 为回测、simulator、live adapter 分别实现显式转换器。
- 对所有价格、金额、费用和数量冻结 scale、舍入与溢出合同；冻结估值、外部出入金和费用最终化事件。

完成标准：同一标准化输入在回测、Shadow 和 LiveRuntime 中逐字段保留 decision/order/execution 关联关系。

### WP2：LiveJournal、快照与确定性回放

施工内容：

- journal record 分类：raw market、normalized market、bar、decision、risk、order command、broker report、
  account、reconcile、runtime/operator event；
- 文件头包含 magic、schema、endianness、run/session、build 和 adapter 版本；每条记录包含单调 sequence、CRC；
- OrderCommand 先 durable append，再调用 SDK；执行回报先 durable inbox，再改变 OMS/账户投影；
- outbox 状态至少包含 `PREPARED`、`DISPATCH_STARTED`、`SDK_ACCEPTED`、`SDK_REJECTED`、`AMBIGUOUS`；每个状态
  变更先持久化。对新单和撤单分别实现恢复规则，不能用一次“发送失败”覆盖调用边界歧义；
- 持久化 dedup key、broker ID、未知状态、outbox 发送结果和 raw report 引用；
- checkpoint 使用“临时文件写入 -> fsync 文件 -> 原子 rename -> fsync 目录”，禁止先删旧快照；
- journal 故障、磁盘满或 CRC 中段损坏时 fail closed；尾部半记录只能按明确规则截断并告警；
- 为 kill/cancel-all 提供独立预留的 emergency journal 或第二持久化目标；主 journal 失败时只允许走
  break-glass 撤单，并在恢复后用 broker 查询补录证据，不能把普通无审计报单当成紧急路径；
- 生成不可变且签名的 LiveVerificationReport，绑定 build/SDK/OS/manifest、脱敏账户、交易日、订单/成交 ID、
  journal hash、券商结单、演练和审批；SDK/schema/关键 OS/风险合同变化后自动失效；
- 从真实 capture 回放同一个 reducer/LiveRuntime，不再维护语义不同的特殊 replay 链路。
- 对涉及真实委托的 journal segment、decision 输入引用（含 Bar/行情快照 payload 或不可变 capture locator）、
  risk 和 broker report 建立连续 hash 链，按固定大小/时间轮转并同步复制到独立故障域；只有本地 durable 和
  独立副本均确认时才允许真实 submit，关键审计 `RPO=0`，
  不能只依赖本机单盘；
- 独立副本不可用、复制延迟超过阈值或 hash 链断裂时立即 disarm；备份校验、恢复到新主机和证据留存期纳入
  Gate C，恢复后必须重新获得 fencing 并完成 broker 全量对账。

完成标准：在任意持久化边界强制终止进程，重启后订单、账户和 dedup 投影一致，未知订单不被盲重发。

### WP3：纯 OMS reducer 与可恢复 Outbox

施工内容：

- 用纯状态归约器处理 pending、ack、partial fill、filled、cancel pending、cancelled、rejected、ambiguous、
  `EXPIRED`、`SESSION_CANCELLED`、`SUSPENDED` 等厂商状态；映射必须保留原始状态和会话阶段；
- 支持 fill-before-ack、重复/乱序回报、cancel/fill race、迟到成交、撤单拒绝和外部订单；
- 状态未发生经济变化时，不向 AccountMirror/Strategy 重复发布；
- SDK 调用超时进入 `AMBIGUOUS`，先 query/reconcile，不能自动重新 submit；
- 在 `PREPARED/DISPATCH_STARTED/SDK_ACCEPTED` 每个边界执行 kill/restart 测试；厂商无法按 client ID 查询或
  不能通过官方语义解除歧义时，Adapter 不得通过 Phase 0；
- 外部/人工订单导入为 `EXTERNAL`，触发对账和风险降级，不能当 duplicate 丢弃；
- 实现 cancel-all；若 SDK 无原子能力，按官方查询结果枚举撤单并记录不完整结果。
- DAY 委托在会话结束进入明确的到期/会话撤销终态，释放未成交量对应的资金/仓位预占；若厂商只在次日查询
  才返回到期状态，启动恢复必须补齐该状态后才能重新 arming。
- 优雅停机先 disarm 并冻结新 dispatch，再按冻结策略撤活动单、持久化未决状态和对账；不得为了“清空
  outbox”而发送尚未下发的新风险订单。

完成标准：状态机 property test 和官方 conformance fixture 覆盖所有合法/非法顺序；任何执行报告重复 N 次的经济效果等于一次。

### WP4：AccountMirror 与 ReconciliationEngine

施工内容：

- 建立资金、可用资金、冻结资金、购买力、费用、总仓、可用仓、今/昨仓和活动订单预占投影；
- 对 A 股以 broker 返回的可卖数量为权威，不再用进程本地自然日推导 T+1；
- 启动和重连查询交易日、账户、持仓、活动订单和当日成交；
- 查询开始前缓冲实时回报；每次分页带 request ID/完成标志，并在厂商 snapshot/watermark 后重放缓冲事件；
- 厂商没有原子水位时，按官方语义重复全量查询直至连续快照稳定收敛；超时或无法证明完整性时禁止 arming；
- 对账规则显式分类：matched、broker-only、local-only、quantity/cash mismatch、ambiguous；
- broker-only/未知订单不得静默忽略；默认进入 `DEGRADED + REDUCE_ONLY` 并等待人工或规则化处置；
- 应用 position ownership/allocation，只允许策略改变归属自己的持仓；Canary 残仓必须有次日退出或书面接管计划；
- 周期性与日终重新对账，差异、容差和处置均写 journal。

启动恢复顺序：

```text
获取 fencing
 -> 校验 journal / 加载快照 / replay
 -> Adapter 只读登录
 -> 开始缓冲回报并获取官方 snapshot/watermark
 -> 分页查询交易日、资金、持仓、活动订单、成交
 -> 应用快照并重放水位后的缓冲回报；无水位时循环至稳定
 -> 对账至零未知差异
 -> 行情同步、模型 warmup、风控就绪
 -> 才允许 arming
```

完成标准：任意重启后，四方“本地订单、券商订单、券商成交、账户快照”在同一一致性边界完成对账前，
submit 调用为零；测试覆盖查询中成交、分页中断和重连回报重放。

### WP5：真实行情完整性与 BarAggregator

施工内容：

- 接入版本化 `ReferenceDataProvider`：symbol/venue 映射、交易日历、会话阶段、tick/lot、涨跌停、上市/停牌
  和盘中状态；明确券商、行情和研究三套 ID 的映射及变更事件；
- 开盘前加载并冻结当日 reference snapshot，热路径只传稳定 SymbolId，消除动态 vector 扩容并发风险；盘中
  参考数据变化必须带版本/生效时间，不能静默改写已产生的 Bar；
- 区分 snapshot、incremental、session、sequence、exchange time 和 receive time；
- 永久 gap、checksum 错误、队列溢出或 snapshot 过期进入 `RESYNC_REQUIRED`，不得自行恢复 trusted；
- 正确 resync：请求/加载官方快照，缓存增量，以 snapshot sequence 为基线重放并校验连续性；
- 若厂商没有序列化 snapshot，只有经官方确认且通过故障测试的全量重订阅隔离方案才可替代；无法证明恢复
  完整性时 Phase 0 不得退出；
- BarAggregator 依据交易日历、会话和 watermark 生成与回测相同的完整 `MarketFrameBatch`；
- 缺标的显式带 validity/quality，不得清空其他标的历史；迟到数据策略固定并审计；
- 正式策略禁止同 Bar `CLOSE` 偷价，只允许与实盘可实现的 cutoff/NEXT_OPEN 语义对齐；
- 行情 stale、半截面或 warmup 不足时禁止新风险。

完成标准：永久 gap 后所有新风险订单为零，只有完成官方 snapshot/sequence 或 Phase 0 接受的官方等价恢复
方案并重建完整截面后才恢复。

### WP6：LiveRiskGate、KillSwitch 与额度

最低前置检查：

- readiness、模式、账户/行情新鲜度、序列完整性和交易时段；
- symbol 上市/停牌/涨跌停、tick、lot、订单类型和 TIF 能力；
- 价格 collar、单笔数量、单笔金额、重复意图和订单频率；
- 相反方向活动委托、自成交/交叉风险、撤单频率与券商异常交易限制；
- 可用现金/购买力、T+1 可卖、未完成订单预占；
- 单标的仓位、组合 gross/net exposure、行业/因子集中度；
- 当日亏损、最大回撤、模型/特征过期、推理超时；
- broker/gateway/account/journal/disk/fencing 健康；
- Canary 白名单和剩余额度。

日损与回撤严格使用 WP1 冻结的日初权益、mark、费用和外部出入金口径；所有 freshness、超时、磁盘和队列
判断使用 `operational_threshold_contract`，不能在代码内另设隐式默认值。

风险状态为 `NORMAL / REDUCE_ONLY / KILLED`。每一次批准与拒绝均输出 `RiskDecisionEvent`，包含规则 ID、
输入快照引用、阈值和配置 hash。任何路径都不能直接绕过 LiveRiskGate 调用生产 Adapter。

Kill 默认执行“禁止新风险 + 撤活动单”；是否自动减仓必须单独授权。A 股 T+1 下不能承诺当日自动平仓。

完成标准：所有限额在并发活动订单下仍按 projected exposure 计算；kill 后下一决策周期 submit 新风险次数为零。

### WP7：ExecutionPlanner 与订单执行策略

施工内容：

- 输入 `TargetPositionBatch + AccountMirror + active orders + trusted market + adapter capabilities`，输出
  可持久化的 `ExecutionPlan/OrderIntent`；
- 按 projected position 计算差额，projected position 必须包含已成交、未完成买卖量和待撤订单；
- 冻结限价生成、tick/lot 取整、DAY TIF、最大 child quantity、参与率、有效时间和价格 collar；当前默认
  `MARKET` 的 `OrderIntent` 不得直接进入生产；
- 同一 decision/target 重放必须幂等，相同目标在已有有效 child order 时不得重复报单；
- 处理部分成交、撤单中成交、超时撤单、cancel-replace、新目标覆盖旧目标和“撤完再决策”屏障；
- 每个 child order 保存 originating decision、plan version 和剩余目标，重启后由 OMS/AccountMirror 接管并继续
  向目标收敛，禁止重新从零生成一套重复委托；
- 定价和撤换策略是 DeploymentManifest 的版本化部分，未经 Shadow/Canary 验证不得在线热改。

完成标准：相同 target 重放任意次数只产生一组经济委托；部分成交、目标更新和进程重启后最终仓位仍按规则
收敛，且不存在相反方向活动订单、自成交或重复 rebalance。

### WP8：策略运行时与模型发布

施工内容：

- 将 pybind 中的 ArtifactLoader 下沉为 C++ 共享库，standalone LiveRuntime 原生加载制品；
- 校验模型、feature schema、universe、calendar、训练数据截止、晋级标志和全部 SHA-256；
- 完成 Bar warmup、推理 deadline/watchdog、非有限输出和 stale feature 的 fail-closed；
- 实现 `on_execution()`/订单状态回调，策略投影来自 OMS + AccountMirror，不自行假定成交；
- 修复缺失标的导致特征历史被错误清空的问题；
- 只有 WP3-WP7 readiness 与执行计划已接通后才移除 `context.live` 拒绝；
- 模型切换采用双实例加载、warmup、原子切换和可回滚 manifest；
- 保存 `decision_id -> target -> execution_plan -> intent -> risk -> order -> execution` 全链路。

当前 Phase 5 状态为 `promotion_eligible=false`，所以当前研究制品不得直接使用真实资金。平台实盘验收和具体
模型经济晋级是两条门禁：

- `INFRA_CANARY` 是签名、确定性、单一用途的测试策略，只能生成 Phase 0 冻结的白名单最小目标；它必须完整
  经过真实行情、AccountMirror、ExecutionPlanner、LiveRiskGate、OMS 和 Adapter，禁止人工直调 submit；
- `INFRA_CANARY` 只用于生成 `PLATFORM_LIVE_VERIFIED` 证据，不会把任何研究模型标为 live；
- 某个模型通过预注册 OOS、成本、Shadow 和风险门槛后只能生成 `MODEL_CANARY_ELIGIBLE(model_hash)`；
  继续通过模型专属真实 Canary 后，才能生成 `MODEL_LIVE_APPROVED(model_hash)`。

完成标准：同一真实 capture 在 Shadow 与 LiveRuntime 回放中产生相同 decision、risk 和 order intent。

### WP9：真实 Vendor Adapter

施工内容：

- 按官方 SDK 实现 Market、Broker、Query 三类能力及统一回调 normalizer；
- 厂商回调线程只做字段校验、复制和入有界队列，不执行策略；执行/账户回报使用独立高优先级通道；
- 任一回调队列饱和时原子标记会话失信、立刻禁止新 dispatch 并触发查询对账；市场事件触发 resync，执行/
  账户事件必须能从官方查询恢复。厂商无法恢复遗漏回报时，不满足 Phase 0 能力门槛；
- 完成登录、心跳、流控、重连、交易日切换、错误码映射和 capability negotiation；
- 保存脱敏后的 adapter/SDK/server/session 版本和 raw report；
- 建立官方模拟/认证环境 conformance fixture；若厂商没有模拟环境，必须在 Phase 0 冻结官方认可的替代认证
  方案；两者都没有时不得退出 Phase 0；
- 没有官方保证的行为保留 `AMBIGUOUS` 并 query-before-resend；若使用 VendorBridge，完成 IPC outbox、
  fencing、背压和任一进程崩溃恢复测试。

完成标准：官方模拟/认证环境或 Phase 0 接受的官方替代方案覆盖登录、订阅、行情恢复、报单、确认、部分成交、
撤单、拒单、查询和重启恢复。通用接口或 Mock Adapter 完成不算本 WP 完成。

### WP10：LiveRuntime、SessionController 与 ModeRouter

施工内容：

- 用单一编排层串起行情、Bar、策略、ExecutionPlanner、风险、OMS、账户和 Adapter；
- 实现 readiness 状态机、交易会话、开盘/午休/收盘、优雅停机和跨交易日重置；
- `submit()` 只能由 ModeRouter 调用，添加生产调用计数和审计；
- 实现账户级 fencing、每日 arming、额度消耗和自动过期；
- `DEGRADED` 下保留查询、撤单和受控 reduce-only，不产生新风险；
- 信号处理不直接丢失未决事件；停机时先 disarm、冻结新 dispatch、按策略撤活动单、持久化未决状态并对账，
  只 flush journal，禁止为了清空 outbox 发送新风险。

完成标准：不满足任一 readiness 条件时，真实 Adapter submit 调用严格为零。

### WP11：Python 控制面与实盘只读投影

Python 工程新增：

- 不可变 `DeploymentManifest`：代码、模型、schema、universe、calendar、风险、adapter 和审批 hash；
- 模型状态：research/reference/shadow/model_canary/live/revoked；`INFRA_CANARY` 使用独立 manifest 类型，不能
  改写模型晋级状态；
- `LiveEventProjector` 将 journal/事件流投影到独立 `live.db`，不复用回测 `TradeStore` 作为真值；
- Dashboard 展示 readiness、账户、持仓、活动委托、成交、PnL、对账差异、风险拒绝和告警；
- 控制命令经认证接口进入 C++ 并写 journal；Dashboard 默认只读，禁止通过直接改表实现 kill/arming；
- `Broker` 重命名为 `CostModel`，避免与真实 BrokerAdapter 混淆。

完成标准：即使 Python/Dashboard 停止，C++ 的风险、订单、账户和恢复仍正确；Dashboard 恢复后可从 journal 重建。

### WP12：安全、部署、监控与 Runbook

施工内容：

- 凭据只通过 OS secret store、环境注入或厂商安全设施引用，配置、日志和 crash dump 全部脱敏；
- 最小文件权限、生产账号隔离、配置签名/hash、操作员命令鉴权和审计；
- 提供不依赖 Python、Dashboard 和 `live.db` 的鉴权、防重放本地运维入口，可执行 kill、cancel-all、查询和
  只读恢复；正常与 break-glass 命令使用不同权限并保留证据；
- 用 systemd/Windows Service 等目标 OS 原生 supervisor 管理单实例、自动重启和退出码；
- 定义 journal/配置/验收证据的保留、加密备份和恢复演练；灾备实例默认只读，接管前必须取得新 fencing、
  完成 broker 全量查询和重新对账，不做未经验证的双活；
- 校验时钟同步、磁盘空间、journal fsync 延迟、CPU/内存、网络和 SDK 健康；
- 关键指标：行情 stale/gap、bar lag、账户快照年龄、未知订单、对账差异、pending age、拒单、
  journal/disk、订单/成交延迟、风险状态、日内 PnL；
- 告警和 runbook：断线、行情 resync、未知订单、磁盘满、模型超时、kill、cancel-all、只读重启、
  人工解除、日终对账和回滚。
- 完成程序化交易准入、行情许可、日志留存和必要报备清单；不在本文臆定具体法规条文。

完成标准：运维演练不依赖修改代码；所有高风险操作有身份、原因、时间和结果记录。

### WP13：模拟器、故障注入和测试矩阵

必须具备的自动化测试：

| 层次 | 必测场景 |
| --- | --- |
| 契约 | schema 兼容、enum 显式映射、price/qty scale、未知字段失败 |
| 行情 | 半包/粘包、重复/乱序、永久 gap、snapshot 衔接、队列溢出、迟到 Bar |
| OMS | ack/fill/cancel/reject 任意合法顺序、fill-before-ack、cancel-fill race、外部订单 |
| 会话生命周期 | DAY 到期无回报、会话撤销、挂起后恢复、次日查询才发现终态、预占释放 |
| 幂等 | execution 重放、进程重启、journal 尾损坏、checkpoint 中断、SDK 超时 |
| 账户 | 资金/冻结/总仓/可卖仓、费用、T+1、broker-only/local-only 对账 |
| 一致性恢复 | 查询中成交、分页中断、回报缓冲/重放、无水位时稳定收敛、恢复超时 |
| 执行规划 | target 幂等、限价/tick/lot、部分成交、超时撤换、新旧目标冲突、重启接管 |
| 风控 | 价格/数量/金额/仓位/频率/日亏损、stale/gap、kill、reduce-only、额度并发 |
| 策略 | Bar cutoff、完整截面、warmup、模型超时/非有限、shadow/live parity |
| 并发 | callback 队列饱和、执行高优先级通道、SymbolRegistry 冻结、TSan、背压、优雅停机 |
| Adapter | 官方登录、心跳、重连、查询窗口、错误码、流控和生产认证用例 |
| 运维 | 跨主机双进程、人工/外部订单、主 journal 磁盘满时 break-glass cancel-all、Python 全停、网络中断、进程 kill -9、灾备恢复、跨交易日、Canary 残仓和回滚 |
| 审计灾备 | journal 轮转/hash 链、独立副本不可用、整机/单盘丢失、备份校验、恢复后重新 fencing/对账 |

完成标准：本地 deterministic broker simulator、真实 capture replay 和官方模拟/认证环境或 Phase 0 接受的
官方替代方案三层均通过；只通过 Mock 单元测试不能进入生产。

## 7. 施工顺序与依赖

```text
S0 / Live Phase 0：外部冻结与 Git 基线
 |
 v
S1 合同冻结：WP0 + WP1 + readiness/risk/execution/mode/threshold 合同
 |
 +--------------------+
 v                    v
S2 通用核心                 S3 官方 Adapter
WP2-WP8, WP10, WP13         WP9 + 官方 conformance
 |                          |
 +------------+-------------+
            v
S4 官方非生产联调：完整恢复、断线、kill、对账、全天 soak
            |
            v
S4.5 PRODUCTION_OPERATIONS_READY：WP11 + WP12 完成并演练
            |
            v
S5 生产只读 + Shadow：真实行情/账户/委托/成交同步，submit=0
            |
            v
S6 真实 Canary：受控 submit/ack/cancel + 至少一笔最小合法 fill
            |
            v
S7 日终/次日对账 + 预冻结 Canary 周期通过
            |
            v
PLATFORM_LIVE_VERIFIED=true
```

建议按以下可独立验收的合并单元施工：

1. 基线与隔离的 `live-sim`/`live-prod` 构建；
2. V2 领域契约和 schema tests；
3. journal/checkpoint/replay；
4. OMS reducer/outbox/dedup；
5. AccountMirror/reconciler；
6. market integrity/bar/session；
7. ExecutionPlanner；
8. LiveRiskGate/ModeRouter/fencing；
9. StrategyRuntime/ArtifactLoader/LiveRuntime；
10. broker simulator + fault suite；
11. 真实 Vendor Adapter；
12. Python deployment/live projection、独立控制入口、安全运维和 Dashboard；
13. 官方环境、生产只读、Canary 与最终验收。

Live Phase 0 未退出前不提供虚假日历工期。退出后应依据官方 SDK 语言、OS、认证排期和测试账户可用性，
对 WP9、S4-S7 单独估算；通用核心与 Adapter 可以并行，但必须在 S4 汇合。WP11/WP12 可以并行施工，但
未通过 S4.5 不得连接生产账户。

## 8. 晋级门禁

### Gate A：通用核心完成

- simulator 下所有状态机、幂等、恢复、风控、bar 和回放测试通过；
- sanitizer/TSan 和故障注入通过；
- `live-prod` 无 Mock、simulator 和 PAPER 路由；
- 尚不连接生产账户。

### Gate B：官方非生产环境完成

- 官方 SDK conformance 清单逐项有真实证据；
- 登录、行情、报单、撤单、成交、查询、断线和重启恢复通过；
- 使用官方模拟/认证环境；若厂商没有该环境，只接受 Phase 0 已书面冻结的官方替代认证方案；
- 全天 soak 无未知订单、重复经济成交或不可解释账户差异。

### Gate C：生产运维就绪

- WP11/WP12 已完成，生产凭据、权限、supervisor、监控、告警、控制入口和 runbook 可用；
- Python/Dashboard 全停时，仍可鉴权执行 kill、cancel-all、查询和只读恢复；
- 主 journal 磁盘满时的 break-glass 撤单与事后查询补录演练通过；
- 主机/磁盘完全丢失后，从独立副本恢复 journal、配置和验收证据，审计关键记录 `RPO=0`、恢复时间满足
  `RTO`，并重新完成 fencing 与 broker 对账；
- 专用账户/人工交易政策、生产准入、行情许可、日志留存和必要报备已有可归档证据；
- 所有 `operational_threshold_contract` 数值已冻结并进入统一配置。

### Gate D：生产只读与 Shadow 完成

- 生产账户真实资金、持仓、活动委托、成交和行情同步；
- Readiness/告警/runbook 全天运行；
- Shadow 对相同 Bar 的决策、ExecutionPlan、风险和订单意图可确定性回放；
- 真实 Adapter 的 submit 调用严格为零。

### Gate E：真实 Canary 授权

必须在当次交易日前冻结并由用户明确确认：

```text
账户 + 交易日 + 标的白名单 + 方向 + 最大数量 + 最大金额
+ 价格边界 + 最大损失 + 有效时间窗 + 中止条件
```

平台 P0 阻断项必须全部清零。使用 `INFRA_CANARY` 时不要求当前研究模型晋级，但该策略必须签名、确定性且
只能走完整交易链；使用具体模型时还必须已有 `MODEL_CANARY_ELIGIBLE(model_hash)` 资格。
Canary 先完成真实 submit/ack/cancel，再完成至少一笔最小合法真实成交。买入后受 T+1 或会话规则约束的
残仓必须在授权前冻结次日退出或书面接管方案。

### Gate F：实盘验收

- 成交、手续费、现金、总仓、可卖仓和活动订单与券商结果一致；
- 跨重启和下一交易日 T+1 可卖仓对账一致；
- 预冻结的真实 Canary 周期内，未知订单、重复经济成交、风险绕过和账户残差均为零；
- kill/cancel-all/只读恢复/回滚演练通过；
- 生成并签名不可变 `LiveVerificationReport`，绑定代码、SDK、OS、manifest、账户、交易日、订单/成交证据、
  journal hash、券商结单、阈值和审批；
- 只有有效报告存在时才能派生 `PLATFORM_LIVE_VERIFIED=true`。SDK/schema/关键 OS/风险合同变化后报告自动
  失效并重新验收，禁止直接写布尔值。
- 若本次使用具体模型完成模型 Canary，同时生成该模型的签名审批证据，才能派生
  `MODEL_LIVE_APPROVED(model_hash)=true`。

## 9. 失败关闭矩阵

| 事件 | 自动状态 | 允许操作 | 禁止操作 |
| --- | --- | --- | --- |
| 行情 gap/stale/半截面 | `DEGRADED` | 查询、撤单、受控减仓 | 新增风险 |
| 账户快照过期/对账不一致 | `REDUCE_ONLY` | 查询、撤单、经确认减仓 | 新增风险、盲目修正 |
| 订单结果不确定 | `DEGRADED` | query/reconcile | 重发同一经济订单 |
| 回调队列饱和 | `KILLED` | 官方查询、对账、行情 resync | 新 dispatch、假定事件未丢失 |
| journal/磁盘故障 | `KILLED` | 只读查询、带 emergency journal 的 break-glass 撤单或券商 mass-cancel | 新单、普通无审计命令 |
| 模型/特征超时或非法 | `REDUCE_ONLY` | 现有订单管理 | 使用旧预测静默下单 |
| fencing 丢失/第二实例 | `KILLED` | 第二实例只读退出 | 两实例同时报单 |
| 外部/人工订单出现 | `REDUCE_ONLY` | 导入、查询、ownership 对账 | 忽略外部订单、继续按旧 target 增仓 |
| 风险越限/日损触发 | `KILLED` 或 `REDUCE_ONLY` | 按冻结策略撤单/减仓 | 增仓 |
| SDK 断线 | `DEGRADED` | 重连后查询和对账 | 重连后立即 READY/盲重发 |

## 10. 本版不可妥协的验收清单

- [ ] 已指定真实券商、官方 API/SDK 版本和部署 OS；
- [ ] `live-sim`/`live-prod` 产物隔离，生产构建没有 Mock/simulator/PAPER 路由，provenance 完整；
- [ ] 未达到全部 readiness 条件时，真实 submit 次数为零；
- [ ] 同一 target 不重复下单，部分成交、撤换和重启后由 ExecutionPlanner 正确收敛；
- [ ] OrderCommand 先持久化后发送，AMBIGUOUS 不盲重发；
- [ ] 同一 execution 任意重放只入账一次；
- [ ] broker-only、local-only 和外部订单均有明确处置；
- [ ] 永久行情 gap 只能通过官方 snapshot/sequence 或 Phase 0 接受的官方等价方案恢复 trusted；
- [ ] AccountMirror 在同一 snapshot/watermark 或稳定收敛边界内与 broker 持续对账；
- [ ] 所有订单必须经过 LiveRiskGate 与 ModeRouter；
- [ ] SHADOW/INFRA_CANARY/MODEL_CANARY 无法通过任何旁路直接调用生产 submit；
- [ ] 同账户单实例 fencing、专用账户/人工交易政策和持仓 ownership 生效；
- [ ] 当前未晋级模型不能进入 MODEL_CANARY/LIVE；INFRA_CANARY 不改变模型晋级状态；
- [ ] 回调队列饱和、主 journal 磁盘满和 Python 全停的故障演练通过；
- [ ] 完成生产 submit/ack/cancel 和至少一笔真实 fill；
- [ ] 完成日终及下一交易日账户对账；
- [ ] 运维、告警、kill、cancel-all、只读恢复和回滚演练通过；
- [ ] 已生成有效、签名且可失效的 LiveVerificationReport，由此派生 `PLATFORM_LIVE_VERIFIED=true`，而不是只写 `LIVE_READY=true` 或手工布尔值。

## 11. 下一决策

用户补齐 2.2 的外部输入后，完成 Live Phase 0 退出评审，并将本文件中的建议范围改为冻结范围；随后为目标
SDK 增补：

1. Adapter 字段映射表；
2. 官方状态/错误码映射表；
3. 回调线程与队列模型；
4. 行情 snapshot/sequence 恢复算法；
5. 带 snapshot/watermark 或稳定收敛边界的查询与启动对账时序；
6. ExecutionPlanner 定价、撤换、部分成交与重启合同；
7. 账户隔离、持仓 ownership、Canary 残仓与人工交易政策；
8. operational threshold、官方模拟/替代认证和生产运维用例；
9. 可执行工期、责任人和发布窗口。

在官方文档未确认前，上述厂商语义保持显式未决，不用经验或论文替代官方接口合同。

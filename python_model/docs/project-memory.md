# Quant Backtester 项目记忆与优化方案

更新日期：2026-07-20

## 1. 项目目标

这是一个面向个人使用、同时可用于量化开发岗位展示的 A 股回测引擎。

目标不是伪装成生产交易系统，而是把以下工程问题做深、做实：

- 分钟线历史数据可以在单机上流式回测。
- 多标的截面策略不会看到不完整或顺序依赖的行情。
- Python 适合策略研究，C++ 负责有状态执行热路径。
- 同一输入在 Python/C++ 后端产生一致结果。
- 交易成本、成交时点和绩效口径明确，没有隐式前视偏差。
- 回测结果可以持久化、复现和通过 Dashboard 检查。

## 2. 代码位置

- C++ 核心：`/Users/Zhuanz/CLionProjects/quant-backtester-cpp`
- Python 业务层：`/Users/Zhuanz/PycharmProjects/PythonProject`

## 3. 目标架构

```text
CSV/Parquet 原始行情
        ↓
数据清洗、校验、版本与分区
        ↓
Parquet Data Lake + DuckDB + Arrow
        ↓ 同一 timestamp 的完整截面
Python Strategy.on_cross_section()
        ↓ 批量订单
C++ Execution Engine
        ↓
Fill / Position / Cash / Equity
        ↓
SQLite 结果存储 + Streamlit Dashboard
```

### 3.1 数据层

分钟线主存储使用分区 Parquet，默认：

```text
year/month/symbol_bucket
```

默认 64 个稳定 bucket，文件内按 `timestamp,symbol` 排序。主键为：

```text
(timestamp, symbol)
```

`MinuteBarDataLake` 已支持 CSV/Parquet 导入、字段校验、幂等、catalog、Arrow
查询、截面查询、单标的历史查询、主键冲突检查和按字节限制的缓存。

### 3.2 回测执行层

历史行情不再先全部塞进 C++ 事件队列。`BacktestRunner` 按 timestamp 聚合一个完整
截面后立即处理，内存上限由当前截面和 Arrow batch 决定。

每个 timestamp 的处理顺序固定为：

1. 执行上一根 Bar 的 `NEXT_OPEN` 订单（市价单和限价单）。
2. 更新所有标的行情并触发限价单。
3. 调用一次截面策略。
4. 处理策略订单。
5. 统一 mark-to-market，记录一个权益点。

事件优先队列保留给小规模乱序事件、定时器和兼容接口，不作为历史 Bar 主通道。

### 3.3 Python/C++ 边界

- Python：数据查询、策略、实验编排、结果分析。
- C++：订单状态、撮合、手续费回调、持仓、现金、权益和事件状态。
- `QBT_BACKEND=auto|cpp|python` 显式选择后端。
- auto 只在 `cpp_engine` 未安装时回退；二进制加载失败不能静默回退。
- `on_market_data` 保留兼容旧策略，截面策略使用 `on_cross_section`。

## 4. 当前已完成

### 回测核心

- C++ `OrderBook` 市价/限价、多档、挂单和 Bar 触发。
- Python/C++ `PositionTracker` 平均成本、平仓、反向开仓和已实现盈亏。
- `NEXT_OPEN` / `CLOSE` 成交时点。
- 现金、手续费、权益和交易记录。
- 总收益、年化收益、Sharpe、最大回撤、胜率。
- 同 timestamp 截面批处理和严格递增时间检查。
- Python/C++ parity test。

### 数据系统

- `Bar` 统一数据契约。
- CSV/Parquet 读取和 schema/OHLCV 校验。
- Parquet 分区数据湖。
- DuckDB 查询与 Arrow RecordBatch 流式接口。
- catalog 原子提交、导入锁、幂等和 `(timestamp,symbol)` 冲突检测。
- `CSVDataFeed`、`ParquetDataFeed`、`DataLakeFeed`。
- Arrow Dataset Scanner：predicate pushdown、column projection、RecordBatch streaming。
- Partition-aware replay：月/日自适应窗口、软内存目标、严格有序和完整截面。
- Arrow C Stream C++ bridge：原生消费 RecordBatch，跨 batch 保留截面并拒绝乱序。
- 数据血缘：catalog generation、schema hash、内容级 dataset/query fingerprint。
- 不可变物化特征缓存：绑定代码、参数和 point-in-time 上下文。
- batch event interface：历史 replay 已实现，未来 live adapter 保留抽象边界。
- IO/decode/compute/bridge/execution 分阶段 benchmark。

### 构建与测试

- CMake 4.4 配置成功。
- `cpp_engine` 和 `trading_engine` 均可构建。
- 默认 CTest 7/7 通过。
- 正式 Python 工程：强制 Python 后端 100 passed / 6 skipped，强制 C++ 后端 126 passed。
- Python/C++ 后端一致性已验证。
- `game` 目录和相关构建配置已完全删除。

## 5. 当前完成度

```text
回测核心             90%
数据系统             90%
跨语言边界           90%
结果持久化           85%
Dashboard            80%
A 股真实交易规则     80%
低延迟交易引擎       25%
```

## 6. 下一阶段执行顺序

### P0：让项目形成完整闭环（已完成）

1. `TradeStore.save_equity_curve()` 已实现上海时区日权益聚合。
2. `BacktestRunner` 已接入事务化 run/trades/equity/metrics 持久化。
3. Dashboard 主页已读取真实指标。
4. 权益、交易分析和风险指标页面已实现。
5. 已增加回测到 SQLite 的端到端测试，并用浏览器验证全部页面。
6. 运行开始前冻结 lineage、RunSpec 和随机种子上下文；SQLite 保留
   `RUNNING/SUCCEEDED/FAILED` 状态及失败错误信息。

验收标准：运行一个示例策略后，可以在 SQLite 和 Dashboard 中完整看到同一次回测。

### P1：补齐 A 股回测可信度

当前状态：已完成。

已完成：

1. Python/C++ 统一 `ExecutionConfig`：默认成交量参与率 10%、滑点 0 bp、启用
   涨跌停限制和 T+1。
2. `MarketSnapshot` 接收显式 point-in-time 状态：`upper_limit`、
   `lower_limit`、`is_suspended`；不根据股票代码猜板块或历史 ST 状态。
3. 停牌、零成交量、涨停买入和跌停卖出不成交。
4. 市价单支持滑点、Bar 成交量参与率和跨 Bar 部分成交。
5. T+1 按 Asia/Shanghai 自然日滚动 `sellable_quantity`，并预留未成交卖单数量，
   防止多张订单合计穿透可卖数量。
6. 策略可读取现金、单个/全部持仓、成本、已实现盈亏和可卖数量，并使用统一的市价、
   限价、目标仓位、平仓和撤单方法；`on_order_update()` 避免拒单/部分成交后业务状态失真。
   `ColumnarStrategy` 在 C++ 后端直接消费 `MarketBatchView` 并返回列式订单。
7. CSV/Parquet 保持基础 OHLCV 七字段兼容，同时可透传三个可选交易状态字段。
8. C++ 边界测试和 Python/C++ parity 已覆盖停牌、零成交量、涨跌停、滑点、
   部分成交和 T+1。
9. `ChinaAShareCalendar` 使用显式交易日，支持分钟起始/结束时间戳约定、午休、
   连续竞价、开盘集合竞价和收盘集合竞价校验。
10. `SecurityMaster`、`AdjustmentFactorStore`、`CorporateActionStore` 均采用
    point-in-time 查询；历史股票池不会使用回测结束时仍存续的股票倒推。
11. 原始 OHLC 用于成交，`signal_price = raw_price * adjustment_factor` 仅用于策略；
    现金分红和送转股通过公司行动直接调整现金、数量和成本。
12. 买入最小数量、整手数量、上市状态均由历史证券状态显式提供，不根据当前代码
    规则猜测；卖出允许一次性清理零股。
13. 引擎在接受订单时冻结现金/可卖数量，拒绝资金不足、持仓不足、未上市、非法数量
    和非法订单，并保存拒绝原因。
14. 订单生命周期覆盖 `ACCEPTED / PARTIALLY_FILLED / FILLED / CANCELED /
    REJECTED / EXPIRED`；公司行动会先撤销该标的未完成委托。
15. Portfolio 只读视图提供现金、权益、总/净敞口、最大持仓权重、行业暴露和任意
    风格因子暴露。
16. SQLite 已保存订单最终状态、公司行动和最终组合风险快照，拒单不再从闭环中丢失。
17. `Broker` 使用 point-in-time 费率表，默认覆盖 2023-08-28 印花税由千一降至
    万五；佣金、最低佣金和过户费均可按历史区间配置。
18. 引擎拒绝负初始资金、非法 OHLCV/状态字段和负数/非有限手续费，避免错误输入
    静默污染结果。

当前执行模型的明确取舍：

- Bar 级模型没有委托队列数据，因此对涨停买入、跌停卖出采用保守的“不成交”近似。
- 参与率造成的市价单剩余数量会滚到该标的下一根 Bar，不模拟交易所内具体市价单类型。
- `volume=0` 表示无流动性，不再解释为“成交量缺失时无限成交”。

验收结果：C++ CTest 6/6；强制 Python 后端 50/50；强制 C++ 后端 50/50，
Parquet 和 DuckDB 数据湖测试无跳过。

### P2：提升分钟线规模能力（已完成）

1. `ArrowDatasetScanner` 已支持 predicate pushdown、column projection 和 batch streaming。
2. `PartitionAwareIterator` 已支持月/日自适应窗口、超大窗口继续拆分、软内存目标、
   严格 `(timestamp,symbol)` 排序和 timestamp 截面原子性。
3. C++ 已通过 Arrow C Data Interface 原生消费 `RecordBatchReader`，不构造逐 Bar Python
   对象；Parquet decode 和 symbol 到 `std::string` 的复制仍明确保留。
4. benchmark 已拆分 IO、decode、compute、bridge 和 execution，记录 rows/s、MB/s、
   peak RSS、cache state、batch p50/p95、查询与环境。
5. 已实现 catalog generation、schema hash、fragment 内容 hash、dataset/query fingerprint，
   并随回测结果写入 SQLite。
6. 已实现按 `year/month/bucket` 保存的不可变物化特征缓存，键绑定代码、参数、数据、
   交易日历、历史股票池和复权口径。
7. 已实现历史 batch event replay，并为未来实时行情定义 `LiveMarketSource` 抽象接口；
   P2 不实现网络、重连和柜台。

关键取舍：DuckDB 继续用于研究 SQL；固定 schema 快速主路径使用 Arrow Scanner；Scanner
不保证跨 fragment 全局有序，回放必须经过 partition-aware 层；物理布局继续使用月分区和
稳定 symbol bucket，不改成每日分区；内存目标允许被单个完整截面突破。

基准报告：`docs/benchmarks/p2-baseline.json`，fixture 为 200 标的 × 5 天 × 240 分钟，
共 240,000 行。P2 最终验收：正式 Python 工程强制 Python 后端 103/103、强制 C++ 后端
103/103；默认 CTest 3/3，`all-modules` 6/6，ASan/UBSan 3/3。

### P3：性能与实时实验（进行中）

1. 已把 benchmark 拆成存储 IO、Arrow decode/compute、整段 C Stream、事件回放和完整
   策略回测；warm/cold 不再只是标签，报告记录调用次数、批大小和 RSS 边界。
2. P3 warm 基线使用 240,000 行、1,200 个 timestamp：整段 C Stream 约 20.2 万行/秒，
   事件回放约 19.5 万行/秒，带 20 笔成交和 10 个 round-trip 的策略回测约 14.5 万行/秒。
3. 正式 `BacktestRunner` 已接入 Arrow point-in-time 富化与批量执行；C++ 原生按 timestamp
   使用历史费率，`allow_short` 与 T+1 已拆分。C++ 性能分支已引入 `SymbolId`、定点货币、
   `MarketBatchView`、轻量历史配置和分页查询；下一步按 profiling 结果评估批量费用模型。
4. 实时模块只有在协议 framing、OMS 幂等、断线重连、状态对账和本地交易所模拟器形成
   端到端闭环后，才进入 cache-line、对象池和纳秒级延迟优化。
5. 连续行情现在只在公司行动、timer 和 session 等稀疏事件边界 flush；24 万行基线的
   C Stream 调用从 1,200 次降到 5 次，事件回放约 19.5 万行/秒。
6. 快路径已覆盖证券状态、复权、历史费率、公司行动、行业/风格暴露和订单账本 parity，
   带 `stream_batches()` 的标准数据源默认使用该路径；自定义 Broker 保留对象回调路径。

此前“作品化与求职展示”中的 README、性能报告、RunSpec、策略参数/代码/数据版本保存已
在 P2 交付中完成，不再占用 P3 阶段编号。

## 7. 暂不投入的方向

- Kafka、Spark、微服务和分布式数据库。
- 在没有真实 L2 数据时构建复杂盘口模拟。
- 未经本地模拟器验证的生产实盘网络或柜台接入。

`trading_engine` 保留为独立扩展方向，但应在回测闭环和 A 股规则完成后再投入。

## 8. 已知边界

- 当前数据湖 catalog 适合单机或共享 POSIX 文件系统；对象存储需要版本指针和条件提交。
- 历史数据修订目前采用新建数据版本，不覆盖已有主键。
- DuckDB 对跨大量文件的全局排序可能使用临时磁盘。
- `DataLakeFeed` 兼容路径仍会构造 `MarketSnapshot`；新的 batch replay 主路径使用
  Arrow C Stream，避免逐 Bar Python 对象构造。
- C Stream bridge 不是全链路完全零拷贝：Parquet decode 和 symbol copy 仍存在。
- `PartitionAwareIterator.target_bytes` 是软目标；完整截面和排序临时空间可令峰值 RSS
  高于目标，容量规划应以 benchmark 观察值为准。
- 涨跌停成交采用保守 Bar 级近似；需要逐笔委托队列数据才能模拟封单和排队成交。
- 交易日历必须来自可信数据源，本项目不通过“周一到周五”猜测法定节假日。
- 公司行动的现金分红值按数据源提供值直接入账；红利税和配股参与决策需要上层模型。
- 风格暴露由 point-in-time 数据提供，引擎只做持仓市值加权，不在回测时回看未来因子。

## 9. Transformer Phase E 数据与执行合同（2026-07-29）

1. 正式训练/评审数据使用 `price_adjustment_mode =
   pit_adjusted_signal_raw_execution`。原始 OHLC 仅用于 C++ 成交，`signal_* = raw *
   adjustment_factor` 同时用于 BAR_V1 特征和 NEXT_OPEN 标签。
2. Bar 表必须包含上市、停牌、ST、涨跌停、lot、行业、`industry_known_at`、
   `universe_asof`、`reference_data_known_at_max`；企业行动使用独立 PIT 表，不能从未来
   修订值或当前状态反推。
3. 模型 Benchmark 输出 `prediction_manifest.json`，按模型和 walk-forward window 指向冻结
   NPZ；每个预测文件按 `(timestamp,symbol)` 唯一升序。
4. C++ 预计算预测运行时只替代 ONNX inference，之后仍走生产
   `LongOnlyTopKPolicy -> LongOnlyOrderPlanner -> BasicRiskManager`，再由 C++ 引擎执行费用、
   T+1、涨跌停、现金和部分成交。Python 只负责编排，不使用 Python demo 回测引擎。
5. 正式组合报告固定比较动量、反转、等权、现金、Ridge、MLP、因果 TCN、GRU 和 V1.1，
   至少三个相同窗口，同时运行 0/5/10 bp；输出合同为
   `schemas/phase_e_portfolio_backtest.schema.json`。
6. BaoStock `adjustment=none` 原始 Bar 不能直接用于 Phase E。必须先与状态、PIT 行业、
   adjustment factor 和企业行动表按主键/as-of 规则富化。
7. `enrich-phase-e` 以状态表为日频骨架，明确停牌且缺 Bar 时只允许沿用上一原始收盘价，
   并记录 `bar_observed=false, volume=0`。行业可用时间是
   `max(provider known_at,snapshot_asof)`；月度快照不能回填到快照日期之前。
8. 在缺少历史逐日涨跌停价和 PIT lot/minimum buy quantity 时，富化报告固定为
   `MODEL_READY_EXECUTION_DEFERRED`。模型训练、Leakage、Walk-forward、预测任务消融和
   Attention Analysis 可继续；C++ 仅能运行关闭涨跌停/board-lot 约束的研究模式，报告
   必须标记 `promotion_eligible=false`。正式 Promotion 仍为 `INSUFFICIENT_EVIDENCE`。
   不能用当前板块规则反推历史。

## 10. 定义完成

当以下条件全部满足，回测引擎 Phase 1 视为完成：

- 一个真实分钟线数据集可以导入并查询。
- 截面策略可以运行并且不依赖股票顺序。
- Python/C++ 后端 parity test 通过。
- A 股成交规则测试通过。
- 回测结果自动落 SQLite。
- Dashboard 可以展示权益、交易和风险。
- README 能让新用户在本机复现一次完整回测。

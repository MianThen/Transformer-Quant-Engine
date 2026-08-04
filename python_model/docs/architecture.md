# 回测引擎架构

## 目标

这是个人可长期使用、同时可用于量化开发岗位展示的单机回测引擎。设计优先级是：

1. 无前视偏差和可解释的成交语义。
2. 全市场分钟线可流式运行，不随历史长度线性占用内存。
3. Python/C++ 后端对相同输入产生相同结果。
4. 组件边界清晰，可测试，但不引入个人项目不需要的分布式基础设施。

## 组件边界

```text
CSV/Parquet -> catalog + lineage -> Arrow Dataset Scanner
                                      |
                         PartitionAwareIterator
                         (有序、完整 timestamp 截面)
                                      |
                 +--------------------+--------------------+
                 |                                         |
          Python batch events                    Arrow C Stream bridge
   (calendar/action/timer/market)                          |
                 |                                         v
                 +----------------------------> C++ Execution Engine
                                                      |
                                          fills / positions / equity
                                                      |
                                     SQLite result + data lineage
```

- 数据层负责校验、版本、裁剪、排序和批次读取，不包含策略逻辑。
- Python 负责研究接口、截面策略和实验编排。
- C++ 只负责有状态热路径：待成交订单、成交、现金、持仓和权益。
- 普通策略使用对象截面；`ColumnarStrategy` 直接消费 C++ `MarketBatchView` 并返回列式订单，
  Python 参考后端提供相同语义的兼容视图。
- SQLite 只保存回测运行与结果摘要，不保存数十亿行原始行情。

## 分钟线主路径

`ArrowDatasetScanner` 使用 Arrow Dataset 对 Parquet 执行时间/标的 predicate pushdown、
column projection 和 RecordBatch 流式读取。Scanner 的原始输出不承诺跨 fragment
全局有序，因此不能直接作为确定性历史回放。

`PartitionAwareIterator` 根据 catalog 中文件范围和压缩大小选择月或日窗口；窗口仍过大时
继续拆分。每个窗口只在内存中排序，并把输出 batch 边界扩展到完整 timestamp，确保一个
截面不会被拆开。`target_bytes` 是软目标：若单个完整截面本身超过目标，正确性优先于硬切分。

`process_arrow_stream()` 通过 Arrow C Data Interface 直接消费 `RecordBatchReader`，避免
在 Python/C++ 边界逐行构造 `MarketSnapshot`。这不是全链路零拷贝：Parquet 仍由 Arrow
解码，UTF-8 symbol 仍复制到现有 `std::string` 执行模型。桥接会拒绝非严格
`(timestamp, symbol)` 有序输入，并跨 RecordBatch 保留同一 timestamp 截面。

## 统一事件契约

`HistoricalReplaySource` 将同一批量接口组织为 session、corporate action、timer 和
market batch 事件，`BatchEventRunner` 负责消费。`LiveMarketSource` 只定义未来实盘适配器
边界；P2 不包含行情网络、断线重连和交易柜台实现。

## A 股可信执行顺序

每个完整 timestamp 截面固定执行：

1. 用历史时点证券状态富化上市/ST/停牌/涨跌停、整手、行业和风格暴露。
2. 用显式交易日历校验交易日、午休和竞价时段。
3. 应用上一个截面后生效的现金分红/送转股，并撤销除权标的未完成订单。
4. 按当时生效的佣金、印花税和过户费，执行上一根 Bar 遗留订单；同时受现金、
   T+1、涨跌停、停牌和成交量参与率约束。
5. 更新完整截面，调用一次策略并处理新订单。
6. 记录权益、订单状态和组合风险快照。

原始 OHLC 始终用于成交。复权字段只通过
`signal_price = raw_price * adjustment_factor` 提供给策略，避免用复权价格扣现金。

订单接受时冻结估算现金或可卖数量；最终状态为已成交、部分成交、拒绝、撤销或
回测结束过期。SQLite 同时保存成交和未成交委托，因而可以解释“为什么没有成交”。

## 主运行路径

带 `stream_batches()` 的标准数据源由 `BacktestRunner` 先在 Arrow 列上完成 point-in-time
证券状态、复权和风险因子富化，再把连续行情作为惰性 C Stream 交给引擎；只在公司行动、
timer 和 session 等稀疏事件边界切断。对象数据源仍按 timestamp 聚合完整截面。两条路径
都先处理 next-open/限价成交，再更新截面价格、回调策略并记录权益点；内存上限由当前
Arrow batch/截面决定，而不是由回测历史长度决定。

原事件优先队列保留给乱序小事件、定时器和接口展示，但不是历史 Bar 主通道。

## 后端与复现

`QBT_BACKEND=cpp|python|auto` 控制后端。auto 只在 C++ 模块确实未安装时回退，
二进制损坏不会静默切换。结果记录 backend；核心行为由跨后端 parity test 固定。

数据复现使用不可变 `CatalogSnapshot` 和 `DataLineage`：回放文件列表、catalog generation、
规范 schema hash、Parquet fragment 内容 hash、数据集 fingerprint 和查询 fingerprint
在运行开始时固定。`RunSpec` 进一步保存 backend artifact、策略代码/参数、随机种子、
执行与费率配置、交易日历/reference fingerprint 和环境摘要，并随结果写入 SQLite。
物化特征缓存键绑定同一数据与 point-in-time 上下文，防止跨版本误用。

## 性能证据

benchmark 将 IO、Parquet decode、Arrow compute、C Stream bridge 和 C++ execution
分别计时，并记录 rows/s、MB/s、峰值 RSS、batch p50/p95、cache state、查询和数据血缘。
基线报告见 `docs/benchmarks/p2-baseline.json`；单次小样本结果受操作系统缓存和调度影响，
比较时必须固定数据集、查询、机器和 cache state。

## 有意不做

- 不建设 Kafka、Spark、微服务或分布式数据库；个人单机回测得不到相应收益。
- 不用模拟 L2 盘口来伪装 Bar 数据的精度；成交模型必须声明可用的 OHLCV 假设。
- 不允许逐股票回调承担截面策略，否则同一分钟内会产生顺序依赖。

## P1 完成边界

- 交易日历不猜节假日，必须输入可信的历史开市日期。
- 股票池按指定 timestamp 查询，退市股票不会因为今天已不存在而从历史中消失。
- 科创板、主板等数量规则通过历史 `lot_size/min_buy_quantity` 提供，不根据代码猜。
- 行业与风格暴露由 point-in-time 数据提供；引擎只按当前持仓市值聚合。
- Bar 数据无法还原涨跌停封单排队，因此采用保守的涨停不买、跌停不卖模型。

## P2 完成边界

- Arrow Dataset Scanner、分区感知回放和 C Stream C++ bridge 已实现。
- 数据 lineage、物化特征缓存、分阶段 benchmark 和批量事件接口已实现。
- 保留月/稳定 symbol bucket 物理分区，不改为每日物理分区，避免小文件爆炸。
- DuckDB 继续服务研究 SQL 和复杂查询，Arrow Scanner 服务固定 schema 的快速主路径。
- P3 先比较整段 C Stream、事件回放和完整策略的真实开销，再依据 profiling 引入
  `SymbolId`、定点价格和批量费用；实时模块完成协议、OMS、重连与模拟器闭环前不做
  cache-line、对象池或纳秒级延迟优化。

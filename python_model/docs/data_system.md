# A 股分钟线数据系统

## 规模与目标

按 10,000 个标的、240 根分钟 Bar/交易日、250 个交易日/年和 20 年估算，
理论上限约 120 亿行。本系统优先优化多标的截面查询，同时保留单标的历史查询。

## 物理布局

```text
data_lake/
├── catalog.json
└── bars/
    └── year=2026/
        └── month=07/
            └── bucket=43/
                └── part-<ingestion-id>-0.parquet
```

默认使用 64 个稳定 symbol bucket，文件内部按 `timestamp,symbol` 排序：

- 截面查询只扫描目标月份的 64 组文件，并利用 timestamp row-group 统计裁剪。
- 单标的查询只扫描该 symbol 所在 bucket，读取量约为全市场的 1/64。
- 月分区避免“每日 × bucket”产生约 32 万个小文件。

## 数据契约

唯一主键为 `(timestamp, symbol)`，timestamp 是 UTC epoch 纳秒整数。字段固定为：

```text
timestamp:int64, symbol:string,
open:float64, high:float64, low:float64, close:float64, volume:int64
```

导入会检查 null、正价格、OHLC 关系、非负成交量、批次内重复和历史主键冲突。
CSV 股票代码按字符串读取，因此 `000001` 不会丢失前导零。

## 写入一致性

导入先写入 `.staging/<ingestion-id>`，完成校验后移动为不可变 Parquet fragment，
最后原子更新 `catalog.json`。查询只读取 catalog 已登记文件，因此不会看到半批数据。
源文件内容生成 SHA-256 幂等指纹，重复导入会直接跳过；每个已发布 Parquet fragment
也保存内容 hash。catalog 的 generation、files 和 sources 在同一把锁下生成不可变
`CatalogSnapshot`。一次查询、feed、scanner、分区回放、benchmark 和最终 lineage 都使用
同一个 snapshot，运行中新增 fragment 不会混入已经开始的回放。

## 两条查询路径

DuckDB 直接查询 catalog 选择出的 Parquet 文件，结果按 `timestamp,symbol` 排序。
小结果可使用 `query/history/cross_section` 返回 Arrow Table；大查询必须使用
`iter_batches`，以固定批大小返回 RecordBatch；跨多个 Parquet 文件的全局排序可能由
DuckDB 使用临时磁盘。物化结果缓存按 Arrow 实际字节数
执行 LRU 淘汰，catalog generation 进入缓存键，数据提交后旧缓存不会误命中。

固定 schema 的大规模主路径使用 `ArrowDatasetScanner`。它把 symbol/time predicate 和
column projection 下推到 Arrow Dataset，并流式返回 RecordBatch。Scanner 原始结果只保证
扫描效率，不保证多个 Parquet fragment 之间的全局顺序。Scanner 在构造时固定 catalog；
要读取后续 ingest 的 generation，必须构造新的 Scanner。

```python
from python.market_data import DataLakeConfig, MinuteBarDataLake

lake = MinuteBarDataLake(DataLakeConfig("/data/ashare-minute", bucket_count=64))
lake.ingest("incoming/2026-07-01.csv")

snapshot = lake.cross_section(1782869460000000000)
history = lake.history("000001", start=1704067200000000000)

for batch in lake.iter_batches(start=1704067200000000000, batch_size=65536):
    consume(batch)
```

```python
from python.market_data import ArrowDatasetScanner, PartitionAwareIterator

scanner = ArrowDatasetScanner(lake)

# 研究或聚合：下推过滤与列投影，不依赖全局顺序。
for batch in scanner.iter_batches(
    symbols=["000001", "600000"],
    start=1704067200000000000,
    columns=("timestamp", "symbol", "close", "volume"),
):
    consume(batch)

# 历史回放：严格按 (timestamp, symbol) 排序，且不拆分截面。
replay = PartitionAwareIterator(scanner, target_bytes=256 * 1024 * 1024)
for batch in replay.iter_batches(start=1704067200000000000):
    engine.process_arrow_stream(
        pa.RecordBatchReader.from_batches(batch.schema, [batch])
    )
```

## 分区感知回放与内存目标

回放器根据相关月份的 Parquet 压缩大小估算解码体积：窗口小于目标时按月处理，否则按
上海时区交易日处理；单日仍过大时继续切时间窗口。每个窗口内做
`timestamp,symbol` 排序，输出 batch 自动延伸到 timestamp 边界。

`target_bytes` 是软内存目标而非硬上限。原因是截面策略必须一次看到某一分钟的全部标的；
当单个截面大于目标时，系统会输出完整截面。估算还会受 Parquet 压缩率、Arrow buffer、
排序索引和 Python/Arrow 分配器影响，因此生产 benchmark 应同时观察峰值 RSS。

## Arrow C Stream 边界

C++ `process_arrow_stream(reader)` 原生消费 Arrow C Stream capsule，支持基础 OHLCV 与 P1
可选执行状态列。它会验证必需列、类型、null 和严格有序性，跨 RecordBatch 聚合同一
timestamp 后调用执行引擎。

`MarketReferenceData.enrich_batch()` 在进入 bridge 前批量追加上市/ST/停牌、涨跌停、
整手、行业、复权信号价和 `factor_exposure__*` 动态风格列，不创建逐 Bar Python 对象。
历史事件 Runner 会把稀疏事件之间的多个 RecordBatch 组成一个惰性 reader，因此不会为
每个 timestamp 重建 C Stream，也不会为合并而把整段历史物化进内存。

这里的“直接消费”特指不创建逐 Bar Python 对象。Parquet 解压仍由 Arrow 完成，symbol
仍进入现有 `std::string` 模型，因此不宣称从磁盘到撮合的完全零拷贝。

## 数据血缘

`lake.lineage()` 返回：

- catalog generation；
- 规范 Bar schema hash；
- 基于 fragment 内容 hash 和 source fingerprint 的 dataset fingerprint；
- 绑定 symbols、time range 和 columns 的 query fingerprint。

回测结果会把上述 lineage 写入 SQLite 的 `data_lineage` 表，并把包含该 lineage 的完整
`RunSpec` 写入 `run_config`。数据内容、schema、查询或执行配置改变都会产生不同身份，
支持结果复现和审计。

## 物化特征缓存

`MaterializedFeatureCache` 将因子按 `year/month/bucket` 保存为不可变 Parquet。缓存键绑定：

- 特征定义、代码 hash、版本和参数；
- schema、dataset 和 query fingerprint；
- 交易日历、历史股票池、复权口径和额外上下文。

物化过程使用跨进程文件锁、staging 目录和原子发布，并拒绝重复
`(timestamp, symbol)` 主键。`FeatureContext` 要求显式声明 calendar、universe 和
adjustment 身份，避免错误复用带来前视偏差。

```python
from python.market_data import (
    FeatureContext, FeatureDefinition, MaterializedFeatureCache,
)

definition = FeatureDefinition(
    name="momentum_20", version="1", output_columns=("momentum_20",),
    parameters={"window": 20}, code_hash="git-or-source-hash",
)
context = FeatureContext(
    calendar_fingerprint="calendar-v1",
    universe_fingerprint="universe-pit-v1",
    adjustment_mode="backward-adjusted-signal-only",
)
lineage = scanner.lineage(start=start, end=end)
entry = cache.materialize(definition, feature_table, lineage, context)
```

## Batch 事件接口

`HistoricalReplaySource` 把有序回放转换为完整 timestamp 的 market batch，并合并交易日
开闭、公司行动和 timer 事件。`BatchEventRunner` 对 C++ 后端优先使用 C Stream bridge，
Python 后端保留兼容路径。`LiveMarketSource` 是未来实时行情适配接口，P2 不实现网络连接。

## Benchmark

```bash
python -m python.benchmarks.generate_fixture /tmp/minute.parquet \
  --symbols 1000 --days 5 --minutes 240
python -m python.benchmarks.run_minute_replay /tmp/minute-lake \
  --target-mb 256 --cache-state warm --output /tmp/p3-report.json
```

报告独立记录 IO、decode、compute、整段 C Stream、事件回放和完整策略回测，并包含
rows/s、MB/s、RSS 起止/增量、进程高水位、C Stream 调用次数、batch p50/p95、环境、
查询和 lineage。warm 在每个读盘阶段前显式预热；Linux cold 使用 `posix_fadvise`，其他
平台要求 `--cold-cache-command`，不能可靠控制时直接拒绝运行。P3 基线位于
`docs/benchmarks/p3-baseline.json`，应视为特定机器、数据与 cache state 下的实验记录。

## 独立时态数据集

原始 Bar 不覆盖、不复权。以下数据通过 CSV/Parquet 单独维护：

```text
security_state:
  symbol,effective_from,effective_to,is_listed,is_st,is_suspended,
  upper_limit,lower_limit,board,industry,lot_size,min_buy_quantity,
  factor_exposures_json

adjustment_factor:
  symbol,effective_from,effective_to,factor

corporate_action:
  symbol,timestamp,cash_dividend_per_share,share_multiplier,description

trading_calendar:
  date,is_open
```

有效区间统一为 `[effective_from, effective_to)`；省略 `effective_to` 时由下一条记录
自然接管。查询股票池必须指定 timestamp，默认排除未上市、已退市和 ST 状态。

`MarketReferenceData` 在 Bar 离开数据湖后富化执行快照。成交仍使用原始价格；复权
因子只生成 `signal_*` 字段。公司行动由 Runner 按生效时间送入执行引擎，现金分红
进入现金，送转股调整数量与平均成本。

## 明确边界

- 当前 catalog 适合单机或共享 POSIX 文件系统；对象存储需要把 manifest 提交改成
  带条件写入的版本指针，不能依赖 `os.replace`。
- 修订历史数据当前采用“新建数据湖并切换版本”的方式，不允许覆盖已有主键。
- 官方节假日必须由外部可信交易日历提供，系统不会用工作日规则猜测。
- 截面策略必须按完整 timestamp 批次回调。现有逐 Bar 引擎接口只适合兼容旧策略，
  新的 C Stream 主路径已避免逐 Bar Python 对象构造，但策略回调仍遵循现有引擎契约。

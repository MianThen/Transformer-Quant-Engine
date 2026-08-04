# Quant Backtester

面向 A 股分钟线和多标的截面策略的个人回测引擎。

项目采用 Python/C++ 混合架构：Python 负责策略研究、point-in-time 参考数据和实验编排，
C++20 负责订单、成交、现金、持仓与权益等有状态执行路径。分钟行情保存在分区 Parquet
数据湖中，通过 Arrow Dataset 流式扫描，并使用 Arrow C Data Interface 直接进入 C++。

这不是一个只展示“双均线收益曲线”的玩具项目。项目重点解决三类问题：

- **回测可信度**：A 股 T+1、涨跌停、停牌、成交量约束、历史费率、公司行动和完整订单审计。
- **分钟线规模能力**：过滤下推、列投影、分区感知回放、完整 timestamp 截面和可控工作集。
- **结果可复现**：数据内容 hash、schema/dataset/query fingerprint、SQLite 审计和跨后端一致性测试。

## 项目状态

| 阶段 | 状态 | 内容 |
|---|---|---|
| P0 | 已完成 | DataFeed → 回测 → SQLite → Streamlit Dashboard 完整闭环 |
| P1 | 已完成 | A 股交易规则、point-in-time 参考数据、订单生命周期与风险审计 |
| P2 数据路径 | 已完成 | Arrow Scanner、分区回放、C Stream bridge、lineage、特征缓存与 benchmark |
| P2 结果交付 | 已完成 | round-trip 账本、SQLite v2、分页聚合、审计 Dashboard、安装包、CI 与质量门禁 |
| P3 性能与实时实验 | 进行中 | 真实 cache 控制、三条执行路径 benchmark；`SymbolId` 与实时闭环待实施 |
| ML 可选能力 | 基础协议已接入 | `BAR_V1`、`NEXT_OPEN`、外部训练代码与模型制品校验；默认关闭 |

当前验收结果：

- 强制 Python 后端：104/104 tests passed
- 强制 C++ 后端：104/104 tests passed
- C++ 默认核心构建：3/3 CTest passed
- C++ `all-modules` 构建：6/6 CTest passed
- ASan/UBSan 核心构建：3/3 CTest passed
- Python 分支覆盖率：76%（门禁 60%）

## 系统架构

```text
CSV / Parquet
      |
      v
校验 + 不可变分区数据湖 + 原子 catalog
      |
      +---- DuckDB ----------------------> 研究 SQL / 小结果查询
      |
      +---- Arrow Dataset Scanner -------> 过滤下推 / 列投影 / batch streaming
                                              |
                                   PartitionAwareIterator
                                   月/日窗口 + 局部排序
                                   完整 timestamp 截面
                                              |
                              Arrow C Stream / batch events
                                              |
                                              v
Python Strategy --------------------> C++ Execution Engine
                                              |
                         orders / fills / positions / equity
                                              |
                                              v
                               SQLite audit + Streamlit Dashboard
```

组件职责：

| 组件 | 职责 |
|---|---|
| Python | 策略、数据查询、交易日历、历史股票状态、复权和实验编排 |
| C++20 | 撮合、订单状态、资金冻结、持仓、T+1、费用、round-trip 和权益计算 |
| Parquet + Arrow | 全市场分钟线存储、裁剪和批量传输 |
| DuckDB | 研究 SQL、单标的历史和物化查询 |
| SQLite | 保存回测运行、委托、成交、权益、风险、公司行动和数据血缘 |
| Streamlit | 回测结果、交易分析和风险指标展示 |

详细设计见 [架构说明](docs/architecture.md) 和
[分钟线数据系统](docs/data_system.md)。

## 源码布局

当前源码由两个工程组成：

```text
quant-backtester-cpp/       # C++ 核心和 pybind11 模块
├── cpp_engine/             # 当前回测执行引擎
├── trading_engine/         # 独立低延迟实验模块，默认不构建
├── CMakePresets.json       # dev/release/cpp-only/sanitize/all-modules
└── pyproject.toml          # cpp_engine wheel 构建入口

PythonProject/              # Python 业务层和项目主 README
├── python/                 # 回测运行器、策略、数据系统和测试
├── storage/                # SQLite schema 与访问层
├── dashboard/              # Streamlit 页面
├── data/sample/            # 可直接运行的样例行情
├── docs/                   # 架构、数据系统、项目记忆与 benchmark
├── pyproject.toml          # Python 安装包与质量工具配置
└── requirements.lock       # Python 3.9/3.12 完整锁定环境
```

后续发布为单一仓库时，可以把两个目录合并；当前构建命令使用 `QBT_CPP_ROOT` 和
`QBT_PY_ROOT`，不依赖本机的绝对路径。

## 环境要求

- Python 3.9+
- 支持 C++20 的编译器
- CMake 3.21+（使用 presets；手工配置最低仍为 3.15）
- macOS 或 Linux

纯 Python 后端不要求编译 C++，适合先验证功能。C++ 后端需要使用同一个 Python
解释器安装 pybind11、配置 CMake 和运行回测。

## 快速开始

先设置两个源码目录：

```bash
export QBT_CPP_ROOT=/path/to/quant-backtester-cpp
export QBT_PY_ROOT=/path/to/PythonProject

# cmake/ctest 已在 PATH 时保留默认值即可。
export CMAKE_BIN="${CMAKE_BIN:-cmake}"
export CTEST_BIN="${CTEST_BIN:-ctest}"

# 若使用 macOS CMake.app，改为：
# export CMAKE_BIN=/Applications/CMake.app/Contents/bin/cmake
# export CTEST_BIN=/Applications/CMake.app/Contents/bin/ctest
```

### 1. 安装 Python 依赖

```bash
cd "$QBT_PY_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

### 2. 先运行纯 Python 版本

```bash
cd "$QBT_PY_ROOT"
QBT_BACKEND=python QBT_DB_PATH=/tmp/qbt-demo.db \
  python -m python.examples.ma_cross_strategy
```

### 3. 安装并运行 C++ 后端

推荐直接构建并安装 wheel：

```bash
cd "$QBT_CPP_ROOT"
CMAKE_BIN="$CMAKE_BIN" python -m pip install .

cd "$QBT_PY_ROOT"
QBT_BACKEND=cpp QBT_DB_PATH=/tmp/qbt-demo.db \
  python -m python.examples.ma_cross_strategy
```

需要开发构建时使用 CMake preset：

```bash
cd "$QBT_CPP_ROOT"
"$CMAKE_BIN" --preset dev \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DQBT_WARNINGS_AS_ERRORS=ON

"$CMAKE_BIN" --build --preset dev -j

export PYTHONPATH="$QBT_PY_ROOT:$QBT_CPP_ROOT/build/dev/cpp_engine"
cd "$QBT_PY_ROOT"
QBT_BACKEND=cpp QBT_DB_PATH=/tmp/qbt-demo.db \
  python -m python.examples.ma_cross_strategy
```

默认 `QBT_BUILD_PYTHON=ON`，因此 Python/pybind11 缺失时 CMake 会直接失败，
不会产生“构建成功但没有扩展模块”的结果。只构建纯 C++ 测试时可显式传入
`-DQBT_BUILD_PYTHON=OFF`。实验性的 `trading_engine` 默认关闭，只有
`all-modules` preset 或显式 `-DQBT_BUILD_LIVE_ENGINE=ON` 才会构建。

已验证的示例输出（Python/C++ 后端一致）：

```text
成交笔数: 7
期末权益: 1,196,138.76
总收益率: 19.61%
Sharpe:   3.85
最大回撤: 1.96%
```

`QBT_BACKEND` 支持：

- `cpp`：必须加载 C++ 模块，加载失败立即报错。
- `python`：强制使用纯 Python 参考实现。
- `auto`：默认值；仅在 `cpp_engine` 未安装时回退到 Python，不会吞掉二进制损坏错误。

## 策略业务接口

普通策略继承 `Strategy`，可直接使用 `market_order()`、`limit_order()`、
`target_position_order()`、`close_position_order()`、`cancel_order()` 和 `get_positions()`。
Runner 会把 `on_order_update()` 接到引擎的接受、部分成交和终态通知，策略不应在信号发出时
自行假定已经持仓。

高频截面研究可继承 `ColumnarStrategy`。C++ 后端直接传入只读 `MarketBatchView`，策略以
`symbol_index/side/quantity/type/price` 列返回订单，避免每个 timestamp 构造完整的 Python
`MarketSnapshot` 列表；纯 Python 后端会用兼容视图执行同一策略。内置均线和均值回归示例
已经使用该路径。

```python
from python.strategy import ColumnarStrategy

class EqualWeightSignal(ColumnarStrategy):
    def on_cross_section_view(self, batch):
        selected = [index for index in range(len(batch)) if batch.close(index) > 10]
        return {"symbol_index": selected, "quantity": [100] * len(selected)}
```

C++ 现金账本按 `1/10000` 货币单位逐成交量化；分析用成交价格仍保留浮点精度。因此由展示
价格反算的 round-trip PnL 与现金账本可能相差若干个最小货币单位。

### 4. 查看 Dashboard

```bash
cd "$QBT_PY_ROOT"
QBT_BACKEND=cpp \
QBT_DB_PATH=/tmp/qbt-demo.db \
QBT_DATA_LAKE_PATH=/tmp/qbt-data-lake \
  streamlit run dashboard/app.py
```

Dashboard 可查看同一次回测的总收益、年化收益、Sharpe、最大回撤、权益曲线、成交分析和
风险指标。成交、平仓轮次、委托和公司行动均使用 SQL 分页；运行审计页同时展示订单状态
聚合、组合风险、数据血缘和完整 RunSpec。

侧栏进入“数据实验”后，可以导入自己的 CSV、Parquet 或 PQ 行情文件，再选择标的和策略
运行回测。文件超过 100 MiB 时应选择“本机文件路径”：CSV 使用 Arrow 流式分块解析，
Parquet 使用 RecordBatch 迭代读取，校验、分区写入和回测回放都不会一次性加载完整数据集。
浏览器上传只用于 100 MiB 以内的小文件，因为上传组件本身会在进程中保留文件内容。

运行回测支持两种多标的业务模式：

- **独立批量**：每个标的独立使用界面中的初始资金，分别产生 run_id，完成后展示横向对比表。
- **共享资金组合**：所有标的共享一笔本金和一条权益曲线，策略为每个标的维护独立信号状态，
  按资金利用率等权分配预算，并由引擎统一执行现金、持仓和交易约束。

推荐的大文件操作流程：

1. 把行情文件保存在本机磁盘。
2. 打开“数据实验”并保留“本机文件路径”选项。
3. 输入单个文件或目录的绝对路径；单文件可先预览，目录会递归批量导入。
4. 切换到“运行回测”，选择标的、策略和参数后开始回测。

爬虫全量数据应选择单一数据源目录，例如：

```text
/Users/Zhuanz/PycharmProjects/scrapy/data/bars/provider=baostock
```

目录导入只保存有限路径和 source fingerprint，行情数据继续按 Arrow RecordBatch 流式读取；
同一批目录只提交一次 catalog，并自动跳过已导入文件。不要选择同时包含多个
`provider=*` 的上级目录，否则不同数据源可能包含重复 `(timestamp, symbol)`。

数据湖目录由 `QBT_DATA_LAKE_PATH` 控制；导入批次大小、CSV 读取块和 DuckDB 校验内存上限
可分别通过 `DataLakeConfig.ingest_batch_rows`、`ingest_csv_block_size_bytes` 和
`ingest_memory_limit_mb` 调整。

## 行情数据契约

CSV 与 Parquet 使用相同的基础字段：

```text
Bar(timestamp, symbol, open, high, low, close, volume)
```

| 字段 | 类型 | 约束 |
|---|---|---|
| `timestamp` | int64 | UTC epoch nanoseconds |
| `symbol` | string | 非空；保留 `000001` 等前导零 |
| `open/high/low/close` | float64 | 有限正数，满足 OHLC 关系 |
| `volume` | int64 | 非负 |

唯一主键是 `(timestamp, symbol)`。导入时会检查 schema、null、重复主键、OHLCV 合法性
以及与已有数据的主键冲突。

小文件和兼容策略可以直接使用 `CSVDataFeed` 或 `ParquetDataFeed`：

```python
from python.data_feed import CSVDataFeed

feed = CSVDataFeed(
    "data/sample/sample_ohlcv.csv",
    symbols=["000001"],
    start=1704067200000000000,
    end=1711929600000000000,
)
```

## 全市场分钟线数据湖

设计容量按 10,000 个标的、240 分钟/交易日、250 个交易日/年、20 年估算，理论上限约
120 亿行。物理布局为 `year/month/bucket`，默认使用 64 个稳定 symbol bucket：

```text
data_lake/
├── catalog.json
└── bars/
    └── year=2026/month=07/bucket=43/part-<id>-0.parquet
```

月分区避免每日分区造成小文件爆炸；symbol bucket 同时兼顾全市场截面扫描与单标的历史
查询。

```python
from python.market_data import (
    ArrowDatasetScanner,
    DataLakeConfig,
    MinuteBarDataLake,
    PartitionAwareIterator,
)
from python.engine_api import BacktestEngine

lake = MinuteBarDataLake(
    DataLakeConfig(
        "/data/ashare-minute",
        bucket_count=64,
        ingest_max_open_files=64,
    )
)
result = lake.ingest("incoming/2026-07-01.parquet")

# Arrow Dataset 负责过滤下推、列投影和原始 batch streaming。
scanner = ArrowDatasetScanner(lake)

# 历史回放负责全局确定性顺序与完整 timestamp 截面。
replay = PartitionAwareIterator(
    scanner,
    target_bytes=256 * 1024 * 1024,
)

reader = replay.reader(
    start=1782869400000000000,
    end=1782955800000000000,
    columns=("timestamp", "symbol", "open", "high", "low", "close", "volume"),
)
engine = BacktestEngine()
stats = engine.process_arrow_stream(reader)
```

需要明确的边界：

- Arrow Scanner 不保证多个 Parquet fragment 之间的全局顺序；历史回放必须经过
  `PartitionAwareIterator`。
- `DataLakeFeed` 和 `ArrowDatasetScanner` 在构造时固定 `CatalogSnapshot`；后续 ingest
  不会改变已经创建的回放及其 lineage。需要读取新 generation 时应创建新的 feed/scanner。
- `target_bytes` 是软目标。一个 timestamp 的完整截面不能拆开，因此单截面可能超过目标。
- Arrow C Stream bridge 消除了逐 Bar Python 对象构造，但 Parquet 仍由 Arrow 解码，
  `symbol` 仍会复制到现有 C++ `std::string` 模型；这不是全链路完全零拷贝。
- DuckDB 保留用于研究 SQL 和复杂物化查询，Arrow Scanner 用于固定 schema 的快速主路径。

## A 股回测可信度

执行模型已覆盖：

- T+1 可卖数量和卖单数量冻结；
- 订单接受时冻结预计现金，防止同一截面内资金透支；
- 停牌、零成交量、涨停买入和跌停卖出不成交；
- Bar 成交量参与率、滑点和跨 Bar 部分成交；
- 历史整手、最小买入数量、上市和 ST 状态；
- `NEXT_OPEN` 与 `CLOSE` 成交时点；
- T+1 与做空权限独立配置；关闭 T+1 不会隐式允许裸卖，只有 `allow_short=True`
  才允许负持仓；
- 市价单、限价单和 `ACCEPTED / PARTIALLY_FILLED / FILLED / CANCELED /
  REJECTED / EXPIRED` 完整生命周期；
- point-in-time 费率：原生引擎按委托/成交 timestamp 选择费率，默认佣金万三、最低 5 元，
  印花税覆盖 2023-08-28 由千一降至万五；
- 现金分红、送转股、持仓成本调整和公司行动前撤单；
- 总/净敞口、最大持仓权重、行业和风格因子暴露。

交易日历、证券状态、复权因子和公司行动均为独立的 point-in-time 数据集。成交始终使用
原始 OHLC；策略信号使用 `raw_price * adjustment_factor`，避免用复权价错误扣减现金。

```python
from python.backtest_runner import BacktestRunner
from python.market_data import (
    AdjustmentFactorStore,
    ChinaAShareCalendar,
    CorporateActionStore,
    MarketReferenceData,
    SecurityMaster,
)

calendar = ChinaAShareCalendar.from_file("reference/trading_calendar.csv")
reference = MarketReferenceData(
    security_master=SecurityMaster.from_file("reference/security_state.parquet"),
    adjustment_factors=AdjustmentFactorStore.from_file("reference/adjustment_factor.parquet"),
    corporate_actions=CorporateActionStore.from_file("reference/corporate_action.parquet"),
)

result = BacktestRunner(
    strategy,
    feed,
    calendar=calendar,
    reference_data=reference,
).run()
```

Bar 级数据无法还原涨跌停封单队列，因此当前采用保守近似：涨停不买、跌停不卖。若要模拟
排队成交，需要可靠的逐笔委托和成交数据，而不是在分钟 Bar 上伪造精度。

## 数据血缘与特征缓存

每次数据查询可生成以下身份：

- catalog generation；
- 规范 Bar schema hash；
- 基于源文件和 Parquet fragment 内容 SHA-256 的 dataset fingerprint；
- 绑定 symbols、时间范围和列投影的 query fingerprint。

血缘随回测结果写入 SQLite。`RunSpec` 还会保存实际 backend artifact SHA-256、策略代码
hash、显式策略参数、随机种子、执行配置、完整费率区间、交易日历/reference fingerprint
和环境摘要。物化特征缓存进一步绑定特征代码 hash、版本、参数、交易日历、历史股票池和
复权口径，避免修改数据或 point-in-time 上下文后误用旧因子。

## Benchmark

benchmark 将存储 IO、Arrow decode/compute、整段 C Stream、稀疏事件边界回放和完整
策略回测分开计时，并记录 rows/s、MB/s、RSS 起止/增量、进程高水位、调用次数、batch
p50/p95、cache 控制方法、查询与数据 fingerprint。报告同时保存实际 backend artifact
绝对路径、SHA-256、大小、修改时间和 Arrow/DuckDB/NumPy/pybind11 版本。

基线数据：200 标的 × 5 天 × 240 分钟，共 240,000 行；64 buckets；64 MiB 工作集目标；
arm64 macOS；warm cache。

P3 warm-cache 历史基线：

| 阶段 | 吞吐量 | C Stream 调用 |
|---|---:|---:|
| Arrow decode | 367,844 rows/s | - |
| 整段 C Stream | 201,805 rows/s | 1 |
| 事件回放 | 195,287 rows/s | 5 |
| 完整策略回测 | 144,642 rows/s | 5 |

完整报告见 [P3 benchmark JSON](docs/benchmarks/p3-baseline.json)，旧版阶段拆分保留在
[P2 benchmark JSON](docs/benchmarks/p2-baseline.json)。P3 完整策略默认在首个截面买入
10 个标的、下一截面卖出，产生 20 笔成交和 10 个 round-trip；它用于覆盖策略回调、订单、
费用和结果查询，不代表具体生产策略的性能。该历史报告未记录 CMake build type，不能作为
Release 性能回归门禁；新版 benchmark 会拒绝路径明确属于 dev/debug/sanitize 的扩展。

`warm` 会在每个读盘阶段前显式顺序预读；Linux 的 `cold` 使用
`POSIX_FADV_DONTNEED`，其他平台必须传 `--cold-cache-command`，无法可靠清缓存时会失败，
不会把未受控缓存标成 cold。`uncontrolled` 只用于探索性运行。

对已有数据湖运行 benchmark：

```bash
cd "$QBT_CPP_ROOT"
"$CMAKE_BIN" --preset release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
"$CMAKE_BIN" --build --preset release -j

cd "$QBT_PY_ROOT"
export PYTHONPATH="$QBT_PY_ROOT:$QBT_CPP_ROOT/build/release/cpp_engine"

QBT_BACKEND=cpp python -m python.benchmarks.run_minute_replay \
  /path/to/minute-lake \
  --target-mb 256 \
  --cache-state warm \
  --output /tmp/qbt-benchmark.json
```

## 测试

```bash
# Python 参考后端
cd "$QBT_PY_ROOT"
QBT_BACKEND=python python -m pytest -q python/tests

# C++ 后端及跨后端 parity
export PYTHONPATH="$QBT_PY_ROOT:$QBT_CPP_ROOT/build/dev/cpp_engine"
QBT_BACKEND=cpp python -m pytest -q python/tests

# 默认原生核心（3 项）
cd "$QBT_CPP_ROOT"
"$CTEST_BIN" --preset dev

# 显式包含实验模块（6 项）
"$CMAKE_BIN" --preset all-modules \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
"$CMAKE_BIN" --build --preset all-modules -j
"$CTEST_BIN" --preset all-modules

# ASan + UBSan
"$CMAKE_BIN" --preset sanitize
"$CMAKE_BIN" --build --preset sanitize -j
ASAN_OPTIONS=detect_leaks=1 "$CTEST_BIN" --preset sanitize

# 覆盖率门禁
cd "$QBT_PY_ROOT"
QBT_BACKEND=python coverage run -m pytest -q
coverage report
```

macOS 的 AddressSanitizer 不支持 LeakSanitizer，本机运行最后一条 CTest 时使用
`ASAN_OPTIONS=detect_leaks=0`；Linux CI 保留泄漏检测。

测试重点不是覆盖率数字本身，而是固定两种后端的业务语义，包括截面顺序、手续费、滑点、
部分成交、不可交易 Bar、T+1、拒单、订单过期、公司行动、风险暴露、数据血缘与 Arrow
C Stream 输入约束。固定 JSON golden 场景锁定部分成交后的手续费分摊和 round-trip
净损益；24 组固定随机种子执行轻量 property/fuzz 检查，验证现金守恒、持仓归零、净损益
守恒和订单不过量成交。

两个工程均提供 GitHub Actions：Python CI 覆盖 Python 3.9/3.12、锁文件安装、覆盖率、
源码编译检查和 wheel；C++ CI 覆盖严格告警、核心与全模块 CTest、wheel 安装冒烟以及
ASan/UBSan。

## SQLite 审计结果

一次回测会事务化保存：

```text
backtest_runs
├── trades
├── round_trips
├── orders
├── daily_equity
├── performance_metrics
├── corporate_actions
├── portfolio_risk
├── data_lineage
└── run_config
```

运行开始前会先保存 `RUNNING` 状态和冻结的 RunSpec；成功后原地更新为 `SUCCEEDED`，
异常则保留 `FAILED`、错误信息和实验配置。不仅成交会被保存，拒绝、撤销、部分成交和
过期订单也会保留，因此可以回答“为什么没有成交”。schema 当前为
`PRAGMA user_version=3`：旧库会自动逐版迁移，若数据库版本高于程序支持版本则拒绝打开。
大表查询提供有界 `limit/offset`、计数和 SQL 侧聚合，常用排序路径具有复合索引。

## 设计取舍

- **单机优先**：不引入 Kafka、Spark、微服务或分布式数据库，个人项目没有对应收益。
- **截面优先**：策略一次看到完整 timestamp，避免逐股票回调产生顺序依赖。
- **正确性优先于硬内存限制**：完整截面不可拆分，内存目标只能是软约束。
- **显式 point-in-time**：交易日历、股票状态、费率、复权和公司行动都不能用当前状态
  倒推历史。
- **不伪造盘口精度**：分钟 Bar 只能支持可解释的 Bar 级成交近似。
- **实盘边界明确**：历史回放与未来实时源共用 batch event 接口，但当前不实现行情网络、
  重连、柜台和生产级风控。

## 后续方向

- 增加完整的多标的截面策略案例与可复现实验配置；
- 在更大、公开可描述的数据规模上重复 benchmark；
- 按 profiling 结果评估 `SymbolId`、字符串复制与费用批量化；
- 合并 C++ 与 Python 源码布局，并在正式仓库发布跨工程集成 wheel。

项目的阶段记录和已知边界见 [项目记忆](docs/project-memory.md)。

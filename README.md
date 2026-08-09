# Transformer Quant Engine

> 总仓库：<https://github.com/MianThen/Transformer-Quant-Engine>

面向 **A 股日频/分钟 Bar** 研究的量化回测与 Transformer 模型接入工程，Python + C++ 双层架构。

## 当前状态：Live V1 施工中

Live V1 的施工边界和验收门禁见 [`docs/live_trading_v1_construction_plan.md`](docs/live_trading_v1_construction_plan.md)。当前项目仍处于 `LIVE_PHASE_0 / EXTERNAL_INPUTS_PENDING`：可以继续完成通用领域契约、Shadow 运行、回放和故障注入，但在券商、账户、行情、SDK 版本、部署环境和官方恢复语义冻结前，不会接入生产报单，也不会把 Mock 实现伪装成实盘能力。

本仓库是 C++ 执行层；Python 研究与控制面位于配套的 `PythonProject` 工程。两者通过版本化模型制品、行情/特征协议、运行配置和审计证据衔接。Python 控制面不直接调用券商报单 API，所有 Shadow 和未来实盘订单都必须经过 C++ 的策略、执行规划、风险门禁、ModeRouter 与 OMS 链路。

当前推进顺序：

1. 建立两个工程的独立 Git 基线并上传到明确的 GitHub 仓库；
2. 完成 `live-sim` 与 `live-prod` 产物边界、统一领域事件、可恢复 OMS/Journal、账户镜像和最后一跳物理门禁；
3. 使用真实行情和账户只读同步运行 Shadow，验证决策、执行计划、风险判断和订单意图的确定性回放，且 `submit=0`；
4. 只有取得目标券商官方 SDK 与生产授权资料后，才实现厂商 Adapter、生产只读和受控 Canary。

`SHADOW_READY`、`PAPER_READY`、`PROD_READ_ONLY` 和 `LIVE_READY` 都不是 `LIVE_VERIFIED`；任何真实资金操作必须经过不可变 manifest、当日 arming、风险额度、审计和人工批准。

### Shadow 冒烟测试

live 演示程序默认运行在 `SHADOW`。候选订单会写入 `runs/live-shadow.replay`，再由 `ModeRouter` 在最后一跳拦截，底层 Adapter 的 submit 次数必须保持为 0；审计文件可用 `--audit=PATH` 覆盖。

```bash
cmake -S . -B /tmp/qbt-live-shadow -DCMAKE_BUILD_TYPE=Debug \
  -DQBT_BUILD_PYTHON=OFF -DQBT_BUILD_LIVE_ENGINE=ON -DQBT_ENABLE_ML=OFF
cmake --build /tmp/qbt-live-shadow --parallel 2
ctest --test-dir /tmp/qbt-live-shadow -R test_mode_router --output-on-failure
```

`INFRA_CANARY`、`MODEL_CANARY` 和 `LIVE` 还需要 readiness、显式 arming、审计和对应白名单策略；当前命令行不会绕过这些门禁。

项目将**训练与执行分离**：


- **C++ 执行层**（本仓库 `quant-backtester-cpp`）：C++20 负责订单、成交、现金、持仓、风险以及可选 ONNX Runtime 推理——正式回测引擎；
- **Python 研究层**（`python_model`）：负责 point-in-time 数据、特征、标签、训练、校准、消融和 walk-forward 评估，以及数据湖、结果落库与审计 Dashboard。

项目的核心不是展示一条收益曲线，而是把回测这件事做得**可信、可复现、可审计**：T+1、涨跌停、停牌、整手、历史费率、公司行动全部建模且使用 point-in-time 历史数据；一次运行从行情来源、策略代码到成交、拒单、权益、风险的完整链条全部落库，任何结果都能回答"为什么"。

**当前状态**：基础 `BAR_V1`、多任务 Temporal Transformer、模型制品协议（Manifest V2）、Mock/ONNX 推理、C++ 模型策略回测入口、Leakage Detection、Ridge Walk-forward/Benchmark/Feature Ablation、Transformer Walk-forward、PyTorch ↔ ONNX Runtime Python ↔ ONNX Runtime C++ 六输出 golden parity、Phase D 原生 ArtifactLoader（lineage/SHA-256/schema 校验）、确定性推理分块与失败关闭、M4 Pro Release/LTO 性能基线均已就绪。**仓库不包含训练好的生产模型。**

## 整体架构

```text
Point-in-time Bars
  -> Python Feature/Label/Dataset        （BAR_V1 协议，NEXT_OPEN 对齐）
  -> TemporalTransformer                  （多任务：收益/波动率/方向/分位数/排序）
  -> ONNX Model Artifact                  （model.onnx + manifest + feature_schema + metrics）
  -> ONNX Runtime C++ 推理                （6 命名输出 + 置信度）
  -> Portfolio Policy                     （TopK / 风险预算）
  -> Order Planner / Risk                 （逐单风控）
  -> C++ Backtest Execution               （撮合 / 现金 / T+1 / 涨跌停 / 费用）
```

关键边界：

- 模型只输出预测，**不直接下单**；
- 信号使用复权价格，撮合、现金和涨跌停使用原始价格；
- `QBT_ENABLE_ML=OFF` 时 C++ 默认构建不依赖 ONNX Runtime；
- 模型、特征、标签、日历、股票池和数据截止时间必须通过版本化制品追踪（SHA-256 哈希链）；
- 纯 Python 回测实现仅保留为 demo，**正式模型回测以 C++ 引擎为准**。

---

# 一、C++ 执行引擎（本仓库）

C++20 高性能回测引擎，以 pybind11 扩展（`cpp_engine` 模块）供 Python 调用，并配套可选扩展模块。版本 `0.3.0`，CMake ≥ 3.15（Presets 需 ≥ 3.21），支持 Python 3.9+。

## 模块总览

| 目录 | 模块 | 职责 | 构建开关 |
| --- | --- | --- | --- |
| `engine_common/` | 公共基础设施 | 共享类型（SymbolId、定点数、MarketEvent/OrderIntent 等）、策略/模型接口（`IStrategyRuntime`）、回放文件格式（ReplayWriter/Reader）、SymbolRegistry，纯头文件 | 总是构建 |
| `cpp_engine/` | **核心回测引擎** | `BacktestEngine` + 撮合（OrderBook）、持仓（PositionTracker）、PnL（PnLTracker）、Arrow 接入、pybind11 绑定模块 `cpp_engine` | 总是构建 |
| `quant_math/` | 矩阵工具 | 矩阵视图/有限性/对称性/半正定校验（Eigen） | `QBT_ENABLE_PORTFOLIO_MATH` |
| `portfolio_math/` | 组合数学 | 7 种协方差估计器（含 QuEST 非线性收缩）、风险预算/风险平价求解、BH/BY/Storey FDR 多重检验校正、经验 CVaR | `QBT_ENABLE_PORTFOLIO_MATH` |
| `performance_analytics/` | 绩效分析 | 收益账本、实现落差（IS）核算、证券贡献账本、Brinson-Fachler 归因 + Menchero 链接、DSR/Newey-West 等统计 | `QBT_ENABLE_PERFORMANCE_ANALYTICS` |
| `ml_runtime/` | ML 推理运行时 | BarV1 特征管线（23 特征）、特征窗口存储、推理后端抽象（ONNX Runtime CPU / mock） | `QBT_ENABLE_ML` |
| `strategy_runtime/` | 策略运行时 | `ModelStrategyRuntime`（实现 `IStrategyRuntime`）、TopK/风险预算组合策略、基础风控 | `QBT_ENABLE_ML` |
| `trading_engine/` | 实盘交易引擎（实验性） | 行情接入（Decoder/FeedHandler）、订单网关（WAL 容错）、SPSC 无锁队列、epoll/IOCP 事件循环 | `QBT_BUILD_LIVE_ENGINE` |
| `python/qbt_ml/` | 研究侧 Python 工具 | 冻结 C++ 产物的只读校验、梯度冲突诊断（PCGrad/GradNorm）、漂移监控 artifact | —（pip 可选） |
| `benchmarks/` | 基准体系 | 黄金回放、性能基准与预算、归档报告 | `QBT_BUILD_BENCHMARKS` |
| `tools/` | 工具脚本 | 基准运行/预算检查/黄金对比/Phase 1E challenger 编排 | — |

## 核心回测引擎

### 特性

- **事件驱动、bar 级截面回测**：同一时间戳的一组标的行情作为一个截面原子处理，策略永远不会看到"半个截面"；策略订单只与 bar 的 OHLCV 外部流动性成交，挂单之间不互为对手盘
- **A 股撮合与交易规则**：`NEXT_OPEN`（默认）/ `CLOSE` 两种成交时点、市价/限价单、成交量参与率限制（默认 10%）、滑点、涨跌停拦截、T+1 可卖额度（含未成交卖单预留）、整手/最小买入数量约束、可选做空
- **严谨的订单校验与风控**：下单时预留现金与 T+1 可卖额度，成交时按实际成交价复核资金；非法订单、非上市、现金不足等拒绝原因可审计
- **多种费用模型**：`commission_fn` 回调或 `FeeSchedule` 分段费率表（佣金 = max(成交额×费率, 最低佣金) + 过户费 + 卖出印花税）
- **完整 PnL 与绩效统计**：加权平均成本持仓、实现盈亏、round-trip 配对（开平仓费用按比例分摊）、Sharpe、最大回撤、年化收益、胜率、权益曲线
- **公司行为处理**：现金分红、拆股/送股（乘数），自动撤销未完成订单、调整持仓与未平 round-trip、修正权益曲线
- **零拷贝 Arrow 接入**：通过 Arrow C Stream ABI 直接消费 PyArrow 数据（无需链接 libarrow），解码阶段释放 GIL，严格 schema 校验与 (timestamp, symbol) 有序性校验
- **确定性工程**：golden replay 黄金回放逐字节比对、回测引擎 ↔ 实盘链路 dual replay 双重校验、libFuzzer 模糊测试、ASan/UBSan/TSan 消毒构建

### 撮合模型（bar 级）

- **成交时点**：`FillTiming.NEXT_OPEN`（默认）— 订单在下一 bar 开盘价成交；`CLOSE` — 本 bar 收盘价成交
- **市价单**：按参考价 ± 滑点（`slippage_bps`）成交，受成交量参与率限制（`max_volume_participation`，默认 10%），未成交部分按成交时点顺延
- **限价单**：先检查本 bar 价格是否可触发（买入 `low ≤ limit`，卖出 `high ≥ limit`），未成交部分进入价格档位队列，后续 bar 触发后按 `max(open, limit)` / `min(open, limit)` 成交
- **涨跌停**：买入价 ≥ 涨停价、卖出价 ≤ 跌停价时被拦截（保守侧）；停牌/退市/零成交量不可成交
- 执行参数（`ExecutionConfig`）：`enforce_price_limits`、`enforce_t_plus_one`、`enforce_board_lot`、`enforce_cash`、`allow_short`、`market_order_price_buffer_bps`

### 订单校验、费用与绩效

- 订单校验：未知标的、非上市、非整手（卖出清仓可破整手）、T+1 可卖额度（含未成交卖单预留）、现金不足（按现价 + 滑点 + 缓冲预留）均被拒绝并记录 `RejectReason`；成交时按**实际成交价**复核资金。任何状态变更异常后引擎进入 "poisoned" 不可用状态，防止脏状态继续回测
- 费用：`set_commission_fn` 回调与 `set_fee_schedules` 分段费率表二选一（互斥）
- PnL：`PnLTracker` 按日权益序列计算 Sharpe（日收益年化，无风险利率 2%/252）、最大回撤、年化收益（365 天）、胜率；持仓实现盈亏以定点数（`MoneyMinor`，scale 10000）累积，避免浮点漂移

### 公司行为

`apply_corporate_action`：先撤销该标的所有未完成订单 → 按乘数调整持仓数量/均价/可卖数量（乘数必须产生整数股数，否则抛异常）→ 每股现金分红入账 → 同步修正未平 round-trip 与权益曲线 → 写入公司行为历史并通知绩效分析 sink。

## ML 运行时与模型策略

三层数据流：`BarV1FeaturePipeline`（固定 23 特征：log 收益、波动率、量能、动量的跨截面排序等）→ `FeatureWindowStore`（按标的维护 lookback 窗口、批量组装 + valid_mask）→ `IInferenceBackend` 推理产出 6 个命名输出（`expected_return` / `expected_volatility` / `direction_probability` / `lower_quantile` / `upper_quantile` / `confidence`）。

- **后端**：`OnnxRuntimeBackend`（CPU，PImpl 封装；加载时校验 descriptor 与 ONNX 图 schema 完全一致）与 `MockInferenceBackend`（确定性、无模型文件，便于无模型测试）
- **策略运行时**：`ModelStrategyRuntime` 实现 `IStrategyRuntime` 挂载进回测主循环，支持 `LongOnlyTopKPolicy` 与 `ResearchPortfolioPolicy`（`TOPK_EQUAL_WEIGHT` / `RISK_BUDGET`），`BasicRiskManager` 逐单风控（kill switch、行情可信、可交易、lot 整数倍、数量上限）
- 所有推理错误以 status 上报（`INSUFFICIENT_HISTORY` / `NON_FINITE_OUTPUT` / `SCHEMA_MISMATCH` 等），**绝不静默回退到旧预测**；shadow 模式只更新决策快照不下单
- **CUDA 暂不支持**：首个 ONNX Runtime 集成为 CPU-only，CUDA 需在 CPU golden parity 之后作为独立部署路径开启

## 组合数学与绩效分析（可选）

- **协方差估计**：样本协方差、Ledoit-Wolf 线性收缩（常数相关系数目标）、**LW-NLS-MV-QUEST 非线性收缩**（Ledoit-Wolf 2015/2017/2021 论文的 QuEST 二次特征值求解器，含 p>n 奇异分支与完整诊断）、RMT 谱去噪（Marchenko-Pastur）
- **风险预算**：`solve_long_only_risk_budget` / `solve_bounded_long_only_risk_budget`（风险平价为其特例），带 KKT 残差/换手率诊断
- **多重检验**：BH / BY / Storey FDR 校正与配对块自助法
- **尾部风险**：经验 VaR / CVaR（Rockafellar-Uryasev），可序列化为带 `promotion_eligible` 门控的 artifact
- **绩效分析**承载 "C++ Replay 权威收益核算合同"：`ReturnLedger`（收益桥/会计残差）、`ImplementationShortfallLedger`（验证 `paper_pnl - real_net_pnl = IS`）、`SecurityContributionLedger`（三条会计恒等式）、`PeriodContributionCoordinator/ReplaySink`、Brinson-Fachler + Menchero 归因、Deflated Sharpe Ratio、配对平稳自助、Newey-West HAC、有效试验数估计、因子打分分桶检验；所有产物确定性哈希，未通过门控时强制 `promotion_eligible=false`

## 实验性实盘引擎（Phase 2）

独立的低延迟实盘原型，将回测的"CSV 回放"升级为"实时接入"；线协议只存在于 Adapter 内部，策略/风控只消费 `engine_common` 的统一事件类型：

```text
行情侧:  TcpTransport → Decoder(模拟 ITCH 协议, 粘包/乱序/重排) → MockMarketDataAdapter → SPSC 无锁队列 → 策略线程
下单侧:  OrderIntent → MockOrderAdapter → OrderGateway(状态机 + WAL 预写日志 + 指数退避重连 + 心跳) → TcpTransport
```

- 自研 Mock 二进制定长行情协议（仿 ITCH，大端、pack(1)、定点价格）；解码器内置粘包/半包切分、64 槽乱序重排、gap/重复/畸形计数、永久 gap 跳档重同步
- `OrderGateway`：14 态订单状态机、WAL（CRC32 校验，SYNC_EACH/GROUP_COMMIT/ASYNC 三档持久化，尾部损坏截断恢复）、重连后 QUERY_ORDER 恢复、`reconcile()` 对账
- 无锁 SPSC 环形缓冲（cache-line 对齐、批量 push/pop）、`ObjectPool`、对数分桶延迟直方图、epoll/IOCP 双后端事件循环、CPU 绑核/实时调度
- 可执行 `trading_engine`（端到端演示/压测，输出各阶段 p99 延迟）与 `replay_feed`（抓包重放校验工具）
- 真实 ITCH/OUCH/FIX/CTP 协议按设计需以独立 Adapter 接入，当前 Decoder 只解 Mock 协议

---

# 二、Python 研究层（python_model）

面向 **A 股分钟线**的量化回测研究平台（`quant-backtester`，版本 `0.3.0`），与 C++ 引擎同属一个业务闭环：

```text
行情数据导入 → 校验入湖 → 配置并运行回测 → 结果落库 → Dashboard 审计 →（可选）ML 研究
```

## 行情数据导入与数据湖

- **数据契约**：`Bar(timestamp, symbol, open, high, low, close, volume)`，`timestamp` 为 UTC epoch 纳秒，`symbol` 保留前导零，主键 `(timestamp, symbol)`；CSV/Parquet 可透传 `upper_limit` / `lower_limit` / `is_suspended`
- **导入方式**：单文件导入（可预览 20 行）、目录递归批量导入（自动跳过已导入文件）、浏览器上传（限 100 MiB）；导入先写 staging，完整校验后才发布为不可变 Parquet 分区文件并原子更新 catalog——查询永远看不到半批数据；源文件内容生成 SHA-256 指纹，重复导入直接跳过
- **分区布局**：`data_lake/bars/year=YYYY/month=MM/bucket=N/part-*.parquet`（默认 64 个 symbol bucket），月分区避免小文件爆炸，bucket 让截面查询与单标的查询都只读必要文件

## 回测执行（Python 侧）

纯 Python 回测实现**仅保留为 demo**（双均线、均值回归两个示例策略），正式模型回测以 C++ 引擎为准。两种业务模式：**独立批量**（各标的独立资金，横向对比）与**共享资金组合**（共享本金 + 等权预算分配）。A 股规则与 C++ 引擎一致：T+1、涨跌停保守近似、资金冻结、参与率/滑点、整手、NEXT_OPEN/CLOSE 成交时点、历史费率、公司行动、组合风险暴露，全部使用 point-in-time 参考数据（交易日历显式开市日期、复权因子只用于信号价格）。

## 结果持久化与审计

一次回测以事务方式保存到 SQLite（星型结构，以 `backtest_runs` 为中心）：`trades` / `round_trips` / `orders`（含拒绝、撤销、过期）/ `daily_equity` / `performance_metrics` / `corporate_actions` / `portfolio_risk` / `data_lineage` / `run_config`（冻结的完整 RunSpec）。

- 运行状态机：`RUNNING → SUCCEEDED / FAILED`，失败原因可见；
- 未成交委托也会保存——可以回答"为什么没有成交"；
- 可复现：每次运行冻结 `RunSpec`（后端制品 SHA-256、策略代码 hash、参数、随机种子、执行配置、费率区间、日历/参考数据指纹），同一输入任何时候重跑得到同一结果。

## Dashboard

`streamlit run dashboard/app.py`，包含：主页（指标总览与运行列表）、数据实验（导入 + 配置回测）、权益与回撤、交易分析、风险指标、运行审计（订单状态聚合、数据血缘、完整 RunSpec）。大表查询使用有界分页与 SQL 侧聚合。

## ML 研究管线

```text
build-dataset → train → phase1b-ablation → export → validate-artifact → backtest-artifact
```

1. **构建数据集**：分钟 Bar 特征（`BAR_V1` 协议）与标签（`NEXT_OPEN` 对齐），逐 Bar 波动率、下行半波动、截面 rank 效用等；
2. **训练**：时间 Transformer / 基线模型（Ridge、MLP、TCN、GRU），walk-forward 交叉验证；
3. **排序消融（Phase 1B）**：`legacy / ListMLE / LambdaLoss@K` 三种排序损失在三个 purged 样本外窗口上配对比较；只有通过模型门禁的挑战者才进入 C++ 成本回测与经济门禁；
4. **共享梯度消融（Phase 1E）**：按 `diagnostics → pcgrad → gradnorm` 顺序评估多任务梯度冲突；
5. **导出与验证**：导出 ONNX（opset 17）+ manifest + 特征 schema + 指标，`validate-artifact` 核验 SHA-256、协议、对齐、输入形状；
6. **制品回测**：直接用已导出模型对 CSV/Parquet 行情回测，输出模型版本、订单、成交、现金、权益与收益摘要。

研究流程内置防"调参数调到显著"机制：**假设预注册**（必须在测试数据可用之前注册）、**多重检验校正**（FDR + block bootstrap）、**窗口净化**（purge 间隔防标签泄漏）。

**制品契约**：模型制品最小包含 `model.onnx` / `manifest.json` / `feature_schema.json` / `metrics.json`。生产导出使用 Manifest V2，要求标签、训练集 fingerprint、归一化、validation-only Platt 校准和泄漏报告全部进入哈希链。schema/hash 不匹配、输出非有限、输入过期或模型不可用时，引擎**停止产生新风险**，绝不静默回退到旧预测。

---

# 三、构建与安装

## C++ 引擎

默认构建（纯回测核心）：

```bash
pip install .
```

启用可选模块（构建前设置环境变量）：

```bash
QBT_ENABLE_ML=ON QBT_ML_BACKEND=mock pip install .          # ML 运行时（mock 后端，无外部依赖）
QBT_ENABLE_ML=ON QBT_ML_BACKEND=onnxruntime ONNXRUNTIME_ROOT=/path/to/onnxruntime pip install .  # ONNX Runtime 后端
QBT_ENABLE_PORTFOLIO_MATH=ON QBT_ENABLE_PERFORMANCE_ANALYTICS=ON pip install .
```

wheel 可用 `cpp_engine.__build_type__`、`__compiler_id__`、`__ml_enabled__`、`__ml_backend__` 等模块属性确认构建配置。

CMake 构建（C++ / 测试 / 基准）：

```bash
cmake --preset dev          # Debug + Python 扩展
cmake --build --preset dev
ctest --preset dev
```

常用 Preset：`dev` / `release` / `cpp-only` / `all-modules` / `phase1c-closure` / `benchmark-release`（Release + LTO + 基准）/ `live-release` / `sanitize` / `ci-gcc` / `ci-clang-sanitize` / `ci-clang-tsan`（完整列表见 `CMakePresets.json`）。可选选项：`QBT_ENABLE_ML`、`QBT_ML_BACKEND`（mock|onnxruntime）、`QBT_ENABLE_PORTFOLIO_MATH`、`QBT_ENABLE_PERFORMANCE_ANALYTICS`、`QBT_BUILD_LIVE_ENGINE`、`QBT_BUILD_BENCHMARKS`、`QBT_BUILD_FUZZERS`、`QBT_ENABLE_LTO`、`QBT_WARNINGS_AS_ERRORS`。

## Python 研究层

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
streamlit run dashboard/app.py    # 启动 Dashboard
```

ML 研究另装 `pip install -e '.[ml]'`。M4 Pro 上可使用锁定的 Python 3.9 arm64 环境（PyTorch 2.2.2、ONNX 1.16.2、ONNX Runtime 1.17.3、NumPy 1.26.4），避免版本漂移。

## Python 回测示例（C++ 引擎）

```python
import cpp_engine as ce

engine = ce.BacktestEngine(initial_cash=1_000_000.0)

# 截面回调：每个截面行情后调用，返回本截面订单（列式 dict 或 Order 列表）
def on_cross_section_view(view):
    return [
        {
            "symbol": view.symbol(i),
            "side": ce.Side.BUY,
            "type": ce.OrderType.MARKET,
            "quantity": 100,
        }
        for i in range(len(view))
    ]

engine.set_on_cross_section_view(on_cross_section_view)

def make_snapshot(ts, symbol, o, h, l, c, v):
    snap = ce.MarketSnapshot()
    snap.timestamp, snap.symbol = ts, symbol
    snap.open, snap.high, snap.low, snap.close, snap.volume = o, h, l, c, v
    return snap

# 逐截面注入行情：同一批必须同一时间戳，截面之间时间戳严格递增
for ts in range(1, 61):  # 60 根日线
    engine.process_market_data_batch([
        make_snapshot(ts, "000001", 10.0, 10.5, 9.8, 10.2, 100_000),
        make_snapshot(ts, "600000", 8.0, 8.3, 7.9, 8.1, 200_000),
    ])

engine.finalize()  # 未完成订单全部 EXPIRED 并关闭账本
print("cash:", engine.get_cash())
print("equity:", engine.get_equity())
```

也可以通过 Arrow C Stream 零拷贝接入行情（任何实现 `__arrow_c_stream__` 的对象，如 PyArrow Table）：

```python
stats = engine.process_arrow_stream(pa_table)   # 返回 ArrowBridgeStats（行数/批数/耗时）
```

## ML 模型策略示例（需 `QBT_ENABLE_ML=ON` + onnxruntime 后端）

```python
import cpp_engine as ce
assert ce.__ml_enabled__

engine = ce.BacktestEngine()
engine.set_model_strategy(
    artifact_path="artifacts/best-v1/",   # 目录需含 manifest.json / model.onnx / feature_schema.json（SHA-256 校验）
    policy_config={"max_positions": 20, "max_position_weight": 0.05},
    risk_config={},
    runtime_config={},
)
```

策略运行时在回测主循环中自动完成：行情 → BarV1 特征 → 推理 → 目标持仓 → 风控 → 订单。

---

# 四、ML 研究管线使用（V1.1）

基础命令（`python.qbt_ml.cli`，各配置默认 `enabled=false` 且数据路径为占位值，正式运行前必须审核并显式启用）：

```bash
python -m python.qbt_ml.cli build-dataset --config configs/ml/bar_v1.yaml --output data/dataset.npz
python -m python.qbt_ml.cli detect-leakage --config configs/ml/leakage_detection_v1_1.json --dataset data/dataset.npz --output analysis/<run-id>/leakage
python -m python.qbt_ml.cli walk-forward --config configs/ml/walk_forward_v1_1.json --dataset data/dataset.npz --output runs/<run-id>/walk-forward
python -m python.qbt_ml.cli walk-forward-transformer --config configs/ml/transformer_walk_forward_v1_1.json --dataset data/dataset.npz --output runs/<run-id>/transformer-walk-forward
python -m python.qbt_ml.cli benchmark-models --config configs/ml/model_benchmark_v1_1.json --dataset data/dataset.npz --output benchmarks/ml/<run-id>
python -m python.qbt_ml.cli ablate-features --config configs/ml/feature_ablation_v1_1.json --dataset data/dataset.npz --output analysis/<run-id>/feature-ablation
python -m python.qbt_ml.cli ablate-transformer-features --config configs/ml/transformer_feature_ablation_v1_1.json --dataset data/dataset.npz --output analysis/<run-id>/transformer-feature-ablation
python -m python.qbt_ml.cli export --config configs/ml/bar_v1.yaml --run runs/<run-id> --output models/<model-id>
python -m python.qbt_ml.cli validate-artifact models/<model-id>
```

V1.1 数据契约：数据源必须包含 `universe_asof` 与 `reference_data_known_at_max`，并在配置中明确 `calendar_id`、`universe_id` 与 `data.point_in_time_required=true`。

V1.1 已交付能力（摘要）：冻结 `NEXT_OPEN` 标签语义（信号日 `t`，`open[t+1]` 入场，`close[t+h]` 退出）；持有期波动率使用逐 Bar 子收益标准差；数据集写入 LabelSpec、PIT provenance、lineage 与内容 fingerprint；Future Mutation 与 Prefix Invariance 训练前审计；train-only normalizer；多任务损失覆盖收益/波动率/方向/q10/q90/截面排序；confidence 由校准方向概率派生；`CrossSectionBatchSampler` 保证同一 timestamp 截面不被拆分；expanding Walk-forward 强制至少三个非重叠 test 窗口，每窗独立执行泄漏门禁与 validation 冻结；非神经 Benchmark 在同一窗口比较 Ridge、20 日动量、5 日反转和全现金；Transformer Feature Ablation 支持多 seed group-drop retrain。

注意：`ablate-features` 是 Ridge 快速诊断，`ablate-transformer-features` 才是正式 group-drop retrain（正式结论要求至少三个固定 seed）；所有缩减特征 checkpoint 是分析制品，导出命令会拒绝伪装成完整 `BAR_V1` 生产模型；模型晋级仍需后续 C++ 扣费组合消融指标。

---

# 五、测试、基准与 CI（C++ 引擎）

- **单元测试**：各模块 CTest 自写断言框架，覆盖撮合（含 A 股约束边界）、引擎全生命周期、PnL/round-trip、ML 特征 golden 数值对照（2e-6 容差）、网关故障注入（WAL 损坏/磁盘故障/半包）、回环 TCP 等
- **模糊测试**（Clang + `QBT_BUILD_FUZZERS`）：`fuzz_decoder`、`fuzz_wire_protocol`、`fuzz_arrow_schema`
- **确定性验证**：`qbt_golden_replay`（固定脚本回放，`--verify` 与参照 JSON 逐字节比对）、`qbt_dual_replay`（回测引擎与实盘链路双重回放一致性）
- **性能基准**（Release + LTO）：`qbt_benchmark`（行情/截面/撮合/历史微基准）、`te_benchmark`（解码吞吐 vs legacy 实现）、`te_pipeline_benchmark`（端到端管线 + 性能预算检查）；规范见 `benchmarks/README.md`、`benchmarks/VNEXT_COMPLETION.md`
- **CI**（`.github/workflows/ci.yml`）：Linux GCC 严格构建 + 管线预算检查、Clang ASan/UBSan + fuzz、Clang TSan、Windows MSVC、Python wheel 矩阵（3.9/3.13）冒烟 + 绑定边界基准

## 工具脚本（`tools/`）

| 脚本 | 用途 |
| --- | --- |
| `run_cpp_benchmarks.py` | 运行 C++ 基准，聚合 median/p95/allocations/RSS，生成含构建信息与 SHA-256 的 JSON 报告，可选 baseline 回退检测 |
| `check_pipeline_budget.py` | 端到端性能预算检查（ns/event 回退、p99、RSS 上限，默认 5% 告警 / 10% 失败） |
| `run_python_boundary_benchmarks.py` | Python 绑定边界开销基准（5 种回调模式） |
| `compare_golden.py` | 两个 golden replay JSON 逐字段递归比对 |
| `run_phase1e_challenger.py` | ML 研究：冻结 OOS 窗口上运行 PCGrad/GradNorm challenger，产出 preregistered contract + 报告（report-only，不晋级） |
| `audit_phase3a_exposure_source.py` | 全量 parquet metadata-only 审计 PIT industry/style exposure、available-at 字段和 source schema hash |

---

# 六、第三方依赖

| 依赖 | 版本 | 许可 | 用途 |
| --- | --- | --- | --- |
| Eigen | 3.4.0 | MPL-2.0 | portfolio_math 线性代数（FetchContent，SHA-256 校验） |
| pybind11 | ≥ 3.0 | BSD | Python 绑定 |
| ONNX Runtime | ≥ 1.17 | MIT | ML 推理后端（可选） |
| PyTorch | ≥ 2.2 | BSD | Python 研究层训练（可选 `[ml]` extra） |
| numpy / pandas / pyarrow / duckdb / plotly / streamlit | — | BSD/Apache | Python 研究层运行依赖 |

组合数学中 QuEST 非线性收缩算法依据 Ledoit & Wolf 的三篇论文独立实现（未复制其 MATLAB 代码），详见 `THIRD_PARTY.md`。

# 七、相关文档

- `ml_runtime/README.md` — ML 运行时构建与后端契约
- `performance_analytics/README.md` — 绩效分析模块定位与合同约束
- `benchmarks/README.md`、`benchmarks/VNEXT_COMPLETION.md` — 基准规范与 vNext 优化完成报告
- `docs/phase1e_pytorch.md` — Phase 1E PyTorch 训练环境说明（Apple Silicon / CPU）
- `THIRD_PARTY.md` — 第三方依赖与研究文献
- Python 研究层：`docs/architecture.md`、`docs/data_system.md`、`docs/ml_integration.md`、`docs/phase1b_experiment_report.md`、`docs/transformer_integration_plan.md`
- 总仓库：`docs/TRANSFORMER_V1_1_INTEGRATION_PLAN.md`、`docs/decision_log.md`

# 免责声明

本项目用于研究和工程验证，不构成投资建议。历史回测、预测分数或模型指标不代表未来收益。

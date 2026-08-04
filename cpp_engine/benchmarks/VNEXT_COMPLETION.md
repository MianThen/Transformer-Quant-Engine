# vNext 优化完成报告

生成日期：2026-07-22

## 发布结论

`NEXT_VERSION_OPTIMIZATION_PLAN.md` 的 P0、P1 已实现并通过 Release 验证。P2
遵循“数据证明后再做”的约束完成评估；未获得收益证据的 intrusive order
book、NUMA/大页/锁页、组合内并行和 TSC 未进入默认生产路径。

当前 Windows 验证制品使用 MinGW GNU 13.1.0。该工具链不支持本项目请求的
LTO 组合，因此所有报告如实记录 `lto=false`，不将其标记为 LTO 构建。

## 验证制品

| 制品 | SHA-256 | 构建 |
|---|---|---|
| `qbt_benchmark.exe` | `ff6fcb34927b75e4747e6309f30f2b21a1814738e53efee61d036cbee3ecf59f` | Release, GNU 13.1.0, LTO=false |
| `te_benchmark.exe` | `c896978a87d89d3f1f144d8d992d7f8aced2b8852fc3a150c48981fb6a36e8c7` | Release, GNU 13.1.0, LTO=false |
| `cpp_engine.cp312-win_amd64.pyd` | `65cb56cce47de09251e0a178bd5c1ee3c947df669cab1e08b5aa689262ece024` | Release, GNU, LTO=false |

硬件：Windows 11，AMD64 Family 23 Model 113，12 logical CPUs。C++ 数据集为
`qbt-m0-deterministic` v1，seed 0；每个归档报告均运行五次并保存原始样本、
median、p95 和峰值 RSS。

## C++ 引擎结果

| 场景 | median | p95 | 相对 M0 | median RSS |
|---|---:|---:|---:|---:|
| 单 symbol 纯行情，10,000,000 条 | 107.411 ns/op | 116.339 ns/op | -71.47% | 3.73 MiB |
| 1,000 symbol 有序 span 截面 | 140.720 ns/op | 148.900 ns/op | 新增路径 | 4.96 MiB |
| 10,000 symbol 安全有序截面 | 238.288 ns/op | 340.968 ns/op | 新增路径 | 14.95 MiB |
| 高频成交/部分成交/撤单 | 1,065.090 ns/op | 1,261.610 ns/op | -31.25% | 17.61 MiB |
| 1,000 持仓盯市 | 158.000 ns/op | 196.100 ns/op | -27.26% | 17.61 MiB |
| 10,000 持仓盯市 | 292.370 ns/op | 352.470 ns/op | -41.28% | 23.12 MiB |
| 50,000 持仓盯市 | 373.272 ns/op | 447.238 ns/op | -56.64% | 93.63 MiB |
| 100,000 未触发限价档 | 800.905 ns/op | 819.529 ns/op | +9.72% | 101.96 MiB |
| 百万历史后的退市/公司行动 | 17.750 us/op | 22.250 us/op | +143.15% | 234.53 MiB |

订单开放量 1k/2k/4k/8k、限价档 10k/50k/100k、公司行动和市场规则的完整
逐场景结果保存在 `benchmarks/reports/vnext-current.json`。百万历史场景每次仅
计时两个操作，百分比对微秒级调度噪声敏感，因此发布判断同时使用绝对值和
黄金行为结果。50k/100k 限价档的约 10% 回退已归档，后续改动可单独跟踪。

## 交易链路结果

| Decoder 场景 | median | p95 | 相对 legacy | 计时区分配 |
|---|---:|---:|---:|---:|
| 连续消息 | 12.302 ns/op | 17.565 ns/op | -77.81% | 0 |
| 半包/粘包 | 30.204 ns/op | 32.782 ns/op | -62.54% | 0 |
| 乱序恢复 | 23.850 ns/op | 25.755 ns/op | -73.64% | 0 |

报告：`benchmarks/reports/te-decoder-current.json`。Feed、Strategy、Gateway send、
ACK RTT 和端到端延迟使用在线分桶统计；运行时同时提供 CPU、RSS、队列水位、
连接、重连、错误、订单和线程心跳指标。

## Python 边界

Release wheel 的 10,000 symbol、五次测试结果：纯 C++ 无回调 1,007.89 ns/row，
截面 Python 回调 2,317.89 ns/row，逐行 Python 回调 2,203.31 ns/row。报告保存在
`benchmarks/reports/python-boundary-current.json`，用于将 Python 边界成本与
C++ 核心成本分开归因。

## 行为与可靠性

- 黄金回放验证通过：输入、订单、成交、现金、持仓、权益和最终指标一致。
- Release CTest 13/13 通过，覆盖回测、PnL、Arrow schema、SPSC、Decoder、网络、Gateway 和 Feed replay。
- Gateway 覆盖 READY 门禁、心跳/重连、WAL 恢复、三方订单对账、未知订单、missed fill 和重复回报。
- Feed 覆盖固定乱序窗口、背压安全状态、捕获/离线回放和恢复门禁。
- CI 覆盖 Linux GCC Release、Clang ASan/UBSan、Clang TSan、Windows MSVC、Python 3.9/3.13 wheel 和 fuzz。

## P2 评估结论

- 自定义价格档位容器/intrusive order book：未启用；现有稠密 SymbolId 与索引结构已覆盖主要收益，当前数据不足以证明重写收益大于复杂度。
- NUMA、锁页和大页：未启用；当前单机规模和 RSS 未显示内存局部性为首要瓶颈。CPU pinning 与可选实时优先级已作为受控运行选项提供。
- 单组合内 symbol 分片并行：未启用；为保持确定性和黄金回放可复现，继续采用单组合顺序核心。
- TSC 和更激进编译选项：未启用；默认使用 `steady_clock`，并保留真实编译器/LTO 元数据。换用支持 LTO 的工具链后可通过同一报告格式重新评估。

## 发布护栏

后续每个 C++ 改动应同时提交新的五次 JSON 报告、旧报告 baseline 对比和黄金
逐字段行为对照。性能报告必须能确认编译器、Release/LTO 状态、artifact
SHA-256、数据集和硬件；纯行情、高订单量和高持仓量必须保持独立归因。
# 2026-07-22 全目标完成补充

- `cpp_engine` 百万订单峰值 RSS：135,524,352 字节。
- Python `MarketBatchView`/列式订单路径：中位数 806～880 ns/行。
- OMS 异步 WAL 10 万单发送吞吐回退：约 4.8%。
- CTest 覆盖增加到 16 项，包含双引擎 replay 与 Gateway 故障注入。
- `te_pipeline_benchmark` 与 `check_pipeline_budget.py` 已接入 Linux GCC CI。

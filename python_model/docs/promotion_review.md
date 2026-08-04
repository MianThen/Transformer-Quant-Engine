# Phase E 模型晋级评审

Phase E 使用 `promotion-review` 将软件正确性与模型质量分开裁决。最终结论只有三种：

- `PROMOTE`：两个门槛都通过；
- `REJECT`：至少一个已具备充分证据的门槛失败；
- `INSUFFICIENT_EVIDENCE`：没有失败证据，但正式评审材料不完整。

历史涨跌停或 PIT board-lot 数据延期时，模型评估可以继续，但 C++ 研究报告必须包含
`execution_reference_mode=optional_for_model_evaluation` 和
`promotion_eligible=false`。Promotion Review 将其视为缺少执行证据并返回
`INSUFFICIENT_EVIDENCE`，不会使用研究模式的 Sharpe 或回撤作正式晋级。

缺少真实全 A 数据、C++ 扣费回测或候选模型性能报告时，不允许用合成数据、Python demo
回测或 Phase D 小型测试制品替代。

## 输入证据

配置文件为 `configs/ml/promotion_review_v1_1.json`。评审读取以下只读产物：

| 证据 | 用途 |
|---|---|
| Transformer walk-forward | 窗口、数据 fingerprint、逐窗口 Leakage |
| Model Benchmark | 同窗口 RankIC 与完整基线集合 |
| Transformer Feature Ablation | 全特征组、至少三个种子的分组重训练 |
| C++ Portfolio Ablation | 相同成本下的组合层消融 |
| C++ Portfolio Backtest | 0/5/10 bp、净收益、风险、贡献度和稳定性 |
| ORT C++ parity | 六输出、top-k 和目标仓位一致性 |
| C++ runtime benchmark | 生产候选制品的 Release/LTO p99 与 lineage |

完整预测层基线先在相同窗口上运行：

```bash
python -m python.qbt_ml.cli walk-forward-deep-baselines \
  --config configs/ml/deep_baselines_v1_1.json \
  --output runs/<experiment-id>/deep-baselines

python -m python.qbt_ml.cli benchmark-models \
  --config configs/ml/model_benchmark_v1_1.json \
  --output runs/<experiment-id>/model-benchmark
```

Deep Baseline Suite 固定包含 MLP、因果 TCN 和 GRU。三者与 V1.1 共享数据、窗口、种子调度、
多任务输出头、损失、优化器、早停和训练 epoch，只替换编码器。仓库历史中没有可复现的
Transformer V1 实现或冻结制品，因此它记录在 `unavailable_legacy_models`，不允许在看到
V1.1 结果后反向构造一个所谓的旧基线。

C++ Portfolio Backtest 的固定 JSON 合同见
`schemas/phase_e_portfolio_backtest.schema.json`。三档滑点必须包含相同的至少三个不重叠 test
窗口；每个窗口必须包含候选模型和所有冻结基线。正式成本模型必须启用 point-in-time 费率、
T+1、涨跌停、整手和不高于 10% 的成交量参与率。

模型 Benchmark 完成后运行 C++ 组合层：

```bash
python -m python.qbt_ml.cli benchmark-portfolios-cpp \
  --config configs/ml/portfolio_benchmark_v1_1.json \
  --output runs/<experiment-id>/portfolio-benchmark
```

该命令固定执行 9 个模型 × 至少 3 个窗口 × 0/5/10 bp。Python 只读取和对齐预测；top-k、
换手限制、目标仓位、订单规划、风险、费用、T+1、涨跌停、部分成交、现金和持仓均由 C++
执行。正式运行要求 Release/LTO 扩展。

正式 Bar 表不是 BaoStock 原始 OHLCV 的直接别名，必须先完成 PIT 富化并包含：

- 原始 `open/high/low/close`，仅用于成交；
- `signal_open/high/low/close = raw * adjustment_factor`，用于特征和标签；
- 上市、停牌、ST、涨跌停、lot、行业及各自 known-at/as-of 字段；
- 独立 point-in-time 企业行动表，至少包含现金分红和送转比例；
- `universe_asof` 和 `reference_data_known_at_max`。

标签与特征都使用 `signal_*`，C++ 成交使用原始价格并显式应用企业行动。缺少任一层时命令
拒绝生成正式报告；不能把未复权标签或当前行业映射带入 Phase E。

Transformer 分组消融训练完成后，使用相同 C++ 组合层生成组合消融证据：

```bash
python -m python.qbt_ml.cli benchmark-ablation-portfolios-cpp \
  --config configs/ml/portfolio_ablation_v1_1.json \
  --output runs/<experiment-id>/portfolio-ablation
```

它固定覆盖至少三个种子、全部 BAR_V1 特征组、相同 walk-forward 窗口和 0/5/10 bp，输出
合同为 `schemas/phase_e_portfolio_ablation.schema.json`。该报告回答“删除特征后组合层表现
如何变化”，不会单独把某个特征组的好坏当成模型晋级结论。

## 运行

先将配置中的 `CONFIGURE_...` 替换为冻结后的证据路径和业务门槛，并将 `enabled` 改为
`true`：

```bash
cd python_model
python -m python.qbt_ml.cli promotion-review \
  --config configs/ml/promotion_review_v1_1.json \
  --output runs/<experiment-id>/promotion-review
```

输出包括：

```text
promotion-review/
├── promotion_report.json
├── software_gate.json
├── model_quality_gate.json
├── multi_window_portfolio.json
├── concentration_stability.json
├── evidence_manifest.json
└── summary.md
```

`multi_window_portfolio.json` 和 `concentration_stability.json` 只有在组合证据可解析时生成。

## 性能报告边界

Phase D 的 `ml-runtime-m4-pro-phase-d.json` 使用小型测试制品，只能证明工程链路。正式候选
模型必须重新运行：

```bash
python3 ../cpp_engine/tools/run_ml_runtime_benchmarks.py \
  --executable <release-lto-qbt_ml_benchmark> \
  --artifact <production-candidate-artifact> \
  --output <candidate-runtime.json> \
  --iterations 100 --chunk-size 512 \
  --benchmark-scope production_candidate
```

评审会核对 runtime 与 parity 的 model id/version，并核对 runtime、walk-forward 和 C++ 回测
的数据 fingerprint。

## 已冻结的补充门槛

文档已确认的首版门槛已写入配置：至少三个窗口、RankIC 中位数大于 0.02、净 Sharpe 大于
1.0 且领先最强基线至少 10%、最大回撤不高于 20%、至少 2/3 窗口净收益优于基线。

以下值已于 2026-07-29 在第一次正式冻结 test 前确认：

- 95% CVaR 不低于 `-3%`；
- 单一时期的正收益贡献不超过 `50%`；
- 单股票 Top-1 正收益贡献不超过 `10%`；
- 单股票 Top-5 正收益贡献不超过 `30%`；
- 单行业正收益贡献不超过 `30%`；
- 日频全 A、CPU、batch=4096 的候选模型端到端 p99 不超过 `50 ms`。

这些门槛只适用于日频 V1.1，不能在查看最终 test 结果后反复修改。分钟模型使用独立配置和
独立性能门槛，不能直接继承日频的 `50 ms`。

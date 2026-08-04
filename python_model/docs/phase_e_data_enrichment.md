# Phase E PIT 数据富化

`enrich-phase-e` 把 BaoStock 原始日频数据转换为 Transformer V1.1 训练和 C++ 组合
Benchmark 共用的不可变 Parquet 数据集。它不覆盖原始 Bar、状态或参考表。

## 连接顺序

1. 以 `BAOSTOCK_DAILY_STATE_V1` 的 `(timestamp,symbol)` 作为交易日状态骨架。
2. 原始未复权 OHLCV 精确左连接；只有明确停牌的缺失 Bar 可以沿用上一原始收盘价作为
   OHLC，并固定 `volume=0, bar_observed=false`。
3. `back_adjust_factor` 按生效时间向后 as-of，生成
   `signal_OHLC = raw_OHLC * adjustment_factor`；原始 OHLC 仍供 C++ 成交。
4. 行业先要求 `snapshot_asof <= timestamp`，可用时间记为
   `max(provider known_at, snapshot_asof)`，禁止从未来快照回填。
5. 历史涨跌停价按交易日精确连接；lot/minimum buy quantity 也必须来自带 `known_at` 的
   PIT 规则表。企业行动保持独立表，由 C++ 引擎显式应用。

结果增加 `frequency`、`calendar_id`、`industry_known_at`、
`reference_data_known_at_max` 和输入内容指纹。输出目录存在时拒绝覆盖。

## 两种执行数据模式

默认配置 `execution_reference_mode=optional_for_model_evaluation`。历史涨跌停和 lot
缺失时，富化结果仍可用于 Transformer、深度基线、Leakage Detection、Walk-forward、
Feature Ablation 和 Attention Analysis。报告分别记录：

```text
model_evaluation_status=READY
execution_promotion_status=DEFERRED
promotion_eligible=false
```

C++ 研究回测必须同时关闭 `enforce_price_limits` 和 `enforce_board_lot`；缺失涨跌停用
0 表示不限制，缺失 lot/minimum buy quantity 仅使用 100 股占位值。报告会列出这些假设，
其 Sharpe、回撤、换手和成交结果不能作为 Promotion 证据。

配置 `execution_reference_mode=required_for_promotion` 时，以下任一条件都会终止，不写
半成品：

- 未确认复权历史从上市起完整；
- 缺 PIT 行业或独立企业行动表；
- 任意上市状态缺原始 Bar，且不是明确停牌；
- 任意行缺行业、涨跌停价、lot 或最小买入数量；
- 任意参考数据 `known_at > timestamp`。

## 运行

先复制并填写 `configs/ml/phase_e_enrichment_v1_1.json` 中的绝对数据路径和可信
`calendar_id`，然后执行：

```bash
python -m python.qbt_ml.cli enrich-phase-e \
  --config configs/ml/phase_e_enrichment_v1_1.json
```

模型评估可以直接使用默认可选模式。只有在历史执行规则数据补齐后，才把模式改为
`required_for_promotion` 并重新运行 C++ 组合 Benchmark 和 Promotion Review。通常不需要
重新训练模型，除非未来把涨跌停状态加入特征或样本过滤。生成目录可直接作为
`build-dataset` 的 `data.source`。

# Phase 1F 实验报告

报告日期：2026-08-03
状态：`RESEARCH_PROXY_EXIT_COMPLETE / PRODUCTION_GATE_DEFERRED`

## 1. 结论

Phase 1F `LEGACY-TOPK-STABILITY-V1` 已按预注册合同完成一次新的 validation/untouched 研究。
`stability` 候选通过 validation gate 并冻结为本次 winner；untouched 只运行一次冻结 winner。
研究报告、预测快照和 C++ Replay/CVaR proxy 已形成可复核闭环，但不改变
`FROZEN_CHAMPION=legacy + fixed`，也不允许生产晋级。

| Gate | 结果 |
| --- | --- |
| `median N >= 4K` | PASS，`109 >= 80` |
| validation/untouched 新窗口 | PASS，各 59 sessions |
| top-k stability oracle | PASS |
| validation candidate gate | PASS，winner=`stability` |
| untouched winner evaluation | PASS |
| C++ Replay/CVaR | PASS，58 periods，`RESEARCH_PROXY` |
| 生产经济 gate | DEFERRED_TO_LIVE |

## 2. 预注册与数据

- `K=20`、`temperature=0.20`、`lambda_stability=0.01`、`purge=6`
- `seed=20260803`、`train_timestamp_stride=4`、`batch_cross_sections=8`、`epochs=3`
- validation：2026-01-15–2026-04-16，59 sessions
- untouched：2026-04-17–2026-07-14，59 sessions
- dataset SHA-256：`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`
- contract SHA-256：`edefb09a9a49d858248f7be2d487332e70f496de189530738ae4fd5faf519671`

候选只增加 temporal/top-k stability regularizer；encoder、label、MTL 权重、policy 和成本假设未改动。
失败候选不形成 runtime fallback。

## 3. 研究结果

validation 上 legacy 与 stability 的 NDCG、precision、utility spread 和 turnover 相同，stability
的 RankIC 略高、稳定性 penalty 略低，因此按预注册非劣 gate 选择 `stability`。untouched winner 指标为：

| 指标 | untouched stability |
| --- | ---: |
| NDCG@20 | 0.498825 |
| RankIC | 0.075329 |
| precision@20 | 0.156780 |
| top-k turnover | 0.285345 |
| top-bottom utility spread | 0.009500 |
| stability penalty | 2.095047e-07 |

untouched 结果只用于冻结 winner 的一次评估，不反向调参。

## 4. C++ Replay/CVaR proxy

winner 使用 C++ `BacktestEngine`、`PeriodContributionReplaySink` 和 empirical CVaR。回放结果：

- 58 个 period returns，99 个共同股票；
- `var_loss=0.023912636326028913`；
- `expected_shortfall_loss=0.02469592429258582`；
- `return_CVaR=-0.02469592429258582`；
- proxy report SHA-256：`23b47f6c3542d17722e6949902c95cc850f3e8ffc598829bf0b66fb662462793`；
- training report SHA-256：`0f3b8ab6e29187c90de5663f98f6fd4fa88cfa80c55e5ce4465bd0435a37d795`。

该回放使用 bar-close reference/fill proxy、共同股票交集、关闭涨跌停与 board-lot enforcement、
零滑点机械假设。费用/税费、涨跌停、公司行动/复权、lot 和真实 reference/slippage provenance
仍为 `UNAVAILABLE/PROXY`，所以该 CVaR 只描述代理收益序列的尾部风险。

## 5. 退出判定

Phase 1F 研究工程退出：**PASS**。`phase_exit_eligible=true`，`research_comparison_eligible=true`。
生产经济晋级：**NO-GO**。`promotion_eligible=false`，必须等待实盘接入后的费用/税费、PIT 限价、
公司行动/复权、lot 和带 provenance 的 reference/slippage，并完成 `LIVE_SHADOW` 对账。

## 6. Artifact 索引

- 预注册合同：`runs/phase1f-topk-stability-real-sampled/preregistered_contract.json`
- 训练报告：`runs/phase1f-topk-stability-real-sampled/phase1f_training_report.json`
- C++ proxy：`runs/phase1f-topk-stability-real-sampled/cpp_replay_winner.json`
- winner checkpoint：`runs/phase1f-topk-stability-real-sampled/winner_checkpoint.pt`
- untouched predictions：`runs/phase1f-topk-stability-real-sampled/untouched_winner_predictions.npz`

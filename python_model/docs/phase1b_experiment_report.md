# Phase 1B Transformer 目标对齐实验报告

## 摘要

本实验比较了三种截面排序损失，并在排序损失冻结后独立比较固定多任务权重与 Kendall
同方差不确定性动态权重。所有候选使用相同真实 OHLCV 数据集、LabelSpec、RankingScoreSpec、
Transformer encoder、seed、训练预算和 purged walk-forward fold；首次排序实验中唯一变化项是
ranking loss，动态权重实验中唯一变化项是四个非排序任务的权重机制。

实验结论如下：

1. 冻结 `legacy rank loss`，淘汰 ListMLE 和 LambdaLoss@20。
2. 冻结固定多任务权重，淘汰 Kendall 动态权重。
3. 当前研究基线为 `legacy rank loss + fixed multitask weights`。
4. 本结论只支持“保留 incumbent”，不支持新的生产经济性声明。原始数据缺少完整 PIT 股票池、
   停牌/ST、涨跌停、公司行动、复权、费用和 C++ replay 收益序列，因此不能据此宣称净收益或
   CVaR 改善。

这是一个负结果，但它是有价值的负结果：ListMLE 在 Precision@20 和换手上有局部优势，Kendall
也降低了换手；然而二者都损害了预注册主指标或 realized utility。LambdaLoss 则在本数据与当前
尺度下全面退化。按事前规则，不能用次要指标覆盖主指标和经济代理的失败。

## 1. 实验问题

Phase 1B 回答两个顺序固定的问题：

1. 在生产决策 cutoff 为 `K=20` 时，legacy correlation loss、ListMLE 和显式 LambdaLoss@20
   哪一个应作为冻结的 ranking loss？
2. ranking loss 冻结后，Kendall 任务级同方差动态权重是否优于当前固定权重？

第二个问题不能与第一个问题同时搜索。先看排序实验结果再改标签、encoder、seed、epoch、K 或
temperature，都会破坏单变量配对，因此这些量全部写入实验合同并由 SHA-256 固定。

## 2. 数据与适用范围

### 2.1 数据来源

本地数据湖包含 14,105,167 行 A 股日线记录，catalog 覆盖 2006-01-04 至 2026-07-22。
Phase 1B 使用的研究切片为：

| 项目 | 数值 |
|---|---:|
| 原始切片 | 38,193 行 |
| 原始 timestamp | 1,091 |
| 原始区间 | 2020-01-02 至 2024-07-04 |
| 有效模型样本 | 35,475 |
| 有效 timestamp | 1,025 |
| 有效股票 | 39 |
| 每个截面股票数 | 最小 29 / 中位 32 / 最大 39 |

股票池只使用实验期之前的信息构造：在 2019 年至少有 220 条记录的股票按代码升序排列，冻结前
40 只。该规则没有使用 2020 年后的收益挑选股票，但它仍不等价于正式 PIT index membership。

### 2.2 数据限制

输入只有 `timestamp/symbol/OHLCV`，价格口径为 `adjustment=none`。以下正式经济 gate 输入缺失：

- point-in-time universe membership；
- `is_listed`、`is_suspended`、`is_st`；
- 涨跌停、最小交易单位和其他可交易状态；
- 公司行动与复权事件；
- 佣金、印花税、滑点或 reference-price cost；
- 与模型预测对齐的 C++ replay return series 和 CVaR。

因此本报告可以比较真实 OOS 排序行为，但不能把 top-bottom label utility 当成执行后净收益，也不能
给任何新候选授予 production promotion。由于所有挑战者在更早的模型 gate 已失败，本实验只执行
“保留 incumbent”，没有绕过缺失的经济 gate。

## 3. 预注册实验设计

### 3.1 冻结配置

| 配置 | 冻结值 |
|---|---|
| 模型 | TemporalTransformerV1 |
| Encoder | `d_model=64`, 3 layers, 4 heads, FFN=128 |
| Lookback | 64 bars |
| 训练预算 | 50 epochs |
| Optimizer | AdamW, learning rate `3e-4` |
| Seed | `20260724` |
| Ranking score | raw expected return |
| Production cutoff | Top-20 |
| Rank temperature | 1.0 |
| Explicit rank weight | 0.1 |
| Batch contract | 每个 batch 恰好一个完整 timestamp |
| 固定任务权重 | return 1.0；direction/volatility/quantile 各 0.25 |

三个 ranking variant 只能取：

- `legacy`：截面 prediction 与 `rank_utility` 的负相关损失；
- `listmle`：冻结 symbol-ascending tie policy 的 Plackett-Luce full-list likelihood；
- `lambda`：由 `delta NDCG@20` 加权的显式 pairwise logistic objective。

LambdaLoss 使用预测 Top-K anchor 对全截面的 `O(KN)` 路径；小截面测试与 NumPy `O(N^2)` full-pair
oracle 对齐。其张量化只减少 Python 调度，不改变 pair 集合、权重或 scalar objective。

### 3.2 Purged walk-forward

每个 fold 使用 504 个训练 timestamp、126 个 validation timestamp、126 个 test timestamp，训练与
validation 之间 purge 6 个 timestamp，validation 与 test 之间 purge 6 并 embargo 5 个 timestamp。

| Fold | Train | Validation | Test |
|---|---|---|---|
| 1 | 2020-04-03 至 2022-05-05 | 2022-05-16 至 2022-11-16 | 2022-12-02 至 2023-06-09 |
| 2 | 2020-10-14 至 2022-11-08 | 2022-11-17 至 2023-05-25 | 2023-06-12 至 2023-12-14 |
| 3 | 2021-04-19 至 2023-05-17 | 2023-05-26 至 2023-11-29 | 2023-12-15 至 2024-06-26 |

同一 timestamp 不会跨数据集。每个 OOS 指标先在 timestamp 内计算，再对 126 个 timestamp 等权
平均，避免股票数或 pair 数隐式改变窗口权重。

### 3.3 指标和胜负顺序

排序首要模型指标为 NDCG@20。辅助指标包括 RankIC、Precision@20、Top-K overlap、单边 turnover
和 top-bottom realized utility spread。

挑战者必须同时满足：

1. NDCG@20 不低于 legacy；
2. top-bottom utility spread 不低于 legacy；
3. turnover 不高于 legacy；
4. 若进入模型候选集，仍需 C++ 净收益/CVaR 经济 gate 才能晋级。

若所有挑战者已在 1-3 失败，则无需伪造缺失的 C++ 经济证据，直接保留 incumbent。

## 4. Ranking loss 实验结果

### 4.1 三窗口聚合结果

| Variant | NDCG@20 | Precision@20 | RankIC | Utility spread | Turnover |
|---|---:|---:|---:|---:|---:|
| legacy | **0.639672** | 0.652778 | **0.054843** | **0.002843** | 0.150800 |
| ListMLE | 0.630905 | **0.658069** | 0.041185 | 0.001789 | **0.108667** |
| LambdaLoss@20 | 0.613128 | 0.637831 | -0.022445 | -0.002159 | 0.159733 |

相对 legacy：

| Variant | Delta NDCG | Delta Precision | Delta RankIC | Delta spread | Delta turnover |
|---|---:|---:|---:|---:|---:|
| ListMLE | -0.008767 (-1.37%) | +0.005291 | -0.013658 | -0.001054 (-37.08%) | -0.042133 |
| LambdaLoss@20 | -0.026544 (-4.15%) | -0.014947 | -0.077289 | -0.005003 | +0.008933 |

### 4.2 Fold 稳定性

| Variant | Fold 1 NDCG | Fold 2 NDCG | Fold 3 NDCG |
|---|---:|---:|---:|
| legacy | **0.645576** | **0.640638** | **0.632803** |
| ListMLE | 0.633242 | 0.630752 | 0.628722 |
| LambdaLoss@20 | 0.617608 | 0.617060 | 0.604717 |

legacy 在三个 fold 的 NDCG@20 都是第一。LambdaLoss 在三个 fold 的 RankIC 分别为 -0.0095、
-0.0062 和 -0.0516；Fold 1 和 Fold 3 的 utility spread 也为负。这不是单一 regime 的偶然失误。

### 4.3 为什么 ListMLE 不晋级

ListMLE 的 Precision@20 略高，并把 turnover 从 0.1508 降到 0.1087，说明 full-list likelihood 产生了
更稳定的 Top-K 集合。这是值得保留的诊断结果，但不足以晋级：

- 主指标 NDCG@20 三个 fold 全部低于 legacy；
- RankIC 下降约 24.9%；
- realized utility spread 下降约 37.1%。

换手降低只有在排序质量和经济 utility 不退化时才有价值。这里不能用低换手覆盖主指标失败。

### 4.4 为什么 LambdaLoss@20 不晋级

LambdaLoss 与生产 Top-20/NDCG 的形式最对齐，但形式对齐不等于在金融数据上自动有效。当前结果为：

- NDCG@20 比 legacy 低约 4.15%；
- RankIC 从正值降为负值；
- utility spread 从正值翻为负值；
- turnover 反而增加约 5.92%。

因此 LambdaLoss 同时失败于 NDCG、utility 和 turnover gate，没有资格进入 C++ 经济 gate。可能原因
包括截面只有约 29-39 只股票、`K=20` 占截面比例较高、raw-return relevance 噪声较强，以及冻结的
temperature/score scale 不利于 pairwise margin。但这些只是后续新假设，不能在看完本次 OOS 后
反调参数并覆盖本实验结论。

### 4.5 Ranking 决策

冻结 `legacy rank loss`。ListMLE 和 LambdaLoss 保留为未晋级的 research candidates，不做运行时
blend，也不设置失败 fallback。

## 5. Kendall 动态权重实验

### 5.1 方法

ranking loss 冻结为 legacy 后，按 Kendall et al. 的任务级同方差不确定性框架学习
`s_k = log(sigma_k^2)`：

```text
L_dynamic = sum_k exp(-s_k) * L_k + 0.5 * s_k
            + 0.1 * L_rank_legacy
```

只学习 return、direction、volatility 和 quantile 四个全局 `s_k`。rank weight 继续显式固定为 0.1，
避免模型通过扩大 rank uncertainty 忽略决策目标。`s_k` clamp 为 `[-6, 6]`，并逐 epoch 记录 raw
loss、weighted loss、`s_k`、effective weight 和 `s_k` gradient。

### 5.2 聚合结果

| 指标 | Fixed | Kendall | Delta |
|---|---:|---:|---:|
| NDCG@20 | **0.639672** | 0.638346 | -0.001327 (-0.21%) |
| Precision@20 | 0.652778 | **0.655820** | +0.003042 |
| RankIC | **0.054843** | 0.036816 | -0.018027 (-32.87%) |
| Utility spread | **0.002843** | 0.001342 | -0.001501 (-52.80%) |
| Turnover | 0.150800 | **0.126800** | -0.024000 |
| Return MAE | **0.043031** | 0.046222 | +7.42% |
| Direction Brier | 0.223946 | **0.223372** | -0.26% |
| Volatility MAE | **0.010209** | 0.011250 | +10.19% |

Kendall 降低了换手并轻微改善 direction Brier，但损害了 NDCG、RankIC、utility spread、return MAE
和 volatility MAE。它不是“总体更好但某项略差”，而是决策目标和两个回归主头同时退化。

### 5.3 权重边界诊断

三个 fold 在第 50 epoch 的 effective weights 如下：

| Fold | Return | Direction | Volatility | Quantile |
|---|---:|---:|---:|---:|
| 1 | 206.01 | 0.731 | **403.43** | 20.86 |
| 2 | 197.24 | 0.725 | **403.43** | 20.58 |
| 3 | 216.30 | 0.729 | **403.43** | 21.97 |

三个 fold 的 volatility `s_k` 都触及下界 `-6`，对应 `exp(-s_k)=403.43`，且最后 10 epochs 存在
贴边。这说明动态权重在当前任务损失尺度下试图继续放大 volatility 权重，有限 clamp 正在实质影响
优化。按预注册规则，任一主任务持续贴边即淘汰，不能把 clamp 后得到的低总 loss 解释为稳定最优。

### 5.4 Kendall 决策

淘汰 Kendall 动态权重，继续冻结固定多任务权重：

```text
return=1.0
direction=0.25
volatility=0.25
quantile=0.25
rank=0.1
```

## 6. 最终冻结结论

Phase 1B 最终冻结组合：

```text
Ranking score: raw expected return
Ranking cutoff: Top-20
Ranking loss: legacy correlation loss
Multitask weights: fixed
Model: TemporalTransformerV1, d_model=64, 3 layers, 4 heads
Training budget: 50 epochs
Seed: 20260724
```

选择原因不是 legacy 在所有辅助指标上都最好，而是它在三个真实 purged OOS fold 中稳定守住主指标
NDCG@20、RankIC 和 realized utility spread。ListMLE 的低换手优势不足以覆盖主指标退化；LambdaLoss
全面失败；Kendall 同时失败于 NDCG、utility 和 weight-boundary gate。

## 7. 可复现性

固定权重 ranking 实验合同：

| Artifact | SHA-256 |
|---|---|
| Dataset | `8b16dfa89e20c34c746b1d4806d66bfb128e4cb9499ee97c7a1cb4064e3b726e` |
| Frozen training config | `f43693d64f48e2b202d0b887de143ef1ce0126c2f98e70c994abf7bf1c0a4ab7` |
| Implementation | `1a56f6ad519d298cc61c83549ca8be16d5036253026cb45301a7bbc3d493dbe6` |
| LabelSpec V2 | `951972c270843871b5176769a28fb42f4d41faf0842ee6fed7094cdc07709d42` |
| RankingScoreSpec V1 | `83e0d87d869f060f6da4872f97c7e52e52b6b6f335e0cc2a7b368ec42a573d11` |
| Fixed paired report | `ba1c979f4115d4bd6c695d4c045e84a7e71869c32772634c4ba0d495ece5a0aa` |

Kendall 实现 SHA-256：
`9c65509441b43194db4fe242603ad41792f0deb998e7cfd0846499057dde272d`。

本次共生成 12 个完整 checkpoint：ranking loss 配对 9 个，Kendall 配对 3 个。所有 checkpoint
均为 50 epochs，seed、LabelSpec hash、RankingScoreSpec hash、K=20 和 temperature=1.0 一致。

验证结果：

- Python：147 passed，8 skipped；
- C++ Phase 1A research build：15/15；
- C++ default build：7/7。

## 8. 后续工作

1. 进入 Phase 1C，补齐 C++ period return ledger、毛/净收益对账、reference-price cost、收益归因和
   block-bootstrap interval。
2. 在重新评估任何 ranking challenger 前，先补齐 PIT universe、可交易状态、公司行动、复权和正式
   成本字段。
3. 不在本次 OOS 上调 Lambda temperature、K、gain 或 label winsorization。任何调整必须登记为新的
   hypothesis family，使用新的 validation/untouched period。
4. Kendall 若未来重开，应先预注册 loss-scale normalization 或任务特定 `c_k`，并使用新的 OOS；不能
   因本次 volatility 贴边而在原窗口上缩窄/放宽 clamp 后重新选优。

## 参考

- Kendall, Gal, Cipolla, *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and
  Semantics*, arXiv:1705.07115.
- 本地冻结证据：fixed-weight paired report、Kendall paired report、data audit 和 Phase status JSON。

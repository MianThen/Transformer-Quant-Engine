# Phase 5 InfoTS 监督消融多 seed 复验归档

发布日期：2026-09-12
状态：**RankIC/NDCG 增益不复现（1/3 seed 方向正确）**；volatility MAE 改善稳定（3/3 seed、7/9 fold）；
`winner_selected=false`、家族保持 `REFERENCE_ONLY`，不晋级

本目录归档 seed 20260912 / 20260913 复验（Kaggle Tesla P100，kernel
`mianthen/run-phase5-supervised-ablation-seeds-v1`，2026-09-11 22:50–2026-09-12 00:26）。
seed 变体由 `infots-supervised-ablation-20260911` 的冻结基础配置在运行时派生（仅改四个 seed
字段：supervised + 三个 pretraining spec；run.py 内可审计）。数据集、fold 窗口、gate 合同、
监督预算与首 seed 完全一致。

## 结论（三 seed × 三组 × 三 fold 聚合，见 `analysis/aggregate_seeds.py` 输出）

| seed | infots−from_scratch RankIC | NDCG@20 | vol MAE |
|---|---:|---:|---:|
| 20260911 | +0.0128 | +0.0021 | -0.00046 |
| 20260912 | -0.0209 | -0.0072 | -0.00041 |
| 20260913 | -0.0251 | -0.0123 | -0.00057 |

- **首 seed 的 RankIC +33% 为单 seed 噪声**：两个复验 seed 全部反向且幅度更大；fold 级方向
  一致性 4/9。from_scratch 自身的 seed 间波动（0.039/0.047/0.059）已大于任何组间差。
- **volatility MAE 改善是唯一可复现效应**：3/3 seed、7/9 fold 方向一致，幅度一致（约 -5%
  相对）。按纪律作为**未来新假设**（面向波动率头的预训练）记录，不构成本次晋级理由。
- fixed_augmentation 组各指标同样不稳定，不构成对照优势。
- 两个 seed 的 `RESULT_VALIDATION` 均 PASS（9 artifacts，自哈希 + 合同验证通过）。

## 训练动态诊断（迭代空间判断依据）

对全部 27 个 run 的 `epoch_history` 分析：return-guarded checkpoint 选择的 epoch 中位数为
**2**（最大 7）；**27/27 run 在 epoch 50 前显著过拟合**（验证 return MAE 比最优差 20-40%）。
即三组对照实际比较的是欠训练快照，预训练初始化差异缺乏表达长度——这是实验设计缺陷，
不是 gate 失败后的调参对象。据此授权**一次**正则化修复复训（dropout + weight decay + 降 lr，
同 3 seed 配对比较），预注册决策规则与判读口径见复训归档。

## 目录

- `seed-2026091{2,3}/fold-N/<group>/artifact.json`：各 seed 的 9 个 fold artifact。
- `seed-*/RESULT_VALIDATION.json`、`seed-*/phase5_infots_ablation_report.json`。
- `analysis/aggregate_seeds.py`：三 seed 聚合与方向一致性判定脚本（含 seed 20260911 路径）。

checkpoint 与 npz 不随仓库发布（`.gitignore` 政策）；保留在本地训练输出与 Kaggle kernel 输出。

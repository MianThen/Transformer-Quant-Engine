# Phase 5 正则化修复复训归档（家族终局）

发布日期：2026-09-12
状态：**R1 正控制 FAIL（vol 效应被复训破坏）**；R2 无翻转证据（RankIC/NDCG 均 2/3 混合方向且不满足
|Δ|>0.005）；R3 动态检验通过（selected_epoch 中位 2→4、均值 2.4→11.8，正则化确实改变了训练动态）。
按预注册规则：**该家族终止迭代、封闭，不晋级**。

## 背景与预注册

监督消融多 seed 复验（见 `../infots-supervised-ablation-seeds-2026091x/`）判定排序增益不复现，
但 vol MAE 改善 3/3 seed 稳定；训练动态诊断发现 27/27 run 过拟合、selected_epoch 中位 2。
据此授权一次正则化修复复训：dropout 0.1 / weight_decay 1e-4 / lr 1e-4，其余配置与原配置
逐字段一致，同 3 seed（20260911/12/13）配对。预注册决策规则（训练前冻结）：

- R1 正控制：infots−from_scratch 的 vol MAE 须 3/3 seed < 0；否则复训破坏原效应，终止。
- R2 翻转检验：rank_ic / ndcg 仅当 3/3 seed 同向且 |Δ|>0.005 才构成翻转证据。
- 无论结果，这是该家族最后一次复训。

运行：kernel `mianthen/run-phase5-supervised-reg-ablation-v1`（2026-09-12 07:57–10:20，
3 seed 全量）；结构校验 3×9 artifacts 全 PASS；数据集
`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`。

## 结果（infots − from_scratch，reg vs orig）

| seed | metric | orig | reg |
|---|---|---:|---:|
| 20260911 | rank_ic | +0.0128 | +0.0246 |
| 20260911 | ndcg | +0.0021 | +0.0081 |
| 20260911 | vol_mae | **-0.0005** | **+0.0348** |
| 20260912 | rank_ic | -0.0209 | -0.0188 |
| 20260912 | ndcg | -0.0072 | -0.0065 |
| 20260912 | vol_mae | **-0.0004** | **+0.0207** |
| 20260913 | rank_ic | -0.0251 | +0.0026 |
| 20260913 | ndcg | -0.0123 | +0.0036 |
| 20260913 | vol_mae | **-0.0006** | **+0.0284** |

reg run 三组 vol MAE 均值：from_scratch 0.0099 / fixed_augmentation 0.0157 / infots 0.0378。

## 判定与解释

1. **排序增益不存在**：两次独立多 seed 复验（原配置、正则化配置）均无法复现首 seed 的
   RankIC 正向差异——该家族对主要排序指标无稳定收益，判定终结。
2. **vol 改善是训练配置依赖的脆弱效应**：仅在"无正则化 + 极早停（epoch 1-7）"条件下出现；
   训练动态改变后不仅消失且反向（infots 组 vol MAE 恶化 2-4×）。它不是 InfoTS 预训练的
   稳健属性，不构成可晋级证据；若未来重开须以全新 hypothesis family 登记。
3. 正则化本身按设计生效（R3），因此 R1 失败不能归因于"复训没跑出预算"。

## 目录

- `seed-2026091{1,2,3}/fold-N/<group>/artifact.json`：27 个 fold artifact（自哈希+合同验证 PASS）。
- `seed-*/RESULT_VALIDATION.json`、`seed-*/phase5_infots_ablation_report.json`。
- `analysis/aggregate_reg.py`：预注册规则评估脚本（R1/R2/R3，输出如上）。
- `analysis/phase5_infots_supervised_ablation_reg_cuda.json`：冻结的正则化配置
  （supervised spec sha `86252ec399cae04093157e202ed89ae42e80e3ead83bed9b50176bb44920111b`）。

checkpoint 与 npz 不随仓库发布（`.gitignore` 政策），保留在本地与 Kaggle kernel 输出中。

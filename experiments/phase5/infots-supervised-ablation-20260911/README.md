# Phase 5 InfoTS supervised-budget ablation archive

发布日期：2026-09-11
状态：三组对照全部通过 15/15 gate；`winner_selected=false`、`research_gate_passed=false`；
证据级别 `REFERENCE_ONLY`（fold 窗口复用 Phase 2B r6 预注册 3×126 validation fold，项目层面已被观察）

本目录归档 Phase 5 InfoTS 三组等监督预算消融（`from_scratch` / `fixed_augmentation` / `infots`）
在外部 GPU（Kaggle Tesla P100，PyTorch 2.8.0+cu126）的真实训练结果。它是 research-only 模型选型
证据，不是 Phase 5 退出或生产晋级证明。

## 预注册设计（训练前冻结于训练包）

- 监督模型：`InfoTSCausalEncoder`（hidden 128 / embedding 64，与预训练同构）+ 固定六输出头；
  预训练组只加载 `encoder.*` 权重。
- 三组初始化：随机 / 固定增强 `INFOTS-FIXED-NOISE-V1` 预训练 / 因果增强管线
  `INFOTS-CAUSAL-BASELINE-V1` 预训练；预训练语料固定为 fold-1 `train_end`
  （1700550000000000000）之前，不与任何 fold 的 validation/test 窗口重叠；已回传的全语料
  InfoTS checkpoint 因窗口重叠不用于本次消融。
- 监督预算：3 folds × 50 epochs，seed 20260911；损失权重 return 1.0 / direction 0.25 /
  volatility 0.25 / quantile 0.25（pinball 0.25/0.75）+ confidence 0.05；checkpoint 选择为
  validation return-guarded（margin 0.0025）。
- gate spec：`74f2ad9e09b588079d92cbe0ee9f9e13d7940f6041dff90a5c632ef156ebddc7`。
- 数据集：`phase1e_pit_120_dataset.npz`
  `4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`。

## 结论（test 均值 across 3 folds）

| 组 | composite | return MAE | direction Brier | vol MAE | NDCG@20 | RankIC |
|---|---:|---:|---:|---:|---:|---:|
| from_scratch | 0.089348 | 0.045315 | 0.228238 | 0.009169 | 0.536938 | 0.039198 |
| fixed_augmentation | 0.089357 | 0.045416 | 0.228236 | 0.009067 | **0.548877** | 0.024097 |
| infots | 0.089322 | 0.045441 | 0.228442 | **0.008709** | 0.538999 | **0.051951** |

- **infots 组方向性有效**：RankIC 0.039 → 0.052（+33% 相对），NDCG +0.002，vol MAE 三组最优，
  return MAE 代价可忽略（+0.0001）。
- fixed_augmentation 组 NDCG 最高（+0.012）但 RankIC 反向（-0.015），方向不一致。
- 三组 × 五项 gate（clean/stress/three-window/quality/stability）全部 3/3 通过。
- 单 seed、REFERENCE_ONLY：构成"值得多 seed 复验"的证据；seed 20260912/20260913 复验运行中，
  稳定性结论以复验为准。

## 目录

- `phase5_infots_ablation_report.json`：聚合报告（不选 winner）。
- `RESULT_VALIDATION.json`：结构校验（PASS，9 artifacts）。
- `runtime_preflight.json`：CUDA preflight（`CUDA_PREFLIGHT_PASS`，Tesla P100）。
- `fold-N/<group>/artifact.json`：9 个 group/fold artifact（自哈希 + 合同验证通过；含六项指标、
  五项 gate、逐 epoch history、validation 指标与 stress 指标、预测/embedding 产物指纹）。
- `configs/`：冻结的监督消融配置与预算合同。
- checkpoint 与 npz 预测/embedding 不随仓库发布（`.gitignore` 政策），保留在本地训练输出与
  Kaggle kernel 输出中；其 SHA-256 已固化在各 artifact.json 内。

## 复现

训练包 `qbt_phase5_supervised_gpu_v1`（707MB，归档 SHA-256
`a40bfe5adb028696791a44ee99a74511839ffc03d8e56e2e3efec0b7e1e69b42`）；kernel
`mianthen/run-phase5-supervised-ablation-gpu-bundle-20`（version 4，2026-09-11 21:11–21:55）。
运行入口与断点续跑说明见训练包 README 与 `--resume`。

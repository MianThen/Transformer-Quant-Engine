# Phase 2B Feature-PGD V2 r6 P3 validation grid archive

发布日期：2026-09-11
状态：`STOP_NO_CLEAN_GUARD_PASS`；P4 不授权；`phase_exit_eligible=false`、`promotion_eligible=false`

本目录归档 Phase 2B Feature-PGD V2 的 r6 P3 validation-only 网格（PHASE2B_DECISION 授权的唯一
一轮 epsilon/beta 优化）完整证据。它是 research-only 终局记录，不是生产晋级证明。

## 结论

- 网格：baseline（`none`）+ 7 个 pure `feature_pgd` 候选 × 3 个固定 validation fold，seed
  `20260805`，50 epochs，`structured_missing` 已排除；共 24 个 checkpoint，全部在 bundle 内训练
  （禁止外部 checkpoint 复用）。
- 执行环境：Kaggle Tesla P100-PCIE-16GB，PyTorch `2.8.0+cu126`，CUDA 12.6，deterministic
  algorithms。首次 session 在完成 23/24 cell 后中断；resume session 补齐
  `fold-3/pgd-e010-b050-control` 并执行 gate/selection/报告，两次 session 的重叠 cell
  checkpoint SHA-256 完全一致。
- Gate 口径（预注册，`gate_schema_version=2`）：timestamp 级配对序列 + fold 等权 pooled +
  circular moving-block bootstrap（block 6，2000 draws），clean guard 通过条件为
  **95% 单侧 CI 上界 ≤ margin**，stress gate 要求四类压力逐 fold 点估计严格为负且 pooled
  CI 上界严格小于零。
- 判定：7/7 候选至少违反一个 clean guard，`top2_for_p4=[]`、`p4_allowed=false`。
  - `ndcg_at_cutoff` 是系统性失败点：全部候选的 CI 上界越过 0.005 margin（含点估计改善的
    `pgd-e020-b025`，est `-0.0031` 但 CI 上界 `+0.0100`），即 PGD 无法以统计置信度证明排序非劣。
  - 高预算候选另触发 volatility 崩溃：`pgd-e020-b010` 相对退化 est `+0.4353` / CI 上界
    `+0.5036`（margin 0.02）；`pgd-e010-b050-control` est `+0.3119` / CI 上界 `+0.3656`。
  - `rank_ic` CI 上界 `+0.016~+0.034`（5 个候选）、`pgd-e005-b010` 另挂 return/direction。
- stress gate：7/7 候选四类压力（price/volume/missing/extreme_volatility）全部失败。
- 按预注册合同停止本轮、不放宽门槛。结合 `bundle/PHASE2B_DECISION.md`：Phase 2B 不晋级、
  不退出研究状态、不进入生产；该 hypothesis family 后续重开必须以全新 hypothesis family 与
  新 untouched OOS 登记，不得继续调整 epsilon/beta/steps。

## 目录

- `preregistered_contract.json`：r6 网格预注册合同（训练前冻结，含 guard 定义与 variant specs）。
- `P3_GRID_SELECTION.json`：确定性 Top-2 选择器输出（本轮为空选）。
- `RESULT_VALIDATION.json`：回传产物结构校验（`PASS`，24/24 checkpoint）。
- `phase2b_oos_report.json`：完整 OOS 报告，含逐 fold/variant 的 clean/noisy/stress 与
  timestamp 级指标。
- `MANIFEST.sha256`：运行输出 manifest。
- `runtime_preflight_validation_grid.json`：CUDA preflight（`CUDA_PREFLIGHT_PASS`）。
- `validation-grid/fold-N/<variant>/`：每 cell 的 `metrics.json`、`checkpoint_selection.json`
  与 embedding manifest（train-fold-only standardizer、checkpoint 选择 hash 均在内）。
- `bundle/`：训练输入包自带的 `PHASE2B_DECISION.md`、`AUDIT_R6.md`、`README.md`。
- `archives/`：完整结果归档（含 checkpoint/npz）的 SHA-256 sidecar。归档本体 113 MiB，
  超出 GitHub 单文件 100 MiB 硬限制，不入 Git；以 sidecar 固化指纹。

## 数据与复现边界

数据集 `phase1e_pit_120_dataset.npz` SHA-256 为
`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`（1521 个 model-ready
timestamp，features `173242×64×23`），数据审计 hash
`5aedcc0bbbcd5cebda11a5e2e15be0a8420f65810c43cf7e71b77ff22c7c3c47`。数据集与 checkpoint 不随
仓库发布。本目录只收录可审计 JSON 与文档，不含私人路径或凭据。

## Published SHA-256

```text
preregistered_contract.json (r6)  a8190e6520d21eb9e5fc064a1001f7b3c5fcac54eba00e74b7e79fa8b6091b61
P3_GRID_SELECTION.json (selection) c3d119e6cafd9cf1672780545a0fb05689ebb62ac4dd32003be49737e358e089
phase2b_oos_report.json (report)   8972caad57f4eb50fcd8b65b836ac8e6c603748c15cae30a7968a9522cb8b851
training_config_sha256             7ccff8d23ab1f9c1cbacdd8bb80cc761c677c8221e71f6ab5e02e428c2a8493c
implementation_sha256              6b61400ba517fa6ed18f19ae166c46e637b06bd294afabca934a07ab64fa789e
results archive (20260823_r6)      0cb7d7576396bac62818cbaf8dbcf34dd9ba8514876747ff53d078e99b5937b2
```

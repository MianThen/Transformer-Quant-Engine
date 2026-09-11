# Experiments

本目录只保存经过整理、可复现且不包含私人数据的实验说明。

本地训练输出、checkpoint、模型、数据集、attention 矩阵、ablation 中间结果和 benchmark 临时
文件默认不提交。需要发布结果时，应同时提供配置、代码 revision、数据 fingerprint、硬件环境、
Leakage Detection 报告和指标口径。

## 已归档实验

| 目录 | 内容 | 状态 |
|---|---|---|
| `phase1e/` | 共享梯度诊断、PCGrad、GradNorm-4 与 C++ proxy Replay/CVaR | report-only，promotion NO-GO |
| `phase1f/` | Top-K 稳定性与 oracle 合同结果 | 见目录 README |
| `phase2b/baseline-oos/` | Feature-PGD V2 之前的 baseline OOS 记录 | 见目录内文件 |
| `phase2b/feature-pgd-v2-r3/` | r3 外部 GPU 回传（validation + short OOS） | `TRAINING_COMPLETE_GATE_FAILED` |
| `phase2b/feature-pgd-v2-r3-rerun-20260807/` | r3 重跑审计与自包含归档 | validation/stress gate 失败 |
| `phase2b/feature-pgd-v2-r5/` | r5 包执行 r4 预注册假设（validation + 新短 OOS） | volatility guard + 跨窗 stress 失败 |
| `phase2b/feature-pgd-v2-r6-p3-grid-20260830/` | r6 P3 validation-only 网格终局（24 checkpoint） | `STOP_NO_CLEAN_GUARD_PASS`，P4 不授权 |
| `phase5/infots-pretraining-20260809/` | InfoTS 对比预训练 GPU artifact | `GPU_TRAINED_REFERENCE_PENDING_OOS` |
| `phase5/infots-supervised-ablation-20260911/` | InfoTS 三组等监督预算消融（seed 20260911） | 全 gate 通过；infots 组 RankIC +33%（REFERENCE_ONLY，多 seed 复验中） |


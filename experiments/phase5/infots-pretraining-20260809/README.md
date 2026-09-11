# Phase 5 InfoTS pretraining artifact archive

发布日期：2026-09-11
状态：`GPU_TRAINED_REFERENCE_PENDING_OOS`；`phase_exit_eligible=false`、`promotion_eligible=false`

本目录归档 Phase 5 InfoTS 对比预训练（`INFOTS-CAUSAL-SIAMESE-V1`）在 Kaggle Tesla P100 上的
真实 GPU 训练产物记录。它只是预训练 reference，不是监督预测证据，不能用于 Phase 2B 失败后的
补偿论证，也不支持任何晋级。

## 结论

- causal GRU Siamese encoder，embedding 维度 64，对称 InfoNCE，50 epochs，训练样本
  160,446；`train_fold_end = 1768374000000000000`（严格 train-fold-only，拒绝 fold 末端之后
  的 timestamp）。
- loss history 由 1.195 收敛至约 0.089 量级（完整序列见 `artifact.json`）。
- limitations（artifact 原文）：`pretraining only`、`no supervised six-output predictions`、
  `no OOS/cost gate`。
- 下一步是三组等监督预算的监督消融（from-scratch / fixed-augmentation / InfoTS init）与
  purged OOS 六输出对比；在该证据完成前，本 artifact 保持 `REFERENCE` 级别。

## 目录

- `artifact.json`：预训练 artifact（含 dataset/augmentation/training spec/checkpoint/
  global+local embedding 的全部 SHA-256 与 loss history）。
- `runtime_preflight.json`：CUDA preflight 记录。
- `run-phase5-infots-gpu-training-bundle-20260809.log`：Kaggle 会话运行日志。
- `configs/phase5_infots_pretraining_cuda.json`：冻结的 CUDA 训练配置。

## 数据与复现边界

数据集 SHA-256 `4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`，不随仓库
发布。按 `.gitignore` 政策 `*.pt` 不入库：预训练 checkpoint（504 KB）保留在本地训练输出中，
其 SHA-256 `988221b89b7da0226068555cd87abf50f1386c51c8c60f4a6f04655ef99d46ea` 已固化于
`artifact.json`。

## Published SHA-256

```text
artifact.json (artifact_sha256)  b6eb3ed83ea3af9239913db6865783eab05df256d294db6885b7e883ea018887
checkpoint.pt (reference only)   988221b89b7da0226068555cd87abf50f1386c51c8c60f4a6f04655ef99d46ea
```

# Phase 2B Feature-PGD V2 施工状态

更新时间：2026-08-06

## 当前状态

`Feature-PGD V2 r3` 已完成 validation `3×126` 和 short OOS `3×39` 的三折 50-epoch 外部 GPU 训练，产物完整，但研究 gate 失败。当前正式状态为：

```text
TRAINING_COMPLETE_GATE_FAILED
phase_exit_eligible = false
promotion_eligible = false
```

完整分析与下一轮施工合同见 `docs/phase2b_feature_pgd_v2_r3_analysis_and_iteration_plan.md`。

## r3 回传判定

- validation archive SHA-256：`b20e30e9be73d8654db7996c78fb4186ec078d111dcf11688293e40e21446b8d`；内部 48/48 条目、9/9 checkpoints 通过。
- short-OOS archive SHA-256：`f3b3d959a008912af3c4f8445c4be94cb699d2b721c1bfbac701cfe55f84fe57`；内部 75/75 条目、9/9 checkpoints 通过。
- validation：composite `-0.010971`、direction Brier `-0.034392`，但 return MAE `+0.001250`、volatility MAE `+0.000229`，extreme-volatility stress `+0.003639`。
- short OOS：composite `-0.002588`、direction Brier `-0.011666`、volatility MAE `-0.000971`，但 return MAE `+0.004872`；fold 2 的 price/volume/missing 方向不一致。
- 两套报告均为 `research_gate_passed=false`；`RESULT_VALIDATION=PASS` 仅代表结构与产物完整。
- 已证实当前 `feature_pgd` 实际叠加了 10% structured missing，不能视为 pure-PGD 单变量消融；r4 必须先拆分。

## 已完成

- 新增 train-fold-only `FeatureStandardizerV1`；只标准化 19 个连续特征，状态字段和 padding 保持原语义。
- 标准化统计量写入模型配置、checkpoint、metrics，并带 canonical SHA-256。
- 模型保持 raw-input forward 合同；PGD 只在标准化 z-space 运行，ONNX 导出路径不包含训练对抗分支。
- 新增 `continuous_center_impute` 和 `structured_missing`；正式 missing 压力不再使用 raw 全零 + `valid_mask=1`。
- 报告增加 relative stress degradation，保留 V1 gate 结果不变。
- 评估结果增加按交易日的 paired loss 和 circular moving-block bootstrap CI；当前仍只作为诊断，不追溯放宽 V1 gate。
- 旧 checkpoint 缺失 scaler buffer 时显式补 identity；其他权重缺失仍严格失败。
- 新配置：`configs/ml/phase2b_feature_pgd_v2.local.json`。
- 新增 validation-only 配置：`configs/ml/phase2b_feature_pgd_v2_validation.local.json`；validation 训练路径为真正 test-blind，不执行或写出 test metrics、prediction 或 embedding，结果校验器发现 test 制品即失败。
- 已生成 r3 GPU 迁移包：`/Users/Zhuanz/Downloads/phase2b_feature_pgd_v2_gpu_training_bundle_20260805_r3.tar.gz`；归档 SHA-256 为 `df9c9258e5c9bea25a370312fc3054f49e62cebe83190adc1c25b025c959b740`。
- r3 使用显式 CUDA 配置和运行前检查；CUDA 不可用时 fail closed，不静默回退 CPU。
- r3 已执行预注册短 OOS：以 V1 最后 test timestamp `1768374000000000000` 为 cutoff，使用其后 117 个 timestamp，冻结为 `3×39`。该窗口现在已经被观察，只能保留为低功效 `research-only` 证据。
- r3 short OOS 最后 timestamp 为 `1783926000000000000`（2026-07-13 15:00 Asia/Shanghai）。真正新的 r4 `3×39` 需要此后 117 个新 timestamp，新的 `3×126` 需要此后 378 个新 timestamp。

## 验证

- Phase 2B V2 定向测试：`14 passed`（`14/14`）。
- Python 全量测试：`182 passed, 8 skipped`。
- 跳过项仍是可选依赖缺失，不构成 OOS 证据。
- 两个外部结果包的 `RESULT_VALIDATION` 均为 `PASS`，各 9 个 checkpoint、0 failures；研究 gate 仍为 `FAIL`。

## 下一步

1. 先在本机拆分 pure PGD 与 structured missing，加入统一 best-validation checkpoint、per-head attack diagnostics、金融语义投影和真正使用 bootstrap CI 的 gate。
2. 只在 development validation 上运行预注册 epsilon/beta 小网格；Top-2 再使用三个 seeds 复验，禁止选择最好 seed。
3. 冻结 winner 后等待 2026-07-13 之后的新 untouched OOS；旧 `3×39` 不得用于 r4 调参或重新晋级。

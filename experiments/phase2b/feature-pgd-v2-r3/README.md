# Phase 2B Feature-PGD V2 r3

本目录保存 2026-08-06 审计完成的 r3 外部 GPU 实验资料。它们只证明训练和结果链路完整，不代表 Feature-PGD 通过研究 gate。

## 状态

- validation：`development_validation_only`，`research_gate_passed=false`
- short OOS：`research_only_no_promotion`，`research_gate_passed=false`
- 最终登记：`TRAINING_COMPLETE_GATE_FAILED`
- 数据集 SHA-256：`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`

## 归档

- `archives/phase2b_feature_pgd_v2_validation_results.tar.gz`：`b20e30e9be73d8654db7996c78fb4186ec078d111dcf11688293e40e21446b8d`
- `archives/phase2b_feature_pgd_v2_short_oos_results.tar.gz`：`f3b3d959a008912af3c4f8445c4be94cb699d2b721c1bfbac701cfe55f84fe57`
- `validation/` 与 `short-oos/` 保存 contract、report、结构校验和原始 CRLF manifest 副本。

`RESULT_VALIDATION=PASS` 仅表示产物完整；模型效果 gate 失败。完整根因和 r4 施工合同见 `docs/phase2b_feature_pgd_v2_r3_analysis_and_iteration_plan.md`。

## 复现边界

当前 short OOS 最后 timestamp 为 `1783926000000000000`（2026-07-13 15:00 Asia/Shanghai），已被观察，不得再用于调参或宣称 untouched OOS。r4 必须先拆分 pure-PGD 与 structured missing，再运行 validation-only 网格。

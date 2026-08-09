# Phase 2B Feature-PGD V2 rerun archive

本目录归档 2026-08-07 收到的 `OUTPUT.zip` 审计结果。它是 research-only 证据，不是生产晋级证明。

## 结论

- 外层 ZIP 可正常解压，`unzip -t` 通过。
- validation 与 short-OOS 各有 3 folds × 3 variants，共 9 个主 checkpoint；两个 `RESULT_VALIDATION.json` 均为 `PASS`。
- `RESULT_VALIDATION=PASS` 只表示产物结构和引用完整，不表示 Feature-PGD 效果 gate 通过。
- validation 的 `feature_pgd` gate 失败；short-OOS 也因逐折 relative-stress gate 失败，不能晋级或进入生产图。
- 原始 `OUTPUT.zip` 未提交到普通 Git 树；其哈希和两份可下载结果包的哈希见 `inventory.json` 与 `source/`、`archives/`。

## 目录

- `validation/`：validation 合同、报告、结构校验和归档 manifest。
- `short-oos/`：短 OOS 合同、报告、结构校验和归档 manifest。
- `archives/`：包含完整 checkpoint、prediction、embedding、metrics 的结果压缩包。
- `analysis_and_iteration_plan.md`：指标分析、失败原因和下一轮改造/退出条件。

二进制结果只保留在压缩包中，避免把 159 个展开文件重复写入 Git 历史。原始 ZIP 仍保留在用户的 Downloads 目录。

# Phase 2B Feature-PGD V2 r5 validation + new short OOS archive

发布日期：2026-09-11
状态：validation `development_validation_only`；short OOS `research_only_no_promotion`；
两个 `RESULT_VALIDATION.json` 均为 `PASS`（各 9/9 checkpoint）

本目录归档 r5 训练包（执行 r4 预注册假设 `v2-r4-test-blind-validation` 与
`v2-r4-short-new-oos`）在外部 GPU（Kaggle Tesla P100，PyTorch `2.8.0+cu126`）回传的结果。
它是 research-only 证据，不是生产晋级证明。

## 结论

- validation（3×126 窗口）与 short new OOS（cutoff `1768374000000000000` 之后 3×39 窗口）
  各 3 folds × 3 variants（`none` / `feature_pgd` / `structured_missing`），seed `20260805`。
- `RESULT_VALIDATION=PASS` 只表示产物结构与引用完整，不表示效果 gate 通过。
- 判定（见 r6 包内 `PHASE2B_DECISION.md` 的引用）：pure Feature-PGD 改善了 return MAE 与
  排序类指标，但 **clean volatility guard 失败、跨窗口 stress 稳定性不成立**，因此不晋级，
  并据此授权了唯一一轮 r6 P3 validation-only 网格（见
  `../feature-pgd-v2-r6-p3-grid-20260830/`）。
- short OOS 窗口自本轮起已被观察，不再是 untouched OOS，不得用于调参或重解释。

## 目录

- `validation/`：validation 合同、报告、结构校验。
- `short-oos/`：短 OOS 合同、报告、结构校验与归档 manifest。
- `archives/`：自包含结果归档（含完整 checkpoint、prediction、embedding、metrics）及其
  SHA-256 sidecar。
- `runtime_preflight_validation.json`：CUDA preflight 记录（`CUDA_PREFLIGHT_PASS`）。

## 数据与复现边界

数据集 `phase1e_pit_120_dataset.npz` SHA-256
`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`。数据集不随仓库发布；
checkpoint 等二进制只保留在 `archives/` 压缩包内，与 r3 系列的 self-contained 归档约定一致。

## Published SHA-256

```text
preregistered_contract.json (validation) 9c729d73add97584895a153bbd403bf9b24e98bf638ebe9813d99c3176bf6750
preregistered_contract.json (short-oos)  849d915087374b13ea0984fa4ee14936d212844b9232f349ef366597f9704a05
results archive (short oos)              a43591897dcf5cab41070ae6e23e5f32a6362f657548599bc8ba1091b5fb2f42
```

# OUTPUT.zip 审计与 Feature-PGD 迭代方案

## 1. 审计范围与可信边界

输入是 `/Users/Zhuanz/Downloads/OUTPUT.zip`，SHA-256 为 `23a46ff0e356ddf34745d4e24abcba07b9f13e6f4d557fde43fcf9a326d32e00`。ZIP 共 188 个成员（含目录），159 个文件，`unzip -t` 通过。

包内有两套产物：

- validation：3 folds × (`none`、`structured_missing`、`feature_pgd`)，9 个主 checkpoint；`RESULT_VALIDATION.status=PASS`，效果状态为 `development_validation_only`。
- short-OOS：3 folds × 同样 3 个 variant，9 个主 checkpoint；`RESULT_VALIDATION.status=PASS`，效果状态为 `research_only_no_promotion`。

原始 ZIP 没有顶层 `MANIFEST.sha256`，两个结果目录也没有根级 manifest。本归档对展开结果重新生成了 manifest，并对两份 tar.gz 做了独立 SHA-256 sidecar。结构 PASS 不等于研究 gate PASS。

另一个边界是 checkpoint 选择：18 个 `checkpoint_last.pt` 与对应 `checkpoint.pt` 逐一相同，18 个 `checkpoint_best.pt` 均不同；原始 `RESULT_VALIDATION.json` 只覆盖 18 个 `checkpoint.pt`，没有覆盖 `checkpoint_best.pt`。因此本报告不把 best checkpoint 当作已完成的独立评估结果。

## 2. 结果摘要

误差类指标越低越好。下面是聚合 clean 指标，方向为 `none → feature_pgd`。

| 窗口 | Composite error | Return MAE | Direction Brier | Volatility MAE | NDCG@K | Rank IC | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| validation | 0.126672 → 0.126223 | 0.066288 → 0.074799 | 0.303149 → 0.292559 | 0.010578 → 0.011313 | 0.485216 → 0.472724 | 0.031201 → 0.014207 | 失败 |
| short-OOS | 0.132512 → 0.122283 | 0.071633 → 0.068625 | 0.314008 → 0.287183 | 0.011896 → 0.011042 | 0.482548 → 0.489396 | 0.026904 → 0.059209 | 失败 |

### validation 的实质性失败

- Return MAE 增加 `+0.008510`（约 `+12.8%`），Volatility MAE 增加 `+0.000735`。
- NDCG@K 下降 `-0.012492`，Rank IC 下降 `-0.016994`；方向 Brier 虽改善 `-0.010591`，但没有抵消收益/排名损失。
- extreme-volatility composite error `0.136060 → 0.143403`，missing stress `0.126454 → 0.127288`。
- gate flags 全部为 false：`clean_non_degraded=false`、`stress_non_degraded=false`、`all_three_window_consistent=false`、`research_gate_passed=false`。

### short-OOS 的近通过但仍失败

- clean、raw stress 和三窗口 raw consistency 均为 true。
- relative-stress 的逐折门禁仍失败：missing fold 2 退化约 `+0.003516`，extreme-volatility fold 3 退化约 `+0.007856`，均超过预注册 margin `0.0025`。
- 聚合 relative delta 不能掩盖逐折失败；因此不能把 short-OOS 标记为通过。

## 3. 根因判断

1. **窗口/制度不稳定。** 同一合同在 short-OOS 改善收益和排名，在较长 validation 却牺牲收益和排名；这不是单纯“样本少”的问题，而是跨窗口泛化不稳定。
2. **多任务权衡失衡。** PGD 明显改善方向任务，但固定的收益、波动和排名目标没有受到非退化约束，导致 clean return/volatility/ranking 被牺牲。
3. **极端波动保护不足。** validation 的 extreme-volatility 退化是明确的方向性失败，不能通过放宽 gate 掩盖。
4. **训练预算固定打满。** 合同使用 `epsilon=0.01`、`beta=0.5`、3 steps；诊断中的 `feature_perturbation_linf` 约为 `0.0100002`。需要先扫攻击预算和任务权重，再考虑增加训练轮数。
5. **审计链仍有缺口。** 根级 manifest 缺失；`checkpoint_best.pt` 未被结构校验器覆盖，且没有独立评估报告。

## 4. 是否继续优化

**继续，但限定为 research-only validation 迭代；不晋级生产，也不重跑已经观察过的 short-OOS。** 原因是 short-OOS 提供了“方向可行性”信号，而 validation 明确指出收益/排名和极端波动的保护约束尚未满足。若下一轮仍不能满足收益非退化和逐折 stress gate，则冻结 Feature-PGD，保留 `none` 或 `structured_missing` 基线。

## 5. 改造方案

### A. 先修结果与合同链

1. 在 `package_results.py` 中改用显式 `open(..., newline="\\n")` 写 manifest，兼容当前 Python 3.9；每次包都必须生成根级 `MANIFEST.sha256` 和外层 `.sha256`。
2. `validate_results.py` 同时校验 `checkpoint.pt`、`checkpoint_last.pt`、`checkpoint_best.pt`、prediction 和 embedding 的存在性与哈希；best checkpoint 必须有独立评估字段，不能只凭文件名推断最佳。
3. 将 `checkpoint_selection`、selected epoch、评估 split 和报告哈希写入 contract；若 best 未评估，状态明确标为 `UNVERIFIED_BEST`。

### B. 在 validation-only 上做纯 PGD 消融

1. 固定三折、数据、seed、模型结构和 gate，不再触碰已观察的 short-OOS 时间戳。
2. 明确分离 `none`、`structured_missing` 和 `feature_pgd(missing_mode=none)`；每个 variant 单独记录缺失率与攻击预算。
3. 预注册小网格：`epsilon ∈ {0.0025, 0.005, 0.01}`、`beta ∈ {0.10, 0.25, 0.50}`，`pgd_steps=3` 固定。先用 validation 选候选，不以 OOS 反复调参。

### C. 防止收益/排名被方向任务挤掉

1. 给 return MAE、volatility MAE、NDCG/Rank IC 加非退化约束或动态任务权重；方向改善不能抵销收益目标退化。
2. 将 `return_mae` 和 `ndcg_at_cutoff` 纳入候选排序，优先选择满足收益非退化 margin `0.0025` 的 checkpoint，而不是只看总 loss。
3. 对极端波动样本使用预注册、固定强度的 targeted stress/重加权；不在已观察 OOS 上调压力尺度。

### D. 验证与退出条件

1. 候选先跑 validation 三折；通过后再跑至少三 seed 的 validation 稳定性检查。
2. 只有同时满足 clean non-inferiority、四类 stress non-inferiority、每个 fold 的 relative-stress margin、三窗口方向一致、bootstrap CI 与产物完整性，才允许申请新 untouched OOS。
3. 新 OOS 必须使用训练后才出现的时间戳；当前 short-OOS 已观察，不能因调参后再次使用。
4. 若两轮预注册 validation 网格仍有任一收益或 extreme-volatility gate 失败，停止 Feature-PGD 优化，生产候选回退到 `none`/`structured_missing`，并把 PGD 保留为研究分支。

## 6. 当前状态

`RESULT_VALIDATION=PASS`、`research_gate_passed=false`、`promotion_eligible=false`。本归档保存了可复核证据，但不改变 Phase 2B 的晋级状态。

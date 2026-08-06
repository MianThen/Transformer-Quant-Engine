# Phase 2B Feature-PGD V2 r3 结果分析与 r4 迭代方案

更新时间：2026-08-06

## 1. 结论

本次回传结果完整、可校验，训练链路确实完成；失败不是压缩包损坏，也不是 CUDA 训练未生效。

```text
artifact_integrity = PASS
validation_status = development_validation_only
short_oos_status = research_only_no_promotion
research_gate_passed = false
promotion_eligible = false
r3_final_status = TRAINING_COMPLETE_GATE_FAILED
```

Feature-PGD r3 不是“全面无效”：它稳定改善 direction Brier，在 5%/10%/20% 人工标签翻转下也持续改善，并降低 composite error、top-k turnover；短 OOS 上四类压力的聚合 composite error 均改善。但它同时显著牺牲 clean return，validation 上还牺牲 volatility、NDCG 和 RankIC，并在 extreme-volatility 与跨窗口一致性上失败。因此 r3 不得晋级，也不得与其他方法组合掩盖失败。

当前结果不能简单归因于“数据太小”。短 OOS 每折只有 39 个日期，确实使置信区间较宽；但 short fold 2/3 和 validation fold 1/2 的 return MAE 退化置信区间完全位于零以上，说明至少存在真实的多任务权衡。数据量不足是功效限制，不是全部原因。

## 2. 输入与完整性

输入目录：`/Users/Zhuanz/Downloads/output`

| Artifact | SHA-256 | 验收 |
|---|---|---|
| `phase2b_feature_pgd_v2_validation_results.tar.gz` | `b20e30e9be73d8654db7996c78fb4186ec078d111dcf11688293e40e21446b8d` | 外层 hash 匹配；内部 48/48；9/9 checkpoints |
| `phase2b_feature_pgd_v2_short_oos_results.tar.gz` | `f3b3d959a008912af3c4f8445c4be94cb699d2b721c1bfbac701cfe55f84fe57` | 外层 hash 匹配；内部 75/75；9/9 checkpoints |
| validation report | `368fd0b4c58ff4c1a9c8249ffa7c1a94529bef419d286f39d257e863e44f7d5a` | canonical self-hash 通过 |
| validation contract | `66686e0013cbb180a29b6493ba51952c8f413a0431630193ad6f6f12240c8659` | canonical self-hash 通过 |
| short-OOS report | `3f170b5f2ca23c4e7844ec594a141e7aa9dfe9cf7953dbb6e67eeda8ce3d267a` | canonical self-hash 通过 |
| short-OOS contract | `de8a04f52bfd6a542769c6c9bf4c915bd3f6accf48f0742f26f76367e07cea8d` | canonical self-hash 通过 |
| dataset | `4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554` | preflight、contract、report 一致 |

训练环境为 Windows 11、Python 3.12.10、PyTorch 2.7.1+cu128、CUDA 12.8、RTX 2080 Ti 11 GB；两次 preflight 均为 `CUDA_PREFLIGHT_PASS`。

`RESULT_VALIDATION.status=PASS` 只证明目录、checkpoint 和规定产物齐全，不表示 `research_gate_passed`。两个概念必须继续分开。

### 2.1 回传包的工程瑕疵

- short sidecar 被命名为 `*.tar.g.sha256`，少一个 `z`；validation sidecar 被命名为 `*.tar..sha256`，少 `gz`。内容正确，但按 `*.tar.gz.sha256` 自动发现会失败。
- 包内 `MANIFEST.sha256` 使用 CRLF。去掉行尾 `\r` 后全部条目通过，但 macOS 直接执行 `shasum -c` 会把 `\r` 解释为文件名的一部分。
- 这些问题不改变模型结论，但 r4 打包器必须强制 LF、自动生成准确 sidecar，并由回传校验器检查 manifest、contract/report self-hash 和路径安全。

## 3. 冻结实验合同

| 项目 | Validation | Short OOS |
|---|---:|---:|
| 用途 | 开发选型，不允许晋级 | 新短窗口研究证据 |
| 窗口 | `3 × 126` | `3 × 39` |
| 评估区间 | 2023-11-30 起至 2025-06-25 | 2026-01-15 至 2026-07-13 |
| seed | `20260805` | `20260805` |
| epochs | 50 | 50 |
| PGD | `epsilon=0.01, beta=0.5, steps=3` | 同左 |
| preprocessing | train-fold-only standardizer | 同左 |
| variants | none / structured_missing / feature_pgd | 同左 |

模型为 3 层、`d_model=64`、4 heads、FFN 128；训练使用固定多任务权重和 legacy ranking。PGD 在 19 个标准化连续特征的 z-space 中运行，4 个状态布尔字段和 padding 不扰动。

## 4. 结果分解

以下 delta 均为 `feature_pgd - none`；误差、NDCG/RankIC、coverage/turnover 的方向含义不同，不能只看 composite。

### 4.1 Clean 聚合

| 指标 | Validation：none → PGD | Delta | Short OOS：none → PGD | Delta |
|---|---:|---:|---:|---:|
| composite error ↓ | 0.126819 → 0.115848 | -0.010971 | 0.127014 → 0.124425 | -0.002588 |
| return MAE ↓ | 0.066890 → 0.068140 | **+0.001250** | 0.070529 → 0.075401 | **+0.004872** |
| direction Brier ↓ | 0.303129 → 0.268736 | **-0.034392** | 0.298662 → 0.286996 | **-0.011666** |
| volatility MAE ↓ | 0.010439 → 0.010668 | +0.000229 | 0.011850 → 0.010879 | -0.000971 |
| NDCG@20 ↑ | 0.484737 → 0.472647 | -0.012091 | 0.477097 → 0.472856 | -0.004241 |
| RankIC ↑ | 0.028496 → 0.010357 | -0.018139 | 0.040070 → 0.047384 | +0.007314 |
| interval coverage ↑ | 0.589123 → 0.703995 | +0.114871 | 0.498911 → 0.544101 | +0.045190 |
| interval width ↓ | 0.124541 → 0.155210 | +0.030669 | 0.100202 → 0.118777 | +0.018575 |
| top-k turnover ↓ | 0.273067 → 0.232533 | -0.040533 | 0.262281 → 0.207456 | -0.054825 |

coverage 上升伴随区间明显变宽，不能单独解释为分位数头质量提升。Validation 的 composite 改善主要由 direction Brier 拉动，不能抵消 roadmap 明确要求的 clean return/volatility 不退化。

### 4.2 Clean 逐折 delta

| 数据 | Fold | Composite | Return MAE | Brier | Vol MAE | NDCG@20 | RankIC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 1 | -0.005437 | **+0.009011** | -0.024968 | -0.000355 | -0.026838 | -0.040111 |
| Validation | 2 | -0.007952 | **+0.002939** | -0.026879 | +0.000085 | +0.002050 | -0.029600 |
| Validation | 3 | -0.019525 | -0.008201 | -0.051330 | +0.000957 | -0.011484 | +0.015294 |
| Short OOS | 1 | -0.001694 | -0.002170 | -0.003054 | +0.000142 | -0.023199 | -0.003238 |
| Short OOS | 2 | +0.000775 | **+0.010600** | -0.006763 | -0.001511 | +0.027422 | +0.029723 |
| Short OOS | 3 | -0.006846 | **+0.006185** | -0.025180 | -0.001544 | -0.016946 | -0.004541 |

paired circular moving-block bootstrap 给出的关键 95% CI：

- validation return fold 1：`+0.00909 [ +0.00567, +0.01252 ]`；fold 2：`+0.00296 [ +0.00024, +0.00586 ]`；
- short return fold 2：`+0.01060 [ +0.00649, +0.01463 ]`；fold 3：`+0.00619 [ +0.00084, +0.01112 ]`；
- validation direction Brier 三折的 CI 均完全小于零，说明 direction 改善较稳定；
- short 每折只有 39 个日期，其余多项 CI 较宽，只能形成低功效 research-only 证据。

### 4.3 Noisy direction

| 人工翻转率 | Validation Brier delta | Short OOS Brier delta |
|---:|---:|---:|
| 5% | -0.034831 | -0.012175 |
| 10% | -0.035644 | -0.011376 |
| 20% | -0.037380 | -0.014125 |

方向抗噪改善是一致的，但当前实验不能把改善完全归因于 PGD，因为 `feature_pgd` 同时启用了 structured missing。

### 4.4 Stress composite

| Stress | Validation delta | Short OOS delta | 判定 |
|---|---:|---:|---|
| price | -0.012404 | -0.006215 | 聚合改善；short fold 2 不一致 |
| volume | -0.010672 | -0.003087 | 聚合改善；short fold 2 不一致 |
| missing | -0.011817 | -0.002934 | 聚合改善；short fold 2 不一致 |
| extreme volatility | **+0.003639** | -0.010652 | validation fold 1/2 明显恶化 |

Validation extreme-volatility 相对 clean 的 fold 1/2 bootstrap 估计分别为 `+0.01742` 和 `+0.02588`，CI 全部大于零。因此它不是可以通过修改显示精度或 gate 容差消除的小误差。

## 5. 失败机制

### 5.1 已证实：PGD 与 structured missing 被错误合并

runner 对 `structured_missing` 和 `feature_pgd` 都注入 `missing_mode=continuous_center_impute, missing_rate=0.1`；训练循环先做 missing augmentation，再计算所谓 clean loss 和 PGD。当前对比实际是：

```text
none
vs structured_missing_only
vs structured_missing + feature_pgd
```

而不是单变量的 `none vs pure feature_pgd`。这违反 roadmap 的单变量消融要求，也使 direction 改善、return 退化和 stress 改善都无法独立归因。r4 必须先拆成纯 PGD；组合候选只有在两个单项分别通过后才能另立 hypothesis。

### 5.2 已证实：收益/方向/波动/排序发生任务权衡

PGD 内层使用完整五任务损失寻找同一个扰动，外层使用 `clean_loss + beta × adversarial_loss`。现有 diagnostics 只记录总 clean/adversarial loss，没有记录每个 head 的损失、输入梯度 norm、梯度 cosine 或扰动后增量，所以不能根据结果臆测某一个 head 的梯度必然主导。

能够确认的是：方向头在六个窗口整体改善，收益头在四个窗口显著退化，validation ranking 也退化。下一轮必须先补 per-head attack diagnostics，再决定是否需要 train-fold task-scale normalization 或 head-selective adversarial objective。

### 5.3 已证实：只保存第 50 轮，没有 best-validation checkpoint

当前训练先完成全部 50 epochs，随后只评估一次 validation/test，并直接保存最终模型。六个 PGD 运行的 train loss 都持续下降，但这不能证明第 50 轮在 clean-return、ranking 或 stress 上最优，也无法排除 fold 2/3 的泛化退化来自 checkpoint 选择。

r4 必须对 baseline 和所有 candidate 使用完全相同的、预注册的逐 epoch validation 与选择规则，同时保存 `best` 和 `last`；不能只给 PGD early stopping 而保持 baseline 用最后一轮。

### 5.4 已证实：现有 gate 没有执行“显著改善”

当前 gate 对 clean、stress 和三折一致性只用 `<= 1e-12` 的点估计。报告虽然计算了 bootstrap CI，但 gate 不读取 CI；relative stress degradation 也只报告、不参与判定。这既会让 short fold 2 的 `+0.000252` volume 小差异直接失败，也没有兑现 roadmap 对压力退化“显著减少”的要求。

不能据此追溯修改 r3 结论。正确做法是为 r4 预注册 non-inferiority margin、block bootstrap、窗口一致性和 relative-stress gate，再训练新 hypothesis。

### 5.5 高可信假设：攻击预算与评估压力的几何不匹配

- PGD 在 z-space 使用统一 `L∞ epsilon=0.01`，每个记录到的 batch 都触达约 `0.010000` 的边界；
- adversarial loss 相对 clean loss 的平均增量仅约 2.18%–2.74%；
- price stress 在 raw space 对一组派生特征加入 `sigma=0.02`，volume 为 `sigma=0.25`，extreme volatility 直接乘 2；按 fold-1 scaler 换算，部分 price shock 约为 `0.18–1.54 z`，与 `0.01 z` 不在同一量级。

这支持“训练攻击与验收 stress 不对齐”的假设，但不能用已观察 short OOS 直接反推更大的 epsilon。预算校准只能使用 train/validation 数据，并作为新合同冻结。

### 5.6 已证实：金融语义约束不足

当前 PGD 只冻结四个布尔状态和 padding。它没有保证 `intraday_range/volatility >= 0`、`price_position/cross_section_rank ∈ [0,1]`、`breakout_20 <= 0`，也没有维护多周期收益、波动和价格派生特征之间的关系，尚未完整落实 roadmap 的“不得产生违反行情关系的伪特征”。

r4 至少要加入 feature-domain projection，并冻结无法独立扰动而保持语义的 cross-section rank。更严格版本应在受控 primitive shock 后重算派生特征，而不是直接对全部派生列加独立噪声。

### 5.7 数据限制

当前 short OOS 最后 timestamp 为 `1783926000000000000`，即 2026-07-13 15:00 Asia/Shanghai；截至该时点的 117 个日期已经被观察，永久失去 untouched 身份。

- 新的 `3 × 39` short OOS 需要该 cutoff 之后至少 117 个新交易 timestamp；
- 新的 `3 × 126` 完整 OOS 需要该 cutoff 之后至少 378 个新交易 timestamp；
- 在新数据形成前，r4 最多只能完成 development validation，不能宣布 Phase 2B 退出。

## 6. r4 不可违反的约束

1. 不得根据本次 short OOS 调整 epsilon、beta、steps、stress scale、fold、bootstrap block、gate margin 或 composite 权重后，再把相同 117 个日期称为新 OOS。
2. 不得因为 structured-missing 在 short 聚合上较好就直接选它为 champion；它也没有通过三窗口一致性。
3. 不得事后放宽 r3 gate；r3 永久保持 `TRAINING_COMPLETE_GATE_FAILED`。
4. 不得 cherry-pick seed、epoch 或单个窗口；失败 seed/checkpoint 必须保留。
5. 首轮不得把 pure PGD 与 missing、PCGrad、GradNorm、APL、InfoTS 或新 ranking loss 组合。
6. `RESULT_VALIDATION PASS`、validation winner 和正式 OOS gate 必须继续是三个独立状态。

## 7. r4 分阶段施工

### P0：登记与回传可靠性（CPU）

- 在状态文档登记本报告的 archive/report/contract/dataset hash 和 r3 最终状态。
- 修复 sidecar 命名和 manifest LF；打包器同时生成 `<archive>.sha256`。
- 扩展结果校验器：校验全部 manifest 条目、canonical self-hash、checkpoint 三方 hash、validation 禁止 test artifact、archive 路径安全。
- contract 增加 effective sampler、每个 timestamp 的实际 batch 语义、shuffle、optimizer/weight decay、checkpoint-selection spec。

退出：Windows 生成的 fixture 可在 macOS/Linux 原生命令下 100% 校验，错误 hash/CRLF/缺件测试均 fail closed。

### P1：恢复单变量实验（CPU 代码 + smoke）

- `feature_pgd_pure` 强制 `missing_mode=none, missing_rate=0`。
- `structured_missing_only` 保持 10%，作为独立 attribution control。
- 暂不开放 `feature_pgd_plus_missing`；组合仅在两个单项各自通过后另立 hypothesis。
- baseline 与全部 candidates 每 epoch 在 fold validation 上评价；保存 `checkpoint_best.pt` 和 `checkpoint_last.pt`。
- 预注册 best 选择为：先满足 clean guards，再最小化最坏 relative-stress degradation，最后依次用 clean composite、较早 epoch 打破平局。
- 每个采样 batch 记录各 head 的 clean/adversarial loss、输入梯度 norm、head-pair cosine、每特征扰动利用率和边界命中率。

退出：测试证明 pure PGD 不发生 missing 更新；同一 checkpoint 选择器对 baseline/candidate 完全一致；last 不会静默覆盖 best。

### P2：语义投影与正式 gate（CPU）

- 增加 per-feature domain contract：非负、`[0,1]`、非正以及 protected-derived 字段分别投影或冻结。
- 增加 projection 前后 L∞、有效扰动率和约束命中 diagnostics。
- clean 使用预注册 one-sided non-inferiority CI；stress 使用 `candidate(stress-clean) - baseline(stress-clean)` 的 relative degradation。
- bootstrap 以 fold 为边界做 paired block resampling，再等权聚合 fold；primary block length、draws、seed 在训练前冻结，其他 block length 只做全量 sensitivity，不能择优报告。
- 三窗口一致性按 seed-averaged fold 点估计判断；显著性按 pooled one-sided CI 判断，不能把“每折都显著”与“每折方向一致”混为一项。
- 增加自然 near-zero-return slice 和 train-fold 冻结的 volatility-regime 分层报告；分层只作诊断，不替代主 gate。

建议的开发层 non-inferiority 默认值如下；它们只能用于 r4 新合同，不能追溯应用于 r3，并需在启动 GPU 训练前最终冻结：

| Guard | 建议 margin |
|---|---:|
| return MAE | 相对 baseline 不超过 +1% |
| direction Brier | 相对 baseline 不超过 +1% |
| volatility MAE | 相对 baseline 不超过 +2% |
| NDCG@20 | 绝对 delta 不低于 -0.005 |
| RankIC | 绝对 delta 不低于 -0.005 |

正式 stress gate 仍要求四类 relative degradation 的三折点估计方向一致，且等权 pooled one-sided 95% CI 上界小于 0。若业务不接受上述 clean margin，应在训练前改为更严格值，而不是看到结果后调整。

### P3：Validation-only 小网格（GPU，单 seed）

固定 model、labels、optimizer、learning rate `3e-4`、50 epochs、PGD steps=3、fold 和 stress。只改变 pure-PGD 的 epsilon/beta：

| Candidate | epsilon | beta | missing |
|---|---:|---:|---|
| `none-r4` | — | — | none |
| `missing-only-r4` | — | — | center-impute 10% |
| `pgd-e005-b010-r4` | 0.005 | 0.10 | none |
| `pgd-e005-b025-r4` | 0.005 | 0.25 | none |
| `pgd-e010-b010-r4` | 0.010 | 0.10 | none |
| `pgd-e010-b025-r4` | 0.010 | 0.25 | none |
| `pgd-e020-b010-r4` | 0.020 | 0.10 | none |
| `pgd-e020-b025-r4` | 0.020 | 0.25 | none |
| `pgd-e010-b050-control-r4` | 0.010 | 0.50 | none |

首轮使用 seed `20260805` 和三个 validation folds。候选必须先通过全部 clean guards；可行候选按“最坏 relative-stress delta → clean composite → 较早 best epoch”的固定顺序排名，只保留前两名。若没有候选通过 clean guards，本轮直接停止，不为了产生 winner 而放宽门槛。

steps 已经能够把现有预算打满，因此首轮不增加 steps。epsilon/beta warm-up、随机起点、task-normalized attack 都属于后续独立 hypothesis，不能同时加入本轮网格。

### P4：Top-2 多 seed 复验（GPU）

- 对 baseline 和 Top-2 使用 seeds `{20260805, 20260806, 20260807}`，每 seed 三折；不能挑最好 seed。
- 每折先等权聚合三个 seed，再检查三折方向；同时报告 seed dispersion、最差 seed 和失败率。
- winner 必须在三 seed 聚合后仍通过 clean、relative-stress、窗口一致性、ranking guard；否则 r4 validation 记为无 winner。
- 所有训练共享同一数据、fold、epoch 上限、best-checkpoint selector 和实现 hash。

### P5：失败后的单变量候选（条件触发）

只有 P3/P4 仍显示 return/ranking 与 direction 的稳定冲突时，才按以下顺序一次测试一个新机制：

1. `task-scale-normalized adversarial objective`：normalizer 仅由 train fold clean task loss/gradient 估计并冻结；外层 clean objective 和 frozen scalar weights不变；
2. `epsilon/beta warm-up`：固定 target budget，预注册 warm-up epochs；不同时加入 task normalization；
3. `primitive-shock PGD`：在可保持金融语义的 primitive 上生成扰动并重算派生特征。

若 pure PGD 在多 seed 下仍只改善 direction、持续损害 return，或者 extreme-volatility 在相同 fold 持续失败，应淘汰该 hypothesis family，而不是继续扩大网格。

### P6：新的 untouched OOS（等待数据）

- 在 validation winner、seed、checkpoint selector、stress、margin、bootstrap 和 trial count 全部冻结后，才建立新 cutoff。
- 当前 2026-07-13 之前的数据不得再次作为 untouched OOS。
- 数据不足时明确输出 `WAITING_FOR_NEW_OOS_DATA`；不得用 validation 或旧 short OOS替代。
- 新 OOS 只运行 frozen winner 与 matched baseline，不再跑网格。

### P7：winner-only 下游验证（CPU/C++）

只有新 OOS 预测 gate 通过后才执行：

- ONNX parity、六输出、validation/test embedding snapshot 和 drift reference；
- 同一 policy/risk/cost 假设下的 C++ Replay/CVaR research proxy；
- Return Analysis 与 paired benchmark；
- 继续保持 `promotion_eligible=false`，直到独立经济 gate 和生产数据条件满足。

## 8. r4 状态机

| 状态 | 条件 | 允许的下一步 |
|---|---|---|
| `R3_GATE_FAILED` | 本报告登记完成 | 修工程与 validation 方案 |
| `R4_VALIDATION_NO_WINNER` | 无候选通过 clean/stress guards | 停止或新建单变量 hypothesis |
| `R4_VALIDATION_WINNER_FROZEN` | Top-2 多 seed 后唯一 winner | 等待新 OOS；不得晋级 |
| `WAITING_FOR_NEW_OOS_DATA` | cutoff 后日期不足 | 继续其他独立 phase 工作 |
| `R4_OOS_GATE_FAILED` | 新 OOS 任一硬 gate 失败 | 淘汰；不组合、不 fallback |
| `R4_RESEARCH_GATE_PASSED` | clean、relative stress、三窗、多 seed 全通过 | winner-only C++/经济验证 |

Phase 2B 只有在新的 untouched OOS 和要求的下游研究报告通过后，才能重新判断 phase exit。当前结论仍为：

```text
phase_exit_eligible = false
engineering_scaffold_complete = true
blocking_reason = r3_clean_return_and_window_gates_failed
```

## 9. r4 必须回传的产物

- `preregistered_contract.json`：包含 trial set、seed set、fold、margin、bootstrap、checkpoint selector、domain projection、effective sampler 和实现 hash；
- 每 fold/seed/variant 的 `checkpoint_best.pt`、`checkpoint_last.pt`、`metrics.json`；
- per-head attack gradient/loss 与 per-feature perturbation diagnostics；
- validation-only 六输出、embedding snapshot 与 manifest；
- `phase2b_oos_report.json`：同时报告 absolute stress、relative stress、one-sided CI、block sensitivity、seed dispersion 和三折方向；
- `RESULT_VALIDATION.json`、LF `MANIFEST.sha256`、准确命名的 `<archive>.sha256`；
- CUDA preflight 和依赖锁定信息。

## 10. 推荐施工顺序

立即执行 P0–P2 的 CPU 工程修复与测试，然后生成 r4 validation-only GPU 包。GPU 回传后只做 P3/P4 选型；在 2026-07-13 之后积累足够的新数据前，不再运行或重解释正式 OOS。

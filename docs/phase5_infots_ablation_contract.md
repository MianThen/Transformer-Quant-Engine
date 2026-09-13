# Phase 5 三组监督预算与 OOS 合同

## 固定对照

每个 purged fold 必须同时登记且使用相同 `supervised_budget`：

```text
from_scratch
fixed_augmentation
infots
```

默认要求至少三个唯一 fold（当前合同为 `1,2,3`），不得把不同预算或不同标签合同混入同一报告。

## Fold artifact

`validate_infots_fold_artifact` 要求每个 group/fold artifact 同时具备：

- `test_blind=true`、固定 `gate_spec_sha256` 和严格递增 `train_end < validation_end < test_start <= test_end`；
- `test_start > validation_end + purge_gap`，禁止验证集尾部泄漏到 test；
- prediction artifact 的固定六输出：`expected_return`、`expected_volatility`、
  `direction_probability`、`lower_quantile`、`upper_quantile`、`confidence`；
- validation/test 两个 embedding snapshot，各自带相对路径、行数、维度和 SHA-256；
- composite error、return MAE、direction Brier、volatility MAE、NDCG@20、RankIC 六项有限指标；
- clean/stress/three-window、quality/stability 五个布尔 gate。

## Report semantics

`build_infots_ablation_report` 要求完整的 `3 × fold_count` 输入，输出各组逐 fold 与均值指标、相对
`from_scratch` 的 delta、purged/test-blind 诊断和 gate 汇总。它永远写入：

```text
winner_selected = false
research_gate_passed = false
phase_exit_eligible = false
promotion_eligible = false
```

这一步只负责可审计汇总，不替训练结果选择 winner，也不把 synthetic/reference 输入冒充正式 OOS。只有
外部 GPU 回传、预注册 gate 证据和独立人工登记后，才可更新 Phase 5 判定。

## 聚合命令

```text
python tools/run_phase5_infots_ablation.py \
  --contract configs/ml/phase5_infots_ablation_contract.json \
  --artifact runs/phase5-infots/fold-1/from_scratch/artifact.json \
  --artifact runs/phase5-infots/fold-1/fixed_augmentation/artifact.json \
  --artifact runs/phase5-infots/fold-1/infots/artifact.json \
  --output runs/phase5-infots/phase5_infots_ablation_report.json \
  --evidence-level FORMAL_OOS
```

实际使用时必须提供完整的 9 个 group/fold artifact；缺失、重复、预算不一致或未来泄漏都会非零退出。

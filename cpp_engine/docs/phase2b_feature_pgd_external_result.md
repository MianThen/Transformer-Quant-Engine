# Phase 2B Feature-PGD 外部训练结果登记

更新时间：2026-08-05

## 输入与完整性

- 输入包：`/Users/Zhuanz/Downloads/output.zip`
- archive SHA-256：`0aed4f7638f014adff72084ce0b78a8b0e15012de3be1e394f6ad1a407b14975`
- 包内 `feature_pgd`：fold 1--3，每折 50 epochs。
- 每折包含 checkpoint、ONNX export、metrics、validation/test predictions、validation/test embedding snapshots 和 manifests。
- feature schema：`BAR_V1`、`NTF`、`float32`、23 个特征；`epsilon=0.01`、`beta=0.5`、`pgd_steps=3`。

## Provenance

| Artifact | SHA-256 |
|---|---|
| `phase2b_oos_report.json` | `0adf2a7788bd46ad685a62025b2136fb7fde49b55ff9d5d64bb268cb549c8153` |
| `preregistered_contract.json` | `22e87cbaa0e40284df5d51f2bea6bd179e8eea7a38ada3994f04aff3c823fd7a` |
| dataset | `4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554` |

完整 contract hash 为 `22e87cbaa0e40284df5d51f2bea6bd179e8eea7a38ada3994f04aff3c823fd7a`。

## 三折结果

| Fold | Validation loss | Test loss | Validation NDCG@20 | Test NDCG@20 | Test RankIC |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.175405 | 0.180099 | 0.479951 | 0.518875 | 0.077719 |
| 2 | 0.179669 | 0.181040 | 0.502188 | 0.509891 | 0.058100 |
| 3 | 0.186014 | 0.179128 | 0.517785 | 0.480581 | 0.040181 |

汇总 clean 指标：`composite_error=0.0956173`、`return_mae=0.0493726`、`direction_brier=0.2274944`、`volatility_mae=0.0099849`、`NDCG@20=0.5032469`、`RankIC=0.0586506`。

## Gate 判定

```json
{
  "status": "research_only_no_promotion",
  "clean_non_degraded": false,
  "stress_non_degraded": false,
  "all_three_window_consistent": false,
  "research_gate_passed": false,
  "promotion_eligible": false,
  "largest_observed_stress_delta": {
    "mode": "missing",
    "composite_error_delta": 0.07069240758816402
  }
}
```

结论：外部训练已完成，解决了缺少 PGD checkpoint 的工程阻塞；但 Feature-PGD 未通过 clean/stress/窗口一致性 gate，不能晋级、不能替换 frozen champion，也不能解除 Phase 2B 退出条件。该结果不阻塞后续 Phase 3A/3B。

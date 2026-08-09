# Phase 3C 退出报告（当前版本）

## Gate

| 门槛 | 状态 | 证据 |
|---|---|---|
| 加权 Platt/PAV Isotonic 工程实现与单测 | PASS | `python/qbt_ml/calibration/probability.py`、定向测试 |
| 固定/rolling CQR 数学实现与 future guard | PASS | `python/qbt_ml/calibration/conformal.py`、定向测试 |
| 三窗口 validation-fit/test-apply 防泄漏 | PASS | `runs/phase3c-calibration-real/report.json` |
| 概率校准各 OOS regime 达标 | NOT MET | 无预注册数值 gate；Isotonic OOS 劣于未校准，fold-1 Platt fail-closed |
| CQR coverage 各 OOS regime 达标 | NOT MET | coverage `0.7840 / 0.8998 / 0.8397`，fold-1 低于 `0.80`；无预注册容差 |
| Attention analysis API 与生产六输出 parity | PASS | `python/tests/test_phase3c_attention.py`：`8 passed` |
| 真实冻结模型 Attention artifact | PASS/DIAGNOSTIC | `runs/phase3c-attention-real/report.json`，SHA-256 `e57fdb...702c44` |
| IG completeness 与 feature-group occlusion | PASS/DIAGNOSTIC | 三折、每折两个样本，512 步梯形积分；非因果解释 |
| Attention occlusion 随机对照 | PASS/DIAGNOSTIC | 三折 artifact 含 top-vs-disjoint-random 对照 |
| seed/window/regime stability | NOT MET | 当前仅三折确定性小样本，未形成预注册全 OOS gate |

## 判定

```json
{
  "engineering_scaffold_complete": true,
  "phase_exit_eligible": false,
  "promotion_eligible": false,
  "evidence_level": "RESEARCH_PROXY",
  "blocking_reason": "calibration_oos_gate_unregistered_and_unmet_attention_evidence_scope"
}
```

本报告不把工程实现、诊断 correlation 或小样本 IG 结果解释为模型晋级，也不把失败的概率校准回退为
另一个运行时映射。后续 Phase 4A 只能使用已冻结且可用的 prior/均值 view；校准失败的 head 保持
`UNAVAILABLE_FAIL_CLOSED`。

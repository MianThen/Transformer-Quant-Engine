# Phase 2B 退出报告（2026-08-05）

## 状态

| 门槛 | 结果 | 说明 |
|---|---|---|
| direction noise audit 可重放 | PASS | 固定 seed、翻转计数和 BCE/Brier（若提供概率）可重算 |
| APL direction-only 约束 | PASS | APL 不改变四个连续/分位/排序任务 |
| latent FGM 公式与训练隔离 | PASS | shared latent、L2 归一化、训练专用分支 |
| feature PGD mask/epsilon 约束 | PASS | padding 与状态布尔字段不可扰动，L∞ 投影有界 |
| price/volume/missing/extreme-volatility stress fixture | PASS | NPZ + manifest hash 已生成器化 |
| frozen champion 三窗口 clean OOS 参照 | PASS | fold 1--3 已完成；平均 return MAE 0.0649495、direction Brier 0.2273957、volatility MAE 0.0099349 |
| APL/latent-FGM/feature-PGD 三折配对 OOS 报告 | PASS | 外部 `output.zip`，clean/noisy/stress 均覆盖 3 个 purged 窗口 |
| clean OOS 不退化（候选 vs baseline） | FAIL | `feature_pgd.clean_non_degraded=false` |
| 压力集稳定改善 | FAIL | `feature_pgd.stress_non_degraded=false`，missing delta `+0.0706924` |
| 至少三个 walk-forward 窗口方向一致 | FAIL | `feature_pgd.all_three_window_consistent=false` |
| 失败候选关闭且不自动 fallback | PASS | 单候选配置、显式 hypothesis id、无运行时组合 |
| 生产/ONNX 结构与无对抗分支一致 | PASS | 对抗参数不进入模型导出签名 |

## 研究判定

```json
{
  "phase_exit_eligible": false,
  "promotion_eligible": false,
  "engineering_scaffold_complete": true,
  "evidence_level": "RESEARCH_ONLY",
  "blocking_reason": "candidate_clean_stress_and_window_gates_failed"
}
```

外部三窗口报告已覆盖 clean、5/10/20% 人工 direction 翻转和 price/volume/missing/extreme-volatility 四类派生压力集；它证明训练和评估链路完成，但候选 gates 失败，因此不能证明 APL、FGM 或 PGD 对真实 OOS 有益，也不能按 test 结果回调 epsilon。

`direction` soft label 来自 Phase 1E `label_v2`；noisy 标签在评估器内由固定 seed 的 flip mask 派生，数据集不需要新增 `noisy` 列。完整 provenance 记录在 `preregistered_contract.json`。

## 复现记录

测试命令：

```text
PYTHONPYCACHEPREFIX=/tmp/qbt-pyc .venv/bin/pytest -q -p no:cacheprovider
```

结果：`176 passed, 8 skipped`。当前没有生成正式候选排名或 C++ 经济晋级 artifact，因此 Phase 2A 的 `RESEARCH_PROXY` 结果不会被 Phase 2B 自动覆盖；已生成的 baseline 三窗口汇总仍为 research-only。

本次外部结果已完成 APL/latent-FGM/feature-PGD 三折 checkpoint 和完整配对 OOS；feature-PGD 训练完成但 gates 失败，`phase_exit_eligible` 与 `promotion_eligible` 继续保持 `false`。

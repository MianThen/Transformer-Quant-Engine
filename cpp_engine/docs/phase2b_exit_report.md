# Phase 2B 退出报告（当前版本）

## 状态

| 门槛 | 结果 | 说明 |
|---|---|---|
| direction noise audit 可重放 | PASS | 固定 seed、翻转计数和 BCE/Brier（若提供概率）可重算 |
| APL direction-only 约束 | PASS | APL 不改变四个连续/分位/排序任务 |
| latent FGM 公式与训练隔离 | PASS | shared latent、L2 归一化、训练专用分支 |
| feature PGD mask/epsilon 约束 | PASS | padding 与状态布尔字段不可扰动，L∞ 投影有界 |
| price/volume/missing/extreme-volatility stress fixture | PASS | NPZ + manifest hash 已生成器化 |
| frozen champion 三窗口 clean OOS 参照 | PASS | fold 1--3 已完成；平均 return MAE 0.0649495、direction Brier 0.2273957、volatility MAE 0.0099349 |
| clean OOS 不退化（候选 vs baseline） | NOT RUN | 尚未完成 APL/latent FGM/feature PGD 的配对候选训练 |
| 压力集稳定改善 | NOT RUN | 尚无候选与 frozen champion 的配对 OOS |
| 至少三个 walk-forward 窗口方向一致 | NOT RUN | 尚无冻结窗口结果 |
| 失败候选关闭且不自动 fallback | PASS | 单候选配置、显式 hypothesis id、无运行时组合 |
| 生产/ONNX 结构与无对抗分支一致 | PASS | 对抗参数不进入模型导出签名 |

## 研究判定

```json
{
  "phase_exit_eligible": false,
  "promotion_eligible": false,
  "engineering_scaffold_complete": true,
  "evidence_level": "RESEARCH_ONLY",
  "blocking_reason": "missing_three_window_clean_noisy_stress_oos"
}
```

baseline 三窗口报告已覆盖 clean、5/10/20% 人工 direction 翻转和 price/volume/missing/extreme-volatility 四类派生压力集，但它只描述 frozen champion，不能证明 APL、FGM 或 PGD 对真实 OOS 有益。尤其不能用人工标签翻转改善替代自然近零收益子集和真实压力集的改善，也不能按 test 结果回调 epsilon。

`direction` soft label 来自 Phase 1E `label_v2`；noisy 标签在评估器内由固定 seed 的 flip mask 派生，数据集不需要新增 `noisy` 列。完整 provenance 记录在 `preregistered_contract.json`。

## 复现记录

测试命令：

```text
PYTHONPYCACHEPREFIX=/tmp/qbt-pyc .venv/bin/pytest -q -p no:cacheprovider
```

结果：`176 passed, 8 skipped`。当前没有生成正式候选排名或 C++ 经济晋级 artifact，因此 Phase 2A 的 `RESEARCH_PROXY` 结果不会被 Phase 2B 自动覆盖；已生成的 baseline 三窗口汇总仍为 research-only。

本次推进已完成 APL/latent-FGM 三折 checkpoint，但 feature-PGD 正式训练受单机资源限制未完成；这些 checkpoint 不能单独构成候选 OOS 结论，`phase_exit_eligible` 继续保持 `false`。

Feature-PGD 重试仅完成首 epoch 基准，未产生可用于晋级的 checkpoint；输入梯度优化已通过 Phase 2B smoke 测试，但不改变真实 OOS 缺口。

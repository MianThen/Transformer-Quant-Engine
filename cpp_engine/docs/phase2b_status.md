# Phase 2B 状态：鲁棒训练

更新时间：2026-08-05

## 当前结论

Phase 2B 的 APL、latent-FGM、feature-PGD 三折候选训练和配对 OOS 报告已经完成，但研究退出条件仍未满足：

```text
phase_exit_eligible = false
promotion_eligible = false
```

原因不是训练结果缺失，而是候选没有同时通过 clean 不退化、压力集稳定改善和三个窗口方向一致门槛。所有候选仍是 research-only。

本次外部算力结果包：

- `/Users/Zhuanz/Downloads/output.zip`
- archive SHA-256：`0aed4f7638f014adff72084ce0b78a8b0e15012de3be1e394f6ad1a407b14975`
- 三窗口报告 SHA-256：`0adf2a7788bd46ad685a62025b2136fb7fde49b55ff9d5d64bb268cb549c8153`
- preregistered contract SHA-256：`22e87cbaa0e40284df5d51f2bea6bd179e8eea7a38ada3994f04aff3c823fd7a`
- dataset SHA-256：`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`

已生成 frozen champion 的三窗口 baseline OOS 参照报告，但这不等于 Phase 2B 退出：

- 报告：`/Users/Zhuanz/PycharmProjects/PythonProject/runs/phase2b-oos-baseline-real/phase2b_oos_report.json`
- 合同：`/Users/Zhuanz/PycharmProjects/PythonProject/runs/phase2b-oos-baseline-real/preregistered_contract.json`
- clean/noisy/stress 均覆盖 fold 1--3；报告状态为 `research_only_no_promotion`。

## 已交付

- `direction_noise_audit`：固定 seed 的人工 direction label flip 审计，支持 0/5/10/20% 翻转率。
- `direction` 是数据集原始 soft label（`[0,1]`）；`noisy_direction` 只在评估时按固定 seed 派生，不是爬虫原始列。
- `direction_apl`：只替换 direction head 的 NCE+RCE normalized APL；return/volatility 仍为 Huber，quantile 仍为 Pinball，ranking 仍绑定已冻结实现。
- `latent_fgm`：在 shared latent 上使用 `r_adv = epsilon * g / (||g||_2 + eps)`，训练时计算 clean + beta·adversarial loss。
- `feature_pgd`：标准化连续特征上的 mask-aware L∞ PGD；padding 不变，`is_suspended/is_listed/is_st/is_tradable` 不变。
- `build_stress_sets`：price、volume、missing、extreme-volatility 四类确定性压力集。
- `phase2b-audit`：从 NPZ 数据集写出 noise audit、压力集和带 hash 的 `stress_manifest.json`。
- 训练 checkpoint/metrics 写入 robust mode、hypothesis id、spec 和 batch diagnostics。

实现位置：

- `/Users/Zhuanz/PycharmProjects/PythonProject/python/qbt_ml/training/robust_training.py`
- `/Users/Zhuanz/PycharmProjects/PythonProject/python/qbt_ml/models/temporal_transformer.py`
- `/Users/Zhuanz/PycharmProjects/PythonProject/python/qbt_ml/cli.py`
- `/Users/Zhuanz/PycharmProjects/PythonProject/python/tests/test_phase2b_robust_training.py`

## 使用约束

默认配置仍为 `robust_training.mode = none`。候选必须单独注册 `hypothesis_id`，不能与 Phase 1E PCGrad/GradNorm 组合；`production_eval=true` 会被拒绝。对抗分支只存在于训练循环，不进入评估、导出或 ONNX 图。

生成压力集示例：

```text
/Users/Zhuanz/PycharmProjects/PythonProject/.venv/bin/python -m python.qbt_ml.cli phase2b-audit \
  --dataset <dataset.npz> --output <phase2b-audit-dir>
```

## 验证

在 Python 工程虚拟环境中：

- Phase 2B 单元与三候选训练 smoke：`7 passed`。
- 全量 Python 回归：`176 passed, 8 skipped`。

跳过项来自未安装的可选组件，不构成 Phase 2B OOS 证据。

## 外部 Feature-PGD 结果

- `feature_pgd` 已完成三折、每折 50 epochs；每折都有 checkpoint、ONNX、validation/test predictions 和 embedding snapshots。
- 汇总 clean：`composite_error=0.0956173`、`return_mae=0.0493726`、`direction_brier=0.2274944`、`volatility_mae=0.0099849`、`NDCG@20=0.5032469`、`RankIC=0.0586506`。
- gates：`clean_non_degraded=false`、`stress_non_degraded=false`、`all_three_window_consistent=false`、`research_gate_passed=false`、`promotion_eligible=false`。
- missing stress 的 composite error delta 为 `+0.0706924`，是当前最明显的稳定性退化；不能把 PGD 当作 champion 或进入生产组合。
- 三折训练已解除“缺少 PGD checkpoint”的阻塞，但不解除 Phase 2B 退出或晋级门槛。

## 历史记录：本机资源限制（2026-08-04）

- 已按冻结 50 epochs、3 个 purged fold 启动正式候选训练。
- APL 与 latent-FGM 已各生成 3 个 fold checkpoint，目录为 `/Users/Zhuanz/PycharmProjects/PythonProject/runs/phase2b-oos-real-v2/runs/`；尚未因 PGD 未完成而写入汇总报告。
- 当时 feature-PGD 的 `pgd_steps=3` 三进程并发触发 macOS 内存压缩；改为单 worker 恢复后首个 epoch 仍过慢，已安全停止，未改变正式参数。
- 可用 `/Users/Zhuanz/PycharmProjects/PythonProject/configs/ml/phase2b_oos_resume.local.json` 恢复；runner 会校验既有 preregistered contract hash。

### Feature-PGD 重试

当时已优化 PGD 内层只对输入特征求梯度，并分别用 `cpu_threads=1` 与 `cpu_threads=6` 重试；两次首 epoch 均约 4 分钟，确认瓶颈是 3-step 序列化 PGD 前向/反向而非并发争抢。该本机尝试未生成 PGD checkpoint；后续外部算力结果已补齐正式 50-epoch 三折训练和 OOS 报告。

### 外部训练完成登记（2026-08-05）

- `epsilon=0.01`、`beta=0.5`、`pgd_steps=3` 和三窗口合同与压缩包中的 `robust_training_spec` 一致。
- 外部训练结果已登记为 `TRAINING_COMPLETE_GATE_FAILED`，不再记录为 `DEFERRED_EXTERNAL_TRAINING`。
- 不自动 fallback 到 APL/latent-FGM；后续若重新调 PGD，必须新建 hypothesis/config 并重新运行三窗口 gate。
- 不让失败候选阻塞后续 Phase 3A/3B 独立数学与风险工程。

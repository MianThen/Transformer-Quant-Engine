# Phase 2B 状态：鲁棒训练

更新时间：2026-08-04

## 当前结论

Phase 2B 已完成第一批工程施工，但尚未满足研究退出条件：

```text
phase_exit_eligible = false
promotion_eligible = false
```

原因不是代码缺失，而是尚未在预注册的真实 clean/noisy/stress 数据上完成至少三个 purged walk-forward 窗口的候选配对 OOS 报告。现阶段所有候选仍是 research-only。

已生成 frozen champion 的三窗口 baseline OOS 参照报告，但这不等于 Phase 2B 退出：

- 报告：`/Users/Zhuanz/PycharmProjects/PythonProject/runs/phase2b-oos-baseline-real/phase2b_oos_report.json`
- 合同：`/Users/Zhuanz/PycharmProjects/PythonProject/runs/phase2b-oos-baseline-real/preregistered_contract.json`
- clean/noisy/stress 均覆盖 fold 1--3；`promotion_eligible=false`。

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

## 下一步

用同一冻结训练合同分别运行 APL、latent FGM、feature PGD 与 frozen champion，覆盖至少三个 purged walk-forward 窗口；分别输出 clean、人工噪声、四类压力集的 head/ranking 指标和配对退化。任一候选未同时满足 clean 不退化、压力集稳定改善和窗口方向一致，均保留报告但不得进入下一阶段组合。

当前 baseline 三窗口平均值（仅作 frozen champion 参照）：`return MAE=0.0649495`、`direction Brier=0.2273957`、`volatility MAE=0.0099349`、`NDCG@20=0.5105553`、`RankIC=0.0673630`。候选 robust training 因单机 CPU 运行时间较长，尚未形成正式配对结论；不得将 baseline 结果解释为 APL/FGM/PGD 的改善证据。

## 本次推进记录（2026-08-04）

- 已按冻结 50 epochs、3 个 purged fold 启动正式候选训练。
- APL 与 latent-FGM 已各生成 3 个 fold checkpoint，目录为 `/Users/Zhuanz/PycharmProjects/PythonProject/runs/phase2b-oos-real-v2/runs/`；尚未因 PGD 未完成而写入汇总报告。
- feature-PGD 的 `pgd_steps=3` 三进程并发触发 macOS 内存压缩；改为单 worker 恢复后首个 epoch 仍过慢，已安全停止，未改变正式参数。
- 可用 `/Users/Zhuanz/PycharmProjects/PythonProject/configs/ml/phase2b_oos_resume.local.json` 恢复；runner 会校验既有 preregistered contract hash。

### Feature-PGD 重试

已优化 PGD 内层只对输入特征求梯度，并分别用 `cpu_threads=1` 与 `cpu_threads=6` 重试；两次首 epoch 均约 4 分钟，确认瓶颈是 3-step 序列化 PGD 前向/反向而非并发争抢。两次均安全停止在首 epoch，未生成 PGD checkpoint，也未写入候选 OOS 报告。需要 GPU/更高效运行时后才能继续正式 50-epoch 三 fold 训练。

### 用户决策：延期 Feature-PGD（2026-08-04）

- 本轮不再在本机重试 `feature_pgd`，不改变已注册的 `epsilon=0.01`、`beta=0.5`、`pgd_steps=3` 和三窗口合同。
- `feature_pgd` 记录为 `DEFERRED_EXTERNAL_TRAINING`，不是通过，也不是自动替换为 APL 或 latent-FGM。
- Phase 2B 仍保持 `phase_exit_eligible=false`、`promotion_eligible=false`；缺失的 PGD 配对 OOS 证据必须在外部算力完成后单独补齐。
- 不让该候选阻塞后续 Phase 3A 的独立数学与风险工程；未来补回 checkpoint 后仍需重新执行三窗口 clean/noisy/stress gate。

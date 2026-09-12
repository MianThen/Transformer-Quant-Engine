# Phase 5 状态：InfoTS 对比预训练

更新时间：2026-09-12

## 家族终局判定（2026-09-12 登记封闭）

正则化修复复训（kernel `run-phase5-supervised-reg-ablation-v1`，2026-09-12，3 seed 配对，
预注册规则 R1/R2/R3）判定：

```text
r3_training_dynamics_changed = true       # selected_epoch 中位 2→4、均值 2.4→11.8
r1_positive_control_vol = FAIL            # vol MAE 改善在复训中反向（+0.021~+0.035，3/3 seed）
r2_ranking_flip = false                   # RankIC/NDCG 方向混合，不满足 3/3 同向且 |Δ|>0.005
family_status = CLOSED                    # 按预注册规则终止迭代，不再复训
winner_selected = false
phase_exit_eligible = false
promotion_eligible = false
```

最终结论：InfoTS 对比预训练对监督六输出预测**无可复现收益**——排序增益两次独立复验均不存在；
vol MAE 改善为训练配置依赖的脆弱效应（仅存在于无正则化 + 极早停条件），不构成可晋级证据。
若未来重开必须以全新 hypothesis family 与新 untouched OOS 登记。证据链：
`experiments/phase5/infots-supervised-ablation-20260911/`（首轮）、
`.../infots-supervised-ablation-seeds-2026091x/`（复验）、
`.../infots-supervised-reg-ablation-20260912/`（正则化复训终局，GitHub）。

## 监督消融与多 seed 复验判定（2026-09-12 登记）

预训练已于 2026-08-22 在 Kaggle 完成（`GPU_TRAINED_REFERENCE_PENDING_OOS`）。监督三组等预算
消融（2026-09-11/12，3 seeds × 3 groups × 3 folds，全部结构校验 PASS）判定如下：

```text
supervised_ablation_complete = true
ranking_benefit_replicated = false        # RankIC 增益 1/3 seed 方向正确（fold 级 4/9）
volatility_benefit_replicated = true      # vol MAE 改善 3/3 seed、7/9 fold（约 -5% 相对）
winner_selected = false
research_gate_passed = false
phase_exit_eligible = false               # 无 winner，家族保持 REFERENCE_ONLY
promotion_eligible = false
```

- 证据归档：`experiments/phase5/infots-supervised-ablation-20260911/` 与
  `.../infots-supervised-ablation-seeds-2026091x/`（GitHub）。
- 首 seed 的 RankIC +33% 为单 seed 噪声；volatility 改善登记为未来新假设（面向波动率头的
  预训练），不构成本次晋级理由。
- 训练动态诊断：27/27 run 在 epoch 50 前过拟合并由 return guard 选择 epoch 1-7（中位 2），
  即对照比较的是欠训练快照。据此授权**一次**正则化修复复训（同 3 seed 配对、预注册决策
  规则）；无论结果，不再对该家族做正则网格化迭代。

## 当前判定（2026-08-09 基线）

Phase 5 已完成不依赖 GPU 的研究工程闭环：数据增强、InfoNCE、train-fold-only artifact 合同、可迁移训练入口、
预计算 C++ Replay/CVaR、跨语言 provenance/hash 和九组联合验收；已生成外部 GPU 输入包，但尚无真实训练结果：

```text
engineering_scaffold_complete = true
non_gpu_engineering_complete = true
gpu_package_ready = true
phase_exit_eligible = false
promotion_eligible = false
evidence_level = TRAINING_INPUT_ONLY
```

## 本轮交付

- `python/qbt_ml/research/infots.py`：`InfoTSAugmentationSpecV1`、确定性 causal suffix crop、连续特征
  轻噪声、受控 time-mask、state/padding/future guard、augmentation hash 与 metadata。
- `info_nce_loss`：NumPy 对称 in-batch global/local InfoNCE reference，固定 temperature、正负 cosine
  diagnostics 和 embedding hashes。
- `python/qbt_ml/research/infots_training.py`：可迁移的可选 PyTorch causal GRU Siamese encoder、对称
  InfoNCE training loop、CUDA/deterministic/device/train-fold guards、checkpoint SHA-256 与六输出契约；
  本机缺少 PyTorch 时明确失败，不静默回退 CPU。
- `select_information_aware_augmentation`：只接受通过 future/state/padding guard 的 train-fold candidate，
  使用预注册 minimum improvement 选择或保留 baseline，不做隐式 fallback。
- `build_infots_pretraining_artifact`：拒绝 `train_fold_end` 之后的 timestamp，保存 dataset、augmentation、
  selector hash、监督预算和六输出语义；artifact 固定为 `REFERENCE_ONLY`，不允许 phase/promotion 晋级。
- `python/qbt_ml/evaluation/infots_ablation.py`：三组等监督预算合同、六输出/双 embedding provenance、
  test-blind/purged-OOS guard、逐 fold/均值指标和 gate 汇总；不自动选择 winner。
- `performance_analytics/include/performance_analytics/infots_replay.h`、
  `performance_analytics/src/infots_replay.cpp`：预计算六输出研究回放、ReturnLedger/CVaR、未来数据门禁、
  代理执行状态和跨语言固定二进制 hash；不推理模型、不静默回退 Python。
- `python/qbt_ml/evaluation/infots_replay.py`、`tools/run_phase5_infots_cpp_replay.py`、
  `tools/run_phase5_infots_joint_acceptance.py`：C++ report/hash 独立复算和九个 group/fold 联合验收，固定
  `promotion_eligible=false`。
- `python/tests/test_phase5_infots.py`：determinism、状态/填充保护、future mutation、InfoNCE、selector 和
  artifact hash/future guard 测试。
- `python/tests/test_phase5_infots_ablation.py`：三组完整性、预算一致、未来泄漏、六输出/embedding hash 和
  deterministic report 测试。
- `python/tests/test_phase5_infots_replay.py`、`performance_analytics/tests/test_infots_replay.cpp`：跨语言
  输入/report hash、future leakage、六输出范围、代理状态、C++ ledger/CVaR 和联合验收失败关闭测试。
- `tools/run_phase5_infots_pretraining.py`、`tools/phase5_infots_gpu_preflight.py`、冻结 CUDA 配置和
  `requirements-phase5-infots-cuda.txt` 已纳入训练输入包：
  `work/phase5_infots_gpu_training_bundle_20260809.tar.gz`。
- 输入包 SHA-256：`ee9e28b0806cd5c969556228bc930133a6a31d65b9ebd049eb7dcd14a6434c2e`；sidecar 与归档一致，
  包内 24 个 manifest 条目、数据 SHA-256 和路径安全审计通过。

## 验证

```text
python3 -m pytest python/tests/test_phase5_infots.py -q
=> 4 passed

python3 -m pytest python/tests/test_phase5_infots_ablation.py -q
=> 3 passed

此前基础全量 Python：29 passed, 1 skipped。

本轮全量 Python：31 passed, 1 skipped；带 `QBT_ENABLE_PERFORMANCE_ANALYTICS=ON` 的 CTest：17/17 passed。

bundle manifest/data/path audit => PASS
CUDA preflight without PyTorch/CUDA => exit 1 (fail-closed)
```

## 仍待 GPU 结果

- 输入包已具备 PyTorch global/local encoder、pretraining loop、冻结 CUDA 配置、CLI、preflight、pinned
  GPU requirements 和数据；本机未执行真实 CUDA 训练，不能把输入包当成训练结果。
- 尚无 train-fold-only 真实 GPU run、from-scratch/fixed-augmentation/InfoTS 三组等监督预算比较、
  purged OOS、六输出 prediction/embedding artifact；C++ Replay 仅已完成接口和研究代理验收，尚未消费真实
  GPU 回传的九组结果。
- 仍需在 GPU 训练完成后执行 selector gate、质量/稳定性门槛和 Phase 5 退出判定；CPU reference 不替代
  真实训练证据。

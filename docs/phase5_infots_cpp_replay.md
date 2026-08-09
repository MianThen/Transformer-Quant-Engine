# Phase 5 InfoTS C++ 预计算 Replay 合同

## 目的

本接口消费外部 GPU 返回的预计算六输出和研究代理收益，不在 C++ 中推理模型，也不把训练输入包当成训练结果。它复用现有 `ReturnLedger`、净收益会计恒等式和经验 CVaR 语义，输出 `RESEARCH_PROXY` 范围的可审计 artifact。

## 输入

`performance_analytics/infots_replay.h` 定义 `InfoTSReplaySpecV1` 和 `InfoTSReplayRowV1`。

- 每个 fold 必须声明 `group_id`（`from_scratch`、`fixed_augmentation` 或 `infots`）、`fold_id`、policy、合同/数据/预测/两个 embedding/source snapshot SHA-256。
- 每行必须提供 `prediction_available_at <= decision_at < realized_at`、严格递增 session/实现时间、代理收益和固定顺序的六输出：`expected_return`、`expected_volatility`、`direction_probability`、`lower_quantile`、`upper_quantile`、`confidence`。
- `claim_scope=RESEARCH_PROXY`、reference/执行缺失状态、`lot=1`、涨跌停 disabled、费用 assumed zero 和 slippage unavailable 均为硬冻结值。

输入 hash 使用跨语言固定的大端二进制合同；Python 的 `compute_infots_replay_input_sha256` 与 C++ `infots_replay_input_sha256` 必须一致。任何 future leakage、非有限值、概率/分位数越界、非单调时间或声明 hash 不一致都会失败关闭。

## 输出和联合验收

`run_infots_precomputed_replay` 输出 ledger SHA-256、累计收益、Sharpe、最大回撤、VaR loss、expected shortfall loss 和 `return_cvar`。`artifact_sha256` 使用固定 binary report contract，Python `compute_infots_cpp_report_sha256` 可独立复算。

单 fold 可通过：

```text
tools/run_phase5_infots_cpp_replay.py --spec spec.json --rows rows.json --output cpp_replay.json
```

九个 group/fold 结果由 `tools/run_phase5_infots_joint_acceptance.py` 与 Python fold artifact 联合验收。它要求 prediction hash、validation/test embedding hash、contract hash、row count、test-blind/purge 状态一致，并固定 `promotion_eligible=false`。

该接口完成研究工程闭环，但不替代真实 CUDA 训练、三窗口 OOS、质量/稳定性 gate 或生产执行数据；因此当前 Phase 5 仍不得自动退出或晋级。

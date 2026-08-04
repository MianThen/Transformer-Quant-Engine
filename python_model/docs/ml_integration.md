# 模型训练与推理边界

当前 ML 功能默认关闭。现有 Python 回测环境不安装 PyTorch、ONNX 或 ONNX Runtime，
也不会在回测、Dashboard 或安装过程中触发训练。

## Phase 1B 排序合同

- `LabelSpecV2` 保留原始 log return 输出，同时生成 soft direction、逐 Bar
  realized volatility、downside semivol、risk-adjusted return、截面
  `rank_utility` 和 `[0,1]` relevance。
- `RankingScoreSpecV1` 冻结 raw-return 或 risk-adjusted score、生产 Top-K、
  risk floor、cost proxy、winsorization、rank temperature 和 `lambda_rank`。
- 每个 ranking batch 是一个完整 timestamp。legacy correlation、ListMLE 和
  LambdaLoss@K 一次只启用一个，并共享 encoder、标签、K、seed 和其他固定权重。
- LambdaLoss 使用预测 Top-K anchor 对全截面的 `O(KN)` 路径；小截面测试用
  全 pair oracle 校验。NDCG 对预测并列使用平均 discount。
- LabelSpec/RankingScoreSpec hash、loss variant、K、temperature 和 rank weight
  写入 checkpoint 与 model manifest。C++ candidate selection 使用同一 score 公式，
  V2 manifest 与 policy K/score 不一致时拒绝启动。
- 在至少三个 purged OOS 窗口完成 legacy/ListMLE/LambdaLoss 固定权重配对并冻结
  胜出项前，不启用 Kendall 动态权重。

## Phase 1E 共享梯度合同

Phase 1E 固定按 `diagnostics -> pcgrad -> gradnorm` 顺序执行，每次训练只能启用一种模式。
所有模式都要求 `legacy` rank loss、fixed weighting，以及
`return=1.0/direction=0.25/volatility=0.25/quantile=0.25/rank=0.1`；训练入口会拒绝
Kendall、新 ranking loss、非冠军权重、AMP 或未实现的 gradient accumulation 组合。

- `diagnostics` 只采样共享 backbone 的五任务 norm、dot/cosine、conflict、dominance 和
  relative training rate，不改 forward、backward 或 optimizer update，并输出
  `gradient_conflict_artifact.json`。
- `pcgrad` 只替换共享 backbone 梯度，任务头继续使用原始 scalar-loss 梯度；固定 seed、
  pair traversal、zero-norm guard 和 projection frequency 写入 checkpoint/metrics。
- `gradnorm` 只自适应四个非排序任务，使用正值且和为 4 的权重；legacy rank 永远固定 `0.1`，
  权重路径、`G_k/r_k/G*_k`、zero-gradient rate 和 controller state 写入 checkpoint/metrics。
- `pcgrad` 与 `gradnorm` 都必须配置新的 `hypothesis_id`。工程 smoke 通过不代表实验晋级；
  每个 challenger 仍需三个新的 purged OOS 窗口、Phase 1C 经济报告和 Phase 1D drift baseline。

默认配置保持 `{"gradient_optimization": {"mode": "none"}}`。诊断时把 mode 改为
`diagnostics`；挑战者分别使用例如
`{"mode": "pcgrad", "hypothesis_id": "MTL-PCGRAD-..."}` 或
`{"mode": "gradnorm", "hypothesis_id": "MTL-GRADNORM-4-...", "alpha": 1.5}`，不得合并字段。

## 固定决策

- 第一版协议是 `BAR_V1`，执行对齐为 `NEXT_OPEN`。
- Python 负责 point-in-time 数据集、特征、标签、训练、评估和 ONNX 导出。
- C++ 只消费版本化制品；模型预测不能绕过组合、订单规划和既有交易规则。
- 特征 schema、模型、日历、股票池和数据截止时间都必须写入制品。
- schema/hash 不匹配、输出非有限、输入过期或模型不可用时停止产生新风险。

## 在训练机启用

```bash
python -m pip install -e '.[ml]'
python -m python.qbt_ml.cli build-dataset --config configs/ml/bar_v1.yaml
python -m python.qbt_ml.cli train --config configs/ml/temporal_transformer_v1.yaml
python -m python.qbt_ml.cli phase1b-ablation --config configs/ml/temporal_transformer_v1.yaml
python -m python.qbt_ml.cli phase1b-kendall-ablation --config configs/ml/temporal_transformer_v1.yaml
python -m python.qbt_ml.cli export --config configs/ml/bar_v1.yaml --run runs/<run-id> --output models/<model-id>
python -m python.qbt_ml.cli validate-artifact models/<model-id>
```

配置中的 `enabled` 默认为 `false`，训练数据路径、股票池、交易日历和设备必须在训练机
显式填写。训练完成后只迁移只读模型制品，不迁移临时 checkpoint 或原始训练数据。

## 制品最小内容

`model.onnx`、`manifest.json`、`feature_schema.json`、`metrics.json` 是必需文件。
生产晋级还需要 golden 输入输出，并完成 PyTorch、ONNX Runtime Python 和 C++ 三方一致性验证。

## 在回测 C++ 模块中激活

普通 wheel 始终保持 ML 关闭。迁移到具备 ONNX Runtime SDK 的平台后，在 C++ 项目目录显式构建：

```bash
QBT_ENABLE_ML=ON \
QBT_ML_BACKEND=onnxruntime \
ONNXRUNTIME_ROOT=/opt/onnxruntime \
python -m pip install .
```

安装后先检查 `cpp_engine.__ml_enabled__` 和 `cpp_engine.__ml_backend__`，再在任何行情处理前
调用 `engine.set_model_strategy(...)`。加载过程会核验模型和特征 schema 的 SHA-256、
`BAR_V1`、`NEXT_OPEN`、输入形状和 CPU provider；失败时不会退回旧预测或产生订单。

可直接对 CSV/Parquet Bar 数据回测已导出的制品：

```bash
python -m python.qbt_ml.cli backtest-artifact models/<model-id> \
  --data /path/to/bars.parquet \
  --config configs/ml/artifact_backtest.json \
  --output runs/artifact-backtest.json
```

命令要求 `cpp_engine.__ml_backend__ == "onnxruntime"`，按 timestamp 传入完整截面，
并输出模型版本、数据范围、订单、成交、现金、权益、收益、Sharpe 和最大回撤摘要。

Phase 1B 的首轮消融固定为 `legacy`、`listmle`、`lambda` 三个变体。运行器要求数据集和窗口参数
恰好产生三个 purged walk-forward fold，并冻结 dataset/spec/config hash；每个 fold 只允许改变
`ranking.loss_variant`。报告按 timestamp 等权给出 NDCG@K、RankIC、Precision@K、top-k overlap、
单边 turnover 和 top-bottom realized utility spread。挑战者通过模型 gate 后，仍需对齐的 C++
成本回放与 CVaR 证据才能写入 `frozen_winner`；若所有挑战者已在模型 gate 失败，则直接冻结并保留
incumbent，随后才允许运行 Kendall 动态权重消融。

固定权重报告冻结 ranking loss 后，Kendall 消融只学习 return、direction、volatility 和 quantile
四个任务级 `log(sigma^2)`；rank 继续使用冻结的显式 `lambda_rank`。报告逐 epoch 保存 raw loss、
weighted loss、log variance、有效权重和 log-variance gradient，并对主头、排序指标和 clamp 贴边
状态做独立 gate。挑战者在模型 gate 已失败时，允许直接保留并冻结 incumbent；只有挑战者进入模型
候选集时才要求 C++ 经济 gate 后再冻结。

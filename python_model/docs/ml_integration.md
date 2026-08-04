# 模型训练与推理边界

当前 ML 功能默认关闭。现有 Python 回测环境不安装 PyTorch、ONNX 或 ONNX Runtime，
也不会在回测、Dashboard 或安装过程中触发训练。

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

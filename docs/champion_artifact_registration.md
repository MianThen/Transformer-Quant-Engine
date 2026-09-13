# 冠军模型制品登记（champion-temporal-transformer-v1）

登记时间：2026-09-12
制品位置：`/Users/Zhuanz/PycharmProjects/PythonProject/models/champion-legacy-fold3/`

## 制品身份

| 字段 | 值 |
|---|---|
| model_id | `champion-temporal-transformer-v1` |
| model_version | `legacy-fold3-20260912` |
| model.onnx SHA-256 | `2337cb08b179b9c44ef70d9e97d7fe2e517e3fb86e526adee8ad66a24e1e301c` |
| 架构 | TemporalTransformerV1（lookback 64，d_model 64×3L×4H，FFN 128，BAR_V1，23 特征） |
| 排序合同 | legacy correlation loss + 固定权重（return 1.0 / direction 0.25 / volatility 0.25 / quantile 0.25，rank 0.1）；Top-20，raw_return score |
| 来源 checkpoint | `runs/phase1b-real-ablation-v2/fold-3/legacy/checkpoint.pt`（Phase 1B 冻结现场，seed 20260724，50 epochs） |
| 训练窗口 | 1618815600000000000 → 1684306800000000000（≈2021-04-19 至 2023-05-17 UTC） |
| data_cutoff_utc | `2023-05-17T07:00:00Z`（训练数据截止；validation/test 仅用于评估与冻结选择） |
| universe_id / calendar_id | `pit-120-symbols` / `sse-minute-pit-v1`（首次登记的描述性标识） |
| 冻结现场 test 指标 | NDCG@20 = 0.6328，RankIC = 0.0300（fold-3 test 窗 1702623600000000000–1719385200000000000） |

## 验证状态

```text
validate_artifact        = PASS          # qbt-ml CLI：manifest/schema/文件齐全性
pytorch_vs_ort_parity    = PASS_BITEXACT # 六输出 max|diff| = 0.000e+00（golden I/O）
cpp_onnxruntime_thirdway = PASS          # 1587 截面全量零错误（safe 制品）
economic_gate            = NOT_RUN       # 生产晋级仍按合同保持关闭
```

### C++ 三方一致性：PASS（2026-09-12 修复后通过）

首次运行发现 ONNX 导出对部分零 valid_mask 产生 NaN（attention softmax 全 -inf）。
修复：`SafeExportWrapper` 在导出时加入 `torch.where(isfinite, x, 0)` 守卫，
正常 mask 下与原模型**逐位一致**（max|diff|=0.000e+00）。

修复后制品 `champion-legacy-fold3-safe`（sha256 `0ef338fe...`）：
- C++ ORT 1.19.2 (arm64) 全量 1587 截面推理**零错误**
- 同时修复 C++ 引擎两处 binding 缺失（`get_order_count`/`get_trade_count` → `len(get_*_history())`）
- 附加修复：C++ BAR_V1 管线 zero-batch guard + ModelStrategyRuntime `features.batch_size==0` 早退

零订单为策略过滤所致（`minimum_confidence=0.55`），非推理故障。

## 边界与使用

- 这是**首个通过两方一致性验证的冠军制品候选**：PyTorch 导出 → ONNX → ONNX Runtime Python
  逐位一致，golden 输入输出已固化在 `golden/`。
- **未达成生产晋级**：C++ ONNX Runtime 三方一致性未跑（本机无 ORT SDK 构建）；成本回放与
  经济 gate 未运行。晋级前禁止进入任何实盘链路。
- C++ 侧激活路径（具备 ORT SDK 的平台）：`QBT_ENABLE_ML=ON QBT_ML_BACKEND=onnxruntime
  ONNXRUNTIME_ROOT=... python -m pip install .`，随后核验 `cpp_engine.__ml_backend__`，
  并可用 `python -m python.qbt_ml.cli backtest-artifact models/champion-legacy-fold3 ...`
  直接对该制品跑 C++ 回测。
- fold-1/fold-2 的同族 checkpoint 保留在原 run 目录，如需可按同一流程导出为对照制品。

## 证据链

- Phase 1B 冻结报告：`phase1b_experiment_report.md`（winner=legacy，多窗口配对）
- 导出配置：`work/champion-export-config.json`（C++ 仓库，身份字段与冻结现场一致）
- 后续挑战者（1E/2B/5）全部被证据链拒绝——本制品即当前唯一合法的预测模型部署候选。

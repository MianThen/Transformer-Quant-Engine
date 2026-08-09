# Phase 3C 状态：校准、Conformal 与 Attention

更新时间：2026-08-06

## 当前判定

Phase 3C 的工程实现和诊断产物已补齐，但阶段退出条件尚未满足：

```text
engineering_scaffold_complete = true
phase_exit_eligible = false
promotion_eligible = false
evidence_level = RESEARCH_PROXY
```

## 已完成

- `python/qbt_ml/calibration/probability.py`：加权 Platt、PAV Isotonic、Brier/NLL/ECE、reliability
  bins、knot table、边界和时间可用性 guard。
- `python/qbt_ml/calibration/conformal.py`：固定/rolling CQR、有限样本 higher quantile、指数衰减、future
  guard、artifact hash。
- `python/qbt_ml/analysis/attention.py`：逐层/head 权重、last-token、rollout、entropy/span、时间遮挡
  对照、稳定性、straight-line trapezoidal Integrated Gradients 和 feature-group occlusion 交叉验证。
  分析路径保持生产六输出不变。
- `tools/run_phase3c_calibration.py`：三窗口严格 `(timestamp,symbol)` 对齐，validation 拟合、test 只应用。
- `tools/run_phase3c_attention_real.py`：三折冻结 `none` checkpoint 的真实小样本 Attention/IG 诊断 runner。

## 真实产物

Attention 报告位于 `runs/phase3c-attention-real/report.json`，SHA-256：
`e57fdb3445de68fa6f8c987f50a5d8dac442373f0edb96da43f894e847702c44`。

- 三折共同股票为 `000001`、`000002`，每折两个完整 64 步 test 窗口样本。
- 三折 IG completeness 均通过（512 步梯形积分）；生产六输出 parity 三折均通过且最大差为 `0.0`。
- Attention 与 IG 的时间 profile 相关性仅作诊断：cosine 均值约 `0.172 / 0.457 / 0.321`，top-2
  Jaccard 为 `0 / 0.167 / 0`，没有预注册晋级阈值，不据此挑选模型。
- IG 与 feature-group occlusion 在三折均高度一致，group cosine 约 `0.993--1.000`；这仍不构成因果解释。
- 跨折 Attention rollout cosine 为 `0.905--0.960`，top-2 overlap 在 fold-3 参与的 pair 降至 `0.333`，
  因而只登记为 stability diagnostic。

## 仍未退出的原因

- 概率校准真实报告为诊断-only：fold-1 Platt 失败关闭；Isotonic OOS 平均 Brier/NLL/ECE 劣于未校准；
  CQR fold coverage 为 `0.7840 / 0.8998 / 0.8397`，且没有事前注册的数值容差和 regime gate。
- 真实 Attention 仅为确定性小样本，不是全 OOS faithfulness/stability 证据；路线图也未预注册 Attention
  数值 winner gate。
- 因此校准 head 不得进入 FFV view，生产/实盘晋级仍保持关闭。

## 验证

- Phase 3C 校准、CQR、报告定向测试：`20 passed`。
- Attention 定向测试：`8 passed`（实际 PythonProject `.venv`，仅有 PyTorch `norm_first` 性能提示）。

下一步是建立 Phase 4A 的 `ViewSpec V1`、PIT prior-scenario 合同和独立 Gaussian posterior oracle；在
校准未退出前只允许合成/均值 view，未校准 direction/quantile view 必须失败关闭。

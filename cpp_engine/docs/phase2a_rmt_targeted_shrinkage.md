# Phase 2A：RMT Targeted-Shrinkage

## 定义

`RMT_TARGETED_SHRINKAGE` 与 `RMT_CONSTANT_RESIDUAL` 使用相同的收益窗口、去均值、标准化相关矩阵
和 MP noise boundary，但 estimator id 独立。对落在 noise bulk 内的特征值使用：

```text
lambda_clean = lambda_raw + alpha * (lambda_noise_mean - lambda_raw)
```

其中 `alpha = targeted_shrinkage_intensity`，固定在 `[0, 1]`；signal 子空间特征值不变。`alpha=1`
才等价于 noise 子空间的 constant residual，不能因此复用 constant-residual estimator id。

## 输入与数值合同

- `alpha` 必须有限且位于 `[0, 1]`；默认值为 `0.5`。
- noise mean 必须不小于 `eigenvalue_floor`；否则返回 `NUMERICAL_FAILURE`，不通过抬高噪声特征值
  破坏 trace preservation。
- cleaned eigenvalues 必须有限且不低于 floor；恢复后的 correlation 必须对称、PSD、单位对角。
- 协方差使用固定输入收益的原始波动率向量映射，不把 targeted correlation 写入 official risk 或
  cluster provenance。

## 诊断与状态

结果保留 raw/cleaned eigenvalues、`targeted_shrinkage_intensity`、signal rank、noise count、MP
boundary、trace drift、diagonal drift、condition number 和 PSD repair amount。`eligible_for_official_risk`
保持 `false`，直到路线图要求的 synthetic spectrum、独立 oracle、OOS variance forecast、组合稳定性
和成本门槛全部通过。

当前实现入口为 `portfolio_math::rmt_targeted_shrinkage_denoising`，研究选择器也保留独立的
`RMT_TARGETED_SHRINKAGE` estimator identity。

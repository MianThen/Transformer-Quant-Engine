# Phase 2A：HRP policy anchor

## 目的

`solve_hrp_policy` 实现路线图中的 long-only、recursive-bisection HRP baseline。它只产生
risk-only anchor，不消费 expected return、Transformer view 或 confidence，也不把 cluster discovery
correlation 当作正式风险协方差。

## 输入与边界

- `official_covariance` 必须是有限、对称、PSD 且对角线严格为正的官方协方差；计算簇方差前只做
  对称化，不做矩阵求逆。
- `quasi_diagonal_order` 必须是完整、无重复的资产 permutation，由 raw/denoised/detoned correlation
  的独立 linkage 结果提供。
- 递归二分和最终 `predicted_risk` 始终使用 `official_covariance`。
- V1 返回 long-only 权重，默认全投资 `sum(weights)=1`；`target_investment<1` 时，未分配部分由
  上层显式 cash sleeve 解释，不在 HRP 内部隐式生成资产。
- 成本、行业上限、换手、成交量和其他硬约束不在 anchor 内处理，统一交给后续 reconciler。

## 算法合同

对每个准对角区间 `[begin,end)` 按位置中点拆成左右子簇。子簇方差使用官方协方差上的 inverse-
variance portfolio：

```text
v_C = w_C' Sigma_official w_C
w_i ∝ 1 / Sigma_official[i,i],  i ∈ C
```

左右簇分配为：

```text
a_left  = v_right / (v_left + v_right)
a_right = v_left  / (v_left + v_right)
```

从根到叶重复乘比分配，最后归一化到 `target_investment`。每次 split 的区间、簇方差和分配因子
进入 `HrpPolicyDiagnostics::bisection_steps`，用于 composition replay 和实验审计。

## 失败关闭

以下情况不会回退到 Top-K、Risk Budget 或上一期权重：输入维度/顺序非法、非对称或非有限矩阵、
非 PSD 官方风险模型、非正对角线、零/非有限簇方差、分配因子或最终风险非有限。失败通过
`OptimizationStatus` 返回，权重保持为空；`eligible_for_official_risk=false`，本切片不接入正式
CVaR、Replay 或 reconciler。

## 退出测试

`test_hrp_policy` 覆盖：

- 对角协方差 inverse-variance 解析 oracle；
- 两个独立块的簇间分配与簇内 composition oracle；
- linkage 生成的 quasi-diagonal order 到 HRP 的边界；
- 协方差和资产置换等变性；
- long-only、全投资、finite predicted risk 和 split 诊断；
- 重复顺序、非对称、非 PSD、零方差、非有限输入和非法 options 的失败关闭。

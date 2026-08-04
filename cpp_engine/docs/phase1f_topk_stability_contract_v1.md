# Phase 1F `LEGACY-TOPK-STABILITY-V1` 合同

日期：2026-08-02（预注册）
状态：合同、oracle 与一次正式研究运行已完成；生产经济 gate `DEFERRED_TO_LIVE`

## 1. 研究范围

本候选只在冻结的 `legacy + fixed` 模型上增加显式 temporal/top-k stability
regularizer。它不重开 ListMLE/LambdaLoss，不改变 LabelSpec、RankingScoreSpec、encoder、
optimizer、MTL 权重、C++ policy 或 cost model。当前 `FROZEN_CHAMPION` 不变。

候选 hypothesis id：`LEGACY-TOPK-STABILITY-V1`。

正式实验使用 Phase 1C/1D 之后的新 validation/untouched period，并在训练前冻结本合同。C++
经济输出仍按 `RESEARCH_PROXY` 口径审计；它可以支持研究比较和阶段退出，但不能据此生产晋级。

## 2. Frozen forward reference

给定一个决策时点的截面 score `s_i`，按 score 降序得到 `v_j`。symbol 只用于精确并列时的
确定性排序，不参与非并列数值。温度 `tau > 0` 时：

```text
P[j,i] = exp(-|v_j - s_i| / tau) / sum_l exp(-|v_j - s_l| / tau)
w_i    = (1 / K) * sum_{j=1..K} P[j,i]
```

因此 `w_i >= 0` 且 `sum_i w_i = 1`；当 `tau -> 0` 且没有 tie 时，前 K 个 symbol 的 `w_i`
趋近 `1/K`，其他 symbol 趋近零。跨相邻决策时只允许使用两期共同、在当时可知的 symbol
intersection。

## 3. Frozen stability objective

对当前截面 `t` 和上一决策截面 `t-1`：

```text
L_stability(t) = (1 / |I_t|) * sum_i (w_t,i - stop_gradient(w_t-1,i))^2
L_total       = L_legacy + lambda_stability * mean_t[L_stability(t)]
```

`w_t-1` 在训练实现中必须作为前一决策的冻结 snapshot 使用，不能读取 realized return、未来
label、未来价格或未来 universe。`K=20` 继承现有生产 top-k；`tau`、`lambda_stability` 的
实验值必须在新 validation/untouched period 之前单独登记，不能从 Phase 1B/1E OOS 反调。

## 4. 参考资料

实现采用 SoftSort/可微排序的前向结构作为研究参考，不把论文实现直接当作生产代码：

- Prillo & Eisenschlos, *SoftSort: A Continuous Relaxation for the argsort Operator*,
  ICML 2020，arXiv:2006.16038，<https://arxiv.org/abs/2006.16038>。
- Blondel et al., *Fast Differentiable Sorting and Ranking*, NeurIPS 2020，
  arXiv:2002.08871，<https://arxiv.org/abs/2002.08871>。
- Cuturi et al., *Differentiable Ranking and Sorting using Optimal Transport*, NeurIPS 2019，
  arXiv:1905.11819，<https://arxiv.org/abs/1905.11819>。

本仓库的独立前向和有限差分 oracle 位于
`python/qbt_ml/research/topk_stability.py`，验证入口为
`tools/verify_phase1f_topk_oracle.py`。oracle 只用于公式、置换/平移不变性、低温 top-k
极限和有限差分有限性校验；未来 PyTorch/ONNX/C++ 实现仍需单独做三方 fixture parity。

当前只登记了论文题录和公开链接，没有把原始 PDF、版本许可和内容 hash 固化为生产依赖；
因此本合同的论文状态仍是 `RESEARCH_REFERENCE_ONLY`，不能作为正式模型晋级的数学冻结证明。

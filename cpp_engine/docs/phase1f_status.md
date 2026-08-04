# Phase 1F 状态

日期：2026-08-03
状态：`RESEARCH_PROXY_EXIT_COMPLETE / PRODUCTION_GATE_DEFERRED`

## Gate 判定

| Gate | 结果 | 证据 |
|---|---|---|
| Phase 1C 完成 | `ENGINEERING_COMPLETE_RESEARCH_PROXY` | 六个 C++ Replay/CVaR proxy 已统一审计；生产经济晋级延后，历史执行字段保持 unavailable |
| Phase 1D real baseline/report | `RESEARCH_PROXY EXIT COMPLETE` | 23 个 BAR_V1 features、六个 model heads、validation/test embedding snapshot、manifest hash 和 CKA 已通过；`phase_exit_eligible=true` |
| 稀疏 universe | `PASS` | `K=20`，`4K=80`；新 OOS `N=min/median/max=106/109/112` |
| 新 validation/untouched period | `PASS` | validation 59 sessions（2026-01-15–2026-04-16）；untouched 59 sessions（2026-04-17–2026-07-14） |
| differentiable sorting/top-k oracle | `PASS` | `LEGACY-TOPK-STABILITY-V1` 合同、独立 oracle 和有限差分检查已通过；论文 PDF/许可 hash 仍为 reference-only |
| stability challenger | `PASS / WINNER=stability` | 预注册 `tau=0.20`、`lambda=0.01`、`purge=6`、`seed=20260803`；validation gate 通过，untouched 只评估冻结 winner |
| C++ Replay/CVaR | `PASS / PROXY` | 58 periods；`return_CVaR=-0.02469592429258582`；`promotion_eligible=false` |

数量门槛的计算为 `109 >= 4 * 20 = 80`，因此 Phase 1F 不受 universe 稀疏度阻塞。正式
validation/untouched 窗口、top-k 合同、训练报告和 winner 的 C++ proxy Replay/CVaR 均已落盘；研究
工程退出完成，但长期不可得执行字段仍阻止生产经济晋级。核心证据位于
`runs/phase1c-closure/phase1c_closure_report.json`、
`runs/phase1d-real-baseline/phase1d_real_baseline_report.json` 和
`runs/phase1f-topk-stability-real-sampled/phase1f_training_report.json`。

## 已完成的前置施工

- 固定 SoftSort-style top-k mass、temporal stability penalty 和 causal snapshot 语义。
- 增加纯 Python 独立 oracle、置换/平移不变性、低温极限和有限差分梯度校验。
- 保留 `FROZEN_CHAMPION=legacy + fixed`；没有训练、OOS 调参、C++ policy 或经济结果改动。

验证命令：

```text
python3 tools/verify_phase1f_topk_oracle.py
```

## 研究退出与生产门禁

Phase 1F 已完成一次预注册单变量研究：validation 通过后冻结 `stability` winner，只在 untouched
窗口评估一次，并将 winner 的 C++ Replay/CVaR 链接到训练报告。报告哈希为
`0f3b8ab6e29187c90de5663f98f6fd4fa88cfa80c55e5ce4465bd0435a37d795`，C++ proxy report 哈希为
`23b47f6c3542d17722e6949902c95cc850f3e8ffc598829bf0b66fb662462793`。

该结果只证明当前预注册窗口中的研究可复现性和相对比较资格，不改变冻结冠军，也不构成生产
promotion。费用/税费、PIT 涨跌停、公司行动/复权、lot、真实滑点/reference provenance 继续为
`UNAVAILABLE/PROXY`；因此 `phase_exit_eligible=true`、`research_comparison_eligible=true`，而
`promotion_eligible=false`。

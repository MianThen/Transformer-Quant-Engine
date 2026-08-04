# Phase 1E 实验报告

报告日期：2026-08-02
路线图：`QUANT_MATH_TRANSFORMER_V1_2_ROADMAP+5.md`  研究环境：`PythonProject/.venv`，PyTorch `2.8.0`，CPU

## 1. 结论

| 项目 | 判定 |
| --- | --- |
| Phase 1E 代码与实验施工 | **完成** |
| `GradientConflictArtifact V1` | **完成** |
| `MTL-PCGRAD` 三窗口实验 | **完成，report-only** |
| `MTL-GRADNORM-4` 三窗口实验 | **完成，report-only** |
| C++ Replay/CVaR 研究输出 | **已完成，PROXY** |
| Phase 1E 严格退出/正式晋级 | **未完成** |
| 当前冻结冠军 | `legacy + fixed`，保持 `FROZEN_CHAMPION` |

严格退出未完成的原因不是没有数值输出，而是两个门槛仍未满足：

1. C++ Replay/CVaR 使用 bar-close/reference proxy，费用、涨跌停、公司行动、lot、滑点和正式
   reference-price provenance 仍缺失，因此所有经济 artifact 强制 `promotion_eligible=false`。
2. 当前 PythonProject 没有可复用的 Phase 1D real drift baseline/report；本报告不伪造该证据。

## 2. 预注册合同与数据

- Phase 1E dataset SHA-256：`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`
- data audit SHA-256：`5aedcc0bbbcd5cebda11a5e2e15be0a8420f65810c43cf7e71b77ff22c7c3c47`
- 数据审计：181,273 行、120 只预注册股票、1,587 个 timestamp、PIT violations=0、security-state missing rows=0
- 新 OOS：502 个 timestamp，横截面数量 min/median/max=`106/109/112`
- 固定 scalar weights：return=`1.0`、direction=`0.25`、volatility=`0.25`、quantile=`0.25`、legacy rank=`0.1`
- 三个窗口均使用 purge=6、embargo=5；PCGrad 与 GradNorm 使用相同 split，不使用 Phase 1B 原 OOS 调参。

| Fold | Train | Validation | Test |
| --- | --- | --- | --- |
| 1 | 2021-10-27–2023-11-21 | 2023-11-30–2024-06-11 | 2024-06-27–2024-12-30 |
| 2 | 2022-05-06–2024-05-31 | 2024-06-12–2024-12-13 | 2024-12-31–2025-07-10 |
| 3 | 2022-11-09–2024-12-05 | 2024-12-16–2025-06-25 | 2025-07-11–2026-01-14 |

本轮 challenger 是短周期 `3 epoch` 预注册机制实验，目的是验证优化机制和完整证据链，不能作为
50 epoch 生产训练替代品。

## 3. 诊断基线

- 三个 fold 每个 `788` 个梯度样本。
- `baseline_diagnostic_exact_parity=true`：diagnostics 开关前后 history 与 model state 完全一致。
- 诊断报告 SHA-256：`d0e492b0c12070005bc18e69f6942a202b5eb8bcfe53bf8d70769603f89aa2aa`
- 诊断保持 observation-only，没有形成 runtime fallback 或覆盖冻结冠军。

## 4. 主模型指标

指标为三个 test window 的 timestamp-equal-weighted 均值；delta 为 challenger 减去配对 baseline。

| 指标 | Paired baseline | PCGrad | PCGrad delta | GradNorm-4 | GradNorm-4 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| NDCG@K | 0.503214 | 0.500191 | -0.003022 | 0.497440 | -0.005773 |
| RankIC | 0.058836 | 0.065868 | +0.007032 | 0.065944 | +0.007108 |
| precision@K | 0.165608 | 0.156217 | -0.009392 | 0.156878 | -0.008730 |
| top-bottom utility spread | 0.004876 | 0.005020 | +0.000145 | 0.004733 | -0.000142 |
| top-k turnover | 0.168667 | 0.176933 | +0.008267 | 0.174933 | +0.006267 |
| return MAE | 0.070688 | 0.066946 | -0.003742 | 0.066630 | -0.004057 |
| direction Brier | 0.226624 | 0.227569 | +0.000946 | 0.227057 | +0.000433 |

结论：两个 challenger 都改善了 RankIC 或部分回归头，但 NDCG/precision/turnover 存在退化，不能
满足路线图要求的主指标非劣与经济 gate。

## 5. 机制指标

### PCGrad

- hypothesis：`MTL-PCGRAD-REAL-OOS-20260802-E3`
- 每 fold：1,512 条机制诊断记录，zero-norm skip=`0`
- projection frequency：fold 1=`0.5501`，fold 2=`0.5506`，fold 3=`0.5458`
- 只替换 shared backbone gradient；没有与 GradNorm 组合。
- challenger report SHA-256：`d25172d607c6ae84afb6ea842c94a97e80bdad73cb10330fb30b54c7499f61da`

### GradNorm-4

- hypothesis：`MTL-GRADNORM-4-REAL-OOS-20260802-E3`
- 每 fold：1,512 条机制诊断记录。
- 四个自适应任务权重始终为正，归一化和为 `4`；legacy rank weight 始终为 `0.1`。
- 三个 fold 末期均出现任务权重重新分配，未修改 rank loss 或 encoder。
- challenger report SHA-256：`7384c559ae7ed502da023d595b0b311f85c5270b5dd6d1c2fa6d5dde8566d429`

## 6. C++ Replay/CVaR 研究输出

所有 proxy report 均由 C++ `BacktestEngine`、`PeriodContributionReplaySink` 和
`estimate_empirical_cvar` 生成，125 个 period return，reference/fill 为 bar-close proxy。

| Challenger | Fold 1 return_CVaR | Fold 2 return_CVaR | Fold 3 return_CVaR | 状态 |
| --- | ---: | ---: | ---: | --- |
| PCGrad | -0.036858 | -0.027778 | -0.020448 | `PROXY`, no promotion |
| GradNorm-4 | -0.037149 | -0.028996 | -0.018756 | `PROXY`, no promotion |

Proxy artifacts：

- `cpp-proxy/pcgrad-fold-1.json`
- `cpp-proxy/pcgrad-fold-2.json`
- `cpp-proxy/pcgrad-fold-3.json`
- `cpp-proxy/gradnorm-fold-1.json`
- `cpp-proxy/gradnorm-fold-2.json`
- `cpp-proxy/gradnorm-fold-3.json`

统一 limitations：`REFERENCE_PRICE_PROXY_BAR_CLOSE`、无 commission/tax schedule、无 price-limit
字段、无 corporate-action adjustment、无 board-lot provenance、无 slippage provenance，以及共同股票
交集 proxy。C++ replay/CVaR 已有输出，但不等价于正式净收益或正式 economic gate。

## 7. Go/No-Go

| Gate | 结果 |
| --- | --- |
| 诊断不改变 baseline 数值 | PASS |
| PCGrad 三个新 purged OOS 窗口 | PASS |
| GradNorm-4 三个新 purged OOS 窗口 | PASS |
| 机制指标可重放 | PASS |
| 主模型指标报告 | PASS，两个 challenger 均有退化项 |
| C++ Replay/CVaR artifact | PASS，PROXY only |
| 正式费用/涨跌停/公司行动/reference gate | BLOCKED |
| Phase 1D real drift baseline/report | BLOCKED |
| 自动晋级/覆盖 `FROZEN_CHAMPION` | NO-GO |

最终决策：**Phase 1E 研究施工与实验报告完成；正式 Phase 1E exit 和 challenger promotion 不通过。**
`legacy + fixed` 继续作为唯一冻结冠军，PCGrad 与 GradNorm-4 仅保留为可审计研究 artifact。

## 8. Artifact 索引

- 诊断报告：`external/phase1e-diagnostics-real/diagnostic_report.json`
- PCGrad 报告：`pcgrad-real-e3/challenger_report.json`
- GradNorm-4 报告：`gradnorm4-real-e3/challenger_report.json`
- 数据审计：`external/phase1e_pit_120_audit.json`
- Proxy 回放工具：`source-tool-not-published`

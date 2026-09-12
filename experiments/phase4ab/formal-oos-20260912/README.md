# Phase 4A/4B 正式 OOS 归档（终局判定）

发布日期：2026-09-12
状态：**无 challenger 通过预注册 gate；incumbent（NCO risk-only）冻结**；
`formal_oos_complete = true`；Phase 4A/4B 按 `docs/phase4_ab_formal_oos_contract.md` 判定封闭

## 运行

- 数据：PIT 120-symbol 源表（119 symbols × 1587 决策点），预备产物 manifest SHA-256
  `5670b8a53935695d...`（scenarios 回看 / forward 仅度量，端点无效置零）。
- Walk-forward：估计窗 252、每 5 点调仓、266 期 OOS、0 跳步；视图降级阶梯占比 13.5%
  （<20% 合同上限）；cost 10 bps 单边。
- Driver：`portfolio_math/apps/formal_oos_driver.cpp`（fail-closed 全程，任何 arm 失败即整 run 作废），
  报告 SHA-256 `1462a08f9466260d06a431adfb982072e5130d79ce7159f5b1185ee3e642cc4e`。

## 结果（266 期净值口径）

| arm | 年化 Sharpe | 累计净收益 | 最大回撤 | 平均换手 |
|---|---:|---:|---:|---:|
| **risk_only（incumbent）** | **+0.483** | **+37.4%** | 21.9% | 24.0% |
| posterior_bl | −0.545 | −73.6% | 81.5% | 93.5% |
| posterior_ffv | −0.601 | −75.6% | 82.4% | 95.1% |
| nco_ffv | −0.006 | −6.6% | 30.8% | 83.6% |

预注册统计（block bootstrap 2000、block 10、seed 20260912；3 子窗口；BH-FDR；DSR）：

- 三组 challenger 对 incumbent 的 ΔSharpe 全部为负（−1.03 / −1.08 / −0.49），CI95 全负，
  子窗口同向 0/3，FDR 全拒，DSR≈0.002。
- **Q1（BL vs FFV）**：两后验的 Posterior Direct 均显著为负，FFV 略差（−0.60 vs −0.55）——
  两种后验引擎对该视图内容均无增益。
- **Q2（NCO-FFV vs Posterior Direct）**：nco_ffv 明显好于两个 direct arm（均值盲的 MinVar
  下游对视图内容不敏感），但仍不敌 incumbent。
- 机制解读：机械 20 点动量视图在周度调仓 + 10 bps 成本下产生 84-95% 换手，成本与噪声 views
  共同摧毁净值；incumbent 的协方差驱动 MinVar（24% 换手）是唯一正 Sharpe。

## 判定

```text
formal_oos_complete = true
winner              = incumbent (NCO risk-only, prior-covariance)
posterior_family    = NOT_PROMOTED（按合同 §5 冻结；重开需新 view 来源假设与新窗口）
phase_exit_eligible = true（对照已完成）；promotion_eligible = false（经济/生产 gate 独立）
```

## 目录

- `formal_oos_report_20260912.json`：完整报告（逐期收益、view tier、universe/clusters）。
- `docs/phase4_ab_formal_oos_contract.md`（仓库）：预注册合同（含三次运行前修订记录：
  decile 视图、收敛参数与 complete-linkage 切割 0.70、FFV 数值降级阶梯）。
- `tools/prepare_formal_oos_data.py` / `tools/verify_formal_oos.py`（仓库）：预备与统计校验。
- 工程注记：修复了 union-find 聚类节点空间（合并树内部节点 ≥n 导致的堆损坏）与前向收益
  端点掩码两处缺陷，均在有效产出之前修复并连同合同修订记录在案。

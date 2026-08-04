# Phase 3B 退出报告（当前版本）

## Gate

| 门槛 | 状态 | 证据 |
|---|---|---|
| `TAIL-EMPIRICAL-ES` 离散/带权 parity | PASS | `test_portfolio_math` |
| VaR 符号、平移和 return-CVaR 转换 | PASS | `test_portfolio_math` |
| VaR/ES 联合 backtest 输入 future guard | PASS | `backtest_tail_risk` |
| Kupiec exception diagnostics | PASS | 异常率、LR、p-value 和 hash |
| Christoffersen transition diagnostics | PASS | 00/01/10/11 转移计数、LR、p-value |
| Proxy artifact promotion gate | PASS | `reference_price_quality=PROXY` 强制 false |
| GARCH-FHS single-portfolio synthetic recovery | PASS | `test_portfolio_math`：GARCH 参数约束、残差诊断、确定性 residual replay |
| GARCH-FHS future/missing/unsupported-path guards | PASS | `test_portfolio_math`：future timestamp、NaN、未冻结 spec、asset-vector fail closed |
| GARCH-FHS asset-vector synchronized replay | NOT RUN | 当前只交付单组合 portfolio-return oracle |
| POT-GPD threshold/shape/finite-ES gate | PASS | `test_portfolio_math`：阈值、超额样本、shape guard、有限 ES、确定性 replay |
| POT-GPD asset-vector synchronized splice | NOT RUN | 当前只交付单组合 portfolio-return oracle |
| Direct Expectile single-portfolio training solve | PASS | `test_portfolio_math`：加权 ALS/二分、future/invalid mapping guard、artifact hash |
| Taylor-mapped Expectile-ES | NOT RUN | 映射规范尚未冻结，当前 fail closed |
| 三个 purged OOS tail-risk 窗口 | NOT RUN | 尚无正式 OOS 报告 |

## 判定

```json
{
  "engineering_scaffold_complete": true,
  "phase_exit_eligible": false,
  "promotion_eligible": false,
  "evidence_level": "RESEARCH_PROXY",
  "blocking_reason": "missing_expectile_factor_oos_and_production_data_evidence"
}
```

本报告确认经验固定组合尾部风险和单组合 GARCH-FHS 过滤/replay 合同已可重放；不声称已经完成资产向量相关性、极值理论、Expectile、三个 purged OOS 窗口或生产经济晋级。

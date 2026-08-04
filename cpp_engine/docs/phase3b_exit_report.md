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
| GARCH-FHS synthetic recovery | NOT RUN | 下一阶梯 |
| POT-GPD threshold/shape/ES gate | NOT RUN | FHS 单独通过后才允许 |
| Expectile calibration/Taylor mapping | NOT RUN | 独立候选 |
| 三个 purged OOS tail-risk 窗口 | NOT RUN | 尚无正式 OOS 报告 |

## 判定

```json
{
  "engineering_scaffold_complete": true,
  "phase_exit_eligible": false,
  "promotion_eligible": false,
  "evidence_level": "RESEARCH_PROXY",
  "blocking_reason": "missing_garch_evt_expectile_oos_evidence"
}
```

本报告只确认经验固定组合尾部风险和联合回测的可重放合同，不声称已经完成条件波动率、极值理论、Expectile 或生产经济晋级。

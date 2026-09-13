# Phase 3B 退出报告（当前版本）

## Gate

| 门槛 | 状态 | 证据 |
|---|---|---|
| `TAIL-EMPIRICAL-ES` 离散/带权 parity | PASS | `test_portfolio_math` |
| VaR 符号、平移和 return-CVaR 转换 | PASS | `test_portfolio_math` |
| VaR/ES 联合 backtest 输入 future guard | PASS | `backtest_tail_risk` |
| Kupiec exception diagnostics | PASS | 异常率、LR、p-value 和 hash |
| Christoffersen transition diagnostics | PASS | 00/01/10/11 转移计数、LR、p-value |
| Orientation-adapted FZ0 joint score | PASS | upper-loss 公式 golden；`ES<=0` domain fail closed；artifact hash/serialization |
| Phase 3B canonical preregistration | PASS | `runs/phase3b-preregistered/preregistered_contract.json`；SHA-256 `4a81f722465ac3e3124e6ad8d770a6305cc04dea408393b2a3308247f0267cf7`；self-check 通过 |
| Observed proxy / future OOS evidence separation | PASS | 三个旧 proxy 窗口只允许 validation/research；future new OOS 数组为空、不可用于 exit/promotion |
| Regression-based ESR robust suite | PASS | `test_tail_risk_esr`：strict/auxiliary/strict-intercept、kernel bread、empirical score、Newey-West HAC、correct-spec reference、hash 和 fail-closed |
| ESR robust observed-proxy diagnostic | PASS/NO-GO | 六份均计算成功；strict 4/6、auxiliary 5/6 在 5% 拒绝 empirical VaR/ES；只作 research diagnostic |
| ESR formal new-OOS inference | NOT RUN | robust 工程已完成，但注册后未来 OOS 仍为 0 个 |
| Proxy artifact promotion gate | PASS | `reference_price_quality=PROXY` 强制 false |
| GARCH-FHS single-portfolio synthetic recovery | PASS | `test_portfolio_math`：GARCH 参数约束、残差诊断、确定性 residual replay |
| GARCH-FHS future/missing/unsupported-path guards | PASS | `test_portfolio_math`：future timestamp、NaN、未冻结 spec、非同步 residual-row fail closed |
| GARCH-FHS asset-vector synchronized replay | PASS | N≤200 reference path；逐资产过滤、同历史 residual-row 重放、单组合 parity、逐资产 diagnostics 和确定性 hash |
| POT-GPD threshold/shape/finite-ES gate | PASS | `test_portfolio_math`：阈值、超额样本、shape guard、有限 ES、确定性 replay |
| POT-GPD asset-vector synchronized splice | PASS | 四点加权阈值网格、有效超额样本、稳定性、连续归一、GPD 极限、非同质相关资产 parity 和 artifact |
| Direct Expectile unconditional oracle | PASS | `test_portfolio_math`：加权标量二分、future/invalid mapping guard、artifact hash |
| Conditional Expectile fixed-PIT ALS | PASS | `test_conditional_expectile`：条件系数恢复、score/一阶条件、平移/正齐次、PIT future guard 和 replay |
| Conditional Expectile real PIT validation / CARE-SAV | NOT RUN | 当前只有合成 reference；没有真实 PIT validation，未声称 CARE-SAV |
| Taylor-mapped Expectile-ES | NOT RUN | 映射规范尚未冻结，当前 fail closed |
| 三个未来新 purged OOS tail-risk 窗口 | NOT RUN | 当前 `registered_windows=[]`；已有三个 proxy 窗口在注册前已观察，不能重标为 formal OOS |

## 判定

```json
{
  "engineering_scaffold_complete": true,
  "phase_exit_eligible": false,
  "promotion_eligible": false,
  "evidence_level": "RESEARCH_PROXY",
  "blocking_reason": "robust_proxy_esr_rejections_and_missing_expectile_coverage_taylor_factor_future_new_oos_and_production_evidence"
}
```

本报告确认经验固定组合尾部风险、N≤200 同步资产向量 GARCH-FHS/EVT、固定 PIT conditional
Expectile ALS 和 ESR robust covariance 工程可重放，并确认 observed/future evidence 分区可自校验。
旧 proxy robust ESR 多窗口拒绝 empirical baseline，且它们不是注册后的未来 OOS。当前不声称
CARE-SAV、Taylor mapping、full-A factor-specific FHS、三个未来新 purged OOS 窗口或生产经济晋级已经完成。

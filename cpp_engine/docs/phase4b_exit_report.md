# Phase 4B 退出报告（当前版本）

## Gate

| 门槛 | 状态 | 证据 |
|---|---|---|
| Posterior covariance 接入 NCO-MinVar | PASS/REFERENCE | solve_nco_ffv_minvar、test_nco_ffv |
| posterior / cluster provenance hash | PASS/REFERENCE | deterministic artifact hash 与 JSON serialization |
| cluster shape、future/status、options fail-closed | PASS/REFERENCE | test_nco_ffv |
| NCO permutation / cluster objective parity | INHERITED REFERENCE | 由既有 test_nco_policy 覆盖，尚无 FFV formal OOS |
| Posterior Direct vs NCO-FFV vs NCO risk-only | NOT RUN | 尚无注册后的 OOS |
| policy-return correlation / effective trial count | NOT RUN | governance diagnostics 尚未接入 |
| FDR / Deflated Sharpe / economic gate | NOT RUN | 生产晋级保持关闭 |

## 判定

{
  "engineering_scaffold_complete": true,
  "phase_exit_eligible": false,
  "promotion_eligible": false,
  "evidence_level": "REFERENCE_ONLY",
  "blocking_reason": "missing_policy_family_oos_governance_and_economic_evidence"
}

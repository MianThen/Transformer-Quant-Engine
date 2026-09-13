# Phase 4B 退出报告（当前版本）

## Gate

| 门槛 | 状态 | 证据 |
|---|---|---|
| Posterior covariance 接入 NCO-MinVar | PASS/REFERENCE | solve_nco_ffv_minvar、test_nco_ffv |
| posterior / cluster provenance hash | PASS/REFERENCE | deterministic artifact hash 与 JSON serialization |
| cluster shape、future/status、options fail-closed | PASS/REFERENCE | test_nco_ffv |
| NCO permutation / cluster objective parity | PASS/REFERENCE | `test_nco_policy` 覆盖 permutation；`FULL`、`INTRA_ONLY`、`INTER_ONLY` 单层消融由 `test_nco_policy` 与 `test_nco_ffv` 覆盖，FFV artifact hash/JSON 固化 mode provenance |
| Posterior Direct vs NCO-FFV vs NCO risk-only | PASS/REFERENCE | `compare_nco_policy_family` 与 `test_nco_policy_comparison`：三路 anchor 统一经过同一 reconciler；risk-only 使用 prior covariance；无 winner 自动选择 |
| policy-return correlation / effective trial count | PASS/REFERENCE | `PolicyFamilyGovernanceArtifactV1` + ONC/effective-trials 定向测试 |
| FDR / Deflated Sharpe | PASS/REFERENCE | BH/BY/Storey 与逐 policy DSR 已接入，尚无真实注册后 policy returns |
| Governance artifact full audit payload | PASS/REFERENCE | 序列化保存 policy IDs、correlation、ONC candidates/partition、FDR、逐策略 DSR 和 winner gate inputs；`test_policy_family_governance` |
| economic gate | NOT RUN | 生产晋级保持关闭 |
| CPU/C++ deterministic regression | PASS/REFERENCE | CLion bundled CMake 全量 CTest `46/46` 通过 |

## 判定

{
  "engineering_scaffold_complete": true,
  "phase_exit_eligible": false,
  "promotion_eligible": false,
  "evidence_level": "REFERENCE_ONLY",
  "blocking_reason": "missing_registered_policy_family_oos_and_economic_evidence"
}

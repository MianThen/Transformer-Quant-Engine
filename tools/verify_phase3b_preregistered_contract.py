#!/usr/bin/env python3
"""Verify the Phase 3B preregistration contract and observed proxy sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROLE = "phase3b_conditional_tail_risk_preregistered_contract"
LADDER = [
    "TAIL-EMPIRICAL-ES",
    "TAIL-GARCH-FHS-ES",
    "TAIL-GARCH-FHS-EVT-ES",
    "TAIL-EXPECTILE",
    "TAIL-EXPECTILE-ES",
    "TAIL-FACTOR-SPECIFIC-SYNCHRONIZED-FHS",
]
ALLOWED_OBSERVED_USES = ["VALIDATION", "RESEARCH_DIAGNOSTIC"]


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON 根节点必须是对象")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_report(repo_root: Path, source: dict[str, Any], window: dict[str, Any]) -> None:
    relative_path = source.get("path")
    _require(isinstance(relative_path, str) and relative_path.startswith("runs/cpp-proxy/"),
             "observed source path 必须位于 runs/cpp-proxy")
    report_path = repo_root / relative_path
    _require(report_path.is_file(), f"缺少 observed proxy report: {report_path}")
    report = _load_object(report_path)
    supplied_hash = report.get("report_sha256")
    unsigned_report = dict(report)
    unsigned_report.pop("report_sha256", None)
    _require(
        isinstance(supplied_hash, str)
        and len(supplied_hash) == 64
        and _canonical_hash(unsigned_report) == supplied_hash,
        f"{report_path}: report_sha256 校验失败",
    )
    _require(source.get("report_sha256") == supplied_hash,
             f"{report_path}: 合同 source hash 与报告不一致")
    _require(report.get("evidence_tier") == "RESEARCH_PROXY",
             f"{report_path}: evidence_tier 必须为 RESEARCH_PROXY")
    _require(report.get("economic_claim_scope") == "RESEARCH_PROXY_ONLY",
             f"{report_path}: economic_claim_scope 无效")
    _require(report.get("promotion_eligible") is False,
             f"{report_path}: proxy report 不得允许 promotion")
    _require(report.get("first_timestamp") == window.get("first_timestamp")
             and report.get("last_timestamp") == window.get("last_timestamp")
             and report.get("periods") == window.get("periods"),
             f"{report_path}: observed window 元数据不一致")


def verify(contract_path: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = _load_object(contract_path)
    repo_root = Path(__file__).resolve().parents[1]

    supplied_hash = contract.get("contract_sha256")
    unsigned_contract = dict(contract)
    unsigned_contract.pop("contract_sha256", None)
    _require(
        isinstance(supplied_hash, str)
        and len(supplied_hash) == 64
        and _canonical_hash(unsigned_contract) == supplied_hash,
        "contract_sha256 校验失败",
    )
    _require(contract.get("schema_version") == 1 and contract.get("role") == ROLE,
             "Phase 3B contract schema/role 无效")
    canonicalization = contract.get("canonicalization", {})
    _require(
        canonicalization == {
            "encoding": "UTF-8",
            "ensure_ascii": False,
            "excluded_root_fields": ["contract_sha256"],
            "json_separators": [",", ":"],
            "sort_keys": True,
        },
        "canonicalization 规则被修改",
    )
    _require(
        contract.get("registration_state")
        == "REGISTERED_AFTER_PROXY_WINDOWS_OBSERVED_BEFORE_FUTURE_NEW_OOS",
        "registration_state 未区分 observed proxy 与 future OOS",
    )

    claim_scope = contract.get("claim_scope", {})
    _require(claim_scope.get("economic_claim_scope") == "RESEARCH_PROXY_ONLY",
             "claim scope 必须为 RESEARCH_PROXY_ONLY")
    _require(claim_scope.get("phase_exit_eligible") is False
             and claim_scope.get("promotion_eligible") is False,
             "预注册本身不得关闭 phase exit/promotion gate")

    common_spec = contract.get("common_tail_risk_spec", {})
    _require(common_spec.get("confidence_level") == 0.95
             and common_spec.get("tail_probability") == 0.05,
             "alpha/tail probability 必须冻结为 0.95/0.05")
    _require(common_spec.get("forecast_horizon_periods") == 1,
             "首版 forecast horizon 必须为 1")

    ladder = contract.get("estimator_ladder")
    _require(isinstance(ladder, list) and len(ladder) == len(LADDER),
             "estimator ladder 长度无效")
    _require([item.get("order") for item in ladder] == list(range(1, len(LADDER) + 1)),
             "estimator ladder order 必须连续")
    _require([item.get("estimator_id") for item in ladder] == LADDER,
             "estimator ladder 顺序被修改")

    backtest_spec = contract.get("backtest_spec", {})
    fz0 = backtest_spec.get("fz0", {})
    _require(
        fz0.get("variant") == "FZ0_UPPER_TAIL_LOSS_V1"
        and fz0.get("formula")
        == "I(loss>var)*(loss-var)/((1-alpha)*es)+var/es+log(es)-1"
        and fz0.get("exception_tie_policy")
        == "STRICT_LOSS_GREATER_THAN_VAR_WITH_NUMERICAL_TOLERANCE"
        and fz0.get("domain")
        == (
            "expected_shortfall_loss > 0 and "
            "expected_shortfall_loss >= value_at_risk_loss"
        ),
        "FZ0 variant/formula/domain/tie policy 未冻结",
    )
    esr = backtest_spec.get("esr", {})
    _require(
        esr.get("joint_loss") == "FZ_G1_ZERO_G2_NEGATIVE_RECIPROCAL"
        and esr.get("covariance_estimator")
        == "KERNEL_BREAD_EMPIRICAL_SCORE_NEWEY_WEST_HAC_V1"
        and esr.get("reference_covariance_estimator")
        == "KERNEL_IND_CORRECT_SPEC_SANDWICH_V1"
        and esr.get("misspecification_robust") is True
        and esr.get("hac_kernel") == "BARTLETT"
        and esr.get("hac_lag") == 4
        and esr.get("score_centering") is True
        and esr.get("lower_tail_return_orientation") is True,
        "ESR joint loss/covariance/orientation 未冻结",
    )
    variants = esr.get("variants")
    _require(
        isinstance(variants, list)
        and [item.get("variant") for item in variants]
        == ["STRICT_ESR", "AUXILIARY_ESR", "STRICT_INTERCEPT_ESR"],
        "ESR strict/auxiliary/strict-intercept 顺序无效",
    )

    estimator_specs = contract.get("estimator_specs", {})
    pot_gpd = estimator_specs.get("pot_gpd", {})
    _require(
        pot_gpd.get("threshold_quantile_grid") == [0.75, 0.8, 0.85, 0.9]
        and pot_gpd.get("fit_method") == "WEIGHTED_MOMENTS_V1"
        and pot_gpd.get("minimum_exceedances") == 12
        and pot_gpd.get("minimum_effective_exceedances") == 12
        and pot_gpd.get("minimum_valid_grid_points") == 3
        and pot_gpd.get("maximum_shape_spread") == 0.35
        and pot_gpd.get("maximum_relative_es_spread") == 0.25
        and pot_gpd.get("selection_rule")
        == "LOWEST_QUANTILE_AMONG_VALID_STABLE_GRID_V1"
        and pot_gpd.get("threshold_selection_data") == "TRAIN_ONLY",
        "POT-GPD weighted grid/stability contract 未冻结",
    )
    expectile = estimator_specs.get("expectile", {})
    _require(
        expectile.get("model_kind") == "FIXED_PIT_LINEAR_ALS"
        and expectile.get("feature_set")
        == ["LAGGED_LOSS_1", "LAGGED_ABSOLUTE_LOSS_1", "FILTERED_VOLATILITY_1"]
        and expectile.get("availability_policy")
        == (
            "TRAIN_FEATURE_AVAILABLE_STRICTLY_BEFORE_TARGET_AND_"
            "FORECAST_FEATURE_AVAILABLE_AT_DECISION"
        ),
        "conditional Expectile PIT feature contract 未冻结",
    )

    partitions = contract.get("evidence_partitions", {})
    observed = partitions.get("observed_proxy_windows", {})
    _require(observed.get("status") == "OBSERVED_BEFORE_REGISTRATION",
             "proxy windows 必须标记为注册前已观察")
    _require(observed.get("allowed_uses") == ALLOWED_OBSERVED_USES,
             "observed proxy 只能用于 validation/research diagnostic")
    _require(observed.get("formal_oos_eligible") is False
             and observed.get("phase_exit_evidence_eligible") is False
             and observed.get("promotion_evidence_eligible") is False,
             "observed proxy 被错误升级为 OOS/exit/promotion 证据")
    windows = observed.get("windows")
    _require(isinstance(windows, list) and len(windows) == 3
             and observed.get("window_count") == 3,
             "observed proxy window 数必须为 3")
    previous_last = 0
    for expected_fold, window in enumerate(windows, start=1):
        _require(window.get("fold") == expected_fold, "observed fold 顺序无效")
        first = window.get("first_timestamp")
        last = window.get("last_timestamp")
        _require(isinstance(first, int) and isinstance(last, int)
                 and previous_last < first <= last,
                 "observed proxy timestamps 重叠或无效")
        previous_last = last
        sources = window.get("source_reports")
        _require(isinstance(sources, list) and len(sources) == 2,
                 "每个 observed window 必须同时登记 GradNorm/PCGrad report")
        for source in sources:
            _verify_report(repo_root, source, window)

    future = partitions.get("future_new_oos", {})
    _require(future.get("status") == "UNAVAILABLE",
             "future new OOS 当前必须为 UNAVAILABLE")
    _require(future.get("registered_windows") == []
             and future.get("window_count") == 0
             and future.get("minimum_required_windows") == 3,
             "future OOS 必须为空且最低要求为 3 个新窗口")
    _require(future.get("formal_oos_eligible") is False
             and future.get("phase_exit_evidence_eligible") is False
             and future.get("allowed_for_selection") is False
             and future.get("must_begin_after_contract_registration") is True,
             "不存在的 future OOS 不得关闭 gate")

    fail_closed = contract.get("fail_closed_policy", {})
    _require(fail_closed.get("runtime_fallback") is False
             and fail_closed.get("runtime_estimator_averaging") is False
             and fail_closed.get("failed_candidate_may_advance") is False,
             "fail-closed policy 被放宽")
    prohibited = set(fail_closed.get("disallowed_behaviors", []))
    _require(
        {
            "RUNTIME_ESTIMATOR_FALLBACK",
            "ESTIMATOR_AVERAGING",
            "POST_HOC_ALPHA_TAU_OR_THRESHOLD_SELECTION",
            "INDEPENDENT_PER_ASSET_RESIDUAL_RESAMPLING",
            "FUTURE_VOLATILITY_RESIDUAL_THRESHOLD_OR_REGIME",
            "DIRECT_EXPECTILE_LABELED_AS_VAR_ES_OR_CVAR",
            "PROXY_RESULT_LABELED_AS_PRODUCTION",
        }.issubset(prohibited),
        "fail-closed 禁止行为不完整",
    )

    self_check = contract.get("self_check", {})
    _require(self_check.get("expected_observed_proxy_windows") == 3
             and self_check.get("expected_future_oos_windows") == 0
             and self_check.get("expected_claim_scope") == "RESEARCH_PROXY_ONLY",
             "self_check 预期值无效")
    _require(self_check.get("verifier") == "tools/verify_phase3b_preregistered_contract.py",
             "self_check verifier 路径无效")

    return {
        "valid": True,
        "contract_sha256": supplied_hash,
        "observed_proxy_windows": len(windows),
        "future_new_oos_windows": 0,
        "economic_claim_scope": "RESEARCH_PROXY_ONLY",
        "phase_exit_eligible": False,
        "promotion_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("runs/phase3b-preregistered/preregistered_contract.json"),
    )
    args = parser.parse_args()
    result = verify(args.contract)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

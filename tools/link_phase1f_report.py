#!/usr/bin/env python3
"""Link a Phase 1F C++ proxy Replay/CVaR artifact into its training report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON 顶层必须是对象")
    return value


def _validate_proxy(proxy: dict[str, Any], path: Path) -> None:
    unsigned = dict(proxy)
    supplied = unsigned.pop("report_sha256", None)
    if not isinstance(supplied, str) or _canonical_hash(unsigned) != supplied:
        raise ValueError(f"{path}: C++ proxy report_sha256 校验失败")
    required = {
        "status": "proxy_replay_complete",
        "evidence_tier": "RESEARCH_PROXY",
        "economic_claim_scope": "RESEARCH_PROXY_ONLY",
        "reference_price_quality": "PROXY",
        "phase_exit_eligible": True,
        "research_comparison_eligible": True,
        "promotion_eligible": False,
        "cpp_replay_executed": True,
        "cvar_available": True,
    }
    for key, expected in required.items():
        if proxy.get(key) != expected:
            raise ValueError(f"{path}: {key} 不满足 Phase 1F proxy 合同")
    if not isinstance(proxy.get("period_returns"), list) or len(proxy["period_returns"]) < 20:
        raise ValueError(f"{path}: period return 样本不足")
    missing = proxy.get("missing_execution_data")
    expected_missing = {
        "fee_state": "UNAVAILABLE",
        "tax_state": "UNAVAILABLE",
        "limit_state": "UNAVAILABLE",
        "corporate_action_state": "UNAVAILABLE",
        "adjustment_state": "UNKNOWN",
        "lot_state": "UNAVAILABLE",
        "reference_quality": "PROXY",
        "slippage_state": "UNAVAILABLE",
    }
    if missing != expected_missing:
        raise ValueError(f"{path}: missing_execution_data 不符合冻结 proxy 状态")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--proxy-report", required=True, type=Path)
    args = parser.parse_args()

    training_path = args.training_report.resolve()
    proxy_path = args.proxy_report.resolve()
    report = _read_json(training_path)
    proxy = _read_json(proxy_path)
    _validate_proxy(proxy, proxy_path)
    if report.get("role") != "phase1f_topk_stability_training_report":
        raise ValueError(f"{training_path}: 不是 Phase 1F training report")
    if report.get("promotion_eligible") is not False:
        raise ValueError(f"{training_path}: promotion gate 必须保持关闭")

    report["status"] = "research_proxy_complete_production_gate_deferred"
    report["phase_exit_eligible"] = True
    report["economic_claim_scope"] = "RESEARCH_PROXY_ONLY"
    report["evidence_tier"] = "RESEARCH_PROXY"
    report["promotion_eligible"] = False
    report["cpp_economic_gate"] = {
        "status": "research_proxy_complete_production_gate_deferred",
        "evidence_tier": proxy["evidence_tier"],
        "economic_claim_scope": proxy["economic_claim_scope"],
        "phase_exit_eligible": proxy["phase_exit_eligible"],
        "research_comparison_eligible": proxy["research_comparison_eligible"],
        "cpp_replay_executed": proxy["cpp_replay_executed"],
        "cvar_available": proxy["cvar_available"],
        "promotion_eligible": False,
        "reference_price_quality": proxy["reference_price_quality"],
        "proxy_report_path": str(proxy_path),
        "proxy_report_sha256": proxy["report_sha256"],
        "ledger_sha256": proxy["ledger_sha256"],
        "periods": proxy["periods"],
        "symbols": proxy["symbols"],
        "return_cvar": proxy["cvar"].get("return_cvar"),
        "var_loss": proxy["cvar"].get("var_loss"),
        "expected_shortfall_loss": proxy["cvar"].get("expected_shortfall_loss"),
        "limitations": proxy["limitations"],
        "missing_execution_data": proxy["missing_execution_data"],
        "promotion_block_reason": proxy["promotion_block_reason"],
    }
    report.pop("report_sha256", None)
    report["report_sha256"] = _canonical_hash(report)
    training_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "training_report": str(training_path),
        "report_sha256": report["report_sha256"],
        "return_cvar": report["cpp_economic_gate"]["return_cvar"],
        "phase_exit_eligible": report["phase_exit_eligible"],
        "promotion_eligible": report["promotion_eligible"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

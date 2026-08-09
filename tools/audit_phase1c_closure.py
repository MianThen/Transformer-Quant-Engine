#!/usr/bin/env python3
"""Audit C++ Phase 1C Replay/CVaR proxy artifacts without recomputing returns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_LIMITATIONS = {
    "REFERENCE_PRICE_PROXY_BAR_CLOSE",
    "NO_COMMISSION_OR_TAX_SCHEDULE",
    "NO_PRICE_LIMIT_FIELDS",
    "NO_CORPORATE_ACTION_ADJUSTMENT_FIELDS",
    "NO_BOARD_LOT_PROVENANCE",
    "NO_SLIPPAGE_PROVENANCE",
}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"),
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: report 必须是对象")
    supplied = value.get("report_sha256")
    unsigned = dict(value)
    unsigned.pop("report_sha256", None)
    if not isinstance(supplied, str) or len(supplied) != 64 or _canonical_hash(unsigned) != supplied:
        raise ValueError(f"{path}: report_sha256 校验失败")
    if value.get("status") != "proxy_replay_complete":
        raise ValueError(f"{path}: status 不是 proxy_replay_complete")
    if value.get("evidence_tier") != "RESEARCH_PROXY":
        raise ValueError(f"{path}: evidence_tier 不是 RESEARCH_PROXY")
    if value.get("economic_claim_scope") != "RESEARCH_PROXY_ONLY":
        raise ValueError(f"{path}: economic_claim_scope 无效")
    if value.get("phase_exit_eligible") is not True or value.get(
            "research_comparison_eligible") is not True:
        raise ValueError(f"{path}: research proxy exit/comparison gate 无效")
    if value.get("cpp_replay_executed") is not True or value.get("cvar_available") is not True:
        raise ValueError(f"{path}: 缺少 C++ Replay/CVaR 输出")
    if value.get("reference_price_quality") != "PROXY" or value.get("promotion_eligible") is not False:
        raise ValueError(f"{path}: proxy promotion gate 未关闭")
    returns = value.get("period_returns")
    timestamps = value.get("period_end_timestamps")
    if not isinstance(returns, list) or not isinstance(timestamps, list) or len(returns) != len(timestamps):
        raise ValueError(f"{path}: period return/timestamp 长度不一致")
    limitations = value.get("limitations")
    if not isinstance(limitations, list) or not REQUIRED_LIMITATIONS.issubset(limitations):
        raise ValueError(f"{path}: 缺少正式经济字段 limitation")
    missing_data = value.get("missing_execution_data")
    expected_states = {
        "fee_state": "UNAVAILABLE",
        "tax_state": "UNAVAILABLE",
        "limit_state": "UNAVAILABLE",
        "corporate_action_state": "UNAVAILABLE",
        "adjustment_state": "UNKNOWN",
        "lot_state": "UNAVAILABLE",
        "reference_quality": "PROXY",
        "slippage_state": "UNAVAILABLE",
    }
    if missing_data != expected_states:
        raise ValueError(f"{path}: missing_execution_data 状态不符合冻结 proxy 合同")
    ledger = value.get("ledger_artifact")
    analysis = value.get("return_analysis_report")
    if not isinstance(ledger, dict) or ledger.get("manifest", {}).get("promotion_eligible") is not False:
        raise ValueError(f"{path}: ledger artifact promotion gate 无效")
    if not isinstance(analysis, dict) or analysis.get("manifest", {}).get("promotion_eligible") is not False:
        raise ValueError(f"{path}: return analysis promotion gate 无效")
    return value


def audit(reports: list[Path], output: Path) -> dict[str, Any]:
    if not reports:
        raise ValueError("没有找到 Phase 1C proxy report")
    summaries = []
    for path in reports:
        report = _read_report(path)
        summaries.append({
            "path": str(path.resolve()),
            "report_sha256": report["report_sha256"],
            "ledger_sha256": report["ledger_sha256"],
            "periods": len(report["period_returns"]),
            "symbols": report.get("symbols"),
            "first_timestamp": report.get("first_timestamp"),
            "last_timestamp": report.get("last_timestamp"),
            "reference_price_quality": report["reference_price_quality"],
            "evidence_tier": report["evidence_tier"],
            "phase_exit_eligible": report["phase_exit_eligible"],
            "research_comparison_eligible": report["research_comparison_eligible"],
            "promotion_eligible": report["promotion_eligible"],
        })
    result: dict[str, Any] = {
        "schema_version": 1,
        "role": "phase1c_closure_audit_v1",
        "status": "ENGINEERING_COMPLETE_RESEARCH_PROXY",
        "evidence_tier": "RESEARCH_PROXY",
        "reports": sorted(summaries, key=lambda item: item["path"]),
        "report_count": len(summaries),
        "cpp_replay_and_cvar_verified": True,
        "phase_exit_eligible": True,
        "research_comparison_eligible": True,
        "promotion_eligible": False,
        "formal_exit": True,
        "economic_promotion_exit": False,
        "economic_claim_scope": "RESEARCH_PROXY_ONLY",
        "blocking_reasons": [],
        "production_promotion_blockers": [
            "REFERENCE_PRICE_PROXY_BAR_CLOSE",
            "NO_COMMISSION_OR_TAX_SCHEDULE",
            "NO_PRICE_LIMIT_FIELDS",
            "NO_CORPORATE_ACTION_ADJUSTMENT_FIELDS",
            "NO_BOARD_LOT_PROVENANCE",
            "NO_SLIPPAGE_PROVENANCE",
            "NO_FORMAL_BENCHMARK_PROVENANCE",
        ],
        "missing_execution_data": {
            "fee_state": "UNAVAILABLE",
            "tax_state": "UNAVAILABLE",
            "limit_state": "UNAVAILABLE",
            "corporate_action_state": "UNAVAILABLE",
            "adjustment_state": "UNKNOWN",
            "lot_state": "UNAVAILABLE",
            "reference_quality": "PROXY",
            "slippage_state": "UNAVAILABLE",
        },
    }
    result["report_sha256"] = _canonical_hash(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="*", type=Path)
    parser.add_argument("--reports-dir", type=Path, default=Path("runs/cpp-proxy"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/phase1c-closure/phase1c_closure_report.json"))
    args = parser.parse_args()
    reports = args.reports or sorted(args.reports_dir.glob("*.json"))
    result = audit(reports, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "report_count": result["report_count"],
        "formal_exit": result["formal_exit"],
        "promotion_eligible": result["promotion_eligible"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Link C++ proxy Replay/CVaR artifacts into a challenger report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenger-report", required=True)
    parser.add_argument("--proxy-report", action="append", required=True)
    args = parser.parse_args()
    challenger_path = Path(args.challenger_report).resolve()
    report = json.loads(challenger_path.read_text(encoding="utf-8"))
    proxy_reports = []
    for value in args.proxy_report:
        path = Path(value).resolve()
        proxy = json.loads(path.read_text(encoding="utf-8"))
        if proxy.get("status") != "proxy_replay_complete":
            raise ValueError(f"proxy report 状态无效: {path}")
        if proxy.get("promotion_eligible") is not False:
            raise ValueError(f"proxy report promotion gate 无效: {path}")
        proxy_reports.append({
            "path": str(path),
            "report_sha256": proxy["report_sha256"],
            "ledger_sha256": proxy["ledger_sha256"],
            "periods": proxy["periods"],
            "symbols": proxy["symbols"],
            "return_cvar": proxy["cvar"].get("return_cvar"),
            "expected_shortfall_loss": proxy["cvar"].get("expected_shortfall_loss"),
            "reference_price_quality": proxy["reference_price_quality"],
            "promotion_eligible": proxy["promotion_eligible"],
        })
    if len(proxy_reports) != 3:
        raise ValueError("每个 challenger 必须链接三个 proxy fold")
    economic = report["economic_gate"]
    economic.update({
        "status": "proxy_replay_complete_awaiting_economic_gate",
        "cpp_replay_executed": True,
        "cvar_available": all(item["return_cvar"] is not None for item in proxy_reports),
        "promotion_eligible": False,
        "reference_price_quality": "PROXY",
        "proxy_reports": proxy_reports,
    })
    report.setdefault("references", {})["cpp_proxy_replay_cvar"] = {
        "present": True,
        "report_count": len(proxy_reports),
        "report_sha256": _canonical_hash(proxy_reports),
    }
    report["status"] = "challenger_report_with_proxy_economic_artifact_no_promotion"
    report.pop("report_sha256", None)
    report["report_sha256"] = _canonical_hash(report)
    challenger_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "challenger_report": str(challenger_path),
        "report_sha256": report["report_sha256"],
        "cvar_available": economic["cvar_available"],
        "promotion_eligible": economic["promotion_eligible"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

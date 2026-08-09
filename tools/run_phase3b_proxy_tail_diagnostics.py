#!/usr/bin/env python3
"""Run C++ expanding-history VaR/ES and ESR on observed proxy ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON 根节点必须是对象")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _verify_source(path: Path, report: dict[str, Any]) -> None:
    supplied_hash = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if not isinstance(supplied_hash, str) or _canonical_hash(unsigned) != supplied_hash:
        raise ValueError(f"{path}: report_sha256 校验失败")
    if (
        report.get("evidence_tier") != "RESEARCH_PROXY"
        or report.get("economic_claim_scope") != "RESEARCH_PROXY_ONLY"
        or report.get("promotion_eligible") is not False
    ):
        raise ValueError(f"{path}: 只能消费 RESEARCH_PROXY 报告")


def _verify_contract(path: Path) -> dict[str, Any]:
    contract = _load_object(path)
    supplied_hash = contract.get("contract_sha256")
    unsigned = dict(contract)
    unsigned.pop("contract_sha256", None)
    if not isinstance(supplied_hash, str) or _canonical_hash(unsigned) != supplied_hash:
        raise ValueError(f"{path}: contract_sha256 校验失败")
    future = contract.get("evidence_partitions", {}).get("future_new_oos", {})
    esr = contract.get("backtest_spec", {}).get("esr", {})
    if (
        contract.get("claim_scope", {}).get("economic_claim_scope")
        != "RESEARCH_PROXY_ONLY"
        or future.get("registered_windows") != []
        or esr.get("covariance_estimator")
        != "KERNEL_BREAD_EMPIRICAL_SCORE_NEWEY_WEST_HAC_V1"
        or esr.get("hac_lag") != 4
        or esr.get("misspecification_robust") is not True
    ):
        raise ValueError(f"{path}: Phase 3B robust proxy 合同无效")
    return contract


def _run_window(
    cpp_engine: Any,
    path: Path,
    report: dict[str, Any],
    *,
    confidence_level: float,
    minimum_history: int,
    config_hash: int,
) -> dict[str, Any]:
    returns = [float(value) for value in report.get("period_returns", [])]
    timestamps = [int(value) for value in report.get("period_end_timestamps", [])]
    if len(returns) != len(timestamps) or len(returns) <= minimum_history:
        raise ValueError(f"{path}: period return/timestamp 样本不足或未对齐")
    if any(not math.isfinite(value) for value in returns):
        raise ValueError(f"{path}: period returns 含非有限值")
    if any(
        timestamp <= 0
        or (index and timestamp <= timestamps[index - 1])
        for index, timestamp in enumerate(timestamps)
    ):
        raise ValueError(f"{path}: period timestamps 非严格递增")

    realized: list[float] = []
    realization_timestamps: list[int] = []
    value_at_risk: list[float] = []
    expected_shortfall: list[float] = []
    forecast_artifact_hashes: list[int] = []
    for forecast_index in range(minimum_history, len(returns)):
        forecast = cpp_engine.estimate_empirical_cvar(
            returns[:forecast_index],
            timestamps[:forecast_index],
            confidence_level,
            config_hash + forecast_index,
        )
        if int(forecast["status"]) != 0:
            raise RuntimeError(
                f"{path}: C++ empirical tail forecast failed at {forecast_index}"
            )
        current_var = forecast.get("var_loss")
        current_es = forecast.get("expected_shortfall_loss")
        if current_var is None or current_es is None:
            raise RuntimeError(f"{path}: C++ tail forecast 缺少 VaR/ES")
        realized.append(returns[forecast_index])
        realization_timestamps.append(timestamps[forecast_index])
        value_at_risk.append(float(current_var))
        expected_shortfall.append(float(current_es))
        forecast_artifact_hashes.append(int(forecast["artifact_hash"]))

    joint = _plain(cpp_engine.backtest_tail_risk(
        realized,
        realization_timestamps,
        value_at_risk,
        expected_shortfall,
        confidence_level,
        config_hash + 10_000,
    ))
    esr = _plain(cpp_engine.backtest_tail_risk_esr(
        realized,
        realization_timestamps,
        value_at_risk,
        expected_shortfall,
        confidence_level,
        40,
        6000,
        1e-8,
        config_hash + 20_001,
        config_hash + 20_002,
        config_hash + 20_003,
    ))
    return {
        "source_path": path.as_posix(),
        "source_report_sha256": report["report_sha256"],
        "fold": int(path.stem.rsplit("-", 1)[-1]),
        "candidate": path.stem.rsplit("-fold-", 1)[0],
        "source_periods": len(returns),
        "minimum_history_periods": minimum_history,
        "forecast_count": len(realized),
        "first_forecast_realization_timestamp": realization_timestamps[0],
        "last_forecast_realization_timestamp": realization_timestamps[-1],
        "forecast_values_sha256": _canonical_hash({
            "timestamps": realization_timestamps,
            "value_at_risk_loss": value_at_risk,
            "expected_shortfall_loss": expected_shortfall,
            "forecast_artifact_hashes": forecast_artifact_hashes,
        }),
        "joint_backtest": joint,
        "esr_backtest": esr,
        "diagnostic_complete": int(joint["status"]) == 0 and int(esr["status"]) == 0,
    }


def run(
    input_dir: Path,
    output_path: Path,
    contract_path: Path,
    *,
    confidence_level: float,
    minimum_history: int,
    config_hash: int,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"输出文件不可覆盖: {output_path}")
    if not 0.5 <= confidence_level < 1.0 or minimum_history < 40:
        raise ValueError("confidence_level/minimum_history 无效")
    contract = _verify_contract(contract_path)
    try:
        import cpp_engine
    except ImportError as exc:
        raise RuntimeError("需要启用 portfolio math 的 cpp_engine") from exc
    for name in (
        "estimate_empirical_cvar",
        "backtest_tail_risk",
        "backtest_tail_risk_esr",
    ):
        if not hasattr(cpp_engine, name):
            raise RuntimeError(f"cpp_engine 缺少 {name}")

    paths = sorted(input_dir.glob("*-fold-*.json"))
    if len(paths) != 6:
        raise ValueError(f"预期六份 proxy report，实际 {len(paths)}")
    windows = []
    for index, path in enumerate(paths):
        source = _load_object(path)
        _verify_source(path, source)
        windows.append(_run_window(
            cpp_engine,
            path,
            source,
            confidence_level=confidence_level,
            minimum_history=minimum_history,
            config_hash=config_hash + index * 100_000,
        ))
    report: dict[str, Any] = {
        "schema_version": 1,
        "role": "phase3b_observed_proxy_tail_risk_diagnostics",
        "status": "COMPLETE" if all(
            window["diagnostic_complete"] for window in windows
        ) else "DIAGNOSTIC_FAILURE",
        "estimator_id": "TAIL-EMPIRICAL-ES",
        "forecast_policy": "EXPANDING_HISTORY_STRICTLY_BEFORE_REALIZATION",
        "confidence_level": confidence_level,
        "minimum_history_periods": minimum_history,
        "window_count": len(windows),
        "windows": windows,
        "phase3b_contract_sha256": contract["contract_sha256"],
        "esr_covariance_method": (
            "KERNEL_BREAD_EMPIRICAL_SCORE_NEWEY_WEST_HAC_V1"
        ),
        "esr_hac_lag": 4,
        "registration_relation": "SOURCE_WINDOWS_OBSERVED_BEFORE_PHASE3B_REGISTRATION",
        "allowed_use": ["VALIDATION", "RESEARCH_DIAGNOSTIC"],
        "formal_oos_eligible": False,
        "phase_exit_evidence_eligible": False,
        "promotion_eligible": False,
        "limitations": [
            "OBSERVED_PROXY_WINDOWS",
            "EXPANDING_EMPIRICAL_BASELINE_ONLY",
            "NO_REAL_FEES_OR_EXECUTION_REFERENCE_PROVENANCE",
            "NO_REGISTERED_FUTURE_OOS_FOR_FORMAL_ESR_CLAIM",
        ],
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("runs/cpp-proxy"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/phase3b-proxy-diagnostics/report.json"),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("runs/phase3b-preregistered/preregistered_contract.json"),
    )
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-history", type=int, default=40)
    parser.add_argument("--config-hash", type=int, default=2026080501)
    args = parser.parse_args()
    report = run(
        args.input_dir.resolve(),
        args.output.resolve(),
        args.contract.resolve(),
        confidence_level=args.confidence_level,
        minimum_history=args.minimum_history,
        config_hash=args.config_hash,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": report["status"],
        "window_count": report["window_count"],
        "report_sha256": report["report_sha256"],
        "formal_oos_eligible": report["formal_oos_eligible"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..data import BAR_V1_FEATURE_GROUPS


PASS = "PASS"
FAIL = "FAIL"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def _read(path: str | Path | None, label: str, missing: list[str]) -> dict | None:
    if not path or "CONFIGURE_" in str(path):
        missing.append(label)
        return None
    source = Path(path)
    if not source.is_file():
        missing.append(f"{label}: {source}")
        return None
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 {label}: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} 顶层必须是 JSON object")
    return value


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _check(code: str, passed: bool, actual, expected) -> dict:
    return {
        "code": code,
        "status": PASS if passed else FAIL,
        "actual": actual,
        "expected": expected,
    }


def _gate(checks: list[dict], missing: list[str]) -> dict:
    if any(item["status"] == FAIL for item in checks):
        status = FAIL
    elif missing:
        status = INSUFFICIENT
    else:
        status = PASS
    return {"status": status, "checks": checks, "missing_evidence": sorted(set(missing))}


def _runtime_p99(report: dict, batch_size: int) -> int | None:
    for item in report.get("results", []):
        if int(item.get("batch_size", -1)) == batch_size:
            value = item.get("end_to_end", {}).get("p99_ns")
            return int(value) if isinstance(value, (int, float)) else None
    return None


def _software_gate(config: dict, evidence: dict) -> dict:
    missing: list[str] = []
    checks: list[dict] = []
    transformer = evidence.get("transformer_manifest")
    leakage = evidence.get("leakage_reports")
    parity = evidence.get("ort_cpp_parity")
    runtime = evidence.get("runtime_benchmark")
    portfolio = evidence.get("portfolio_backtest")

    if transformer is None:
        missing.append("Transformer experiment manifest")
    else:
        checks.append(_check(
            "transformer_model_family",
            transformer.get("model_family") == "TemporalTransformerV1.1",
            transformer.get("model_family"), "TemporalTransformerV1.1",
        ))
    if leakage is None:
        missing.append("per-window leakage reports")
    else:
        failed = [item.get("window_id") for item in leakage if item.get("status") != PASS]
        checks.append(_check("leakage", not failed, failed, []))
        expected_windows = None if transformer is None else transformer.get("windows")
        if not isinstance(expected_windows, int):
            missing.append("Transformer manifest windows")
        else:
            checks.append(_check(
                "leakage_window_coverage", len(leakage) == expected_windows,
                len(leakage), expected_windows,
            ))
    if parity is None:
        missing.append("ORT C++ parity report")
    else:
        checks.append(_check("ort_cpp_parity", parity.get("status") == PASS,
                             parity.get("status"), PASS))
    if runtime is None:
        missing.append("C++ runtime benchmark")
    else:
        checks.extend([
            _check("runtime_release", runtime.get("build_type") == "Release",
                   runtime.get("build_type"), "Release"),
            _check("runtime_lto", runtime.get("lto_enabled") is True,
                   runtime.get("lto_enabled"), True),
        ])
        batch_size = int(config["runtime_batch_size"])
        p99 = _runtime_p99(runtime, batch_size)
        if p99 is None:
            missing.append(f"runtime batch={batch_size} end_to_end.p99_ns")
        else:
            checks.append(_check("runtime_p99", p99 <= int(config["max_runtime_p99_ns"]),
                                 p99, f"<= {int(config['max_runtime_p99_ns'])}"))
        if "benchmark_scope" not in runtime:
            missing.append("runtime benchmark_scope=production_candidate")
        else:
            checks.append(_check(
                "runtime_candidate_scope",
                runtime.get("benchmark_scope") == "production_candidate",
                runtime.get("benchmark_scope"), "production_candidate",
            ))
        if transformer is not None:
            if "training_dataset_fingerprint" not in runtime:
                missing.append("runtime training_dataset_fingerprint")
            else:
                checks.append(_check(
                    "runtime_dataset_fingerprint",
                    runtime.get("training_dataset_fingerprint")
                    == transformer.get("dataset_fingerprint"),
                    runtime.get("training_dataset_fingerprint"),
                    transformer.get("dataset_fingerprint"),
                ))
        if parity is not None:
            for field in ("model_id", "model_version"):
                if field not in parity or field not in runtime:
                    missing.append(f"parity/runtime {field}")
                else:
                    checks.append(_check(
                        f"parity_runtime_{field}",
                        parity[field] == runtime[field], parity[field], runtime[field],
                    ))
    if portfolio is None:
        missing.append("C++ net-of-cost portfolio backtest")
    else:
        cost_model = portfolio.get("cost_model", {})
        execution_eligible = (
            portfolio.get("execution_reference_mode") == "required_for_promotion"
            and portfolio.get("promotion_eligible") is True
        )
        if not execution_eligible:
            missing.append("promotion-eligible PIT execution reference")
        checks.extend([
            _check("portfolio_schema", portfolio.get("schema_version") == 2,
                   portfolio.get("schema_version"), 2),
            _check("portfolio_engine", portfolio.get("engine") == "cpp",
                   portfolio.get("engine"), "cpp"),
            _check("portfolio_net_of_cost", portfolio.get("net_of_cost") is True,
                   portfolio.get("net_of_cost"), True),
            _check("point_in_time_fees",
                   cost_model.get("point_in_time_fees") is True,
                   cost_model.get("point_in_time_fees"), True),
            _check("t_plus_one", cost_model.get("enforce_t_plus_one") is True,
                   cost_model.get("enforce_t_plus_one"), True),
            _check("volume_participation",
                   isinstance(cost_model.get("max_volume_participation"), (int, float))
                   and cost_model["max_volume_participation"] <= 0.10,
                   cost_model.get("max_volume_participation"), "<= 0.10"),
            _check(
                "dataset_fingerprint_match",
                transformer is not None and portfolio.get("dataset_fingerprint")
                == transformer.get("dataset_fingerprint"),
                portfolio.get("dataset_fingerprint"),
                None if transformer is None else transformer.get("dataset_fingerprint"),
            ),
        ])
        if execution_eligible:
            checks.extend([
                _check("price_limits", cost_model.get("enforce_price_limits") is True,
                       cost_model.get("enforce_price_limits"), True),
                _check("board_lots", cost_model.get("enforce_board_lot") is True,
                       cost_model.get("enforce_board_lot"), True),
            ])
    return _gate(checks, missing)


def _positive_concentration(values: dict) -> tuple[float, float]:
    positive = sorted((float(value) for value in values.values() if float(value) > 0),
                      reverse=True)
    total = sum(positive)
    if total <= 0:
        return 1.0, 1.0
    return positive[0] / total, sum(positive[:5]) / total


def _portfolio_analysis(report: dict, config: dict) -> tuple[dict | None, list[str]]:
    missing: list[str] = []
    required_scenarios = {float(value) for value in config["required_slippage_bps"]}
    scenarios = {float(item.get("slippage_bps")): item for item in report.get("scenarios", [])}
    absent = sorted(required_scenarios - set(scenarios))
    if absent:
        missing.append(f"slippage scenarios: {absent}")
    for slippage in sorted(required_scenarios & set(scenarios)):
        scenario_windows = scenarios[slippage].get("windows", [])
        if len(scenario_windows) < int(config["minimum_windows"]):
            missing.append(
                f"slippage {slippage:g} bp windows: "
                f"{len(scenario_windows)} < {int(config['minimum_windows'])}"
            )
    primary = scenarios.get(float(config["primary_slippage_bps"]))
    if primary is None:
        missing.append("primary slippage scenario")
        return None, missing
    windows = primary.get("windows", [])
    candidate = config["candidate_model"]
    required_models = [candidate, *config["required_baselines"]]
    records = []
    baseline_sharpes = {name: [] for name in config["required_baselines"]}
    baseline_returns = {name: [] for name in config["required_baselines"]}
    all_symbols: dict[str, float] = {}
    all_industries: dict[str, float] = {}
    intervals = []
    for window in windows:
        models = window.get("models", {})
        absent_models = [name for name in required_models if name not in models]
        if absent_models:
            missing.append(f"window {window.get('window_id')} models: {absent_models}")
            continue
        start, end = window.get("test_start"), window.get("test_end")
        if start is None or end is None:
            missing.append(f"window {window.get('window_id')} test interval")
        else:
            intervals.append((start, end, window.get("window_id")))
        metrics = models[candidate]
        required_metrics = {"net_return", "sharpe", "max_drawdown", "turnover", "cvar_95"}
        absent_metrics = sorted(required_metrics - set(metrics))
        if absent_metrics:
            missing.append(f"window {window.get('window_id')} metrics: {absent_metrics}")
            continue
        baseline_name = max(config["required_baselines"],
                            key=lambda name: float(models[name]["net_return"]))
        baseline = models[baseline_name]
        for name in config["required_baselines"]:
            for metric in ("net_return", "sharpe"):
                if metric not in models[name]:
                    missing.append(
                        f"window {window.get('window_id')} baseline {name} {metric}"
                    )
            if "net_return" in models[name] and "sharpe" in models[name]:
                baseline_returns[name].append(float(models[name]["net_return"]))
                baseline_sharpes[name].append(float(models[name]["sharpe"]))
        symbol_values = metrics.get("symbol_contributions")
        industry_values = metrics.get("industry_contributions")
        if not isinstance(symbol_values, dict) or not symbol_values:
            missing.append(f"window {window.get('window_id')} symbol contributions")
            symbol_values = {}
        if not isinstance(industry_values, dict) or not industry_values:
            missing.append(f"window {window.get('window_id')} industry contributions")
            industry_values = {}
        for name, value in symbol_values.items():
            all_symbols[name] = all_symbols.get(name, 0.0) + float(value)
        for name, value in industry_values.items():
            all_industries[name] = all_industries.get(name, 0.0) + float(value)
        records.append({
            "window_id": window.get("window_id"),
            **{name: float(metrics[name]) for name in required_metrics},
            "strongest_baseline": baseline_name,
            "baseline_net_return": float(baseline["net_return"]),
            "baseline_sharpe": float(baseline["sharpe"]),
            "net_return_delta": float(metrics["net_return"] - baseline["net_return"]),
        })
    sorted_intervals = sorted(intervals)
    if any(current[0] <= previous[1] for previous, current in zip(
            sorted_intervals, sorted_intervals[1:])):
        missing.append("non-overlapping test windows")
    if not records:
        return None, missing
    period_top1, _ = _positive_concentration({
        str(item["window_id"]): item["net_return"] for item in records
    })
    symbol_top1, symbol_top5 = _positive_concentration(all_symbols)
    industry_top1, _ = _positive_concentration(all_industries)
    complete_baselines = [
        name for name in config["required_baselines"]
        if len(baseline_returns[name]) == len(records)
    ]
    if not complete_baselines:
        missing.append("complete baseline portfolio metrics")
        return None, missing
    strongest_return_baseline = max(
        complete_baselines, key=lambda name: np.mean(baseline_returns[name])
    )
    strongest_sharpe_baseline = max(
        complete_baselines, key=lambda name: np.mean(baseline_sharpes[name])
    )
    return {
        "windows": records,
        "aggregate": {
            "window_count": len(records),
            "mean_net_return": float(np.mean([item["net_return"] for item in records])),
            "mean_sharpe": float(np.mean([item["sharpe"] for item in records])),
            "strongest_return_baseline": strongest_return_baseline,
            "strongest_mean_baseline_return": float(np.mean(
                baseline_returns[strongest_return_baseline]
            )),
            "strongest_sharpe_baseline": strongest_sharpe_baseline,
            "strongest_mean_baseline_sharpe": float(np.mean(
                baseline_sharpes[strongest_sharpe_baseline]
            )),
            "worst_max_drawdown": max(item["max_drawdown"] for item in records),
            "worst_cvar_95": min(item["cvar_95"] for item in records),
            "winning_windows": sum(item["net_return_delta"] > 0 for item in records),
            "net_return_std": float(np.std([item["net_return"] for item in records])),
        },
        "concentration": {
            "period_top1_positive_share": period_top1,
            "symbol_top1_positive_share": symbol_top1,
            "symbol_top5_positive_share": symbol_top5,
            "industry_top1_positive_share": industry_top1,
        },
    }, missing


def _quality_gate(config: dict, evidence: dict) -> tuple[dict, dict | None]:
    missing: list[str] = []
    checks: list[dict] = []
    benchmark = evidence.get("benchmark_quality")
    benchmark_manifest = evidence.get("benchmark_manifest")
    ablation = evidence.get("ablation_manifest")
    portfolio_ablation = evidence.get("portfolio_ablation")
    portfolio = evidence.get("portfolio_backtest")
    candidate = config["candidate_model"]

    candidate_records = [] if benchmark is None else [
        item for item in benchmark.get("records", []) if item.get("model") == candidate
    ]
    if not candidate_records:
        missing.append("Transformer model-quality window records")
    else:
        rank_ic = [float(item["rank_ic"]) for item in candidate_records if "rank_ic" in item]
        if len(rank_ic) != len(candidate_records):
            missing.append("per-window Transformer RankIC")
        else:
            checks.extend([
                _check("rank_ic_median", float(np.median(rank_ic)) > config["min_rank_ic_median"],
                       float(np.median(rank_ic)), f"> {config['min_rank_ic_median']}"),
                _check("rank_ic_positive_majority", sum(value > 0 for value in rank_ic) > len(rank_ic) / 2,
                       sum(value > 0 for value in rank_ic), f"> {len(rank_ic) / 2}"),
            ])
    if benchmark_manifest is None:
        missing.append("model benchmark manifest")
    else:
        required = {candidate, *config["required_baselines"]}
        absent = sorted(required - set(benchmark_manifest.get("evaluated_models", [])))
        if absent:
            missing.append(f"evaluated benchmark models: {absent}")
        transformer = evidence.get("transformer_manifest")
        if transformer is not None:
            checks.append(_check(
                "benchmark_dataset_fingerprint",
                benchmark_manifest.get("dataset_fingerprint")
                == transformer.get("dataset_fingerprint"),
                benchmark_manifest.get("dataset_fingerprint"),
                transformer.get("dataset_fingerprint"),
            ))
    if ablation is None:
        missing.append("Transformer feature-ablation manifest")
    else:
        groups = set(ablation.get("groups", []))
        checks.extend([
            _check("ablation_model", ablation.get("model_family") == "TemporalTransformerV1.1",
                   ablation.get("model_family"), "TemporalTransformerV1.1"),
            _check("ablation_formal_conclusion",
                   ablation.get("formal_transformer_conclusion") is True,
                   ablation.get("formal_transformer_conclusion"), True),
            _check("ablation_seeds", len(set(ablation.get("seeds", []))) >= 3,
                   len(set(ablation.get("seeds", []))), ">= 3"),
        ])
        absent = sorted(set(BAR_V1_FEATURE_GROUPS) - groups)
        if absent:
            missing.append(f"ablation feature groups: {absent}")
        transformer = evidence.get("transformer_manifest")
        if transformer is not None:
            checks.append(_check(
                "ablation_dataset_fingerprint",
                ablation.get("dataset_fingerprint") == transformer.get("dataset_fingerprint"),
                ablation.get("dataset_fingerprint"),
                transformer.get("dataset_fingerprint"),
            ))
    if portfolio_ablation is None:
        missing.append("C++ net-of-cost portfolio ablation")
    else:
        checks.extend([
            _check("portfolio_ablation_schema",
                   portfolio_ablation.get("schema_version") == 2,
                   portfolio_ablation.get("schema_version"), 2),
            _check("portfolio_ablation_engine", portfolio_ablation.get("engine") == "cpp",
                   portfolio_ablation.get("engine"), "cpp"),
            _check("portfolio_ablation_net_of_cost", portfolio_ablation.get("net_of_cost") is True,
                   portfolio_ablation.get("net_of_cost"), True),
            _check("portfolio_ablation_seeds",
                   len(set(portfolio_ablation.get("seeds", []))) >= 3,
                   len(set(portfolio_ablation.get("seeds", []))), ">= 3"),
        ])
        if not (
            portfolio_ablation.get("execution_reference_mode") == "required_for_promotion"
            and portfolio_ablation.get("promotion_eligible") is True
        ):
            missing.append("promotion-eligible portfolio ablation execution reference")
        absent = sorted(set(BAR_V1_FEATURE_GROUPS) - set(portfolio_ablation.get("groups", [])))
        if absent:
            missing.append(f"portfolio ablation feature groups: {absent}")
        scenarios = portfolio_ablation.get("slippage_scenarios_bps")
        checks.append(_check(
            "portfolio_ablation_slippage_scenarios",
            scenarios == [0.0, 5.0, 10.0] or scenarios == [0, 5, 10],
            scenarios, [0, 5, 10],
        ))
        expected_pairs = {
            (seed, group)
            for seed in portfolio_ablation.get("seeds", [])
            for group in BAR_V1_FEATURE_GROUPS
        }
        results = portfolio_ablation.get("results", [])
        actual_pairs = {
            (item.get("seed"), item.get("group")) for item in results
            if isinstance(item, dict)
        }
        checks.append(_check(
            "portfolio_ablation_seed_group_coverage",
            actual_pairs == expected_pairs,
            len(actual_pairs), len(expected_pairs),
        ))
        complete_results = all(
            len(item.get("scenarios", [])) == 3
            and all(len(scenario.get("windows", [])) >= 3
                    for scenario in item.get("scenarios", []))
            for item in results if isinstance(item, dict)
        ) and bool(results)
        checks.append(_check(
            "portfolio_ablation_window_coverage",
            complete_results, complete_results, True,
        ))
        transformer = evidence.get("transformer_manifest")
        if transformer is not None:
            checks.append(_check(
                "portfolio_ablation_dataset_fingerprint",
                portfolio_ablation.get("dataset_fingerprint")
                == transformer.get("dataset_fingerprint"),
                portfolio_ablation.get("dataset_fingerprint"),
                transformer.get("dataset_fingerprint"),
            ))

    analysis = None
    if portfolio is None:
        missing.append("C++ net-of-cost portfolio backtest")
    else:
        analysis, portfolio_missing = _portfolio_analysis(portfolio, config)
        missing.extend(portfolio_missing)
    if analysis is not None:
        aggregate = analysis["aggregate"]
        concentration = analysis["concentration"]
        minimum_wins = int(np.ceil(aggregate["window_count"] * config["min_winning_window_fraction"]))
        baseline_sharpe = aggregate["strongest_mean_baseline_sharpe"]
        required_sharpe = max(config["min_net_sharpe"], baseline_sharpe * (1 + config["min_sharpe_improvement"]))
        checks.extend([
            _check("walk_forward_windows", aggregate["window_count"] >= config["minimum_windows"],
                   aggregate["window_count"], f">= {config['minimum_windows']}"),
            _check("net_sharpe", aggregate["mean_sharpe"] > required_sharpe,
                   aggregate["mean_sharpe"], f"> {required_sharpe}"),
            _check("mean_net_return_vs_baseline",
                   aggregate["mean_net_return"] > aggregate["strongest_mean_baseline_return"],
                   aggregate["mean_net_return"],
                   f"> {aggregate['strongest_mean_baseline_return']}"),
            _check("max_drawdown", aggregate["worst_max_drawdown"] <= config["max_drawdown"],
                   aggregate["worst_max_drawdown"], f"<= {config['max_drawdown']}"),
            _check("tail_risk", aggregate["worst_cvar_95"] >= config["min_cvar_95"],
                   aggregate["worst_cvar_95"], f">= {config['min_cvar_95']}"),
            _check("winning_windows", aggregate["winning_windows"] >= minimum_wins,
                   aggregate["winning_windows"], f">= {minimum_wins}"),
            _check("period_concentration", concentration["period_top1_positive_share"] <= config["max_period_top1_share"],
                   concentration["period_top1_positive_share"], f"<= {config['max_period_top1_share']}"),
            _check("symbol_top1_concentration", concentration["symbol_top1_positive_share"] <= config["max_symbol_top1_share"],
                   concentration["symbol_top1_positive_share"], f"<= {config['max_symbol_top1_share']}"),
            _check("symbol_top5_concentration", concentration["symbol_top5_positive_share"] <= config["max_symbol_top5_share"],
                   concentration["symbol_top5_positive_share"], f"<= {config['max_symbol_top5_share']}"),
            _check("industry_concentration", concentration["industry_top1_positive_share"] <= config["max_industry_top1_share"],
                   concentration["industry_top1_positive_share"], f"<= {config['max_industry_top1_share']}"),
        ])
        if candidate_records:
            benchmark_windows = {item.get("window_id") for item in candidate_records}
            portfolio_windows = {item.get("window_id") for item in analysis["windows"]}
            checks.append(_check(
                "benchmark_portfolio_windows",
                benchmark_windows == portfolio_windows,
                sorted(benchmark_windows, key=str), sorted(portfolio_windows, key=str),
            ))
    return _gate(checks, missing), analysis


def _validate_thresholds(thresholds: dict) -> None:
    numeric = {
        "minimum_windows", "min_rank_ic_median", "min_net_sharpe",
        "min_sharpe_improvement", "max_drawdown", "min_cvar_95",
        "min_winning_window_fraction", "max_period_top1_share",
        "max_symbol_top1_share", "max_symbol_top5_share",
        "max_industry_top1_share", "primary_slippage_bps",
        "runtime_batch_size", "max_runtime_p99_ns",
    }
    invalid = sorted(
        name for name in numeric
        if isinstance(thresholds.get(name), bool)
        or not isinstance(thresholds.get(name), (int, float))
        or not np.isfinite(thresholds[name])
    )
    if invalid:
        raise ValueError(
            "Promotion Review 门槛必须先冻结为有限数值: " + ", ".join(invalid)
        )
    baselines = thresholds.get("required_baselines")
    if (not isinstance(baselines, list) or not baselines
            or len(set(baselines)) != len(baselines)):
        raise ValueError("required_baselines 必须是非空且不重复的数组")
    if thresholds.get("candidate_model") in baselines:
        raise ValueError("candidate_model 不能同时出现在 required_baselines")
    scenarios = thresholds.get("required_slippage_bps")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("required_slippage_bps 必须是非空数组")
    if thresholds["primary_slippage_bps"] not in scenarios:
        raise ValueError("primary_slippage_bps 必须属于 required_slippage_bps")
    for name in (
        "min_winning_window_fraction", "max_period_top1_share",
        "max_symbol_top1_share", "max_symbol_top5_share",
        "max_industry_top1_share",
    ):
        if not 0 <= thresholds[name] <= 1:
            raise ValueError(f"{name} 必须在 [0, 1] 内")
    if thresholds["minimum_windows"] < 3 or thresholds["runtime_batch_size"] <= 0:
        raise ValueError("minimum_windows 至少为 3，runtime_batch_size 必须为正数")
    if thresholds["max_runtime_p99_ns"] <= 0:
        raise ValueError("max_runtime_p99_ns 必须为正数")


def _evidence(config: dict) -> dict:
    paths = config.get("evidence", {})
    missing: list[str] = []
    transformer_root = paths.get("transformer_run")
    transformer_manifest = _read(
        None if not transformer_root else Path(transformer_root) / "experiment_manifest.json",
        "transformer_run/experiment_manifest.json", missing,
    )
    leakage_reports = None
    if transformer_root and "CONFIGURE_" not in str(transformer_root):
        roots = sorted(Path(transformer_root).glob("window_*/leakage_report.json"))
        if roots:
            leakage_reports = [{"window_id": path.parent.name, **json.loads(path.read_text())}
                               for path in roots]
    benchmark_root = paths.get("model_benchmark")
    ablation_root = paths.get("feature_ablation")
    values = {
        "transformer_manifest": transformer_manifest,
        "leakage_reports": leakage_reports,
        "benchmark_quality": _read(None if not benchmark_root else Path(benchmark_root) / "model_quality.json",
                                   "model_benchmark/model_quality.json", missing),
        "benchmark_manifest": _read(None if not benchmark_root else Path(benchmark_root) / "benchmark_manifest.json",
                                    "model_benchmark/benchmark_manifest.json", missing),
        "ablation_manifest": _read(None if not ablation_root else Path(ablation_root) / "experiment_manifest.json",
                                   "feature_ablation/experiment_manifest.json", missing),
        "portfolio_ablation": _read(paths.get("portfolio_ablation"), "portfolio_ablation", missing),
        "portfolio_backtest": _read(paths.get("portfolio_backtest"), "portfolio_backtest", missing),
        "ort_cpp_parity": _read(paths.get("ort_cpp_parity"), "ort_cpp_parity", missing),
        "runtime_benchmark": _read(paths.get("runtime_benchmark"), "runtime_benchmark", missing),
    }
    values["unreadable_evidence"] = missing
    return values


def run_promotion_review(config: dict, output_path: str | Path) -> Path:
    if config.get("enabled") is not True:
        raise RuntimeError("Promotion Review 配置必须显式设置 enabled=true")
    required = {
        "candidate_model", "required_baselines", "minimum_windows",
        "min_rank_ic_median", "min_net_sharpe", "min_sharpe_improvement",
        "max_drawdown", "min_cvar_95", "min_winning_window_fraction",
        "max_period_top1_share", "max_symbol_top1_share", "max_symbol_top5_share",
        "max_industry_top1_share", "required_slippage_bps", "primary_slippage_bps",
        "runtime_batch_size", "max_runtime_p99_ns",
    }
    absent = sorted(required - set(config.get("thresholds", {})))
    if absent:
        raise ValueError("Promotion Review 缺少显式门槛: " + ", ".join(absent))
    thresholds = config["thresholds"]
    _validate_thresholds(thresholds)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    evidence = _evidence(config)
    software = _software_gate(thresholds, evidence)
    quality, analysis = _quality_gate(thresholds, evidence)
    if FAIL in {software["status"], quality["status"]}:
        decision = "REJECT"
    elif INSUFFICIENT in {software["status"], quality["status"]}:
        decision = INSUFFICIENT
    else:
        decision = "PROMOTE"
    generated = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 1,
        "generated_at_utc": generated,
        "decision": decision,
        "software_gate": software["status"],
        "model_quality_gate": quality["status"],
        "thresholds": thresholds,
    }
    _write(output / "software_gate.json", software)
    _write(output / "model_quality_gate.json", quality)
    _write(output / "promotion_report.json", report)
    if analysis is not None:
        _write(output / "multi_window_portfolio.json", {
            "windows": analysis["windows"], "aggregate": analysis["aggregate"]})
        _write(output / "concentration_stability.json", {
            "concentration": analysis["concentration"],
            "stability": analysis["aggregate"],
        })
    _write(output / "evidence_manifest.json", {
        "generated_at_utc": generated,
        "configured_paths": config.get("evidence", {}),
        "unreadable_evidence": evidence["unreadable_evidence"],
    })
    (output / "summary.md").write_text(
        "# Transformer V1.1 Promotion Review\n\n"
        f"- Decision: **{decision}**\n"
        f"- Software gate: **{software['status']}**\n"
        f"- Model quality gate: **{quality['status']}**\n\n"
        "软件正确性与模型质量独立裁决；任一失败即拒绝，证据不全不会晋级。\n",
        encoding="utf-8",
    )
    return output

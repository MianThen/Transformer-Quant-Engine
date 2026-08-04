import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check end-to-end performance budgets")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--warning-percent", type=float, default=5.0)
    parser.add_argument("--failure-percent", type=float, default=10.0)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    budget = json.loads(args.budget.read_text(encoding="utf-8-sig"))
    results = {item["name"]: item for item in report["results"]}
    failures = []
    warnings = []
    for name, limits in budget["scenarios"].items():
        if name not in results:
            failures.append(f"missing scenario: {name}")
            continue
        result = results[name]
        ns_per_event = result["elapsed_ns"] / max(result["events"], 1)
        baseline = limits["baseline_ns_per_event"]
        regression = (ns_per_event / baseline - 1.0) * 100.0
        if regression > args.failure_percent:
            failures.append(f"{name}: {regression:.2f}% regression")
        elif regression > args.warning_percent:
            warnings.append(f"{name}: {regression:.2f}% regression")
        if result["p99_ns"] > limits["max_p99_ns"]:
            failures.append(f"{name}: p99 {result['p99_ns']} > {limits['max_p99_ns']}")
        if result["rss_bytes"] > limits["max_rss_bytes"]:
            failures.append(
                f"{name}: RSS {result['rss_bytes']} > {limits['max_rss_bytes']}")
        if result["degraded_count"] > limits.get("max_degraded_count", 0):
            failures.append(f"{name}: unexpected degraded state")
    for message in warnings:
        print(f"performance warning: {message}")
    for message in failures:
        print(f"performance failure: {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

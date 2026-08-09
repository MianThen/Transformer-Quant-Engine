from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Release/LTO C++ benchmarks")
    parser.add_argument("executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--require-lto", action="store_true")
    parser.add_argument("--max-regression-percent", type=float)
    args = parser.parse_args()
    if args.runs < 5:
        parser.error("--runs must be at least 5")
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"benchmark executable not found: {executable}")

    command = [str(executable)] + (["--quick"] if args.quick else [])
    runs = [json.loads(subprocess.run(command, check=True, text=True,
                                      capture_output=True).stdout)
            for _ in range(args.runs)]
    build = runs[0]["build"]
    if build["type"] != "Release":
        raise SystemExit("refusing non-Release benchmark artifact")
    if args.require_lto and not build["lto"]:
        raise SystemExit("LTO was required but is unavailable in this artifact")
    if any(run["build"] != build or run["dataset"] != runs[0]["dataset"] for run in runs):
        raise SystemExit("build or dataset metadata changed between runs")

    aggregate = []
    for first in runs[0]["results"]:
        samples = [next(item for item in run["results"] if item["name"] == first["name"])
                   for run in runs]
        if len({item["checksum"] for item in samples}) != 1:
            raise SystemExit(f"behavior checksum changed between runs: {first['name']}")
        timings = [item["ns_per_operation"] for item in samples]
        aggregate.append({
            **{key: first[key] for key in ("name", "category", "scale", "operations", "checksum")},
            "median_ns_per_operation": statistics.median(timings),
            "p95_ns_per_operation": percentile(timings, 0.95),
            "samples_ns_per_operation": timings,
        })
        if "allocations" in first:
            allocations = [item["allocations"] for item in samples]
            aggregate[-1]["median_allocations_per_run"] = statistics.median(allocations)
            aggregate[-1]["samples_allocations_per_run"] = allocations
        if "peak_rss_bytes" in first:
            rss = [item["peak_rss_bytes"] for item in samples]
            aggregate[-1]["median_peak_rss_bytes"] = statistics.median(rss)
            aggregate[-1]["p95_peak_rss_bytes"] = percentile(rss, 0.95)
            aggregate[-1]["samples_peak_rss_bytes"] = rss

    categories = {}
    for item in aggregate:
        categories.setdefault(item["category"], []).append(item["name"])
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": {"path": str(executable),
                     "sha256": hashlib.sha256(executable.read_bytes()).hexdigest()},
        "build": build,
        "dataset": runs[0]["dataset"],
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "processor": platform.processor(), "logical_cpu_count": os.cpu_count()},
        "run_count": args.runs,
        "cost_attribution": categories,
        "results": aggregate,
        "raw_runs": runs,
    }
    if args.baseline:
        old = {item["name"]: item for item in
               json.loads(args.baseline.read_text(encoding="utf-8"))["results"]}
        for item in report["results"]:
            if item["name"] in old:
                previous = old[item["name"]]["median_ns_per_operation"]
                item["baseline_median_ns_per_operation"] = previous
                item["median_change_percent"] = ((item["median_ns_per_operation"] / previous - 1) * 100
                                                 if previous else 0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    if args.max_regression_percent is not None:
        regressions = [item for item in report["results"]
                       if item.get("median_change_percent", 0.0) > args.max_regression_percent]
        if regressions:
            for item in regressions:
                print(f"performance regression: {item['name']} {item['median_change_percent']:.2f}%")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

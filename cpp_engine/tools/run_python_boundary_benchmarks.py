from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * probability + 0.999999) - 1))
    return ordered[index]


def make_batch(module, count: int):
    batch = []
    for index in range(count):
        market = module.MarketSnapshot()
        market.symbol = f"S{index:07d}"
        market.timestamp = 1
        market.open = market.high = market.low = market.close = 100.0
        market.volume = 1_000_000
        batch.append(market)
    return batch


def run_case(module, batch, callback_kind: str) -> float:
    engine = module.BacktestEngine()
    if callback_kind == "per-row-python":
        engine.set_on_market_data(lambda _: [])
    elif callback_kind == "cross-section-python":
        engine.set_on_cross_section(lambda _: [])
    elif callback_kind == "cross-section-view-python":
        engine.set_on_cross_section_view(lambda view: None if len(view) >= 0 else None)
    elif callback_kind == "columnar-orders-python":
        engine.set_on_cross_section_view(
            lambda _: {"symbol_index": [], "quantity": [], "side": [],
                       "type": [], "price": []})
    start = time.perf_counter_ns()
    engine.process_market_data_batch(batch)
    return (time.perf_counter_ns() - start) / len(batch)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--symbols", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import cpp_engine

    if cpp_engine.__build_type__ != "Release":
        raise SystemExit("Python performance tests require a Release wheel")
    module_path = Path(importlib.util.find_spec("cpp_engine").origin).resolve()
    batch = make_batch(cpp_engine, args.symbols)
    results = []
    for name in ("pure-cpp-no-callback", "cross-section-view-python",
                 "columnar-orders-python", "cross-section-python", "per-row-python"):
        samples = [run_case(cpp_engine, batch, name) for _ in range(args.runs)]
        results.append({
            "name": name,
            "median_ns_per_row": statistics.median(samples),
            "p95_ns_per_row": percentile(samples, 0.95),
            "samples_ns_per_row": samples,
        })
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": {"path": str(module_path), "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest()},
        "build": {"type": cpp_engine.__build_type__, "compiler_id": cpp_engine.__compiler_id__,
                  "compiler_version": cpp_engine.__compiler_version__, "lto": cpp_engine.__lto_enabled__},
        "dataset": {"name": "python-boundary", "symbols": args.symbols},
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "run_count": args.runs,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

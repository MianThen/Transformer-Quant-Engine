#!/usr/bin/env python3
"""Verify the Phase 1F differentiable top-k reference oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

repository_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository_root / "python_model"))
sys.path.insert(0, str(repository_root))

from python.qbt_ml.research.topk_stability import (
    finite_difference_temporal_gradient,
    softsort_topk_weights,
    temporal_topk_stability_penalty,
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _assert_close(left: float, right: float, tolerance: float, name: str) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{name}: {left} != {right}")


def _assert_mapping_close(
    left: dict[str, float], right: dict[str, float], tolerance: float, name: str
) -> None:
    if left.keys() != right.keys():
        raise AssertionError(f"{name}: key mismatch")
    for symbol in left:
        _assert_close(left[symbol], right[symbol], tolerance, f"{name}[{symbol}]")


def build_report() -> dict[str, Any]:
    current = {"A": 0.80, "B": 0.20, "C": -0.10, "D": -0.50}
    previous = {"A": 0.70, "B": 0.40, "C": -0.20, "D": -0.40}
    weights = softsort_topk_weights(current, top_k=2, temperature=0.20)
    reordered = softsort_topk_weights(
        {"D": -0.50, "B": 0.20, "A": 0.80, "C": -0.10},
        top_k=2,
        temperature=0.20,
    )
    shifted = softsort_topk_weights(
        {symbol: value + 13.0 for symbol, value in current.items()},
        top_k=2,
        temperature=0.20,
    )
    low_temperature = softsort_topk_weights(current, top_k=2, temperature=1e-3)
    penalty = temporal_topk_stability_penalty(current, previous, 2, 0.20)
    gradient = finite_difference_temporal_gradient(current, previous, 2, 0.20)

    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("top-k masses must sum to one")
    _assert_mapping_close(weights, reordered, 1e-15, "permutation invariance")
    _assert_mapping_close(weights, shifted, 1e-15, "translation invariance")
    _assert_close(low_temperature["A"], 0.5, 1e-12, "low-temperature A")
    _assert_close(low_temperature["B"], 0.5, 1e-12, "low-temperature B")
    for symbol in ("C", "D"):
        _assert_close(low_temperature[symbol], 0.0, 1e-12, f"low-temperature {symbol}")
    if penalty < 0.0 or any(not math.isfinite(value) for value in gradient.values()):
        raise AssertionError("penalty or finite-difference gradient is invalid")

    report: dict[str, Any] = {
        "schema_version": 1,
        "role": "phase1f_topk_stability_oracle",
        "formula": "P[j,i]=softmax_i(-abs(v_j-s_i)/temperature); w_i=sum(P[:K,i])/K",
        "fixture": {
            "top_k": 2,
            "temperature": 0.20,
            "weights": weights,
            "temporal_penalty": penalty,
            "finite_difference_gradient": gradient,
        },
        "checks": {
            "mass_sum": True,
            "permutation_invariance": True,
            "translation_invariance": True,
            "low_temperature_top_k_limit": True,
            "finite_difference_gradient_finite": True,
        },
    }
    report["report_sha256"] = hashlib.sha256(_canonical(report).encode("utf-8")).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

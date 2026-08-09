#!/usr/bin/env python3
"""Run Phase 3C validation-only calibration on three frozen OOS folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.qbt_ml.evaluation.phase3c_calibration import run_phase3c_calibration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/Users/Zhuanz/PycharmProjects/PythonProject/"
            "data/research/phase1e_pit_120_dataset.npz"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/phase1e-gradnorm4-real-e4"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/phase3c-calibration-real/report.json"),
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--ece-bins", type=int, default=10)
    args = parser.parse_args()
    report = run_phase3c_calibration(
        args.dataset,
        args.run_root,
        args.output,
        folds=args.folds,
        top_k=args.top_k,
        ece_bins=args.ece_bins,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "report_sha256": report["report_sha256"],
                "status": report["status"],
                "phase_exit_eligible": report["phase_exit_eligible"],
                "fold_count": len(report["folds"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

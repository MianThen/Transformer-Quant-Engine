#!/usr/bin/env python3
"""Link a completed Phase 1D baseline into an existing Phase 1E report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link(report_path: Path, phase1d_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    references = dict(report.get("references", {}))
    references["phase1d_drift_baseline"] = {
        "present": True,
        "path": str(phase1d_path.resolve()),
        "sha256": _file_hash(phase1d_path),
    }
    report["references"] = references
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    report["report_sha256"] = _canonical_hash(unsigned)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--phase1d", type=Path, required=True)
    args = parser.parse_args()
    report = link(args.report.resolve(), args.phase1d.resolve())
    print(json.dumps({
        "report": str(args.report.resolve()),
        "phase1d": str(args.phase1d.resolve()),
        "report_sha256": report["report_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

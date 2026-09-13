"""Join the nine Python fold artifacts with nine C++ proxy replay reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from python.qbt_ml.evaluation.infots_replay import build_infots_joint_replay_report
from python.qbt_ml.evaluation.infots_ablation import InfoTSBudgetContractV1


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--cpp-report", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract = InfoTSBudgetContractV1(**_object(Path(args.contract)))
    report = build_infots_joint_replay_report(
        [_object(Path(path)) for path in args.artifact],
        [_object(Path(path)) for path in args.cpp_report],
        contract,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "report_sha256": report["report_sha256"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the frozen C++ precomputed InfoTS research replay through pybind11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="InfoTSReplaySpecV1 JSON")
    parser.add_argument("--rows", required=True, help="JSON array of precomputed replay rows")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        import cpp_engine
    except ImportError as exc:
        raise SystemExit("cpp_engine 缺少 performance analytics 绑定，拒绝回退 Python Replay") from exc
    result = cpp_engine.run_infots_precomputed_replay(
        _json(Path(args.spec)), _json(Path(args.rows))
    )
    if result["status"] != "OK" or not result["artifact_json"]:
        raise SystemExit(json.dumps({"status": result["status"], "reason": "C++ replay failed closed"}, ensure_ascii=False))
    artifact = json.loads(result["artifact_json"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "artifact_sha256": artifact["artifact_sha256"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

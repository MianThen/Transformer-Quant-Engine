#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_name() -> str:
    if platform.system() == "Darwin":
        try:
            value = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if value:
                return value
        except (OSError, subprocess.CalledProcessError):
            pass
    return " ".join(filter(None, (platform.machine(), platform.processor()))) or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Release ORT ML runtime baseline")
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--hardware")
    parser.add_argument(
        "--benchmark-scope",
        choices=("engineering_test_artifact", "production_candidate"),
        default="engineering_test_artifact",
    )
    args = parser.parse_args()
    if not args.executable.is_file():
        parser.error(f"benchmark executable not found: {args.executable}")
    manifest_path = args.artifact / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"artifact manifest not found: {manifest_path}")
    if args.iterations <= 0 or args.chunk_size <= 0:
        parser.error("iterations and chunk-size must be positive")

    repository = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.executable), "--artifact", str(args.artifact),
        "--output", str(args.output), "--iterations", str(args.iterations),
        "--chunk-size", str(args.chunk_size), "--hardware",
        args.hardware or hardware_name(),
        "--revision", revision,
    ]
    subprocess.run(command, check=True, env=os.environ.copy())

    report = json.loads(args.output.read_text(encoding="utf-8"))
    if report.get("build_type") != "Release" or report.get("lto_enabled") is not True:
        raise SystemExit("refusing benchmark artifact without Release and LTO")
    if [item.get("batch_size") for item in report.get("results", [])] != [1, 64, 512, 4096]:
        raise SystemExit("benchmark report is missing required batch sizes")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_executable_sha256": sha256_file(args.executable),
        "model_id": manifest["model_id"],
        "model_version": manifest["model_version"],
        "model_sha256": manifest["model_sha256"],
        "feature_schema_sha256": manifest["feature_schema_sha256"],
        "training_dataset_fingerprint": manifest.get("training_dataset_fingerprint"),
        "frequency": manifest.get("frequency", "1d"),
        "calendar_id": manifest["calendar_id"],
        "benchmark_scope": args.benchmark_scope,
    })
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

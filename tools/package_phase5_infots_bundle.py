"""Build a self-contained Phase 5 InfoTS external-GPU input bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


EXPECTED_DATASET_SHA256 = "4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554"
BUNDLE_ROOT = "qbt_phase5_infots_gpu_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source_root: Path, staging_root: Path, relative_path: str) -> None:
    source = source_root / relative_path
    if not source.is_file():
        raise RuntimeError(f"bundle input missing: {source}")
    destination = staging_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_bundle_readme(staging_root: Path) -> None:
    readme = """# Phase 5 InfoTS GPU input bundle

This archive is a frozen external-GPU training input, not a training result.

1. Install `requirements-phase5-infots-cuda.txt` on a CUDA host.
2. Run `python tools/phase5_infots_gpu_preflight.py --dataset data/research/phase1e_pit_120_dataset.npz --output runs/runtime_preflight.json`.
3. Run `python tools/run_phase5_infots_pretraining.py --dataset data/research/phase1e_pit_120_dataset.npz --config configs/ml/phase5_infots_pretraining_cuda.json --output runs/phase5-infots/artifact.json --checkpoint runs/phase5-infots/checkpoint.pt`.
4. After all 9 group/fold artifacts return, run `python tools/run_phase5_infots_ablation.py` with `configs/ml/phase5_infots_ablation_contract.json`.
5. After C++ precomputed proxy reports return, run `python tools/run_phase5_infots_joint_acceptance.py` with the same contract and nine `--cpp-report` files.

The runner fails closed without CUDA, never falls back to CPU, keeps `production_eval=false`, and does not claim Phase 5 exit or promotion. Returned artifacts still require independent supervised-budget, purged-OOS, six-output/embedding, quality and stability review.
"""
    (staging_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve()
    dataset = Path(args.dataset).resolve()
    if sha256_file(dataset) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("dataset SHA-256 mismatch")
    files = [
        "python/__init__.py",
        "python/qbt_ml/__init__.py",
        "python/qbt_ml/research/infots.py",
        "python/qbt_ml/research/infots_training.py",
        "python/qbt_ml/evaluation/infots_ablation.py",
        "python/qbt_ml/evaluation/infots_replay.py",
        "python/tests/test_phase5_infots.py",
        "python/tests/test_phase5_infots_ablation.py",
        "python/tests/test_phase5_infots_replay.py",
        "tools/run_phase5_infots_pretraining.py",
        "tools/run_phase5_infots_ablation.py",
        "tools/run_phase5_infots_joint_acceptance.py",
        "tools/run_phase5_infots_cpp_replay.py",
        "tools/phase5_infots_gpu_preflight.py",
        "configs/ml/phase5_infots_pretraining_cuda.json",
        "configs/ml/phase5_infots_ablation_contract.json",
        "requirements-phase5-infots-cuda.txt",
        "docs/phase5_infots_gpu_runner.md",
        "docs/phase5_infots_ablation_contract.md",
        "docs/phase5_infots_cpp_replay.md",
    ]
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qbt-phase5-bundle-") as temporary:
        staging_root = Path(temporary) / BUNDLE_ROOT
        staging_root.mkdir(parents=True)
        for relative_path in files:
            copy_file(source_root, staging_root, relative_path)
        dataset_destination = staging_root / "data/research/phase1e_pit_120_dataset.npz"
        dataset_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset, dataset_destination)
        minimal_init = """\"\"\"InfoTS GPU bundle package.\"\"\"\n\nfrom .infots import *\nfrom .infots_training import *\n"""
        (staging_root / "python/qbt_ml/research/__init__.py").write_text(minimal_init, encoding="utf-8")
        minimal_evaluation_init = """\"\"\"InfoTS evaluation package.\"\"\"\n\nfrom .infots_ablation import *\nfrom .infots_replay import *\n"""
        (staging_root / "python/qbt_ml/evaluation").mkdir(parents=True, exist_ok=True)
        (staging_root / "python/qbt_ml/evaluation/__init__.py").write_text(minimal_evaluation_init, encoding="utf-8")
        write_bundle_readme(staging_root)
        manifest_path = staging_root / "MANIFEST.sha256"
        manifest_files = sorted(path for path in staging_root.rglob("*") if path.is_file() and path != manifest_path)
        manifest_path.write_text(
            "".join(f"{sha256_file(path)}  ./{path.relative_to(staging_root).as_posix()}\n" for path in manifest_files),
            encoding="utf-8",
        )
        with tarfile.open(output, "w:gz") as archive:
            archive.add(staging_root, arcname=BUNDLE_ROOT, recursive=True)
        archive_hash = sha256_file(output)
    sidecar = Path(str(output) + ".sha256")
    sidecar.write_text(f"{archive_hash}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": archive_hash, "manifest_entries": len(manifest_files)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

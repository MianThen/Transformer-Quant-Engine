from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


DATASET_SHA256 = "4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    dataset = root / "data/research/phase1e_pit_120_dataset.npz"
    if sha256_file(dataset) != DATASET_SHA256:
        raise RuntimeError("数据集 SHA-256 不匹配")
    for config_name in (
        "phase2b_feature_pgd_v2_validation_cuda.json",
        "phase2b_feature_pgd_v2_short_oos_cuda.json",
    ):
        config = json.loads(
            (root / "configs/ml" / config_name).read_text(encoding="utf-8")
        )
        if config["training"]["device"] != "cuda":
            raise RuntimeError(f"{config_name} 未冻结 CUDA")
    forbidden = [path for path in root.rglob("*") if "__pycache__" in path.parts]
    if forbidden:
        raise RuntimeError("源码包含 __pycache__")
    manifest = root / "MANIFEST.sha256"
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path != manifest
    )
    manifest.write_text(
        "".join(
            f"{sha256_file(path)}  ./{path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    output = Path(args.output).resolve()
    with tarfile.open(output, "w:gz") as archive:
        archive.add(root, arcname="qbt_phase2b_feature_pgd_v2_gpu_r4")
    print(f"{output}\nsha256={sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

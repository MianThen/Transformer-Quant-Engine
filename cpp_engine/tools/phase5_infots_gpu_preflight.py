"""Fail-closed CUDA and dataset preflight for the Phase 5 InfoTS input bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


EXPECTED_DATASET_SHA256 = "4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Phase 5 preflight requires PyTorch; 不允许静默回退 CPU") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；Phase 5 bundle 禁止静默回退 CPU")
    dataset = Path(args.dataset)
    dataset_sha256 = sha256_file(dataset)
    if dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"训练数据 SHA-256 不匹配: {dataset_sha256}")
    with np.load(dataset, allow_pickle=False) as loaded:
        for field in ("features", "valid_mask", "timestamps"):
            if field not in loaded.files:
                raise RuntimeError(f"数据集缺少字段: {field}")
        feature_shape = list(loaded["features"].shape)
        sample_count = int(feature_shape[0])
    properties = torch.cuda.get_device_properties(0)
    probe = torch.arange(4096, device="cuda", dtype=torch.float32)
    probe_sum = float(probe.square().sum().item())
    torch.cuda.synchronize()
    payload = {
        "schema_version": 1,
        "status": "CUDA_PREFLIGHT_PASS",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "dataset_sha256": dataset_sha256,
        "feature_shape": feature_shape,
        "sample_count": sample_count,
        "device_name": str(properties.name),
        "device_total_memory_bytes": int(properties.total_memory),
        "probe_sum": probe_sum,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

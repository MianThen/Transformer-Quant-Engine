"""Run the optional Phase 5 InfoTS train-fold-only pretraining job."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from python.qbt_ml.research.infots import InfoTSAugmentationSpecV1
from python.qbt_ml.research.infots import _sha256_json
from python.qbt_ml.research.infots_training import (
    InfoTSTrainingSpecV1,
    train_infots_pretraining,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Phase 5 config schema_version 必须为 1")
    if payload.get("production_eval") is not False:
        raise ValueError("Phase 5 pretraining 禁止 production_eval")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    config = _load_config(Path(args.config))
    dataset_sha256 = _sha256_file(dataset_path)
    expected_dataset_sha256 = config.get("dataset_sha256")
    if not isinstance(expected_dataset_sha256, str) or dataset_sha256 != expected_dataset_sha256:
        raise ValueError(f"dataset SHA-256 mismatch: {dataset_sha256}")
    with np.load(dataset_path, allow_pickle=False) as loaded:
        required = {"features", "valid_mask", "timestamps"}
        if not required.issubset(loaded.files):
            raise ValueError(f"dataset 缺少字段: {sorted(required - set(loaded.files))}")
        features = loaded["features"]
        valid_mask = loaded["valid_mask"]
        timestamps = loaded["timestamps"]
    augmentation_spec = InfoTSAugmentationSpecV1(**config["augmentation"])
    training_spec = InfoTSTrainingSpecV1(**config["training"])
    if timestamps.ndim == 1 and timestamps.shape[0] == features.shape[0]:
        decision_timestamps = timestamps
    elif timestamps.ndim == 2 and timestamps.shape == valid_mask.shape:
        decision_timestamps = np.asarray([
            timestamps[row_index, np.flatnonzero(valid_mask[row_index])[-1]]
            for row_index in range(features.shape[0])
        ])
    else:
        raise ValueError("timestamps 必须为 [N] 或 [N,T]")
    try:
        train_rows = decision_timestamps.astype(np.int64) <= training_spec.train_fold_end
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("decision timestamps 必须为整数 timestamp") from exc
    if not train_rows.any():
        raise ValueError("train_fold_end 之前没有训练样本")
    features = features[train_rows]
    valid_mask = valid_mask[train_rows]
    timestamps = timestamps[train_rows]
    artifact = train_infots_pretraining(
        features,
        valid_mask,
        timestamps,
        augmentation_spec,
        training_spec,
        checkpoint_path=args.checkpoint,
    )
    artifact["dataset_sha256"] = dataset_sha256
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _sha256_json(artifact)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output_path), "artifact_sha256": artifact["artifact_sha256"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

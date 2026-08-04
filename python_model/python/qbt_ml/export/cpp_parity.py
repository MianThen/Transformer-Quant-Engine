from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .manifest import OUTPUT_NAMES, validate_artifact


def validate_ort_cpp_parity(
    artifact_path: str | Path,
    runner_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    artifact = Path(artifact_path)
    runner = Path(runner_path)
    manifest = validate_artifact(artifact)
    if manifest.schema_version != 2:
        raise ValueError("ORT C++ golden parity 只接受 Manifest V2 制品")
    if not runner.is_file():
        raise ValueError("ORT C++ golden runner 不存在")
    with np.load(artifact / "golden" / "input.npz", allow_pickle=False) as value:
        features = np.asarray(value["features"], dtype=np.float32)
        valid_mask = np.asarray(value["valid_mask"], dtype=np.uint8)
    with np.load(
        artifact / "golden" / "pytorch_output.npz", allow_pickle=False
    ) as value:
        expected = np.concatenate([
            np.asarray(value[name], dtype=np.float32).reshape(-1)
            for name in OUTPUT_NAMES
        ])
    decisions = json.loads(
        (artifact / "golden" / "expected_decisions.json").read_text(encoding="utf-8")
    )
    if features.ndim != 3 or valid_mask.shape != features.shape[:2]:
        raise ValueError("golden 输入 shape 无效")
    if expected.size != features.shape[0] * len(OUTPUT_NAMES):
        raise ValueError("golden PyTorch 输出 shape 无效")

    with tempfile.TemporaryDirectory(prefix="qbt-ort-cpp-") as temporary:
        root = Path(temporary)
        feature_path = root / "features.f32"
        mask_path = root / "mask.u8"
        expected_path = root / "expected.f32"
        targets_path = root / "targets.txt"
        features.tofile(feature_path)
        valid_mask.tofile(mask_path)
        expected.tofile(expected_path)
        targets = decisions.get("targets", [])
        targets_path.write_text(
            str(len(targets)) + "\n" + "".join(
                f"{item['symbol_id']} {item['target_quantity']} {item['target_weight']}\n"
                for item in targets
            ),
            encoding="ascii",
        )
        completed = subprocess.run([
            str(runner), str(artifact / "model.onnx"), str(feature_path),
            str(mask_path), str(expected_path), str(targets_path),
            str(features.shape[0]), str(features.shape[1]), str(features.shape[2]),
        ], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ValueError(
            f"ORT C++ golden parity 失败(returncode={completed.returncode}): "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("ORT C++ golden runner 输出不是 JSON") from exc
    if report.get("status") != "PASS":
        raise ValueError("ORT C++ golden runner 未返回 PASS")
    if report.get("batch_size") != features.shape[0]:
        raise ValueError("ORT C++ golden runner batch_size 与输入不一致")
    if report.get("top_k_match") is not True:
        raise ValueError("ORT C++ top-k 决策不一致")
    if report.get("target_positions_match") is not True:
        raise ValueError("ORT C++ 目标仓位不一致")
    report.update({
        "schema_version": 1,
        "model_id": manifest.model_id,
        "model_version": manifest.model_version,
        "atol": 1e-5,
        "rtol": 1e-4,
        "reference": "pytorch_output.npz",
    })
    destination = Path(output_path) if output_path else artifact / "golden" / "cpp_parity.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination

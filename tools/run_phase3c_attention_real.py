#!/usr/bin/env python3
"""Run deterministic, analysis-only Attention diagnostics on frozen Phase 1E models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from python.qbt_ml.analysis.attention import (
    analyze_temporal_attention,
    attention_stability_report,
    cross_validate_attention_attribution,
)
try:
    from work.phase2b_feature_pgd_v2_r3_source.python.qbt_ml.models.temporal_transformer import (
        TemporalTransformerConfig,
        TemporalTransformerV1,
        load_temporal_transformer_state_dict,
    )
except ModuleNotFoundError:
    from python.qbt_ml.models.temporal_transformer import (
        TemporalTransformerConfig,
        TemporalTransformerV1,
        load_temporal_transformer_state_dict,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TemporalTransformerV1(
        TemporalTransformerConfig(**checkpoint["model_config"]),
    )
    load_temporal_transformer_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _select_rows(data: np.lib.npyio.NpzFile, checkpoint_paths: list[Path], count: int):
    timestamps = np.asarray(data["timestamps"], dtype=np.int64)
    symbols = np.asarray(data["symbols"]).astype(str)
    masks = np.asarray(data["valid_mask"], dtype=np.uint8)
    complete = masks.sum(axis=1) == masks.shape[1]
    fold_test_sets = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        fold_test_sets.append(set(int(value) for value in checkpoint["split"]["test_timestamps"]))
    common_symbols: set[str] | None = None
    for test_timestamps in fold_test_sets:
        symbols_for_fold = set(symbols[complete & np.isin(timestamps, list(test_timestamps))])
        common_symbols = symbols_for_fold if common_symbols is None else common_symbols & symbols_for_fold
    if not common_symbols:
        raise RuntimeError("三折测试集没有共同的完整窗口股票")
    selected_symbols = sorted(common_symbols)[:count]
    selections = []
    for fold_index, test_timestamps in enumerate(fold_test_sets, start=1):
        candidates = np.flatnonzero(
            complete
            & np.isin(timestamps, list(test_timestamps))
            & np.isin(symbols, selected_symbols)
        )
        order = sorted(candidates.tolist(), key=lambda index: (int(timestamps[index]), symbols[index]))
        if len(order) < count:
            raise RuntimeError(f"fold-{fold_index} 可用共同股票样本不足")
        selections.append(order[:count])
    return selected_symbols, selections


def _window_timestamps(all_timestamps: np.ndarray, cutoff: int, width: int) -> list[int]:
    end = int(np.searchsorted(all_timestamps, cutoff, side="right"))
    if end == 0 or int(all_timestamps[end - 1]) != cutoff:
        raise RuntimeError(f"cutoff timestamp 不在 parquet 时间轴中: {cutoff}")
    if end < width:
        raise RuntimeError(f"cutoff timestamp 的历史时间点不足 {width}: {cutoff}")
    return [int(value) for value in all_timestamps[end - width:end]]


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_path = args.data.resolve()
    parquet_path = args.parquet.resolve()
    run_root = args.run_root.resolve()
    checkpoint_paths = [
        (run_root / f"fold-{fold}" / "none" / "checkpoint.pt").resolve()
        for fold in (1, 2, 3)
    ]
    for path in [data_path, parquet_path, *checkpoint_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    data = np.load(data_path, allow_pickle=True)
    if data["features"].shape[1:] != (64, 23):
        raise RuntimeError(f"冻结数据形状不是 [N,64,23]: {data['features'].shape}")
    import pandas as pd

    frame = pd.read_parquet(parquet_path, columns=["timestamp"])
    all_timestamps = np.asarray(sorted(frame["timestamp"].unique()), dtype=np.int64)
    selected_symbols, selections = _select_rows(data, checkpoint_paths, args.samples_per_fold)
    feature_schema = json.loads(str(data["feature_schema_json"].item()))
    feature_names = feature_schema["feature_names"]
    feature_groups = {
        "return_and_range": list(range(0, 7)),
        "volume": [7, 8],
        "volatility": [9, 10, 11, 12],
        "technical": [13, 14, 15, 16, 17],
        "cross_section": [18],
        "status": [19, 20, 21, 22],
    }
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_artifacts = []
    fold_reports = []
    for fold_index, (checkpoint_path, row_indices) in enumerate(
        zip(checkpoint_paths, selections), start=1,
    ):
        model, checkpoint = _load_model(checkpoint_path)
        features = torch.from_numpy(np.asarray(data["features"][row_indices], dtype=np.float32).copy())
        valid_mask = torch.from_numpy(np.asarray(data["valid_mask"][row_indices], dtype=np.uint8).copy())
        cutoff_timestamps = [int(data["timestamps"][index]) for index in row_indices]
        timestamp_rows = [
            _window_timestamps(all_timestamps, cutoff, features.shape[1])
            for cutoff in cutoff_timestamps
        ]
        labels = [str(data["symbols"][index]) for index in row_indices]
        artifact = analyze_temporal_attention(
            model,
            features,
            valid_mask,
            timestamps=timestamp_rows,
            target_output=args.target_output,
            top_k=args.top_k,
            random_seed=args.random_seed,
        )
        attribution = cross_validate_attention_attribution(
            model,
            features,
            valid_mask,
            artifact,
            feature_groups=feature_groups,
            integration_steps=args.integration_steps,
            top_k=args.top_k,
        )
        fold_dir = output_dir / f"fold-{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = fold_dir / "attention_artifact.json"
        attribution_path = fold_dir / "attribution_report.json"
        artifact_path.write_text(json.dumps(_json_safe(artifact), ensure_ascii=False, indent=2) + "\n")
        attribution_path.write_text(json.dumps(_json_safe(attribution), ensure_ascii=False, indent=2) + "\n")
        fold_artifacts.append(artifact)
        fold_reports.append(attribution)
        fold_reports[-1]["fold"] = fold_index
        fold_reports[-1]["rows"] = row_indices
        fold_reports[-1]["symbols"] = labels
        fold_reports[-1]["cutoff_timestamps"] = cutoff_timestamps
        fold_reports[-1]["checkpoint_sha256"] = _sha256_file(checkpoint_path)
        fold_reports[-1]["artifact_path"] = str(artifact_path.relative_to(output_dir))
        fold_reports[-1]["attribution_path"] = str(attribution_path.relative_to(output_dir))
        fold_reports[-1]["model_seed"] = checkpoint["seed"]
        fold_reports[-1]["feature_names"] = feature_names
    stability = attention_stability_report(
        fold_artifacts,
        labels=[f"fold-{index}" for index in (1, 2, 3)],
        top_k=args.top_k,
    )
    (output_dir / "attention_stability_report.json").write_text(
        json.dumps(_json_safe(stability), ensure_ascii=False, indent=2) + "\n",
    )
    report = {
        "schema_version": "Phase3CAttentionRealReportV1",
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "phase_exit_eligible": False,
        "analysis_only": True,
        "production_forward_modified": False,
        "target_output": args.target_output,
        "selection": {
            "samples_per_fold": args.samples_per_fold,
            "selected_common_symbols": selected_symbols,
            "selection_rule": "first complete valid_mask test row per common symbol ordered by timestamp then symbol",
        },
        "data": {
            "dataset_path": str(data_path),
            "dataset_sha256": _sha256_file(data_path),
            "parquet_path": str(parquet_path),
            "parquet_sha256": _sha256_file(parquet_path),
            "feature_schema_sha256": str(data["feature_schema_sha256"].item()),
            "feature_count": len(feature_names),
            "lookback": int(data["features"].shape[1]),
        },
        "feature_groups": feature_groups,
        "folds": fold_reports,
        "stability_report_path": "attention_stability_report.json",
        "stability_report_sha256": _sha256_file(output_dir / "attention_stability_report.json"),
        "limitations": [
            "样本为三折共同股票的确定性小样本诊断，不构成全 OOS faithfulness gate",
            "Attention、Integrated Gradients 与 occlusion 均不宣称因果解释",
            "Phase 3C 概率/CQR 尚无预注册数值 gate，且 calibration 报告未达退出条件",
        ],
    }
    report["report_sha256"] = _canonical_sha256(report)
    (output_dir / "report.json").write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2) + "\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/Users/Zhuanz/PycharmProjects/PythonProject/data/research/phase1e_pit_120_dataset.npz"))
    parser.add_argument("--parquet", type=Path, default=Path("/Users/Zhuanz/PycharmProjects/PythonProject/data/research/phase1e_pit_120_2020plus.parquet"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/phase1e-gradnorm4-real-e4"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase3c-attention-real"))
    parser.add_argument("--samples-per-fold", type=int, default=2)
    parser.add_argument("--integration-steps", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--target-output", default="expected_return")
    args = parser.parse_args()
    if args.samples_per_fold <= 0 or args.integration_steps <= 0:
        raise SystemExit("samples-per-fold/integration-steps 必须为正数")
    report = run(args)
    print(json.dumps({"report": str(args.output / "report.json"), "report_sha256": report["report_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

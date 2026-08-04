#!/usr/bin/env python3
"""Build a deterministic Phase 1D drift baseline from available PIT snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_MARKET_COLUMNS = {
    "timestamp", "symbol", "open", "high", "low", "close", "volume",
    "is_listed", "is_suspended", "is_st", "is_tradable",
    "reference_data_known_at_max",
}

PREDICTION_HEADS = (
    "expected_return", "expected_volatility", "direction_probability",
    "lower_quantile", "upper_quantile", "confidence",
)

EMBEDDING_COMPATIBILITY_FIELDS = (
    "schema_version",
    "role",
    "diagnostic_only",
    "encoder_family",
    "layer_id",
    "pooling_spec",
    "pooling_spec_sha256",
    "mask_spec",
    "mask_spec_sha256",
    "dimension",
    "source_dataset_sha256",
    "checkpoint_sha256",
    "anchor_split",
    "anchor_sample_count",
    "anchor_values_sha256",
)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _embedding_compatibility_key(manifest: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in EMBEDDING_COMPATIBILITY_FIELDS if name not in manifest]
    if missing:
        raise ValueError(
            "embedding manifest 缺少跨 split 兼容字段: " + ", ".join(missing)
        )
    return {name: manifest[name] for name in EMBEDDING_COMPATIBILITY_FIELDS}


def _load_market(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_MARKET_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"行情缺少 Phase 1D 字段: {', '.join(missing)}")
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="raise").astype("int64")
    frame["available_at"] = pd.to_numeric(
        frame["reference_data_known_at_max"], errors="raise").astype("int64")
    frame = frame.sort_values(["symbol", "timestamp"], kind="stable")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["open", "high", "low", "close", "volume", "available_at"]].isna().any().any():
        raise ValueError("行情包含无法解析的数值或 available_at")
    frame["return_raw"] = frame.groupby("symbol")["close"].shift(-1) / frame["close"] - 1.0
    frame["label_available_at"] = frame.groupby("symbol")["timestamp"].shift(-1)
    frame["rank_utility"] = frame["return_raw"]
    return frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def _load_prediction(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as values:
        required = {"timestamps", "symbols", "scores"}
        missing = sorted(required - set(values.files))
        if missing:
            raise ValueError(f"预测缺少字段: {', '.join(missing)}")
        frame = pd.DataFrame({
            "timestamp": values["timestamps"].astype(np.int64, copy=False),
            "symbol": values["symbols"].astype(str),
            "ranking_score": values["scores"].astype(np.float64, copy=False),
        })
        for name in PREDICTION_HEADS:
            if name in values.files:
                frame[name] = values[name].astype(np.float64, copy=False)
    if frame.empty or not np.isfinite(frame["ranking_score"].to_numpy()).all():
        raise ValueError("预测 ranking_score 必须非空且有限")
    if frame.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("预测 timestamp/symbol 键重复")
    frame["available_at"] = frame["timestamp"]
    return frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def _load_embedding(path: Path) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
]:
    manifest_name = (
        "validation_embedding_manifest.json"
        if path.name == "validation_embeddings.npz"
        else "embedding_manifest.json"
    )
    manifest_path = path.with_name(manifest_name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supplied = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if not isinstance(supplied, str) or _canonical_hash(unsigned) != supplied:
        raise ValueError(f"embedding manifest hash 校验失败: {manifest_path}")
    with np.load(path, allow_pickle=False) as values:
        required = {
            "embeddings", "timestamps", "symbols", "anchor_embeddings",
            "anchor_timestamps", "anchor_symbols",
        }
        missing = sorted(required - set(values.files))
        if missing:
            raise ValueError(f"embedding snapshot 缺少字段: {', '.join(missing)}")
        embeddings = values["embeddings"].astype(np.float32, copy=False)
        timestamps = values["timestamps"].astype(np.int64, copy=False)
        symbols = values["symbols"].astype(str)
        anchors = values["anchor_embeddings"].astype(np.float32, copy=False)
        anchor_timestamps = values["anchor_timestamps"].astype(np.int64, copy=False)
        anchor_symbols = values["anchor_symbols"].astype(str)
    if embeddings.ndim != 2 or anchors.ndim != 2:
        raise ValueError("embedding snapshot 必须是二维矩阵")
    if embeddings.shape[0] != timestamps.size or timestamps.size != symbols.size:
        raise ValueError("embedding snapshot 与 timestamp/symbol 长度不一致")
    if anchors.shape[0] != anchor_timestamps.size or anchor_timestamps.size != anchor_symbols.size:
        raise ValueError("anchor embedding 与 timestamp/symbol 长度不一致")
    if embeddings.shape[1] != int(manifest["dimension"]):
        raise ValueError("embedding dimension 与 manifest 不一致")
    if _array_hash(embeddings) != manifest["embedding_values_sha256"]:
        raise ValueError("embedding values hash 校验失败")
    if _array_hash(anchors) != manifest["anchor_values_sha256"]:
        raise ValueError("anchor values hash 校验失败")
    return embeddings, anchors, timestamps, symbols, manifest


def _windows(reference: pd.DataFrame, current: pd.DataFrame,
             current_available_at: int, drift_module: Any) -> tuple[Any, Any]:
    reference_start = int(reference["timestamp"].min())
    reference_end = int(reference["timestamp"].max())
    current_start = int(current["timestamp"].min())
    current_end = int(current["timestamp"].max())
    if reference_end >= current_start:
        raise ValueError("reference/current prediction windows 重叠")
    return (
        drift_module.WindowSpec(
            reference_start, reference_end, reference_end,
            drift_module.ReferenceKind.TRAINING_STATIC,
        ),
        drift_module.WindowSpec(current_start, current_end, current_available_at),
    )


def build_baseline(
    market_path: Path,
    reference_prediction_path: Path,
    current_prediction_path: Path,
    output_path: Path,
    python_project: Path,
    reference_embedding_path: Path | None = None,
    current_embedding_path: Path | None = None,
) -> dict[str, Any]:
    sys.path.insert(0, str(python_project))
    from python.qbt_ml.features import build_bar_v1
    from python.qbt_ml.monitoring import drift as drift_module

    market = _load_market(market_path)
    feature_source = market[[
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
        "is_listed", "is_suspended", "is_st", "is_tradable",
    ]].copy()
    feature_frame = build_bar_v1(feature_source)
    feature_values = feature_frame.values.astype(np.float64, copy=False).copy()
    feature_values[feature_frame.valid_mask == 0] = np.nan
    features = feature_frame.table.copy()
    features["available_at"] = market["available_at"].to_numpy(dtype=np.int64)
    for index, name in enumerate(feature_frame.schema.feature_names):
        features[name] = feature_values[:, index]
    feature_columns = tuple(feature_frame.schema.feature_names)
    reference_predictions = _load_prediction(reference_prediction_path)
    current_predictions = _load_prediction(current_prediction_path)
    available_prediction_heads = [
        name for name in PREDICTION_HEADS
        if name in reference_predictions and name in current_predictions
    ]
    unavailable_prediction_heads = [
        name for name in PREDICTION_HEADS if name not in available_prediction_heads
    ]
    if (reference_embedding_path is None) != (current_embedding_path is None):
        raise ValueError("reference/current embedding 必须同时提供")
    reference_embeddings = current_embeddings = None
    reference_anchor_embeddings = current_anchor_embeddings = None
    reference_embedding_spec = current_embedding_spec = None
    reference_embedding_manifest = current_embedding_manifest = None
    embedding_sources: dict[str, Any] = {}
    if reference_embedding_path is not None and current_embedding_path is not None:
        (
            reference_embeddings, reference_anchor_embeddings,
            reference_embedding_timestamps, reference_embedding_symbols,
            reference_embedding_manifest,
        ) = _load_embedding(reference_embedding_path)
        (
            current_embeddings, current_anchor_embeddings,
            current_embedding_timestamps, current_embedding_symbols,
            current_embedding_manifest,
        ) = _load_embedding(current_embedding_path)
        if (_embedding_compatibility_key(reference_embedding_manifest)
                != _embedding_compatibility_key(current_embedding_manifest)):
            raise ValueError("reference/current embedding manifest 的跨 split 兼容字段不一致")
        if not np.array_equal(reference_embedding_timestamps,
                              reference_predictions["timestamp"].to_numpy()):
            raise ValueError("reference embedding 与 prediction timestamp 不一致")
        if not np.array_equal(current_embedding_timestamps,
                              current_predictions["timestamp"].to_numpy()):
            raise ValueError("current embedding 与 prediction timestamp 不一致")
        if not np.array_equal(reference_embedding_symbols,
                              reference_predictions["symbol"].to_numpy()):
            raise ValueError("reference embedding 与 prediction symbol 不一致")
        if not np.array_equal(current_embedding_symbols,
                              current_predictions["symbol"].to_numpy()):
            raise ValueError("current embedding 与 prediction symbol 不一致")
        reference_embedding_spec = drift_module.EmbeddingSpec(
            checkpoint_sha256=reference_embedding_manifest["checkpoint_sha256"],
            encoder_family=reference_embedding_manifest["encoder_family"],
            layer_id=reference_embedding_manifest["layer_id"],
            pooling_spec_sha256=reference_embedding_manifest["pooling_spec_sha256"],
            dimension=int(reference_embedding_manifest["dimension"]),
            mask_spec_sha256=reference_embedding_manifest["mask_spec_sha256"],
        )
        current_embedding_spec = reference_embedding_spec
        embedding_sources = {
            "reference_path": str(reference_embedding_path.resolve()),
            "reference_manifest_sha256": reference_embedding_manifest["manifest_sha256"],
            "current_path": str(current_embedding_path.resolve()),
            "current_manifest_sha256": current_embedding_manifest["manifest_sha256"],
        }
    current_prediction_end = int(current_predictions["timestamp"].max())
    later_sessions = market.loc[
        market["timestamp"] > current_prediction_end, "timestamp"]
    if later_sessions.empty:
        raise ValueError("current window 缺少可用于 label maturity 的后续 session")
    label_available_at = int(later_sessions.min())
    reference_window, current_window = _windows(
        reference_predictions, current_predictions, label_available_at, drift_module)

    prediction_keys = set(zip(
        reference_predictions["timestamp"], reference_predictions["symbol"])) | set(zip(
        current_predictions["timestamp"], current_predictions["symbol"]))
    market_keys = set(zip(market["timestamp"], market["symbol"]))
    if not prediction_keys.issubset(market_keys):
        raise ValueError("预测 timestamp/symbol 不在 PIT 行情快照中")

    labels = market[
        ["timestamp", "symbol", "return_raw", "rank_utility", "label_available_at"]
    ].dropna().copy()
    labels = labels[labels["label_available_at"] <= label_available_at].drop(
        columns=["label_available_at"])
    raw = market[[
        "timestamp", "symbol", "available_at", "open", "high", "low", "close",
        "volume", "is_listed", "is_suspended", "is_st", "is_tradable",
    ]].copy()

    spec = drift_module.DriftMonitorSpecV1(
        report_version="phase1d-real-baseline-v1",
        top_k=20,
        minimum_sessions=3,
        minimum_observations=20,
        fast_window_sessions=5,
        confirm_window_sessions=20,
        bootstrap_replicates=499,
        mean_block_length=5.0,
        bootstrap_seed=20260803,
        persistence_windows=2,
    )
    data_hash = _file_hash(market_path)
    reference_prediction_hash = _file_hash(reference_prediction_path)
    current_prediction_hash = _file_hash(current_prediction_path)
    report = drift_module.build_drift_report(
        reference_raw=raw,
        current_raw=raw,
        reference_features=features,
        current_features=features,
        reference_predictions=reference_predictions,
        current_predictions=current_predictions,
        reference_window=reference_window,
        current_window=current_window,
        raw_columns=("open", "high", "low", "close", "volume", "is_tradable"),
        categorical_raw_columns=("is_listed", "is_suspended", "is_st", "is_tradable"),
        feature_columns=feature_columns,
        prediction_columns=tuple(available_prediction_heads),
        reference_labels=labels,
        current_labels=labels,
        return_column="return_raw",
        utility_column="rank_utility",
        source_snapshot_hashes=(data_hash, reference_prediction_hash, current_prediction_hash),
        artifact_hashes={
            "market_snapshot_sha256": data_hash,
            "reference_prediction_sha256": reference_prediction_hash,
            "current_prediction_sha256": current_prediction_hash,
            "feature_schema_sha256": feature_frame.schema.sha256,
            **({
                "reference_embedding_manifest_sha256": embedding_sources[
                    "reference_manifest_sha256"],
                "current_embedding_manifest_sha256": embedding_sources[
                    "current_manifest_sha256"],
            } if embedding_sources else {}),
        },
        reference_embeddings=reference_embeddings,
        current_embeddings=current_embeddings,
        reference_embedding_spec=reference_embedding_spec,
        current_embedding_spec=current_embedding_spec,
        reference_anchor_embeddings=reference_anchor_embeddings,
        current_anchor_embeddings=current_anchor_embeddings,
        fixed_anchor_sha256=(
            reference_embedding_manifest["anchor_values_sha256"]
            if reference_embedding_manifest is not None else None
        ),
        spec=spec,
    )
    report_dict = report.to_dict()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": "phase1d_real_baseline_v1",
        "status": (
            "REAL_DATA_BASELINE_RESEARCH_PROXY_COMPLETE"
            if not unavailable_prediction_heads and embedding_sources
            else "REAL_DATA_BASELINE_RESEARCH_PROXY_INTERNAL_OUTPUTS_UNAVAILABLE"
        ),
        "evidence_tier": "RESEARCH_PROXY",
        "economic_claim_scope": "RESEARCH_PROXY_ONLY",
        "phase_exit_eligible": not unavailable_prediction_heads and bool(embedding_sources),
        "research_comparison_eligible": True,
        "formal_exit": not unavailable_prediction_heads and bool(embedding_sources),
        "economic_promotion_exit": False,
        "promotion_eligible": False,
        "diagnostic_only": True,
        "feature_source": "python.qbt_ml.features.BAR_V1",
        "source": {
            "market_path": str(market_path.resolve()),
            "market_sha256": data_hash,
            "reference_prediction_path": str(reference_prediction_path.resolve()),
            "reference_prediction_sha256": reference_prediction_hash,
            "current_prediction_path": str(current_prediction_path.resolve()),
            "current_prediction_sha256": current_prediction_hash,
            "feature_schema_sha256": feature_frame.schema.sha256,
            "feature_schema_profile": feature_frame.schema.profile,
            "feature_count": len(feature_frame.schema.feature_names),
            "label_available_at": label_available_at,
            **embedding_sources,
        },
        "available_prediction_heads": available_prediction_heads,
        "unavailable_prediction_heads": unavailable_prediction_heads,
        "unavailable_layers": ([] if embedding_sources else ["embedding_drift"]),
        "missing_execution_data": {
            "fee_state": "UNAVAILABLE",
            "tax_state": "UNAVAILABLE",
            "limit_state": "UNAVAILABLE",
            "corporate_action_state": "UNAVAILABLE",
            "adjustment_state": "UNKNOWN",
            "lot_state": "UNAVAILABLE",
            "reference_quality": "PROXY",
            "slippage_state": "UNAVAILABLE",
        },
        "report": report_dict,
        "blocking_reasons": [
            *(["MODEL_OUTPUT_ARTIFACT_HAS_INCOMPLETE_PREDICTION_HEADS"]
              if unavailable_prediction_heads else []),
            *(["EMBEDDING_SNAPSHOT_UNAVAILABLE"] if not embedding_sources else []),
        ],
        "non_blocking_limitations": [
            "PHASE1C_REFERENCE_PRICE_AND_COST_FIELDS_UNAVAILABLE_RESEARCH_PROXY_ONLY",
            "LONG_TERM_EXECUTION_FIELDS_UNAVAILABLE_DO_NOT_BLOCK_PHASE1D_EXIT",
        ],
    }
    payload["report_sha256"] = _canonical_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", type=Path,
                        default=Path("/Users/Zhuanz/PycharmProjects/PythonProject/data/research/phase1e_pit_120_2020plus.parquet"))
    parser.add_argument("--reference-predictions", type=Path,
                        default=Path("runs/phase1e-pcgrad-real-e4/fold-1/pcgrad/validation_predictions.npz"))
    parser.add_argument("--current-predictions", type=Path,
                        default=Path("runs/phase1e-pcgrad-real-e4/fold-1/pcgrad/test_predictions.npz"))
    parser.add_argument("--reference-embedding", type=Path,
                        default=Path("runs/phase1e-pcgrad-real-e4/fold-1/pcgrad/validation_embeddings.npz"))
    parser.add_argument("--current-embedding", type=Path,
                        default=Path("runs/phase1e-pcgrad-real-e4/fold-1/pcgrad/test_embeddings.npz"))
    parser.add_argument("--python-project", type=Path,
                        default=Path("/Users/Zhuanz/PycharmProjects/PythonProject"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/phase1d-real-baseline/phase1d_real_baseline_report.json"))
    args = parser.parse_args()
    result = build_baseline(
        args.market.resolve(), args.reference_predictions.resolve(),
        args.current_predictions.resolve(), args.output.resolve(),
        args.python_project.resolve(), args.reference_embedding.resolve(),
        args.current_embedding.resolve())
    print(json.dumps({
        "output": str(args.output.resolve()),
        "report_sha256": result["report_sha256"],
        "formal_exit": result["formal_exit"],
        "promotion_eligible": result["promotion_eligible"],
        "alert_state": result["report"]["alert_state"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

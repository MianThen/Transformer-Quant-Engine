from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..leakage import audit_dataset
from ..models import LogisticBaseline, RidgeBaseline
from ..training import expanding_timestamp_splits
from .metrics import prediction_metrics


def _last_valid_features(features, valid_mask) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if features.ndim != 3 or mask.shape != features.shape[:2]:
        raise ValueError("features/valid_mask shape 无效")
    if not mask.any(axis=1).all():
        raise ValueError("Walk-forward 样本必须至少包含一个有效 token")
    last_from_end = np.argmax(mask[:, ::-1], axis=1)
    last = mask.shape[1] - 1 - last_from_end
    return features[np.arange(features.shape[0]), last]


def _fit_normalizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64)
    scale = features.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _normalize(features, mean, scale, clip: float) -> np.ndarray:
    return np.clip((features - mean) / scale, -clip, clip)


def _huber(prediction, target) -> float:
    error = np.abs(np.asarray(prediction) - np.asarray(target))
    return float(np.where(error <= 1.0, 0.5 * error**2, error - 0.5).mean())


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _window_split_json(window) -> dict:
    def interval(values):
        return {
            "first": int(values[0]), "last": int(values[-1]),
            "timestamps": int(len(values)),
        }
    return {
        "window_id": window.window_id,
        "train": interval(window.train_timestamps),
        "validation": interval(window.validation_timestamps),
        "test": interval(window.test_timestamps),
        "samples": {
            "train": int(len(window.train)),
            "validation": int(len(window.validation)),
            "test": int(len(window.test)),
        },
    }


def run_walk_forward_baseline(
    config: dict,
    dataset_path: str | Path,
    output_path: str | Path,
) -> Path:
    if config.get("enabled") is not True:
        raise RuntimeError("Walk-forward 配置必须显式设置 enabled=true")
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    required = {
        "features", "valid_mask", "timestamps", "symbols", "expected_return",
        "direction", "realized_volatility", "signal_asof",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError("Walk-forward 数据集缺少字段: " + ", ".join(missing))

    split_config = config.get("windows", {})
    windows = expanding_timestamp_splits(
        arrays["timestamps"],
        minimum_train_timestamps=int(split_config.get("minimum_train_timestamps", 1260)),
        validation_timestamps=int(split_config.get("validation_timestamps", 252)),
        test_timestamps=int(split_config.get("test_timestamps", 252)),
        step_timestamps=int(split_config.get("step_timestamps", 252)),
        purge_timestamps=int(split_config.get("purge_timestamps", 5)),
        embargo_timestamps=int(split_config.get("embargo_timestamps", 5)),
        minimum_windows=int(split_config.get("minimum_windows", 3)),
    )
    model_config = config.get("model", {})
    alpha_candidates = tuple(float(value) for value in model_config.get(
        "ridge_alpha_candidates", [0.1, 1.0, 10.0]
    ))
    if not alpha_candidates or any(value < 0 or not np.isfinite(value) for value in alpha_candidates):
        raise ValueError("ridge_alpha_candidates 必须为有限非负数")
    clip = float(model_config.get("normalization_clip", 8.0))
    if clip <= 0 or not np.isfinite(clip):
        raise ValueError("normalization_clip 必须为有限正数")

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    source_features = _last_valid_features(arrays["features"], arrays["valid_mask"])
    feature_indices = np.asarray(
        model_config.get("feature_indices", list(range(source_features.shape[1]))),
        dtype=np.int64,
    )
    if (
        feature_indices.ndim != 1 or feature_indices.size == 0
        or np.unique(feature_indices).size != feature_indices.size
        or feature_indices.min() < 0 or feature_indices.max() >= source_features.shape[1]
    ):
        raise ValueError("model.feature_indices 无效")
    source_features = source_features[:, feature_indices]
    returns = np.asarray(arrays["expected_return"], dtype=np.float64)
    direction = np.asarray(arrays["direction"], dtype=np.float64)
    volatility = np.asarray(arrays["realized_volatility"], dtype=np.float64)
    all_metrics = []

    for window in windows:
        root = output / f"window_{window.window_id:03d}"
        root.mkdir()
        normalizer_fit_end = np.asarray(arrays["signal_asof"])[window.train].max()
        leakage = audit_dataset(
            arrays, split=window, normalizer_fit_end=normalizer_fit_end
        )
        leakage.write(root / "leakage_report.json")
        leakage.require_pass()
        _json_write(root / "split.json", _window_split_json(window))

        mean, scale = _fit_normalizer(source_features[window.train])
        train_x = _normalize(source_features[window.train], mean, scale, clip)
        validation_x = _normalize(source_features[window.validation], mean, scale, clip)
        test_x = _normalize(source_features[window.test], mean, scale, clip)

        candidates = []
        for alpha in alpha_candidates:
            candidate = RidgeBaseline(alpha).fit(train_x, returns[window.train])
            candidates.append((
                _huber(candidate.predict(validation_x), returns[window.validation]),
                alpha,
            ))
        validation_huber, selected_alpha = min(candidates, key=lambda item: (item[0], item[1]))
        return_model = RidgeBaseline(selected_alpha).fit(train_x, returns[window.train])
        volatility_model = RidgeBaseline(selected_alpha).fit(
            train_x, volatility[window.train]
        )
        validation_return = return_model.predict(validation_x)
        calibrator = LogisticBaseline(alpha=1.0).fit(
            validation_return[:, None], direction[window.validation]
        )
        residual = returns[window.validation] - validation_return
        lower_offset = min(float(np.quantile(residual, 0.10)), 0.0)
        upper_offset = max(float(np.quantile(residual, 0.90)), 0.0)

        expected_return = return_model.predict(test_x)
        expected_volatility = np.maximum(volatility_model.predict(test_x), 0.0)
        direction_probability = calibrator.predict_proba(expected_return[:, None])
        lower_quantile = expected_return + lower_offset
        upper_quantile = expected_return + upper_offset
        confidence = np.maximum(direction_probability, 1.0 - direction_probability)
        metrics = prediction_metrics(
            expected_return=expected_return,
            realized_return=returns[window.test],
            expected_volatility=expected_volatility,
            realized_volatility=volatility[window.test],
            direction_probability=direction_probability,
            direction=direction[window.test],
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            timestamps=np.asarray(arrays["timestamps"])[window.test],
        )
        metrics.update({
            "window_id": window.window_id,
            "selected_alpha": selected_alpha,
            "validation_return_huber": validation_huber,
            "test_samples": int(len(window.test)),
        })
        all_metrics.append(metrics)
        _json_write(root / "metrics.json", metrics)
        _json_write(root / "model" / "ridge_multitask.json", {
            "model_family": "RidgeMultiTaskBaseline",
            "selected_alpha": selected_alpha,
            "feature_source": "last_valid_BAR_V1_token",
            "feature_indices": feature_indices.tolist(),
            "normalization": {
                "method": "mean_std", "clip": clip,
                "fit_start": int(np.asarray(arrays["signal_asof"])[window.train].min()),
                "fit_end": int(normalizer_fit_end),
                "mean": mean.tolist(), "scale": scale.tolist(),
            },
            "return_coefficients": return_model.coefficients_.tolist(),
            "volatility_coefficients": volatility_model.coefficients_.tolist(),
            "calibration": {
                "method": "platt_validation_only",
                "coefficients": calibrator.coefficients_.tolist(),
            },
            "quantile_residual_offsets": {"q10": lower_offset, "q90": upper_offset},
        })
        np.savez_compressed(
            root / "predictions.npz",
            timestamps=np.asarray(arrays["timestamps"])[window.test],
            symbols=np.asarray(arrays["symbols"])[window.test],
            expected_return=expected_return.astype(np.float32),
            expected_volatility=expected_volatility.astype(np.float32),
            direction_probability=direction_probability.astype(np.float32),
            lower_quantile=lower_quantile.astype(np.float32),
            upper_quantile=upper_quantile.astype(np.float32),
            confidence=confidence.astype(np.float32),
            realized_return=returns[window.test].astype(np.float32),
        )

    metric_names = [
        name for name, value in all_metrics[0].items()
        if isinstance(value, (int, float)) and name not in {"window_id"}
    ]
    aggregate = {
        "windows": len(windows),
        "metrics": {
            name: {
                "mean": float(np.mean([item[name] for item in all_metrics])),
                "std": float(np.std([item[name] for item in all_metrics], ddof=0)),
            }
            for name in metric_names
        },
    }
    _json_write(output / "aggregate" / "summary.json", aggregate)
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    _json_write(output / "experiment_manifest.json", {
        "schema_version": 1,
        "pipeline": "expanding_walk_forward",
        "model_family": "RidgeMultiTaskBaseline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(dataset_path).resolve()),
        "dataset_fingerprint": str(np.asarray(arrays["dataset_fingerprint"]).item()),
        "execution_reference_status": str(np.asarray(
            arrays.get("execution_reference_status", np.asarray("UNDECLARED"))
        ).item()),
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "windows": len(windows),
        "test_evaluation_policy": "once_after_validation_freeze",
    })
    return output

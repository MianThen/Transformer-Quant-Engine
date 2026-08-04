from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..data import BAR_V1
from ..models import LogisticBaseline, RidgeBaseline
from ..training import expanding_timestamp_splits
from .metrics import prediction_metrics
from .walk_forward import _json_write, _last_valid_features, run_walk_forward_baseline


RULE_MODELS = ("momentum_20", "reversal_5", "equal_weight", "cash")
DEEP_BASELINE_MODELS = ("mlp", "tcn", "gru")
DEEP_BASELINE_FAMILIES = {
    "mlp": "MLPMultiTaskBaseline",
    "tcn": "CausalTCNMultiTaskBaseline",
    "gru": "GRUMultiTaskBaseline",
}
TRANSFORMER_MODEL = "transformer_v1_1"


def _rule_score(name: str, features: np.ndarray) -> np.ndarray:
    indices = {feature: index for index, feature in enumerate(BAR_V1.feature_names)}
    if name == "momentum_20":
        return features[:, indices["log_return_20"]]
    if name == "reversal_5":
        return -features[:, indices["log_return_5"]]
    if name in {"equal_weight", "cash"}:
        return np.zeros(features.shape[0], dtype=np.float64)
    raise ValueError(f"未知规则模型: {name}")


def _rule_predictions(name, features, returns, direction, volatility, window):
    validation_score = _rule_score(name, features[window.validation])
    test_score = _rule_score(name, features[window.test])
    if name in {"equal_weight", "cash"}:
        expected_return = np.full(
            len(window.test),
            0.0 if name == "cash" else 1e-6,
            dtype=np.float64,
        )
        probability = np.full(len(window.test), 1.0 if name == "equal_weight" else
                              np.clip(direction[window.validation].mean(), 1e-7, 1 - 1e-7))
        lower = np.full(len(window.test), min(float(np.quantile(
            returns[window.validation], 0.10
        )), float(expected_return[0])))
        upper = np.full(len(window.test), max(float(np.quantile(
            returns[window.validation], 0.90
        )), float(expected_return[0])))
    else:
        return_calibrator = RidgeBaseline(alpha=1.0).fit(
            validation_score[:, None], returns[window.validation]
        )
        validation_return = return_calibrator.predict(validation_score[:, None])
        expected_return = return_calibrator.predict(test_score[:, None])
        direction_calibrator = LogisticBaseline(alpha=1.0).fit(
            validation_return[:, None], direction[window.validation]
        )
        probability = direction_calibrator.predict_proba(expected_return[:, None])
        residual = returns[window.validation] - validation_return
        lower = expected_return + min(float(np.quantile(residual, 0.10)), 0.0)
        upper = expected_return + max(float(np.quantile(residual, 0.90)), 0.0)
    expected_volatility = np.full(
        len(window.test), max(float(volatility[window.train].mean()), 0.0)
    )
    return {
        "expected_return": expected_return,
        "expected_volatility": expected_volatility,
        "direction_probability": probability,
        "lower_quantile": lower,
        "upper_quantile": upper,
        "confidence": np.maximum(probability, 1.0 - probability),
    }


def _deep_records(run_path, arrays, windows, model_name, model_family) -> list[dict]:
    run = Path(run_path)
    manifest_path = run / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("transformer_run 缺少 experiment_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_family") != model_family:
        raise ValueError(f"{model_name} run 的 model_family 不匹配")
    dataset_fingerprint = str(np.asarray(arrays["dataset_fingerprint"]).item())
    if manifest.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("transformer_run 与 Benchmark 数据集 fingerprint 不一致")
    if int(manifest.get("windows", -1)) != len(windows):
        raise ValueError("transformer_run 与 Benchmark 窗口数量不一致")

    records = []
    for window in windows:
        root = run / f"window_{window.window_id:03d}"
        predictions_path = root / "predictions.npz"
        if not predictions_path.is_file():
            raise ValueError(f"transformer_run 缺少窗口 {window.window_id} 的预测")
        with np.load(predictions_path, allow_pickle=False) as prediction:
            prediction = {name: prediction[name] for name in prediction.files}
        required = {
            "timestamps", "symbols", "expected_return", "expected_volatility",
            "direction_probability", "lower_quantile", "upper_quantile",
        }
        missing = sorted(required - set(prediction))
        if missing:
            raise ValueError(
                f"transformer_run 窗口 {window.window_id} 缺少字段: " + ", ".join(missing)
            )
        expected_timestamps = np.asarray(arrays["timestamps"])[window.test]
        expected_symbols = np.asarray(arrays["symbols"])[window.test]
        if (
            not np.array_equal(prediction["timestamps"], expected_timestamps)
            or not np.array_equal(prediction["symbols"], expected_symbols)
        ):
            raise ValueError(
                f"transformer_run 窗口 {window.window_id} 的测试样本与 Benchmark 不一致"
            )
        metrics = prediction_metrics(
            expected_return=prediction["expected_return"],
            realized_return=np.asarray(arrays["expected_return"])[window.test],
            expected_volatility=prediction["expected_volatility"],
            realized_volatility=np.asarray(arrays["realized_volatility"])[window.test],
            direction_probability=prediction["direction_probability"],
            direction=np.asarray(arrays["direction"])[window.test],
            lower_quantile=prediction["lower_quantile"],
            upper_quantile=prediction["upper_quantile"],
            timestamps=expected_timestamps,
        )
        records.append({
            "model": model_name, "window_id": window.window_id, **metrics,
        })
    return records


def _transformer_records(run_path, arrays, windows) -> list[dict]:
    return _deep_records(
        run_path, arrays, windows, TRANSFORMER_MODEL, "TemporalTransformerV1.1"
    )


def _deep_baseline_records(suite_path, arrays, windows) -> list[dict]:
    suite = Path(suite_path)
    manifest_path = suite / "benchmark_suite_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("deep_baseline_run 缺少 benchmark_suite_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = tuple(manifest.get("models", []))
    if models != DEEP_BASELINE_MODELS:
        raise ValueError("正式 Deep Baseline Suite 必须包含 mlp/tcn/gru 且顺序冻结")
    records = []
    for name in models:
        run = manifest.get("runs", {}).get(name, {})
        path = suite / run.get("path", name)
        records.extend(_deep_records(
            path, arrays, windows, name, DEEP_BASELINE_FAMILIES[name]
        ))
    return records


def run_model_benchmark(config: dict, dataset_path: str | Path, output_path: str | Path) -> Path:
    if config.get("enabled") is not True:
        raise RuntimeError("Model Benchmark 配置必须显式设置 enabled=true")
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    walk_config = copy.deepcopy(config.get("walk_forward", {}))
    walk_config["enabled"] = True
    ridge_path = run_walk_forward_baseline(
        walk_config, dataset_path, output / "ridge_multitask"
    )
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    split_config = walk_config.get("windows", {})
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
    features = _last_valid_features(arrays["features"], arrays["valid_mask"])
    returns = np.asarray(arrays["expected_return"], dtype=np.float64)
    direction = np.asarray(arrays["direction"], dtype=np.float64)
    volatility = np.asarray(arrays["realized_volatility"], dtype=np.float64)
    timestamps = np.asarray(arrays["timestamps"])

    records = []
    prediction_artifacts: dict[str, list[dict]] = {
        "ridge_multitask": [], **{name: [] for name in RULE_MODELS},
    }
    for window in windows:
        ridge_predictions = (
            ridge_path / f"window_{window.window_id:03d}" / "predictions.npz"
        )
        prediction_artifacts["ridge_multitask"].append({
            "window_id": window.window_id,
            "path": str(ridge_predictions.resolve()),
        })
        ridge_metrics = json.loads(
            (ridge_path / f"window_{window.window_id:03d}" / "metrics.json").read_text()
        )
        records.append({"model": "ridge_multitask", "window_id": window.window_id, **ridge_metrics})
        for model in RULE_MODELS:
            prediction = _rule_predictions(
                model, features, returns, direction, volatility, window
            )
            metrics = prediction_metrics(
                expected_return=prediction["expected_return"],
                realized_return=returns[window.test],
                expected_volatility=prediction["expected_volatility"],
                realized_volatility=volatility[window.test],
                direction_probability=prediction["direction_probability"],
                direction=direction[window.test],
                lower_quantile=prediction["lower_quantile"],
                upper_quantile=prediction["upper_quantile"],
                timestamps=timestamps[window.test],
            )
            records.append({"model": model, "window_id": window.window_id, **metrics})
            prediction_path = (
                output / "predictions" / model /
                f"window_{window.window_id:03d}.npz"
            )
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                prediction_path,
                timestamps=timestamps[window.test],
                symbols=np.asarray(arrays["symbols"])[window.test],
                expected_return=np.asarray(prediction["expected_return"], dtype=np.float32),
                expected_volatility=np.asarray(
                    prediction["expected_volatility"], dtype=np.float32
                ),
                direction_probability=np.asarray(
                    prediction["direction_probability"], dtype=np.float32
                ),
                lower_quantile=np.asarray(prediction["lower_quantile"], dtype=np.float32),
                upper_quantile=np.asarray(prediction["upper_quantile"], dtype=np.float32),
                confidence=np.asarray(prediction["confidence"], dtype=np.float32),
            )
            prediction_artifacts[model].append({
                "window_id": window.window_id,
                "path": str(prediction_path.resolve()),
            })

    transformer_run = config.get("transformer_run")
    deep_baseline_run = config.get("deep_baseline_run")
    if deep_baseline_run:
        if "dataset_fingerprint" not in arrays:
            raise ValueError("Deep Baseline Benchmark 数据集缺少 dataset_fingerprint")
        records.extend(_deep_baseline_records(deep_baseline_run, arrays, windows))
        suite = Path(deep_baseline_run)
        suite_manifest = json.loads(
            (suite / "benchmark_suite_manifest.json").read_text(encoding="utf-8")
        )
        for name in DEEP_BASELINE_MODELS:
            run = suite / suite_manifest["runs"][name].get("path", name)
            prediction_artifacts[name] = [{
                "window_id": window.window_id,
                "path": str((run / f"window_{window.window_id:03d}" /
                             "predictions.npz").resolve()),
            } for window in windows]
    if transformer_run:
        if "dataset_fingerprint" not in arrays:
            raise ValueError("包含 Transformer 的 Benchmark 数据集缺少 dataset_fingerprint")
        records.extend(_transformer_records(transformer_run, arrays, windows))
        run = Path(transformer_run)
        prediction_artifacts[TRANSFORMER_MODEL] = [{
            "window_id": window.window_id,
            "path": str((run / f"window_{window.window_id:03d}" /
                         "predictions.npz").resolve()),
        } for window in windows]

    metric_names = tuple(
        key for key in records[0]
        if key not in {"model", "window_id", "selected_alpha", "test_samples"}
        and isinstance(records[0][key], (int, float))
    )
    evaluated_models = ["ridge_multitask", *RULE_MODELS]
    if deep_baseline_run:
        evaluated_models.extend(DEEP_BASELINE_MODELS)
    if transformer_run:
        evaluated_models.append(TRANSFORMER_MODEL)
    aggregate = {}
    for model in evaluated_models:
        selected = [record for record in records if record["model"] == model]
        aggregate[model] = {
            metric: {
                "mean": float(np.mean([record[metric] for record in selected])),
                "std": float(np.std([record[metric] for record in selected], ddof=0)),
            }
            for metric in metric_names if metric in selected[0]
        }
    ridge_by_window = {
        record["window_id"]: record for record in records if record["model"] == "ridge_multitask"
    }
    paired = []
    for record in records:
        if record["model"] == "ridge_multitask":
            continue
        ridge = ridge_by_window[record["window_id"]]
        paired.append({
            "model": record["model"],
            "reference": "ridge_multitask",
            "window_id": record["window_id"],
            "delta_model_minus_reference": {
                metric: float(record[metric] - ridge[metric])
                for metric in metric_names if metric in record and metric in ridge
            },
        })
    _json_write(output / "model_quality.json", {"records": records})
    _json_write(output / "aggregate.json", {"models": aggregate})
    _json_write(output / "paired_window_deltas.json", {"pairs": paired})
    _json_write(output / "prediction_manifest.json", {
        "schema_version": 1,
        "dataset_fingerprint": str(np.asarray(arrays["dataset_fingerprint"]).item()),
        "execution_reference_status": str(np.asarray(
            arrays.get("execution_reference_status", np.asarray("UNDECLARED"))
        ).item()),
        "models": prediction_artifacts,
        "alignment": "timestamp_symbol_exact",
        "execution_alignment": "NEXT_OPEN",
    })
    _json_write(output / "benchmark_manifest.json", {
        "schema_version": 1,
        "benchmark_layer": (
            "model_quality_complete_deep_suite"
            if transformer_run and deep_baseline_run
            else "model_quality_partial"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(dataset_path).resolve()),
        "dataset_fingerprint": (
            str(np.asarray(arrays["dataset_fingerprint"]).item())
            if "dataset_fingerprint" in arrays else None
        ),
        "execution_reference_status": str(np.asarray(
            arrays.get("execution_reference_status", np.asarray("UNDECLARED"))
        ).item()),
        "evaluated_models": evaluated_models,
        "not_evaluated_models": [
            *([] if deep_baseline_run else DEEP_BASELINE_MODELS),
            *([] if transformer_run else [TRANSFORMER_MODEL]),
        ],
        "unavailable_legacy_models": ["transformer_v1"],
        "unavailable_legacy_reason": (
            "Repository history contains no frozen Transformer V1 implementation or artifact; "
            "reconstructing one after seeing V1.1 would not be a valid baseline."
        ),
        "not_evaluated_reason": (
            "Remaining models require completed training runs and a fixed deep-model budget."
        ),
        "windows": len(windows),
        "paired_reference": "ridge_multitask",
    })
    return output

from __future__ import annotations

import copy
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..leakage import audit_dataset
from ..models import LogisticBaseline
from ..training import CrossSectionBatchSampler, expanding_timestamp_splits
from .metrics import prediction_metrics
from .walk_forward import _json_write, _window_split_json


DEEP_BASELINES = ("mlp", "tcn", "gru")
MODEL_FAMILIES = {
    "transformer_v1_1": "TemporalTransformerV1.1",
    "mlp": "MLPMultiTaskBaseline",
    "tcn": "CausalTCNMultiTaskBaseline",
    "gru": "GRUMultiTaskBaseline",
}


def _run_deep_walk_forward(
    config: dict,
    dataset_path: str | Path,
    output_path: str | Path,
    model_name: str,
) -> Path:
    if config.get("enabled") is not True:
        raise RuntimeError("Deep Walk-forward 配置必须显式设置 enabled=true")
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("Deep Walk-forward 需要安装锁定的 ML 依赖") from exc
    from ..models.deep_baselines import build_deep_baseline
    from ..models.temporal_transformer import TemporalTransformerConfig, TemporalTransformerV1
    from ..training.train import multitask_loss_components, seed_everything

    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    required = {
        "features", "valid_mask", "timestamps", "symbols", "expected_return",
        "direction", "realized_volatility", "signal_asof", "dataset_fingerprint",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError("Transformer Walk-forward 数据集缺少字段: " + ", ".join(missing))
    features = np.asarray(arrays["features"], dtype=np.float32)
    source_feature_count = features.shape[2]
    valid_mask = np.asarray(arrays["valid_mask"], dtype=np.uint8)
    timestamps = np.asarray(arrays["timestamps"])
    returns = np.asarray(arrays["expected_return"], dtype=np.float32)
    direction = np.asarray(arrays["direction"], dtype=np.float32)
    volatility = np.asarray(arrays["realized_volatility"], dtype=np.float32)

    split_config = config.get("windows", {})
    windows = expanding_timestamp_splits(
        timestamps,
        minimum_train_timestamps=int(split_config.get("minimum_train_timestamps", 1260)),
        validation_timestamps=int(split_config.get("validation_timestamps", 252)),
        test_timestamps=int(split_config.get("test_timestamps", 252)),
        step_timestamps=int(split_config.get("step_timestamps", 252)),
        purge_timestamps=int(split_config.get("purge_timestamps", 5)),
        embargo_timestamps=int(split_config.get("embargo_timestamps", 5)),
        minimum_windows=int(split_config.get("minimum_windows", 3)),
    )
    training = config.get("training", {})
    device_name = str(training.get("device", "cpu"))
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("配置要求 MPS，但当前 PyTorch/Mac 不支持 MPS")
    device = torch.device(device_name)
    epochs = int(training.get("epochs", 50))
    patience = int(training.get("early_stopping_patience", 8))
    timestamps_per_batch = int(training.get("timestamps_per_batch", 1))
    if epochs <= 0 or patience <= 0 or timestamps_per_batch <= 0:
        raise ValueError("epochs/patience/timestamps_per_batch 必须为正数")
    gradient_clip_norm = float(training.get("gradient_clip_norm", 1.0))
    if gradient_clip_norm <= 0 or not np.isfinite(gradient_clip_norm):
        raise ValueError("gradient_clip_norm 必须为有限正数")
    base_seed = int(config.get("seed", 20260724))
    weights_config = training.get("loss_weights", {})
    loss_weights = {
        "return_weight": float(weights_config.get("return", 1.0)),
        "direction_weight": float(weights_config.get("direction", 0.25)),
        "volatility_weight": float(weights_config.get("volatility", 0.25)),
        "quantile_weight": float(weights_config.get("quantile", 0.25)),
        "rank_weight": float(weights_config.get("rank", 0.10)),
    }
    configured_model = config.get("model", {})
    feature_indices = np.asarray(
        configured_model.get("feature_indices", list(range(features.shape[2]))),
        dtype=np.int64,
    )
    if (
        feature_indices.ndim != 1 or feature_indices.size == 0
        or np.unique(feature_indices).size != feature_indices.size
        or feature_indices.min() < 0 or feature_indices.max() >= features.shape[2]
    ):
        raise ValueError("model.feature_indices 无效")
    features = features[:, :, feature_indices]
    if model_name == "transformer_v1_1":
        model_values = {
            name: value for name, value in configured_model.items()
            if name in {
                "static_feature_count", "d_model", "nhead", "num_layers",
                "dim_feedforward", "dropout", "normalization_clip",
            }
        }
        unknown = sorted(
            set(configured_model) - set(model_values) - {"feature_indices"}
        )
        if unknown:
            raise ValueError("Transformer 包含不支持的模型参数: " + ", ".join(unknown))
        model_config = TemporalTransformerConfig(
            feature_count=features.shape[2], lookback=features.shape[1], **model_values
        )
        model_factory = lambda: TemporalTransformerV1(model_config)
    else:
        baseline_values = {
            name: value for name, value in configured_model.items()
            if name != "feature_indices"
        }
        prototype = build_deep_baseline(
            model_name, features.shape[2], features.shape[1], baseline_values
        )
        model_config = prototype.config
        del prototype
        model_factory = lambda: build_deep_baseline(
            model_name, features.shape[2], features.shape[1], baseline_values
        )
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    all_metrics = []

    def tensors(indices):
        return TensorDataset(
            torch.from_numpy(features[indices]),
            torch.from_numpy(valid_mask[indices]),
            torch.from_numpy(timestamps[indices]),
            torch.from_numpy(returns[indices]),
            torch.from_numpy(direction[indices]),
            torch.from_numpy(volatility[indices]),
        )

    def loader(indices, *, shuffle: bool, seed: int):
        return DataLoader(
            tensors(indices),
            batch_sampler=CrossSectionBatchSampler(
                timestamps[indices], timestamps_per_batch=timestamps_per_batch,
                shuffle=shuffle, seed=seed,
            ),
        )

    def target_from(batch, target_device):
        _, _, batch_timestamps, expected, target_direction, target_volatility = batch
        return {
            "timestamps": batch_timestamps.to(target_device),
            "expected_return": expected.to(target_device),
            "direction": target_direction.to(target_device),
            "realized_volatility": target_volatility.to(target_device),
        }

    component_names = ("return", "direction", "volatility", "quantile", "rank", "total")
    for window in windows:
        root = output / f"window_{window.window_id:03d}"
        root.mkdir()
        fit_end = np.asarray(arrays["signal_asof"])[window.train].max()
        leakage = audit_dataset(arrays, split=window, normalizer_fit_end=fit_end)
        leakage.write(root / "leakage_report.json")
        leakage.require_pass()
        _json_write(root / "split.json", _window_split_json(window))

        seed = base_seed + window.window_id - 1
        seed_everything(seed)
        model = model_factory().to(device)
        train_tokens = features[window.train][valid_mask[window.train].astype(bool)]
        mean = train_tokens.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = train_tokens.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        model.set_normalization(mean, scale)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training.get("learning_rate", 3e-4)),
            weight_decay=float(training.get("weight_decay", 1e-2)),
        )
        train_loader = loader(window.train, shuffle=True, seed=seed)
        validation_loader = loader(window.validation, shuffle=False, seed=seed)
        history = []
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        for epoch in range(epochs):
            model.train()
            train_totals = {name: 0.0 for name in component_names}
            train_count = 0
            for batch in train_loader:
                feature, mask = batch[0].to(device), batch[1].to(device)
                target = target_from(batch, device)
                optimizer.zero_grad(set_to_none=True)
                components = multitask_loss_components(
                    model(feature, mask), target, **loss_weights
                )
                components["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                size = feature.shape[0]
                train_count += size
                for name, value in components.items():
                    train_totals[name] += float(value.detach().cpu()) * size

            model.eval()
            validation_totals = {name: 0.0 for name in component_names}
            validation_count = 0
            with torch.no_grad():
                for batch in validation_loader:
                    feature, mask = batch[0].to(device), batch[1].to(device)
                    target = target_from(batch, device)
                    components = multitask_loss_components(
                        model(feature, mask), target, **loss_weights
                    )
                    size = feature.shape[0]
                    validation_count += size
                    for name, value in components.items():
                        validation_totals[name] += float(value.cpu()) * size
            train_epoch = {name: value / train_count for name, value in train_totals.items()}
            validation_epoch = {
                name: value / validation_count for name, value in validation_totals.items()
            }
            history.append({
                "epoch": epoch + 1, "train": train_epoch,
                "validation": validation_epoch,
            })
            if validation_epoch["total"] < best_loss - 1e-10:
                best_loss = validation_epoch["total"]
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        if best_state is None:
            raise RuntimeError("Transformer 训练未产生有效 checkpoint")
        model.load_state_dict(best_state)
        model.to(device).eval()

        def predict(indices):
            values = {
                name: [] for name in (
                    "expected_return", "expected_volatility", "direction_logits",
                    "direction_probability", "lower_quantile", "upper_quantile",
                )
            }
            with torch.no_grad():
                for batch in loader(indices, shuffle=False, seed=seed):
                    prediction = model(batch[0].to(device), batch[1].to(device))
                    for name in values:
                        values[name].append(prediction[name].detach().cpu().numpy())
            return {name: np.concatenate(parts) for name, parts in values.items()}

        validation_prediction = predict(window.validation)
        calibrator = LogisticBaseline(alpha=1.0).fit(
            validation_prediction["direction_logits"][:, None],
            direction[window.validation],
        )
        model.set_direction_calibration(
            calibrator.coefficients_[0], calibrator.coefficients_[1]
        )
        test_prediction = predict(window.test)
        probability = test_prediction["direction_probability"]
        confidence = np.maximum(probability, 1.0 - probability)
        metrics = prediction_metrics(
            expected_return=test_prediction["expected_return"],
            realized_return=returns[window.test],
            expected_volatility=test_prediction["expected_volatility"],
            realized_volatility=volatility[window.test],
            direction_probability=probability,
            direction=direction[window.test],
            lower_quantile=test_prediction["lower_quantile"],
            upper_quantile=test_prediction["upper_quantile"],
            timestamps=timestamps[window.test],
        )
        metrics.update({
            "window_id": window.window_id,
            "best_validation_loss": best_loss,
            "epochs_completed": len(history),
            "test_samples": int(len(window.test)),
        })
        all_metrics.append(metrics)
        _json_write(root / "metrics.json", metrics)
        _json_write(root / "history.json", {"epochs": history})
        _json_write(root / "calibration.json", {
            "method": "platt_validation_only",
            "input": "direction_logits",
            "intercept": float(calibrator.coefficients_[0]),
            "slope": float(calibrator.coefficients_[1]),
            "coefficients": calibrator.coefficients_.tolist(),
        })
        _json_write(root / "normalization.json", {
            "method": "mean_std",
            "clip": model_config.normalization_clip,
            "fit_start": int(np.asarray(arrays["signal_asof"])[window.train].min()),
            "fit_end": int(fit_end),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        })
        (root / "label_spec.json").write_text(
            str(np.asarray(arrays["label_spec_json"]).item()), encoding="utf-8"
        )
        torch.save({
            "model_config": dataclasses.asdict(model_config),
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "feature_indices": feature_indices.tolist(),
            "seed": seed,
            "best_validation_loss": best_loss,
            "label_spec_sha256": str(
                np.asarray(arrays["label_spec_sha256"]).item()
            ),
            "training_dataset_fingerprint": str(
                np.asarray(arrays["dataset_fingerprint"]).item()
            ),
            "model_name": model_name,
            "model_family": MODEL_FAMILIES[model_name],
        }, root / "checkpoint.pt")
        np.savez_compressed(
            root / "predictions.npz",
            timestamps=timestamps[window.test],
            symbols=np.asarray(arrays["symbols"])[window.test],
            expected_return=test_prediction["expected_return"].astype(np.float32),
            expected_volatility=test_prediction["expected_volatility"].astype(np.float32),
            direction_probability=probability.astype(np.float32),
            lower_quantile=test_prediction["lower_quantile"].astype(np.float32),
            upper_quantile=test_prediction["upper_quantile"].astype(np.float32),
            confidence=confidence.astype(np.float32),
            realized_return=returns[window.test],
        )

    metric_names = [
        name for name, value in all_metrics[0].items()
        if isinstance(value, (int, float)) and name != "window_id"
    ]
    _json_write(output / "aggregate" / "summary.json", {
        "windows": len(windows),
        "metrics": {
            name: {
                "mean": float(np.mean([item[name] for item in all_metrics])),
                "std": float(np.std([item[name] for item in all_metrics], ddof=0)),
            }
            for name in metric_names
        },
    })
    _json_write(output / "experiment_manifest.json", {
        "schema_version": 1,
        "pipeline": "deep_expanding_walk_forward",
        "model_name": model_name,
        "model_family": MODEL_FAMILIES[model_name],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(dataset_path).resolve()),
        "dataset_fingerprint": str(np.asarray(arrays["dataset_fingerprint"]).item()),
        "execution_reference_status": str(np.asarray(
            arrays.get("execution_reference_status", np.asarray("UNDECLARED"))
        ).item()),
        "device": device_name,
        "feature_indices": feature_indices.tolist(),
        "analysis_only_feature_subset": feature_indices.tolist() != list(
            range(source_feature_count)
        ),
        "windows": len(windows),
        "test_evaluation_policy": "once_after_validation_freeze",
    })
    return output


def run_transformer_walk_forward(
    config: dict,
    dataset_path: str | Path,
    output_path: str | Path,
) -> Path:
    return _run_deep_walk_forward(
        config, dataset_path, output_path, "transformer_v1_1"
    )


def run_deep_baseline_suite(
    config: dict,
    dataset_path: str | Path,
    output_path: str | Path,
) -> Path:
    if config.get("enabled") is not True:
        raise RuntimeError("Deep Baseline Suite 配置必须显式设置 enabled=true")
    names = tuple(config.get("models", DEEP_BASELINES))
    if not names or len(set(names)) != len(names) or set(names) - set(DEEP_BASELINES):
        raise ValueError("models 必须是 mlp/tcn/gru 的非空、不重复子集")
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    model_configs = config.get("model_configs", {})
    unknown_configs = sorted(set(model_configs) - set(names))
    if unknown_configs:
        raise ValueError("存在未运行模型的配置: " + ", ".join(unknown_configs))
    runs = {}
    for name in names:
        model_config = copy.deepcopy(config)
        model_config.pop("models", None)
        model_config.pop("model_configs", None)
        model_config["model"] = copy.deepcopy(model_configs.get(name, {}))
        path = _run_deep_walk_forward(
            model_config, dataset_path, output / name, name
        )
        manifest = json.loads(
            (path / "experiment_manifest.json").read_text(encoding="utf-8")
        )
        runs[name] = {
            "path": name,
            "model_family": manifest["model_family"],
            "windows": manifest["windows"],
            "dataset_fingerprint": manifest["dataset_fingerprint"],
        }
    _json_write(output / "benchmark_suite_manifest.json", {
        "schema_version": 1,
        "pipeline": "deep_baseline_suite",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": list(names),
        "runs": runs,
        "fairness_control": (
            "Identical dataset, walk-forward windows, seed schedule, loss heads, "
            "optimizer, early stopping, and training budget; only encoder differs."
        ),
    })
    return output

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import BAR_V1, build_windows
from .export.manifest import (
    ModelManifest, OUTPUT_NAMES, sha256_file, validate_artifact,
)
from .features import build_bar_v1
from .labels import (
    LabelSpecV2,
    RankingScoreMode,
    RankingScoreSpecV1,
    build_label_v2,
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _load_config(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("非 JSON 配置需要安装 PyYAML") from exc
        value = yaml.safe_load(text)
        if not isinstance(value, dict):
            raise ValueError("配置根节点必须是对象")
        return value


def _require_enabled(config: dict) -> None:
    if config.get("enabled") is not True:
        raise RuntimeError("ML 配置默认关闭；请在训练机审核配置后显式设置 enabled=true")


def _read_table(path: str | Path):
    import pandas as pd
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"symbol": "string"})
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("训练数据仅支持 CSV、Parquet 或 PQ")


def _load_json_object(path: str | Path | None) -> dict:
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("回测配置根节点必须是对象")
    return value


def _reject_unknown_keys(value: dict, allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} 包含不支持的字段: {', '.join(unknown)}")


def _phase1b_specs(config: dict) -> tuple[LabelSpecV2, RankingScoreSpecV1]:
    ranking = config.get("ranking_score", {})
    if not isinstance(ranking, dict):
        raise ValueError("ranking_score 配置必须是对象")
    score_spec = RankingScoreSpecV1(
        mode=RankingScoreMode(ranking.get("mode", "raw_return")),
        production_top_k=int(ranking.get("production_top_k", 20)),
        risk_floor=float(ranking.get("risk_floor", 1e-4)),
        cost_proxy_bps=float(ranking.get("cost_proxy_bps", 0.0)),
        winsor_lower_quantile=float(ranking.get("winsor_lower_quantile", 0.01)),
        winsor_upper_quantile=float(ranking.get("winsor_upper_quantile", 0.99)),
        rank_temperature=float(ranking.get("rank_temperature", 1.0)),
        lambda_rank=float(ranking.get("lambda_rank", 0.1)),
        target_tie_policy=str(ranking.get("target_tie_policy", "symbol_ascending")),
    )
    labels = config.get("label_v2", {})
    if not isinstance(labels, dict):
        raise ValueError("label_v2 配置必须是对象")
    label_spec = LabelSpecV2(
        horizon_bars=int(labels.get(
            "horizon_bars", config.get("label_horizon_bars", 5)
        )),
        execution_lag_bars=int(labels.get("execution_lag_bars", 1)),
        direction_threshold=float(labels.get("direction_threshold", 0.0)),
        direction_temperature=float(labels.get("direction_temperature", 0.0025)),
        ranking=score_spec,
    )
    return label_spec, score_spec


def _backtest_artifact(
    artifact_path: str | Path,
    data_path: str | Path,
    *,
    initial_cash: float = 1_000_000.0,
    config_path: str | Path | None = None,
    output_path: str | Path | None = None,
    _engine_module=None,
) -> dict:
    import pandas as pd

    manifest = validate_artifact(artifact_path)
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash 必须是有限正数")
    engine_module = _engine_module or importlib.import_module("cpp_engine")
    if not getattr(engine_module, "__ml_enabled__", False):
        raise RuntimeError("cpp_engine 未启用 ML；请使用 QBT_ENABLE_ML=ON 重新构建")
    if getattr(engine_module, "__ml_backend__", None) != "onnxruntime":
        raise RuntimeError("制品回测要求 cpp_engine 使用 ONNX Runtime 后端")

    config = _load_json_object(config_path)
    _reject_unknown_keys(config, {"policy", "risk", "runtime", "execution"}, "回测配置")
    sections = {}
    allowed = {
        "policy": {
            "max_positions", "max_position_weight", "minimum_expected_return",
            "minimum_ranking_score", "minimum_confidence",
        },
        "risk": {"kill_switch", "require_trusted_market", "max_order_quantity"},
        "runtime": {
            "intra_op_threads", "inter_op_threads", "max_batch_size",
            "deadline_ns", "max_order_intents",
        },
        "execution": {
            "max_volume_participation", "slippage_bps", "enforce_price_limits",
            "enforce_t_plus_one", "enforce_board_lot", "allow_short", "enforce_cash",
            "market_order_price_buffer_bps",
        },
    }
    for name, keys in allowed.items():
        section = config.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{name} 配置必须是对象")
        _reject_unknown_keys(section, keys, name)
        sections[name] = section

    execution = engine_module.ExecutionConfig()
    for name, value in sections["execution"].items():
        setattr(execution, name, value)
    engine = engine_module.BacktestEngine(
        initial_cash, engine_module.FillTiming.NEXT_OPEN, execution
    )
    if not hasattr(engine, "set_model_strategy"):
        raise RuntimeError("当前 cpp_engine 缺少模型策略绑定；请重新构建 ML 扩展")
    engine.set_model_strategy(
        str(Path(artifact_path).resolve()), sections["policy"], sections["risk"],
        sections["runtime"],
    )

    table = _read_table(data_path)
    # 复用参考特征实现完成 schema、主键和 OHLCV 输入校验。
    build_bar_v1(table)
    table = table.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)

    def optional(row, name, default):
        if name not in table.columns:
            return default
        value = row[name]
        return default if value is None or bool(pd.isna(value)) else value

    rows = 0
    batch_count = 0
    last_timestamp = None
    for timestamp, cross_section in table.groupby("timestamp", sort=True):
        batch = []
        for _, row in cross_section.iterrows():
            market = engine_module.MarketSnapshot()
            market.timestamp = int(timestamp)
            market.symbol = str(row["symbol"])
            for name in ("open", "high", "low", "close"):
                setattr(market, name, float(row[name]))
                setattr(market, f"signal_{name}", float(optional(
                    row, f"signal_{name}", row[name]
                )))
            market.volume = int(row["volume"])
            market.is_listed = bool(optional(row, "is_listed", True))
            market.is_suspended = bool(optional(row, "is_suspended", False))
            market.is_st = bool(optional(row, "is_st", False))
            market.upper_limit = float(optional(row, "upper_limit", 0.0))
            market.lower_limit = float(optional(row, "lower_limit", 0.0))
            market.lot_size = int(optional(row, "lot_size", 100))
            market.min_buy_quantity = int(optional(
                row, "min_buy_quantity", market.lot_size
            ))
            batch.append(market)
        engine.process_market_data_batch(batch)
        rows += len(batch)
        batch_count += 1
        last_timestamp = int(timestamp)
    if last_timestamp is None:
        raise ValueError("回测行情不能为空")
    engine.finalize(last_timestamp)
    result = {
        "model_id": manifest.model_id,
        "model_version": manifest.model_version,
        "rows": rows,
        "batches": batch_count,
        "symbols": int(table["symbol"].nunique()),
        "start_timestamp": int(table["timestamp"].iloc[0]),
        "end_timestamp": last_timestamp,
        "initial_cash": float(initial_cash),
        "cash": float(engine.get_cash()),
        "equity": float(engine.get_equity()),
        "total_return": float(engine.get_total_return()),
        "sharpe_ratio": float(engine.get_sharpe_ratio()),
        "max_drawdown": float(engine.get_max_drawdown()),
        "orders": int(engine.get_order_count()),
        "trades": int(engine.get_trade_count()),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def _build_dataset(config: dict, output_override: str | None) -> Path:
    _require_enabled(config)
    data = config.get("data", {})
    source = data.get("source")
    output = output_override or data.get("dataset_output")
    if not source or "CONFIGURE_" in str(source) or not output:
        raise ValueError("必须配置 data.source 和 data.dataset_output")
    table = _read_table(source)
    frame = build_bar_v1(table)
    label_spec, ranking_spec = _phase1b_specs(config)
    labels = build_label_v2(table, label_spec)
    windows = build_windows(frame, int(config.get("lookback", 64)))
    aligned = labels.iloc[windows.row_indices].reset_index(drop=True)
    selected = (aligned["label_valid"].to_numpy(dtype=bool) &
                windows.valid_mask[:, -1].astype(bool))
    selected_indices = np.flatnonzero(selected)
    order = np.lexsort((windows.symbols[selected_indices].astype(str),
                        windows.timestamps[selected_indices]))
    selected_indices = selected_indices[order]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=windows.features[selected_indices],
        valid_mask=windows.valid_mask[selected_indices],
        timestamps=windows.timestamps[selected_indices],
        symbols=windows.symbols[selected_indices].astype(str),
        expected_return=aligned.iloc[selected_indices]["expected_return"].to_numpy(np.float32),
        direction=aligned.iloc[selected_indices]["direction"].to_numpy(np.float32),
        realized_volatility=aligned.iloc[selected_indices]["realized_volatility"].to_numpy(np.float32),
        downside_semivol=aligned.iloc[selected_indices]["downside_semivol"].to_numpy(np.float32),
        risk_adjusted_return=aligned.iloc[selected_indices]["risk_adjusted_return"].to_numpy(np.float32),
        rank_utility=aligned.iloc[selected_indices]["rank_utility"].to_numpy(np.float32),
        rank_relevance=aligned.iloc[selected_indices]["rank_relevance"].to_numpy(np.float32),
        label_spec_json=np.asarray(label_spec.canonical_json),
        label_spec_sha256=np.asarray(label_spec.sha256),
        ranking_score_spec_json=np.asarray(ranking_spec.canonical_json),
        ranking_score_spec_sha256=np.asarray(ranking_spec.sha256),
        feature_schema_json=np.asarray(BAR_V1.canonical_json),
        feature_schema_sha256=np.asarray(BAR_V1.sha256),
    )
    return output


def _phase2b_audit(dataset_path: str | Path, output_path: str | Path) -> Path:
    """Materialize deterministic Phase 2B noise and stress-set artifacts."""
    from .training.robust_training import build_stress_sets, direction_noise_audit

    dataset = Path(dataset_path)
    output = Path(output_path)
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(dataset, allow_pickle=False) as values:
        if "features" not in values or "valid_mask" not in values or "direction" not in values:
            raise ValueError("Phase 2B audit 需要 features、valid_mask、direction")
        features = values["features"].astype(np.float32, copy=False)
        valid_mask = values["valid_mask"].astype(np.uint8, copy=False)
        direction = values["direction"].astype(np.float32, copy=False)
        feature_names = (
            BAR_V1.feature_names
            if features.shape[2] == len(BAR_V1.feature_names)
            else tuple(f"feature_{index}" for index in range(features.shape[2]))
        )
    stress_sets = build_stress_sets(
        features, valid_mask, feature_names=feature_names,
    )
    audit = direction_noise_audit(direction)
    (output / "direction_noise_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    stress_manifest = {
        "schema_version": 1,
        "kind": "phase2b_stress_sets",
        "source_dataset_sha256": sha256_file(dataset),
        "feature_schema_sha256": (
            BAR_V1.sha256 if feature_names == BAR_V1.feature_names else None
        ),
        "sets": {},
    }
    for name, value in stress_sets.items():
        destination = output / f"stress_{name}.npz"
        np.savez_compressed(
            destination, features=value["features"], valid_mask=value["valid_mask"],
        )
        stress_manifest["sets"][name] = {
            "path": destination.name,
            "sha256": sha256_file(destination),
            "metadata": value["metadata"],
        }
    stress_manifest["manifest_sha256"] = _canonical_sha256(stress_manifest)
    (output / "stress_manifest.json").write_text(
        json.dumps(stress_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _train(
    config: dict,
    dataset_path: str | None,
    output_override: str | None,
    *,
    _split_override=None,
) -> Path:
    _require_enabled(config)
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("训练机需要安装项目的 ml 可选依赖") from exc
    from .models.temporal_transformer import TemporalTransformerConfig, TemporalTransformerV1
    from .features.scaling import FeatureStandardizerV1
    from .training.ranking import CrossSectionBatchSampler, ranking_oos_metrics
    from .training.gradient_methods import (
        GRADIENT_TASKS,
        GradientConflictRecorder,
        GradientDiagnosticSpecV1,
        GradNormController,
        GradNormSpecV1,
        PCGradSpecV1,
        backward_pcgrad,
        shared_named_parameters,
    )
    from .training.train import (
        KENDALL_TASKS,
        KendallTaskWeights,
        multitask_loss,
        multitask_loss_components,
        seed_everything,
    )
    from .training.robust_training import (
        apl_direction_loss,
        feature_pgd_perturbation,
        latent_fgm_loss,
        structured_missing_augmentation,
        validate_robust_training_config,
    )
    from .training.walk_forward import chronological_timestamp_split

    training = config.get("training", {})
    evaluation_split = str(training.get("evaluation_split", "test"))
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("training.evaluation_split 必须为 validation/test")
    cpu_threads = int(training.get("cpu_threads", 0))
    if cpu_threads < 0:
        raise ValueError("training.cpu_threads 不能为负数")
    if cpu_threads:
        torch.set_num_threads(cpu_threads)
    dataset_path = dataset_path or training.get("dataset")
    output = output_override or training.get("output")
    device = training.get("device")
    if not dataset_path or not output or not device or "CONFIGURE_" in str(device):
        raise ValueError("必须配置 training.dataset、training.output 和 training.device")
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch/CUDA 不可用；禁止静默回退 CPU")
    deterministic_algorithms = bool(training.get("deterministic_algorithms", True))
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_algorithms
    runtime_backend = {
        "requested_device": device,
        "resolved_device": str(torch.device(device)),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime_version": str(torch.version.cuda),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda") else None
        ),
        "deterministic_algorithms": deterministic_algorithms,
    }
    with np.load(dataset_path, allow_pickle=False) as data:
        features = data["features"].astype(np.float32, copy=False)
        valid_mask = data["valid_mask"].astype(np.uint8, copy=False)
        expected_return = data["expected_return"].astype(np.float32, copy=False)
        direction = data["direction"].astype(np.float32, copy=False)
        realized_volatility = data["realized_volatility"].astype(np.float32, copy=False)
        rank_utility = data["rank_utility"].astype(np.float32, copy=False) \
            if "rank_utility" in data else None
        rank_relevance = data["rank_relevance"].astype(np.float32, copy=False) \
            if "rank_relevance" in data else None
        symbols = data["symbols"].astype(str)
        timestamps = data["timestamps"]
        schema_hash = str(data["feature_schema_sha256"])
        label_spec_hash = str(data["label_spec_sha256"]) \
            if "label_spec_sha256" in data else "legacy-v1"
        ranking_spec_hash = str(data["ranking_score_spec_sha256"]) \
            if "ranking_score_spec_sha256" in data else "legacy-v1"
    split_config = config.get("split", {})
    label_spec, _ = _phase1b_specs(config)
    minimum_purge = label_spec.horizon_bars + label_spec.execution_lag_bars
    if int(split_config.get("purge_timestamps", 0)) < minimum_purge:
        raise ValueError(
            f"purge_timestamps 至少为 label horizon + execution lag = {minimum_purge}"
        )
    split = _split_override or chronological_timestamp_split(
        timestamps,
        train_fraction=float(split_config.get("train_fraction", 0.70)),
        validation_fraction=float(split_config.get("validation_fraction", 0.15)),
        test_fraction=float(split_config.get("test_fraction", 0.15)),
        purge_timestamps=int(split_config.get("purge_timestamps", 0)),
        embargo_timestamps=int(split_config.get("embargo_timestamps", 0)),
    )

    symbol_mapping = {
        symbol: index for index, symbol in enumerate(sorted(set(symbols.tolist())))
    }
    symbol_tie_breaker = np.asarray(
        [symbol_mapping[symbol] for symbol in symbols], dtype=np.int64
    )
    ranking_config = config.get("ranking", {})
    ranking_variant = str(ranking_config.get("loss_variant", "none"))
    if ranking_variant not in {"none", "legacy", "listmle", "lambda"}:
        raise ValueError("ranking.loss_variant 必须为 none/legacy/listmle/lambda")
    if ranking_variant != "none" and (rank_utility is None or rank_relevance is None):
        raise ValueError("ranking loss 需要使用 LabelSpec V2 重建数据集")
    if ranking_variant != "none" and int(
        ranking_config.get("cross_sections_per_batch", 1)
    ) != 1:
        raise ValueError("Phase 1B V1 要求每个训练 batch 恰好一个 timestamp")
    _, ranking_score_spec = _phase1b_specs(config)
    loss_kwargs = {
        "ranking_variant": ranking_variant,
        "ranking_cutoff": ranking_score_spec.production_top_k,
        "rank_temperature": ranking_score_spec.rank_temperature,
        "ranking_score_mode": ranking_score_spec.mode.value,
        "risk_floor": ranking_score_spec.risk_floor,
        "rank_weight": ranking_score_spec.lambda_rank if ranking_variant != "none" else 0.0,
        "return_weight": float(ranking_config.get("return_weight", 1.0)),
        "direction_weight": float(ranking_config.get("direction_weight", 0.25)),
        "volatility_weight": float(ranking_config.get("volatility_weight", 0.25)),
        "quantile_weight": float(ranking_config.get("quantile_weight", 0.25)),
    }
    weighting_config = config.get("multitask_weighting", {})
    if not isinstance(weighting_config, dict):
        raise ValueError("multitask_weighting 必须是对象")
    weighting_mode = str(weighting_config.get("mode", "fixed"))
    if weighting_mode not in {"fixed", "kendall"}:
        raise ValueError("multitask_weighting.mode 必须为 fixed/kendall")
    if weighting_mode == "kendall" and str(
        weighting_config.get("frozen_ranking_loss", "")
    ) != ranking_variant:
        raise ValueError("Kendall 必须显式绑定已冻结的 ranking loss")
    gradient_config = config.get("gradient_optimization", {})
    if not isinstance(gradient_config, dict):
        raise ValueError("gradient_optimization 必须是对象")
    _reject_unknown_keys(gradient_config, {
        "mode", "hypothesis_id", "cadence_steps", "seed",
        "zero_norm_epsilon", "alpha", "update_every_steps",
        "weight_learning_rate", "accumulation_steps", "amp_enabled",
        "loss_scale", "fold", "regime",
    }, "gradient_optimization")
    gradient_mode = str(gradient_config.get("mode", "none"))
    if gradient_mode not in {"none", "diagnostics", "pcgrad", "gradnorm"}:
        raise ValueError(
            "gradient_optimization.mode 必须为 none/diagnostics/pcgrad/gradnorm"
        )
    if gradient_mode != "none":
        if weighting_mode != "fixed" or ranking_variant != "legacy":
            raise ValueError("Phase 1E 仅允许 legacy rank + fixed weights")
        frozen_weights = {
            "return_weight": 1.0,
            "direction_weight": 0.25,
            "volatility_weight": 0.25,
            "quantile_weight": 0.25,
            "rank_weight": 0.1,
        }
        if any(loss_kwargs[key] != value for key, value in frozen_weights.items()):
            raise ValueError("Phase 1E 必须保持冻结冠军 scalar weights")
        if int(gradient_config.get("accumulation_steps", 1)) != 1:
            raise ValueError("当前训练入口仅支持 accumulation_steps=1")
        if bool(gradient_config.get("amp_enabled", False)):
            raise ValueError("当前训练入口尚未启用 AMP")
        if float(gradient_config.get("loss_scale", 1.0)) != 1.0:
            raise ValueError("非 AMP 训练的 loss_scale 必须为 1.0")
    hypothesis_id = str(gradient_config.get("hypothesis_id", ""))
    if gradient_mode in {"pcgrad", "gradnorm"} and not hypothesis_id:
        raise ValueError("PCGrad/GradNorm challenger 必须提供新 hypothesis_id")
    robust_config = config.get("robust_training", {})
    if not isinstance(robust_config, dict):
        raise ValueError("robust_training 必须是对象")
    dataset_feature_names = (
        BAR_V1.feature_names
        if features.shape[2] == len(BAR_V1.feature_names)
        else tuple(f"feature_{index}" for index in range(features.shape[2]))
    )
    robust_spec = validate_robust_training_config(
        robust_config, feature_names=dataset_feature_names,
    )
    if robust_spec.mode != "none" and gradient_mode != "none":
        raise ValueError("Phase 2B 候选必须与 Phase 1E 梯度 challenger 分开实验")

    preprocessing_config = config.get("preprocessing", {})
    if not isinstance(preprocessing_config, dict):
        raise ValueError("preprocessing 必须是对象")
    _reject_unknown_keys(
        preprocessing_config, {"mode", "scale_floor"}, "preprocessing",
    )
    preprocessing_mode = str(preprocessing_config.get("mode", "none"))
    if preprocessing_mode not in {"none", "train_fold_standardizer_v1"}:
        raise ValueError(
            "preprocessing.mode 必须为 none/train_fold_standardizer_v1"
        )
    standardizer = None
    if preprocessing_mode == "train_fold_standardizer_v1":
        standardizer = FeatureStandardizerV1.fit(
            features[split.train], valid_mask[split.train], dataset_feature_names,
            scale_floor=float(preprocessing_config.get("scale_floor", 1e-6)),
        )

    def tensors_for(indices):
        return TensorDataset(
            torch.from_numpy(features[indices]), torch.from_numpy(valid_mask[indices]),
            torch.from_numpy(expected_return[indices]), torch.from_numpy(direction[indices]),
            torch.from_numpy(realized_volatility[indices]),
            torch.from_numpy(
                rank_utility[indices] if rank_utility is not None else expected_return[indices]
            ),
            torch.from_numpy(
                rank_relevance[indices] if rank_relevance is not None
                else np.zeros(indices.size, dtype=np.float32)
            ),
            torch.from_numpy(timestamps[indices].astype(np.int64, copy=False)),
            torch.from_numpy(symbol_tie_breaker[indices]),
        )

    train_tensors = tensors_for(split.train)
    seed = int(config.get("seed", 20260724))
    seed_everything(seed)
    allowed = {"static_feature_count", "d_model", "nhead", "num_layers",
               "dim_feedforward", "dropout"}
    model_config = TemporalTransformerConfig(
        feature_count=features.shape[2], lookback=features.shape[1],
        **{key: value for key, value in config.get("model", {}).items() if key in allowed},
        input_mean=standardizer.mean if standardizer is not None else (),
        input_scale=standardizer.scale if standardizer is not None else (),
        input_protected=standardizer.protected if standardizer is not None else (),
    )
    model = TemporalTransformerV1(model_config).to(device)
    kendall = None
    gradient_spec = None
    gradient_recorder = None
    gradnorm = None
    shared_named = None
    mechanism_diagnostics = []
    robust_diagnostics = []
    if gradient_mode != "none":
        shared_named = shared_named_parameters(model)
    if gradient_mode == "diagnostics":
        gradient_spec = GradientDiagnosticSpecV1(
            cadence_steps=int(gradient_config.get("cadence_steps", 1)),
            seed=int(gradient_config.get("seed", seed)),
            accumulation_steps=1,
            amp_enabled=False,
            loss_scale=1.0,
        )
        gradient_recorder = GradientConflictRecorder(gradient_spec)
    elif gradient_mode == "pcgrad":
        gradient_spec = PCGradSpecV1(
            seed=int(gradient_config.get("seed", seed)),
            zero_norm_epsilon=float(gradient_config.get(
                "zero_norm_epsilon", 1e-12
            )),
            accumulation_steps=1,
            gradients_unscaled=True,
            hypothesis_id=hypothesis_id,
        )
    elif gradient_mode == "gradnorm":
        gradient_spec = GradNormSpecV1(
            alpha=float(gradient_config.get("alpha", 1.5)),
            rank_weight=0.1,
            update_every_steps=int(gradient_config.get("update_every_steps", 1)),
            hypothesis_id=hypothesis_id,
        )
        gradnorm = GradNormController(gradient_spec).to(device)
    parameter_groups = [{"params": list(model.parameters())}]
    if weighting_mode == "kendall":
        kendall = KendallTaskWeights(
            initial_log_variance=float(weighting_config.get(
                "initial_log_variance", 0.0
            )),
            regularizer=float(weighting_config.get("regularizer", 0.5)),
            minimum=float(weighting_config.get("minimum_log_variance", -6.0)),
            maximum=float(weighting_config.get("maximum_log_variance", 6.0)),
        ).to(device)
        parameter_groups.append({"params": list(kendall.parameters()), "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(
        parameter_groups, lr=float(training.get("learning_rate", 3e-4))
    )
    gradnorm_optimizer = None
    if gradnorm is not None:
        gradnorm_optimizer = torch.optim.Adam(
            gradnorm.parameters(),
            lr=float(gradient_config.get("weight_learning_rate", 1e-3)),
        )

    def compute_loss(prediction, target, *, use_robust_direction=False):
        if kendall is None and gradient_mode == "none" and not use_robust_direction:
            return multitask_loss(prediction, target, **loss_kwargs), None, None
        components = multitask_loss_components(
            prediction,
            target,
            ranking_variant=loss_kwargs["ranking_variant"],
            ranking_cutoff=loss_kwargs["ranking_cutoff"],
            rank_temperature=loss_kwargs["rank_temperature"],
            ranking_score_mode=loss_kwargs["ranking_score_mode"],
            risk_floor=loss_kwargs["risk_floor"],
        )
        if use_robust_direction:
            components["direction"] = apl_direction_loss(
                prediction["direction_probability"], target["direction"],
                alpha=robust_spec.apl_alpha,
                beta=robust_spec.apl_beta,
                label_clip=robust_spec.apl_label_clip,
            )
        if gradnorm is not None:
            weighted = gradnorm.weighted_task_losses(components)
            return gradnorm.model_loss(components), components, weighted
        if kendall is None:
            weighted = {
                "return": loss_kwargs["return_weight"] * components["return"],
                "direction": loss_kwargs["direction_weight"] * components["direction"],
                "volatility": loss_kwargs["volatility_weight"] * components["volatility"],
                "quantile": loss_kwargs["quantile_weight"] * components["quantile"],
                "rank": loss_kwargs["rank_weight"] * components["rank"],
            }
            return (
                weighted["return"] + weighted["direction"] +
                weighted["volatility"] + weighted["quantile"] + weighted["rank"]
            ), components, weighted
        task_loss, weighted = kendall(components)
        return task_loss + loss_kwargs["rank_weight"] * components["rank"], components, weighted
    if ranking_variant == "none":
        loader = DataLoader(
            train_tensors, batch_size=int(training.get("batch_size", 256)),
            shuffle=False, drop_last=False,
        )
    else:
        loader = DataLoader(
            train_tensors,
            batch_sampler=CrossSectionBatchSampler(
                timestamps[split.train],
                cross_sections_per_batch=int(ranking_config.get(
                    "cross_sections_per_batch", 1
                )),
                shuffle=False,
                seed=int(config.get("seed", 20260724)),
            ),
        )
    epoch_count = int(training.get("epochs", 50))
    history = []
    weight_diagnostics = []
    global_step = 0
    model.train()
    for epoch_index in range(epoch_count):
        total = 0.0
        diagnostic_sums = {
            kind: {task: 0.0 for task in KENDALL_TASKS}
            for kind in ("raw_losses", "weighted_losses", "log_variance_gradients")
        }
        diagnostic_batches = 0
        for (feature, mask, expected, target_direction, volatility, utility,
             relevance, batch_timestamps, tie_breaker) in loader:
            feature, mask = feature.to(device), mask.to(device)
            target = {
                "expected_return": expected.to(device),
                "direction": target_direction.to(device),
                "realized_volatility": volatility.to(device),
                "rank_utility": utility.to(device),
                "rank_relevance": relevance.to(device),
                "timestamp": batch_timestamps.to(device),
                "symbol_tie_breaker": tie_breaker.to(device),
            }
            feature_input = feature
            missing_update = None
            if robust_spec.missing_mode == "continuous_center_impute":
                feature_input, missing_update = structured_missing_augmentation(
                    feature, mask,
                    feature_names=robust_spec.feature_names,
                    center=model.input_mean.reshape(-1),
                    rate=robust_spec.missing_rate,
                )
            optimizer.zero_grad(set_to_none=True)
            clean_loss_fn = lambda prediction, clean_target: compute_loss(
                prediction, clean_target,
            )[0]
            if robust_spec.mode == "direction_apl":
                loss, components, weighted = compute_loss(
                    model(feature_input, mask), target, use_robust_direction=True,
                )
            elif robust_spec.mode == "latent_fgm":
                loss, robust_batch = latent_fgm_loss(
                    model, feature_input, mask, target, clean_loss_fn,
                    epsilon=robust_spec.epsilon, beta=robust_spec.beta,
                )
                components, weighted = None, None
                if len(robust_diagnostics) < 256:
                    robust_diagnostics.append(robust_batch)
            elif robust_spec.mode == "feature_pgd":
                clean_prediction = model(feature_input, mask)
                clean_loss = clean_loss_fn(clean_prediction, target)
                standardized_input = model.normalize_features(feature_input, mask)
                parameter_grad_state = [
                    parameter.requires_grad for parameter in model.parameters()
                ]
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                try:
                    adversarial_features = feature_pgd_perturbation(
                        standardized_input, mask,
                        lambda perturbed: clean_loss_fn(
                            model.forward_standardized(perturbed, mask), target,
                        ),
                        epsilon=robust_spec.epsilon,
                        steps=robust_spec.pgd_steps,
                        step_size=robust_spec.pgd_step_size,
                        feature_names=robust_spec.feature_names,
                    )
                finally:
                    for parameter, requires_grad in zip(
                        model.parameters(), parameter_grad_state
                    ):
                        parameter.requires_grad_(requires_grad)
                adversarial_loss = clean_loss_fn(
                    model.forward_standardized(adversarial_features, mask), target,
                )
                loss = clean_loss + robust_spec.beta * adversarial_loss
                components, weighted = None, None
                if len(robust_diagnostics) < 256:
                    robust_diagnostics.append({
                        "clean_loss": float(clean_loss.detach()),
                        "adversarial_loss": float(adversarial_loss.detach()),
                        "clean_return_mae": float(torch.mean(torch.abs(
                            clean_prediction["expected_return"] - target["expected_return"]
                        )).detach()),
                        "adversarial_return_mae": float(torch.mean(torch.abs(
                            model.forward_standardized(adversarial_features, mask)["expected_return"]
                            - target["expected_return"]
                        )).detach()),
                        "clean_direction_brier": float(torch.mean(torch.square(
                            clean_prediction["direction_probability"] - target["direction"]
                        )).detach()),
                        "feature_perturbation_linf": float(
                            (adversarial_features - standardized_input).abs().max().detach()
                        ),
                    })
            elif robust_spec.mode == "structured_missing":
                loss, components, weighted = compute_loss(
                    model(feature_input, mask), target,
                )
            else:
                loss, components, weighted = compute_loss(
                    model(feature_input, mask), target,
                )
            if missing_update is not None and len(robust_diagnostics) < 256:
                robust_diagnostics.append({
                    "missing_mode": robust_spec.missing_mode,
                    "missing_rate": robust_spec.missing_rate,
                    "missing_token_count": int(missing_update.any(dim=-1).sum().detach()),
                    "missing_feature_count": int(missing_update.sum().detach()),
                })
            if gradient_recorder is not None:
                gradient_recorder.sample(
                    components,
                    shared_named,
                    {
                        "return": loss_kwargs["return_weight"],
                        "direction": loss_kwargs["direction_weight"],
                        "volatility": loss_kwargs["volatility_weight"],
                        "quantile": loss_kwargs["quantile_weight"],
                        "rank": loss_kwargs["rank_weight"],
                    },
                    step=global_step,
                    epoch=epoch_index,
                    fold=int(gradient_config.get("fold", 0)),
                    regime=str(gradient_config.get("regime", "ALL")),
                )
                loss.backward()
            elif gradient_mode == "pcgrad":
                mechanism_diagnostics.append(backward_pcgrad(
                    weighted,
                    loss,
                    tuple(parameter for _, parameter in shared_named),
                    gradient_spec,
                    step=global_step,
                ))
            elif gradnorm is not None:
                if global_step % gradient_spec.update_every_steps == 0:
                    gradnorm_optimizer.zero_grad(set_to_none=True)
                    objective, diagnostics = gradnorm.gradnorm_objective(
                        components,
                        tuple(parameter for _, parameter in shared_named),
                    )
                    weight_gradient, = torch.autograd.grad(
                        objective, gradnorm.raw_weights, retain_graph=True
                    )
                    gradnorm.raw_weights.grad = weight_gradient.detach().clone()
                    gradnorm_optimizer.step()
                    diagnostics.update({
                        "step": global_step,
                        "epoch": epoch_index,
                        "objective": float(objective.detach()),
                        "zero_gradient_rate": sum(
                            value <= gradient_spec.epsilon
                            for value in diagnostics["gradient_norms"].values()
                        ) / len(GRADIENT_TASKS[:-1]),
                    })
                    mechanism_diagnostics.append(diagnostics)
                    loss = gradnorm.model_loss(components)
                    weighted = gradnorm.weighted_task_losses(components)
                loss.backward()
            else:
                loss.backward()
            if kendall is not None:
                for task in KENDALL_TASKS:
                    diagnostic_sums["raw_losses"][task] += float(
                        components[task].detach()
                    )
                    diagnostic_sums["weighted_losses"][task] += float(
                        weighted[task].detach()
                    )
                    diagnostic_sums["log_variance_gradients"][task] += float(
                        kendall.log_variances[task].grad.detach()
                    )
                diagnostic_batches += 1
            optimizer.step()
            if kendall is not None:
                kendall.clamp_()
            if ranking_variant == "none":
                total += float(loss.detach()) * feature.shape[0]
            else:
                total += float(loss.detach())
            global_step += 1
        epoch_loss = total / (
            len(split.train) if ranking_variant == "none" else len(loader)
        )
        history.append(epoch_loss)
        print(json.dumps({
            "event": "training_epoch_completed",
            "epoch": epoch_index + 1,
            "epochs": epoch_count,
            "fold": int(gradient_config.get("fold", 0)),
            "gradient_mode": gradient_mode,
            "train_loss": epoch_loss,
        }, sort_keys=True), flush=True)
        if kendall is not None:
            weight_diagnostics.append({
                **{
                    kind: {
                        task: values[task] / diagnostic_batches
                        for task in KENDALL_TASKS
                    }
                    for kind, values in diagnostic_sums.items()
                },
                "log_variances": {
                    task: float(kendall.log_variances[task].detach())
                    for task in KENDALL_TASKS
                },
                "effective_weights": {
                    task: float(torch.exp(-kendall.log_variances[task].detach()))
                    for task in KENDALL_TASKS
                },
            })

    def evaluate(indices):
        evaluation_dataset = tensors_for(indices)
        evaluation_loader = DataLoader(
            evaluation_dataset,
            batch_size=int(training.get("batch_size", 256)),
            shuffle=False,
            drop_last=False,
        ) if ranking_variant == "none" else DataLoader(
            evaluation_dataset,
            batch_sampler=CrossSectionBatchSampler(
                timestamps[indices],
                cross_sections_per_batch=int(ranking_config.get(
                    "cross_sections_per_batch", 1
                )),
                shuffle=False,
                seed=seed,
            ),
        )
        model.eval()
        total = 0.0
        score_parts = []
        utility_parts = []
        relevance_parts = []
        timestamp_parts = []
        symbol_parts = []
        expected_parts = []
        predicted_return_parts = []
        direction_target_parts = []
        direction_prediction_parts = []
        volatility_target_parts = []
        volatility_prediction_parts = []
        lower_parts = []
        upper_parts = []
        confidence_parts = []
        embedding_parts = []
        with torch.no_grad():
            for (feature, mask, expected, target_direction, volatility, utility,
                 relevance, batch_timestamps, tie_breaker) in evaluation_loader:
                feature, mask = feature.to(device), mask.to(device)
                target = {
                    "expected_return": expected.to(device),
                    "direction": target_direction.to(device),
                    "realized_volatility": volatility.to(device),
                    "rank_utility": utility.to(device),
                    "rank_relevance": relevance.to(device),
                    "timestamp": batch_timestamps.to(device),
                    "symbol_tie_breaker": tie_breaker.to(device),
                }
                prediction = model(feature, mask, return_embedding=True)
                loss, _, _ = compute_loss(prediction, target)
                if ranking_variant == "none":
                    total += float(loss) * feature.shape[0]
                else:
                    total += float(loss)
                if ranking_score_spec.mode is RankingScoreMode.RAW_RETURN:
                    ranking_score = prediction["expected_return"]
                else:
                    ranking_score = prediction["expected_return"] / prediction[
                        "expected_volatility"
                    ].clamp_min(ranking_score_spec.risk_floor)
                score_parts.append(ranking_score.detach().cpu().numpy())
                utility_parts.append(utility.numpy())
                relevance_parts.append(relevance.numpy())
                timestamp_parts.append(batch_timestamps.numpy())
                symbol_parts.append(symbols[indices][
                    sum(part.size for part in symbol_parts):
                    sum(part.size for part in symbol_parts) + batch_timestamps.numel()
                ])
                expected_parts.append(expected.numpy())
                predicted_return_parts.append(
                    prediction["expected_return"].detach().cpu().numpy()
                )
                direction_target_parts.append(target_direction.numpy())
                direction_prediction_parts.append(
                    prediction["direction_probability"].detach().cpu().numpy()
                )
                volatility_target_parts.append(volatility.numpy())
                volatility_prediction_parts.append(
                    prediction["expected_volatility"].detach().cpu().numpy()
                )
                lower_parts.append(prediction["lower_quantile"].detach().cpu().numpy())
                upper_parts.append(prediction["upper_quantile"].detach().cpu().numpy())
                confidence_parts.append(
                    prediction["confidence"].detach().cpu().numpy()
                )
                embedding_parts.append(
                    prediction["embedding"].detach().cpu().numpy()
                )
        evaluation_loss = total / (
            len(indices) if ranking_variant == "none" else len(evaluation_loader)
        )
        all_scores = np.concatenate(score_parts)
        all_utility = np.concatenate(utility_parts)
        all_relevance = np.concatenate(relevance_parts)
        all_timestamps = np.concatenate(timestamp_parts)
        all_symbols = np.concatenate(symbol_parts)
        ranking_metrics = ranking_oos_metrics(
            all_scores, all_utility, all_relevance, all_timestamps, all_symbols,
            cutoff=ranking_score_spec.production_top_k,
        )
        all_expected = np.concatenate(expected_parts)
        all_predicted_return = np.concatenate(predicted_return_parts)
        all_direction_target = np.concatenate(direction_target_parts)
        all_direction_prediction = np.concatenate(direction_prediction_parts)
        all_volatility_target = np.concatenate(volatility_target_parts)
        all_volatility_prediction = np.concatenate(volatility_prediction_parts)
        all_lower = np.concatenate(lower_parts)
        all_upper = np.concatenate(upper_parts)
        all_confidence = np.concatenate(confidence_parts)
        all_embeddings = np.concatenate(embedding_parts)
        return {
            "loss": evaluation_loss,
            **dataclasses.asdict(ranking_metrics),
            "head_metrics": {
                "return_mae": float(np.mean(np.abs(
                    all_predicted_return - all_expected
                ))),
                "direction_brier": float(np.mean(np.square(
                    all_direction_prediction - all_direction_target
                ))),
                "volatility_mae": float(np.mean(np.abs(
                    all_volatility_prediction - all_volatility_target
                ))),
                "interval_coverage": float(np.mean(
                    (all_expected >= all_lower) & (all_expected <= all_upper)
                )),
                "interval_width": float(np.mean(all_upper - all_lower)),
            },
            "predictions": {
                "scores": all_scores, "utility": all_utility,
                "relevance": all_relevance, "timestamps": all_timestamps,
                "symbols": all_symbols,
                "expected_return": all_predicted_return,
                "expected_volatility": all_volatility_prediction,
                "direction_probability": all_direction_prediction,
                "lower_quantile": all_lower,
                "upper_quantile": all_upper,
                "confidence": all_confidence,
                "embedding": all_embeddings,
            },
        }

    validation_metrics = evaluate(split.validation)
    test_metrics = evaluate(split.test) if evaluation_split == "test" else None
    gradient_artifact = None
    if gradient_recorder is not None:
        model_contract = {
            "model": dataclasses.asdict(model_config),
            "feature_schema_sha256": schema_hash,
            "label_spec_sha256": label_spec_hash,
            "ranking_score_spec_sha256": ranking_spec_hash,
            "ranking_loss_variant": ranking_variant,
            "loss_weights": {
                task: float(weighted_value)
                for task, weighted_value in zip(GRADIENT_TASKS, (
                    loss_kwargs["return_weight"],
                    loss_kwargs["direction_weight"],
                    loss_kwargs["volatility_weight"],
                    loss_kwargs["quantile_weight"],
                    loss_kwargs["rank_weight"],
                ))
            },
        }
        model_contract_sha256 = hashlib.sha256(json.dumps(
            model_contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        gradient_artifact = gradient_recorder.artifact(
            shared_named,
            model_contract_sha256=model_contract_sha256,
            dataset_sha256=sha256_file(dataset_path),
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validation_predictions = validation_metrics.pop("predictions")
    validation_embeddings = validation_predictions.pop("embedding")
    np.savez_compressed(output / "validation_predictions.npz", **validation_predictions)
    test_predictions = None
    test_embeddings = None
    if test_metrics is not None:
        test_predictions = test_metrics.pop("predictions")
        test_embeddings = test_predictions.pop("embedding")
        np.savez_compressed(output / "test_predictions.npz", **test_predictions)

    anchor_candidates = np.flatnonzero(
        valid_mask.all(axis=1) & np.isfinite(features).all(axis=(1, 2))
    )
    if anchor_candidates.size < 2:
        raise RuntimeError("embedding anchor 可用样本不足")
    anchor_count = min(64, anchor_candidates.size)
    anchor_indices = anchor_candidates[:anchor_count].astype(np.int64, copy=False)
    with torch.no_grad():
        anchor_embedding = model(
            torch.from_numpy(features[anchor_indices]).to(device),
            torch.from_numpy(valid_mask[anchor_indices]).to(device),
            return_embedding=True,
        )["embedding"].detach().cpu().numpy()
    if test_predictions is not None and test_embeddings is not None:
        np.savez_compressed(
            output / "test_embeddings.npz",
            embeddings=test_embeddings,
            timestamps=test_predictions["timestamps"],
            symbols=test_predictions["symbols"],
            anchor_embeddings=anchor_embedding,
            anchor_timestamps=timestamps[anchor_indices].astype(np.int64, copy=False),
            anchor_symbols=symbols[anchor_indices],
        )
    np.savez_compressed(
        output / "validation_embeddings.npz",
        embeddings=validation_embeddings,
        timestamps=validation_predictions["timestamps"],
        symbols=validation_predictions["symbols"],
        anchor_embeddings=anchor_embedding,
        anchor_timestamps=timestamps[anchor_indices].astype(np.int64, copy=False),
        anchor_symbols=symbols[anchor_indices],
    )
    if gradient_artifact is not None:
        (output / "gradient_conflict_artifact.json").write_text(
            json.dumps(gradient_artifact, indent=2) + "\n", encoding="utf-8"
        )
    checkpoint_payload = {
        "model_config": dataclasses.asdict(model_config),
        "model_state_dict": model.cpu().state_dict(), "seed": seed,
        "runtime_backend": runtime_backend,
        "evaluation_split": evaluation_split,
        "preprocessing_mode": preprocessing_mode,
        "preprocessing_spec": (
            standardizer.canonical_payload if standardizer is not None else {
                "mode": "none",
            }
        ),
        "preprocessing_spec_sha256": (
            standardizer.sha256 if standardizer is not None else None
        ),
        "feature_schema_sha256": schema_hash, "history": history,
        "label_spec_sha256": label_spec_hash,
        "ranking_score_spec_sha256": ranking_spec_hash,
        "ranking_loss_variant": ranking_variant,
        "ranking_cutoff": ranking_score_spec.production_top_k,
        "ranking_temperature": ranking_score_spec.rank_temperature,
        "multitask_weighting_mode": weighting_mode,
        "multitask_weighting_state_dict": (
            kendall.cpu().state_dict() if kendall is not None else None
        ),
        "loss_weight_diagnostics": weight_diagnostics,
        "gradient_optimization_mode": gradient_mode,
        "gradient_optimization_hypothesis_id": hypothesis_id or None,
        "gradient_optimization_spec": (
            dataclasses.asdict(gradient_spec) if gradient_spec is not None else None
        ),
        "gradient_optimization_spec_sha256": (
            gradient_spec.sha256 if gradient_spec is not None else None
        ),
        "gradient_optimization_state_dict": (
            gradnorm.cpu().state_dict() if gradnorm is not None else None
        ),
        "gradient_mechanism_diagnostics": mechanism_diagnostics,
        "robust_training_mode": robust_spec.mode,
        "robust_training_hypothesis_id": robust_spec.hypothesis_id,
        "robust_training_spec": dataclasses.asdict(robust_spec),
        "robust_training_diagnostics": robust_diagnostics,
        "gradient_conflict_report_sha256": (
            gradient_artifact["report_sha256"] if gradient_artifact is not None else None
        ),
        "split": {
            "train_timestamps": split.train_timestamps.tolist(),
            "validation_timestamps": split.validation_timestamps.tolist(),
            "test_timestamps": split.test_timestamps.tolist(),
        },
    }
    torch.save(checkpoint_payload, output / "checkpoint.pt")
    shutil.copy2(output / "checkpoint.pt", output / "checkpoint_last.pt")
    best_epoch = int(np.argmin(np.asarray(history, dtype=np.float64))) + 1
    checkpoint_payload["selected_epoch"] = best_epoch
    checkpoint_payload["checkpoint_selection"] = "minimum_train_loss"
    torch.save(checkpoint_payload, output / "checkpoint_best.pt")
    pooling_spec = {
        "layer_id": "shared",
        "sequence_pooling": "last_valid_token",
        "static_feature_merge": "after_sequence_pooling",
    }
    mask_spec = {
        "input": "valid_mask",
        "padding": "masked",
        "causal_attention": True,
        "empty_row_fallback": "last_position",
    }
    def write_embedding_manifest(split_name, embedding_values, prediction_path,
                                 manifest_path):
        manifest = {
            "schema_version": 1,
            "role": "embedding_snapshot_v1",
            "diagnostic_only": True,
            "encoder_family": "TemporalTransformerV1",
            "layer_id": "shared",
            "pooling_spec": pooling_spec,
            "pooling_spec_sha256": _canonical_sha256(pooling_spec),
            "mask_spec": mask_spec,
            "mask_spec_sha256": _canonical_sha256(mask_spec),
            "dimension": int(embedding_values.shape[1]),
            "sample_count": int(embedding_values.shape[0]),
            "embedding_values_sha256": _array_sha256(embedding_values),
            "anchor_sample_count": int(anchor_embedding.shape[0]),
            "anchor_values_sha256": _array_sha256(anchor_embedding),
            "source_dataset_sha256": sha256_file(dataset_path),
            "preprocessing_spec_sha256": (
                standardizer.sha256 if standardizer is not None else None
            ),
            "prediction_artifact_sha256": sha256_file(prediction_path),
            "checkpoint_sha256": sha256_file(output / "checkpoint.pt"),
            "snapshot_split": split_name,
            "anchor_split": "dataset_stable_head",
        }
        manifest["manifest_sha256"] = _canonical_sha256(manifest)
        destination = output / manifest_path
        destination.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

    validation_embedding_manifest = write_embedding_manifest(
        "validation", validation_embeddings, output / "validation_predictions.npz",
        "validation_embedding_manifest.json",
    )
    embedding_manifest = None
    if test_embeddings is not None:
        embedding_manifest = write_embedding_manifest(
            "test", test_embeddings, output / "test_predictions.npz",
            "embedding_manifest.json",
        )
    metrics_payload = {
        "train_loss": history,
        "preprocessing_mode": preprocessing_mode,
        "preprocessing_spec": (
            standardizer.canonical_payload if standardizer is not None else {
                "mode": "none",
            }
        ),
        "preprocessing_spec_sha256": (
            standardizer.sha256 if standardizer is not None else None
        ),
        "multitask_weighting_mode": weighting_mode,
        "loss_weight_diagnostics": weight_diagnostics,
        "gradient_optimization_mode": gradient_mode,
        "gradient_optimization_hypothesis_id": hypothesis_id or None,
        "gradient_optimization_spec_sha256": (
            gradient_spec.sha256 if gradient_spec is not None else None
        ),
        "gradient_mechanism_diagnostics": mechanism_diagnostics,
        "robust_training_mode": robust_spec.mode,
        "robust_training_hypothesis_id": robust_spec.hypothesis_id,
        "robust_training_spec": dataclasses.asdict(robust_spec),
        "robust_training_diagnostics": robust_diagnostics,
        "gradient_conflict_report_sha256": (
            gradient_artifact["report_sha256"] if gradient_artifact is not None else None
        ),
        "runtime_backend": runtime_backend,
        "evaluation_split": evaluation_split,
        "validation_loss": validation_metrics["loss"],
        "validation_ndcg_at_k": validation_metrics["ndcg_at_cutoff"],
        "validation_cross_sections": validation_metrics["cross_sections"],
        "validation_ranking": {
            key: value for key, value in validation_metrics.items() if key != "loss"
        },
        "train_samples": len(split.train),
        "validation_samples": len(split.validation),
        "validation_prediction_artifact": {
            "schema_version": 1,
            "path": "validation_predictions.npz",
            "sha256": sha256_file(output / "validation_predictions.npz"),
            "outputs": list(OUTPUT_NAMES),
        },
        "validation_embedding_snapshot": {
            "path": "validation_embeddings.npz",
            "manifest_path": "validation_embedding_manifest.json",
            "manifest_sha256": validation_embedding_manifest["manifest_sha256"],
        },
        "split": {
            "train_first": int(split.train_timestamps[0]),
            "train_last": int(split.train_timestamps[-1]),
            "validation_first": int(split.validation_timestamps[0]),
            "validation_last": int(split.validation_timestamps[-1]),
        },
    }
    if test_metrics is not None and embedding_manifest is not None:
        metrics_payload.update({
            "test_loss": test_metrics["loss"],
            "test_ndcg_at_k": test_metrics["ndcg_at_cutoff"],
            "test_cross_sections": test_metrics["cross_sections"],
            "test_ranking": {
                key: value for key, value in test_metrics.items() if key != "loss"
            },
            "test_samples": len(split.test),
            "prediction_artifact": {
                "schema_version": 1,
                "path": "test_predictions.npz",
                "sha256": sha256_file(output / "test_predictions.npz"),
                "outputs": list(OUTPUT_NAMES),
            },
            "embedding_snapshot": {
                "path": "test_embeddings.npz",
                "manifest_path": "embedding_manifest.json",
                "manifest_sha256": embedding_manifest["manifest_sha256"],
            },
            "test_split": {
                "first": int(split.test_timestamps[0]),
                "last": int(split.test_timestamps[-1]),
            },
        })
    (output / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _export(config: dict, run_path: str, output_path: str) -> Path:
    _require_enabled(config)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("导出机需要安装项目的 ml 可选依赖") from exc
    from .export.onnx_export import export_temporal_transformer
    from .export.validate import validate_onnx_parity
    from .models.temporal_transformer import TemporalTransformerConfig, TemporalTransformerV1

    for key in ("calendar_id", "universe_id", "data_cutoff_utc"):
        if not config.get(key) or "CONFIGURE_" in str(config[key]):
            raise ValueError(f"导出前必须配置 {key}")
    run, output = Path(run_path), Path(output_path)
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = TemporalTransformerV1(TemporalTransformerConfig(**checkpoint["model_config"]))
    from .models.temporal_transformer import load_temporal_transformer_state_dict
    load_temporal_transformer_state_dict(model, checkpoint["model_state_dict"])
    output.mkdir(parents=True, exist_ok=False)
    opset = int(config.get("export", {}).get("opset", 18))
    model_path = export_temporal_transformer(model, output / "model.onnx", opset=opset)
    BAR_V1.write(output / "feature_schema.json")
    shutil.copyfile(run / "metrics.json", output / "metrics.json")
    golden = output / "golden"
    golden.mkdir()
    generator = np.random.default_rng(int(checkpoint["seed"]))
    features = generator.normal(
        size=(2, model.config.lookback, model.config.feature_count)
    ).astype(np.float32)
    valid_mask = np.ones((2, model.config.lookback), dtype=np.uint8)
    np.savez_compressed(golden / "input.npz", features=features, valid_mask=valid_mask)
    outputs = validate_onnx_parity(model, model_path, features, valid_mask)
    np.savez_compressed(golden / "python_output.npz", **outputs)
    label_spec, ranking_score_spec = _phase1b_specs(config)
    horizon = label_spec.horizon_bars
    ranking_config = config.get("ranking", {})
    ranking_variant = str(ranking_config.get("loss_variant", "none"))
    manifest = ModelManifest(
        schema_version=1,
        model_id=config.get("model_id", "bar-temporal-transformer-v1"),
        model_version=config.get(
            "model_version", datetime.now(timezone.utc).strftime("%Y%m%d.%H%M%S")
        ),
        model_sha256=sha256_file(model_path), feature_profile="BAR_V1",
        feature_schema_sha256=sha256_file(output / "feature_schema.json"),
        calendar_id=config["calendar_id"], universe_id=config["universe_id"],
        data_cutoff_utc=config["data_cutoff_utc"], lookback=model.config.lookback,
        feature_count=model.config.feature_count,
        static_feature_count=model.config.static_feature_count,
        outputs=tuple({
            "name": name,
            "unit": "probability" if name in {"direction_probability", "confidence"}
                    else "return_std" if name == "expected_volatility" else "log_return",
            "horizon_bars": horizon,
        } for name in OUTPUT_NAMES),
        onnx_opset=opset,
        label_spec_version="V2",
        label_spec_sha256=label_spec.sha256,
        ranking_score_spec=json.loads(ranking_score_spec.canonical_json),
        ranking_score_spec_sha256=ranking_score_spec.sha256,
        ranking_loss_variant=ranking_variant,
        ranking_cutoff=ranking_score_spec.production_top_k,
        ranking_temperature=ranking_score_spec.rank_temperature,
        rank_weight=(
            ranking_score_spec.lambda_rank if ranking_variant != "none" else 0.0
        ),
    )
    manifest.write(output / "manifest.json")
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="qbt-ml")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-artifact")
    validate.add_argument("artifact")
    show = commands.add_parser("show-config")
    show.add_argument("config")
    dataset = commands.add_parser("build-dataset")
    dataset.add_argument("--config", required=True)
    dataset.add_argument("--output")
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--dataset")
    train.add_argument("--output")
    ablation = commands.add_parser("phase1b-ablation")
    ablation.add_argument("--config", required=True)
    ablation.add_argument("--dataset")
    ablation.add_argument("--output")
    kendall = commands.add_parser("phase1b-kendall-ablation")
    kendall.add_argument("--config", required=True)
    kendall.add_argument("--dataset")
    kendall.add_argument("--fixed-report")
    kendall.add_argument("--output")
    phase1e_source = commands.add_parser("phase1e-build-source")
    phase1e_source.add_argument("--config", required=True)
    phase1e_diagnostics = commands.add_parser("phase1e-diagnostics")
    phase1e_diagnostics.add_argument("--config", required=True)
    phase1e_diagnostics.add_argument("--dataset")
    phase1e_diagnostics.add_argument("--output")
    phase2b_audit = commands.add_parser("phase2b-audit")
    phase2b_audit.add_argument("--dataset", required=True)
    phase2b_audit.add_argument("--output", required=True)
    phase2b_oos = commands.add_parser("phase2b-oos")
    phase2b_oos.add_argument("--config", required=True)
    phase2b_oos.add_argument("--dataset")
    phase2b_oos.add_argument("--output")
    phase2b_oos.add_argument("--baseline-root")
    export = commands.add_parser("export")
    export.add_argument("--config", required=True)
    export.add_argument("--run", required=True)
    export.add_argument("--output", required=True)
    backtest = commands.add_parser("backtest-artifact")
    backtest.add_argument("artifact")
    backtest.add_argument("--data", required=True)
    backtest.add_argument("--initial-cash", type=float, default=1_000_000.0)
    backtest.add_argument("--config")
    backtest.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "validate-artifact":
        manifest = validate_artifact(args.artifact)
        print(json.dumps({
            "valid": True, "model_id": manifest.model_id,
            "model_version": manifest.model_version,
        }, ensure_ascii=False))
    elif args.command == "show-config":
        print(json.dumps(_load_config(args.config), ensure_ascii=False, sort_keys=True, indent=2))
    elif args.command == "build-dataset":
        print(_build_dataset(_load_config(args.config), args.output))
    elif args.command == "train":
        print(_train(_load_config(args.config), args.dataset, args.output))
    elif args.command == "phase1b-ablation":
        from .training.ablation import run_phase1b_ablation
        config = _load_config(args.config)
        dataset_path = args.dataset or config.get("training", {}).get("dataset")
        output_path = args.output or config.get("phase1b_ablation", {}).get("output")
        if not dataset_path or not output_path:
            raise ValueError("必须配置 ablation dataset 和 output")
        print(run_phase1b_ablation(config, dataset_path, output_path, _train))
    elif args.command == "phase1b-kendall-ablation":
        from .training.ablation import run_kendall_ablation
        config = _load_config(args.config)
        settings = config.get("kendall_ablation", {})
        dataset_path = args.dataset or config.get("training", {}).get("dataset")
        fixed_report = args.fixed_report or settings.get("fixed_report")
        output_path = args.output or settings.get("output")
        if not dataset_path or not fixed_report or not output_path:
            raise ValueError("必须配置 Kendall dataset、fixed_report 和 output")
        print(run_kendall_ablation(
            config, dataset_path, fixed_report, output_path, _train
        ))
    elif args.command == "phase1e-build-source":
        from .training.phase1e import Phase1EDataSpecV1, build_phase1e_source
        config = _load_config(args.config)
        settings = config.get("phase1e_data", {})
        spec = Phase1EDataSpecV1(
            bars_root=settings["bars_root"],
            security_state_root=settings["security_state_root"],
            selection_year=int(settings.get("selection_year", 2019)),
            minimum_selection_sessions=int(settings.get(
                "minimum_selection_sessions", 220
            )),
            universe_size=int(settings.get("universe_size", 120)),
            source_start_year=int(settings.get("source_start_year", 2020)),
            phase1b_last_timestamp=int(settings["phase1b_last_timestamp"]),
        )
        print(build_phase1e_source(
            spec, settings["source_output"], settings["audit_output"]
        ))
    elif args.command == "phase1e-diagnostics":
        from .training.phase1e import run_phase1e_diagnostics
        config = _load_config(args.config)
        settings = config.get("phase1e_diagnostics", {})
        dataset_path = args.dataset or config.get("training", {}).get("dataset")
        output_path = args.output or settings.get("output")
        audit_path = settings.get("data_audit")
        if not dataset_path or not output_path or not audit_path:
            raise ValueError("必须配置 Phase 1E dataset、data_audit 和 output")
        print(run_phase1e_diagnostics(
            config, dataset_path, audit_path, output_path, _train
        ))
    elif args.command == "phase2b-audit":
        print(_phase2b_audit(args.dataset, args.output))
    elif args.command == "phase2b-oos":
        from .training.phase2b import run_phase2b_oos
        config = _load_config(args.config)
        settings = config.get("phase2b_oos", {})
        dataset_path = args.dataset or config.get("training", {}).get("dataset")
        output_path = args.output or settings.get("output")
        baseline_root = args.baseline_root or settings.get("baseline_root")
        if not dataset_path or not output_path:
            raise ValueError("必须配置 Phase 2B dataset 和 output")
        print(run_phase2b_oos(
            config, dataset_path, output_path, _train,
            baseline_root=baseline_root,
        ))
    elif args.command == "export":
        print(_export(_load_config(args.config), args.run, args.output))
    else:
        print(json.dumps(_backtest_artifact(
            args.artifact, args.data, initial_cash=args.initial_cash,
            config_path=args.config, output_path=args.output,
        ), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

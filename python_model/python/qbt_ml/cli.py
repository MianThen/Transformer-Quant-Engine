from __future__ import annotations

import argparse
import dataclasses
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
from .labels import LabelSpec, build_next_open_labels
from .leakage import audit_dataset, audit_feature_time_invariance, dataset_fingerprint


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
    if path.is_dir():
        import pyarrow.parquet as pq
        files = sorted(path.rglob("*.parquet"))
        if not files:
            raise ValueError(f"数据目录不含 Parquet: {path}")
        return pd.concat(
            [pq.ParquetFile(file).read().to_pandas() for file in files],
            ignore_index=True,
        )
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"symbol": "string"})
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("训练数据仅支持 CSV、Parquet 或 PQ")


def _enrich_phase_e(config: dict, output_override: str | None) -> Path:
    from .data import enrich_phase_e

    return enrich_phase_e(config, output_override)


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
    _reject_unknown_keys(
        config, {"policy", "risk", "runtime", "execution", "fees"}, "回测配置"
    )
    sections = {}
    allowed = {
        "policy": {
            "max_positions", "max_position_weight", "minimum_expected_return",
            "minimum_confidence",
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

    fees = config.get("fees", {})
    if not isinstance(fees, dict):
        raise ValueError("fees 配置必须是对象")
    _reject_unknown_keys(fees, {"point_in_time", "schedules"}, "fees")
    schedules = fees.get("schedules", [])
    if not isinstance(schedules, list):
        raise ValueError("fees.schedules 必须是数组")
    if schedules and fees.get("point_in_time") is not True:
        raise ValueError("使用费率表时必须声明 fees.point_in_time=true")

    execution = engine_module.ExecutionConfig()
    for name, value in sections["execution"].items():
        setattr(execution, name, value)
    engine = engine_module.BacktestEngine(
        initial_cash, engine_module.FillTiming.NEXT_OPEN, execution
    )
    if schedules:
        if not hasattr(engine_module, "FeeSchedule"):
            raise RuntimeError("当前 cpp_engine 缺少 point-in-time FeeSchedule 绑定")
        allowed_fee_keys = {
            "effective_from", "effective_to", "commission_rate",
            "min_commission", "stamp_tax_rate", "transfer_fee_rate",
        }
        engine_schedules = []
        for item in schedules:
            if not isinstance(item, dict):
                raise ValueError("fees.schedules 每项必须是对象")
            _reject_unknown_keys(item, allowed_fee_keys, "fee schedule")
            missing = sorted({
                "effective_from", "commission_rate", "min_commission",
                "stamp_tax_rate",
            } - set(item))
            if missing:
                raise ValueError("fee schedule 缺少字段: " + ", ".join(missing))
            engine_schedules.append(engine_module.FeeSchedule(
                int(item["effective_from"]), item.get("effective_to"),
                float(item["commission_rate"]), float(item["min_commission"]),
                float(item["stamp_tax_rate"]),
                float(item.get("transfer_fee_rate", 0.0)),
            ))
        engine.set_fee_schedules(engine_schedules)
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
    equity_curve = list(engine.get_equity_curve()) if hasattr(
        engine, "get_equity_curve"
    ) else []
    equity_values = np.asarray(
        [float(point.equity) for point in equity_curve], dtype=np.float64
    )
    returns = (
        equity_values[1:] / equity_values[:-1] - 1.0
        if len(equity_values) >= 2 and np.all(equity_values[:-1] > 0)
        else np.asarray([], dtype=np.float64)
    )
    tail_count = max(1, int(np.ceil(len(returns) * 0.05))) if len(returns) else 0
    cvar_95 = float(np.sort(returns)[:tail_count].mean()) if tail_count else None
    trades = list(engine.get_trade_history()) if hasattr(
        engine, "get_trade_history"
    ) else []
    traded_notional = sum(
        abs(float(trade.quantity) * float(trade.price)) for trade in trades
    )
    average_equity = float(equity_values.mean()) if len(equity_values) else initial_cash
    turnover = traded_notional / average_equity if average_equity > 0 else None
    positions = list(engine.get_positions()) if hasattr(engine, "get_positions") else []
    last_rows = table.drop_duplicates("symbol", keep="last").set_index("symbol")
    symbol_contributions = {}
    for position in positions:
        symbol = str(position.symbol)
        if symbol not in last_rows.index:
            continue
        market_value_pnl = float(position.quantity) * (
            float(last_rows.loc[symbol, "close"]) - float(position.avg_cost)
        )
        symbol_contributions[symbol] = (
            float(position.realized_pnl) + market_value_pnl
        ) / initial_cash
    industry_contributions = {}
    if "industry" in last_rows.columns:
        for symbol, contribution in symbol_contributions.items():
            industry = str(last_rows.loc[symbol, "industry"])
            industry_contributions[industry] = (
                industry_contributions.get(industry, 0.0) + contribution
            )
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
        "sharpe": float(engine.get_sharpe_ratio()),
        "max_drawdown": float(engine.get_max_drawdown()),
        "turnover": turnover,
        "cvar_95": cvar_95,
        "symbol_contributions": symbol_contributions,
        "industry_contributions": industry_contributions,
        "net_of_cost": bool(schedules),
        "fee_schedule_count": len(schedules),
        "slippage_bps": float(getattr(execution, "slippage_bps", 0.0)),
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
    if data.get("point_in_time_required") is not True:
        raise ValueError("V1.1 数据集必须显式声明 data.point_in_time_required=true")
    lineage = {}
    for name in ("calendar_id", "universe_id"):
        value = config.get(name)
        if not value or "CONFIGURE_" in str(value):
            raise ValueError(f"V1.1 数据集必须配置 {name}")
        lineage[name] = str(value)
    source_path = Path(source)
    enrichment_report = None
    if source_path.is_dir() and (source_path / "enrichment_report.json").is_file():
        enrichment_report = json.loads(
            (source_path / "enrichment_report.json").read_text(encoding="utf-8")
        )
        if enrichment_report.get("model_evaluation_status") != "READY":
            raise ValueError("Phase E 富化数据未达到模型评估就绪状态")
        configured_mode = data.get("execution_reference_mode")
        if configured_mode and configured_mode != enrichment_report.get(
            "execution_reference_mode"
        ):
            raise ValueError("训练配置与富化数据 execution_reference_mode 不一致")
    table = _read_table(source)
    adjustment_mode = str(config.get("price_adjustment_mode", "raw_unadjusted"))
    if adjustment_mode == "pit_adjusted_signal_raw_execution":
        adjustment_fields = {
            "signal_open", "signal_high", "signal_low", "signal_close",
            "adjustment_factor",
        }
        missing_adjustment = sorted(adjustment_fields - set(table.columns))
        if missing_adjustment:
            raise ValueError(
                "正式训练数据缺少 PIT 复权字段: " + ", ".join(missing_adjustment)
            )
    provenance_columns = {"universe_asof", "reference_data_known_at_max"}
    missing_provenance = provenance_columns - set(table.columns)
    if missing_provenance:
        raise ValueError(
            "point-in-time 数据缺少审计字段: " + ", ".join(sorted(missing_provenance))
        )
    provenance = table.sort_values(
        ["timestamp", "symbol"], kind="stable"
    ).reset_index(drop=True)
    time_audit = audit_feature_time_invariance(table, build_bar_v1)
    frame = build_bar_v1(table)
    label_spec = LabelSpec.next_open(int(config.get("label_horizon_bars", 5)))
    labels = build_next_open_labels(
        table, label_spec.horizon_bars, label_spec=label_spec
    )
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
    selected_labels = aligned.iloc[selected_indices]
    signal_asof = selected_labels["signal_asof"].to_numpy(np.int64)
    source_rows = provenance.iloc[windows.row_indices].reset_index(drop=True).iloc[selected_indices]
    arrays = {
        "features": windows.features[selected_indices],
        "valid_mask": windows.valid_mask[selected_indices],
        "timestamps": windows.timestamps[selected_indices],
        "symbols": windows.symbols[selected_indices].astype(str),
        "expected_return": selected_labels["expected_return"].to_numpy(np.float32),
        "direction": selected_labels["direction"].to_numpy(np.float32),
        "realized_volatility": selected_labels["realized_volatility"].to_numpy(np.float32),
        "signal_asof": signal_asof,
        "feature_source_max_timestamp": signal_asof.copy(),
        "label_entry_timestamp": selected_labels["label_entry_timestamp"].to_numpy(np.int64),
        "label_exit_timestamp": selected_labels["label_exit_timestamp"].to_numpy(np.int64),
        "universe_asof": source_rows["universe_asof"].to_numpy(np.int64),
        "reference_data_known_at_max": source_rows[
            "reference_data_known_at_max"
        ].to_numpy(np.int64),
        "feature_schema_json": np.asarray(BAR_V1.canonical_json),
        "feature_schema_sha256": np.asarray(BAR_V1.sha256),
        "feature_code_sha256": np.asarray(sha256_file(
            Path(__file__).resolve().parent / "features" / "bar_v1.py"
        )),
        "label_spec_json": np.asarray(label_spec.canonical_json),
        "label_spec_sha256": np.asarray(label_spec.sha256),
        "calendar_id": np.asarray(lineage["calendar_id"]),
        "frequency": np.asarray(str(config.get("frequency", "1d"))),
        "price_adjustment_mode": np.asarray(adjustment_mode),
        "universe_id": np.asarray(lineage["universe_id"]),
        "point_in_time_required": np.asarray(True),
        "execution_reference_status": np.asarray(
            "UNDECLARED" if enrichment_report is None
            else enrichment_report["execution_promotion_status"]
        ),
        "execution_reference_mode": np.asarray(
            "undeclared" if enrichment_report is None
            else enrichment_report["execution_reference_mode"]
        ),
        "execution_source_fingerprint": np.asarray(
            "" if enrichment_report is None
            else enrichment_report["source_fingerprint_sha256"]
        ),
        "prefix_invariance_pass": np.asarray(time_audit["prefix_invariance_pass"]),
        "future_mutation_pass": np.asarray(time_audit["future_mutation_pass"]),
    }
    arrays["dataset_fingerprint"] = np.asarray(dataset_fingerprint(arrays))
    report = audit_dataset(arrays)
    report.require_pass()
    np.savez_compressed(output, **arrays)
    report.write(output.with_suffix(".leakage_report.json"))
    return output


def _detect_leakage(
    config: dict,
    dataset_path: str | None,
    output_override: str | None,
) -> Path:
    _require_enabled(config)
    configured = config.get("data", {}).get("dataset") or config.get("dataset")
    dataset_path = dataset_path or configured
    if not dataset_path:
        raise ValueError("Leakage Detection 必须提供 dataset")
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    split = None
    split_config = config.get("split")
    if split_config:
        from .training.walk_forward import chronological_timestamp_split
        split = chronological_timestamp_split(
            arrays["timestamps"],
            train_fraction=float(split_config.get("train_fraction", 0.70)),
            validation_fraction=float(split_config.get("validation_fraction", 0.15)),
            test_fraction=float(split_config.get("test_fraction", 0.15)),
            purge_timestamps=int(split_config.get("purge_timestamps", 5)),
            embargo_timestamps=int(split_config.get("embargo_timestamps", 5)),
        )
    report = audit_dataset(arrays, split=split)
    output = Path(output_override or config.get("output", "analysis/leakage"))
    report_path = output if output.suffix == ".json" else output / "leakage_report.json"
    report.write(report_path)
    report.require_pass()
    return report_path


def _walk_forward(
    config: dict,
    dataset_path: str | None,
    output_override: str | None,
) -> Path:
    from .evaluation import run_walk_forward_baseline

    _require_enabled(config)
    dataset_path = dataset_path or config.get("dataset")
    output = output_override or config.get("output")
    if not dataset_path or "CONFIGURE_" in str(dataset_path) or not output:
        raise ValueError("Walk-forward 必须配置 dataset 和 output")
    return run_walk_forward_baseline(config, dataset_path, output)


def _walk_forward_transformer(
    config: dict,
    dataset_path: str | None,
    output_override: str | None,
) -> Path:
    from .evaluation import run_transformer_walk_forward

    _require_enabled(config)
    dataset_path = dataset_path or config.get("dataset")
    output = output_override or config.get("output")
    if not dataset_path or "CONFIGURE_" in str(dataset_path) or not output:
        raise ValueError("Transformer Walk-forward 必须配置 dataset 和 output")
    return run_transformer_walk_forward(config, dataset_path, output)


def _walk_forward_deep_baselines(
    config: dict,
    dataset_path: str | None,
    output_override: str | None,
) -> Path:
    from .evaluation import run_deep_baseline_suite

    _require_enabled(config)
    dataset_path = dataset_path or config.get("dataset")
    output = output_override or config.get("output")
    if not dataset_path or "CONFIGURE_" in str(dataset_path) or not output:
        raise ValueError("Deep Baseline Suite 必须配置 dataset 和 output")
    return run_deep_baseline_suite(config, dataset_path, output)


def _ablate_features(
    config: dict,
    dataset_path: str | None,
    output_override: str | None,
) -> Path:
    from .evaluation import run_feature_ablation

    _require_enabled(config)
    dataset_path = dataset_path or config.get("dataset")
    output = output_override or config.get("output")
    if not dataset_path or "CONFIGURE_" in str(dataset_path) or not output:
        raise ValueError("Feature Ablation 必须配置 dataset 和 output")
    return run_feature_ablation(config, dataset_path, output)


def _ablate_transformer_features(
    config: dict,
    dataset_path: str | None,
    output_override: str | None,
) -> Path:
    from .evaluation import run_transformer_feature_ablation

    _require_enabled(config)
    dataset_path = dataset_path or config.get("dataset")
    output = output_override or config.get("output")
    if not dataset_path or "CONFIGURE_" in str(dataset_path) or not output:
        raise ValueError("Transformer Feature Ablation 必须配置 dataset 和 output")
    return run_transformer_feature_ablation(config, dataset_path, output)


def _benchmark_models(
    config: dict,
    dataset_path: str | None,
    output_override: str | None,
) -> Path:
    from .evaluation import run_model_benchmark

    _require_enabled(config)
    dataset_path = dataset_path or config.get("dataset")
    output = output_override or config.get("output")
    if not dataset_path or "CONFIGURE_" in str(dataset_path) or not output:
        raise ValueError("Model Benchmark 必须配置 dataset 和 output")
    return run_model_benchmark(config, dataset_path, output)


def _promotion_review(config: dict, output_override: str | None) -> Path:
    from .evaluation import run_promotion_review

    _require_enabled(config)
    output = output_override or config.get("output")
    if not output:
        raise ValueError("Promotion Review 必须配置 output")
    return run_promotion_review(config, output)


def _portfolio_benchmark(
    config: dict,
    dataset_path: str | None,
    benchmark_path: str | None,
    bars_path: str | None,
    output_override: str | None,
) -> Path:
    from .evaluation import run_cpp_portfolio_benchmark

    _require_enabled(config)
    dataset_path = dataset_path or config.get("dataset")
    benchmark_path = benchmark_path or config.get("model_benchmark")
    bars_path = bars_path or config.get("bars")
    output = output_override or config.get("output")
    values = (dataset_path, benchmark_path, bars_path, output)
    if any(not value or "CONFIGURE_" in str(value) for value in values):
        raise ValueError(
            "C++ Portfolio Benchmark 必须配置 dataset/model_benchmark/bars/output"
        )
    return run_cpp_portfolio_benchmark(
        config, dataset_path, benchmark_path, bars_path, output
    )


def _portfolio_ablation(
    config: dict,
    dataset_path: str | None,
    ablation_path: str | None,
    bars_path: str | None,
    output_override: str | None,
) -> Path:
    from .evaluation import run_cpp_portfolio_ablation

    _require_enabled(config)
    dataset_path = dataset_path or config.get("dataset")
    ablation_path = ablation_path or config.get("feature_ablation")
    bars_path = bars_path or config.get("bars")
    output = output_override or config.get("output")
    values = (dataset_path, ablation_path, bars_path, output)
    if any(not value or "CONFIGURE_" in str(value) for value in values):
        raise ValueError(
            "C++ Portfolio Ablation 必须配置 dataset/feature_ablation/bars/output"
        )
    return run_cpp_portfolio_ablation(
        config, dataset_path, ablation_path, bars_path, output
    )


def _train(config: dict, dataset_path: str | None, output_override: str | None) -> Path:
    _require_enabled(config)
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("训练机需要安装项目的 ml 可选依赖") from exc
    from .models.temporal_transformer import TemporalTransformerConfig, TemporalTransformerV1
    from .models import LogisticBaseline
    from .training.train import multitask_loss_components, seed_everything
    from .training.sampler import CrossSectionBatchSampler
    from .training.walk_forward import chronological_timestamp_split

    training = config.get("training", {})
    dataset_path = dataset_path or training.get("dataset")
    output = output_override or training.get("output")
    device = training.get("device")
    if not dataset_path or not output or not device or "CONFIGURE_" in str(device):
        raise ValueError("必须配置 training.dataset、training.output 和 training.device")
    with np.load(dataset_path, allow_pickle=False) as data:
        dataset_arrays = {name: data[name] for name in data.files}
        features = data["features"].astype(np.float32, copy=False)
        valid_mask = data["valid_mask"].astype(np.uint8, copy=False)
        expected_return = data["expected_return"].astype(np.float32, copy=False)
        direction = data["direction"].astype(np.float32, copy=False)
        realized_volatility = data["realized_volatility"].astype(np.float32, copy=False)
        timestamps = data["timestamps"]
        schema_hash = str(data["feature_schema_sha256"])
    split_config = config.get("split", {})
    split = chronological_timestamp_split(
        timestamps,
        train_fraction=float(split_config.get("train_fraction", 0.70)),
        validation_fraction=float(split_config.get("validation_fraction", 0.15)),
        test_fraction=float(split_config.get("test_fraction", 0.15)),
        purge_timestamps=int(split_config.get("purge_timestamps", 0)),
        embargo_timestamps=int(split_config.get("embargo_timestamps", 0)),
    )
    normalizer_fit_end = np.asarray(dataset_arrays["signal_asof"])[split.train].max()
    leakage_report = audit_dataset(
        dataset_arrays, split=split, normalizer_fit_end=normalizer_fit_end
    )
    leakage_report.require_pass()

    def tensors_for(indices):
        return TensorDataset(
            torch.from_numpy(features[indices]), torch.from_numpy(valid_mask[indices]),
            torch.from_numpy(timestamps[indices]),
            torch.from_numpy(expected_return[indices]), torch.from_numpy(direction[indices]),
            torch.from_numpy(realized_volatility[indices]),
        )

    train_tensors = tensors_for(split.train)
    seed = int(config.get("seed", 20260724))
    seed_everything(seed)
    allowed = {"static_feature_count", "d_model", "nhead", "num_layers",
               "dim_feedforward", "dropout", "normalization_clip"}
    model_config = TemporalTransformerConfig(
        feature_count=features.shape[2], lookback=features.shape[1],
        **{key: value for key, value in config.get("model", {}).items() if key in allowed},
    )
    model = TemporalTransformerV1(model_config).to(device)
    train_token_mask = valid_mask[split.train].astype(bool)
    train_tokens = features[split.train][train_token_mask]
    if train_tokens.size == 0:
        raise ValueError("训练窗口没有有效 token，无法拟合 normalizer")
    normalization_mean = train_tokens.mean(axis=0, dtype=np.float64).astype(np.float32)
    normalization_scale = train_tokens.std(axis=0, dtype=np.float64).astype(np.float32)
    normalization_scale[normalization_scale < 1e-6] = 1.0
    model.set_normalization(normalization_mean, normalization_scale)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training.get("learning_rate", 3e-4))
    )
    timestamps_per_batch = int(training.get("timestamps_per_batch", 1))
    loader = DataLoader(
        train_tensors,
        batch_sampler=CrossSectionBatchSampler(
            timestamps[split.train], timestamps_per_batch=timestamps_per_batch,
            shuffle=True, seed=seed,
        ),
    )
    configured_weights = training.get("loss_weights", {})
    loss_weights = {
        "return_weight": float(configured_weights.get("return", 1.0)),
        "direction_weight": float(configured_weights.get("direction", 0.25)),
        "volatility_weight": float(configured_weights.get("volatility", 0.25)),
        "quantile_weight": float(configured_weights.get("quantile", 0.25)),
        "rank_weight": float(configured_weights.get("rank", 0.10)),
    }
    if any(value < 0 or not np.isfinite(value) for value in loss_weights.values()):
        raise ValueError("loss_weights 必须为有限非负数")
    component_names = ("return", "direction", "volatility", "quantile", "rank", "total")
    history = []
    model.train()
    for _ in range(int(training.get("epochs", 50))):
        totals = {
            name: 0.0
            for name in component_names
        }
        for feature, mask, batch_timestamps, expected, target_direction, volatility in loader:
            feature, mask = feature.to(device), mask.to(device)
            target = {
                "timestamps": batch_timestamps.to(device),
                "expected_return": expected.to(device),
                "direction": target_direction.to(device),
                "realized_volatility": volatility.to(device),
            }
            optimizer.zero_grad(set_to_none=True)
            components = multitask_loss_components(
                model(feature, mask), target, **loss_weights
            )
            loss = components["total"]
            loss.backward()
            optimizer.step()
            for name, value in components.items():
                totals[name] += float(value.detach()) * feature.shape[0]
        history.append({name: value / len(split.train) for name, value in totals.items()})

    def evaluate(indices):
        evaluation_loader = DataLoader(
            tensors_for(indices),
            batch_sampler=CrossSectionBatchSampler(
                timestamps[indices], timestamps_per_batch=timestamps_per_batch,
                shuffle=False,
            ),
        )
        model.eval()
        totals = {
            name: 0.0
            for name in component_names
        }
        with torch.no_grad():
            for feature, mask, batch_timestamps, expected, target_direction, volatility in evaluation_loader:
                feature, mask = feature.to(device), mask.to(device)
                target = {
                    "timestamps": batch_timestamps.to(device),
                    "expected_return": expected.to(device),
                    "direction": target_direction.to(device),
                    "realized_volatility": volatility.to(device),
                }
                components = multitask_loss_components(
                    model(feature, mask), target, **loss_weights
                )
                for name, value in components.items():
                    totals[name] += float(value) * feature.shape[0]
        return {name: value / len(indices) for name, value in totals.items()}

    validation_loss = evaluate(split.validation)
    test_loss = evaluate(split.test)
    validation_logits = []
    model.eval()
    with torch.no_grad():
        for feature, mask, *_ in DataLoader(
            tensors_for(split.validation),
            batch_sampler=CrossSectionBatchSampler(
                timestamps[split.validation], timestamps_per_batch=timestamps_per_batch,
                shuffle=False,
            ),
        ):
            validation_logits.append(
                model(feature.to(device), mask.to(device))["direction_logits"]
                .detach().cpu().numpy()
            )
    calibrator = LogisticBaseline(alpha=1.0).fit(
        np.concatenate(validation_logits)[:, None], direction[split.validation]
    )
    model.set_direction_calibration(
        calibrator.coefficients_[0], calibrator.coefficients_[1]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    label_spec_json = str(np.asarray(dataset_arrays["label_spec_json"]).item())
    torch.save({
        "model_config": dataclasses.asdict(model_config),
        "model_state_dict": model.cpu().state_dict(), "seed": seed,
        "feature_schema_sha256": schema_hash, "history": history,
        "training_dataset_fingerprint": str(
            np.asarray(dataset_arrays["dataset_fingerprint"]).item()
        ),
        "label_spec_sha256": str(np.asarray(dataset_arrays["label_spec_sha256"]).item()),
        "normalizer_fit_end": int(normalizer_fit_end),
        "split": {
            "train_timestamps": split.train_timestamps.tolist(),
            "validation_timestamps": split.validation_timestamps.tolist(),
            "test_timestamps": split.test_timestamps.tolist(),
        },
    }, output / "checkpoint.pt")
    (output / "normalization.json").write_text(json.dumps({
        "method": "mean_std",
        "clip": model_config.normalization_clip,
        "fit_start": int(np.asarray(dataset_arrays["signal_asof"])[split.train].min()),
        "fit_end": int(normalizer_fit_end),
        "mean": normalization_mean.tolist(),
        "scale": normalization_scale.tolist(),
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "calibration.json").write_text(json.dumps({
        "method": "platt_validation_only",
        "input": "direction_logits",
        "intercept": float(calibrator.coefficients_[0]),
        "slope": float(calibrator.coefficients_[1]),
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "label_spec.json").write_text(label_spec_json, encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps({
        "train_loss": history,
        "validation_loss": validation_loss,
        "test_loss": test_loss,
        "train_samples": len(split.train),
        "validation_samples": len(split.validation),
        "test_samples": len(split.test),
        "split": {
            "train_first": int(split.train_timestamps[0]),
            "train_last": int(split.train_timestamps[-1]),
            "validation_first": int(split.validation_timestamps[0]),
            "validation_last": int(split.validation_timestamps[-1]),
            "test_first": int(split.test_timestamps[0]),
            "test_last": int(split.test_timestamps[-1]),
        },
    }, indent=2) + "\n", encoding="utf-8")
    leakage_report.write(output / "leakage_report.json")
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
    if checkpoint.get("model_family", "TemporalTransformerV1.1") != "TemporalTransformerV1.1":
        raise ValueError("深度基线 checkpoint 仅用于离线比较，不能导出为 V1.1 生产制品")
    feature_indices = checkpoint.get(
        "feature_indices", list(range(len(BAR_V1.feature_names)))
    )
    if feature_indices != list(range(len(BAR_V1.feature_names))):
        raise ValueError("特征消融 checkpoint 仅用于分析，不能导出为 BAR_V1 生产制品")
    model = TemporalTransformerV1(TemporalTransformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    required_run_files = (
        "label_spec.json", "normalization.json", "calibration.json",
        "leakage_report.json", "metrics.json",
    )
    missing_run_files = [name for name in required_run_files if not (run / name).is_file()]
    if missing_run_files:
        raise ValueError("Manifest V2 导出缺少训练文件: " + ", ".join(missing_run_files))
    label_spec = LabelSpec(**json.loads(
        (run / "label_spec.json").read_text(encoding="utf-8")
    ))
    if checkpoint.get("label_spec_sha256") != label_spec.sha256:
        raise ValueError("checkpoint 与 label_spec.json 不一致")
    output.mkdir(parents=True, exist_ok=False)
    opset = int(config.get("export", {}).get("opset", 17))
    model_path = export_temporal_transformer(model, output / "model.onnx", opset=opset)
    BAR_V1.write(output / "feature_schema.json")
    for name in required_run_files:
        shutil.copyfile(run / name, output / name)
    golden = output / "golden"
    golden.mkdir()
    generator = np.random.default_rng(int(checkpoint["seed"]))
    features = generator.normal(
        size=(8, model.config.lookback, model.config.feature_count)
    ).astype(np.float32)
    valid_mask = np.ones(features.shape[:2], dtype=np.uint8)
    np.savez_compressed(golden / "input.npz", features=features, valid_mask=valid_mask)
    with torch.no_grad():
        pytorch_prediction = model(
            torch.from_numpy(features), torch.from_numpy(valid_mask)
        )
    np.savez_compressed(golden / "pytorch_output.npz", **{
        name: pytorch_prediction[name].detach().cpu().numpy() for name in OUTPUT_NAMES
    })
    expected_returns = pytorch_prediction["expected_return"].detach().cpu().numpy()
    ranked = sorted(range(features.shape[0]), key=lambda index: (-expected_returns[index], index))
    expected_targets = []
    for index in ranked[:3]:
        symbol_id = index + 1
        close = 10.0 + symbol_id
        raw_quantity = int((1_000_000.0 * 0.05) // close)
        quantity = raw_quantity - raw_quantity % 100
        expected_targets.append({
            "symbol_id": symbol_id, "target_quantity": quantity,
            "target_weight": 0.05,
        })
    expected_targets.sort(key=lambda value: value["symbol_id"])
    (golden / "expected_decisions.json").write_text(json.dumps({
        "policy": "long_only_top_k", "max_positions": 3,
        "max_position_weight": 0.05, "equity": 1_000_000.0,
        "price_rule": "10_plus_symbol_id", "lot_size": 100,
        "targets": expected_targets,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    onnx_outputs = validate_onnx_parity(model, model_path, features, valid_mask)
    np.savez_compressed(golden / "onnx_output.npz", **onnx_outputs)
    horizon = label_spec.horizon_bars
    manifest = ModelManifest(
        schema_version=2,
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
        frequency=config.get("frequency", "1d"),
        outputs=tuple({
            "name": name,
            "unit": "probability" if name in {"direction_probability", "confidence"}
                    else "return_std" if name == "expected_volatility" else "log_return",
            "horizon_bars": horizon,
            **({"quantile": 0.10} if name == "lower_quantile" else
               {"quantile": 0.90} if name == "upper_quantile" else {}),
        } for name in OUTPUT_NAMES),
        onnx_opset=opset,
        model_family="TemporalTransformer",
        architecture_version="V1.1",
        label_spec_sha256=sha256_file(output / "label_spec.json"),
        normalization_method="mean_std",
        normalization_sha256=sha256_file(output / "normalization.json"),
        calibration_method="platt_validation_only",
        calibration_sha256=sha256_file(output / "calibration.json"),
        training_dataset_fingerprint=checkpoint["training_dataset_fingerprint"],
        leakage_report_sha256=sha256_file(output / "leakage_report.json"),
        minimum_valid_tokens=int(config.get("minimum_valid_tokens", 1)),
        dynamic_batch=True,
    )
    manifest.write(output / "manifest.json")
    return output


def _validate_ort_cpp(
    artifact_path: str, runner_path: str, output_path: str | None,
) -> Path:
    from .export import validate_ort_cpp_parity

    return validate_ort_cpp_parity(artifact_path, runner_path, output_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="qbt-ml")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-artifact")
    validate.add_argument("artifact")
    cpp_parity = commands.add_parser("validate-ort-cpp")
    cpp_parity.add_argument("--artifact", required=True)
    cpp_parity.add_argument("--runner", required=True)
    cpp_parity.add_argument("--output")
    show = commands.add_parser("show-config")
    show.add_argument("config")
    dataset = commands.add_parser("build-dataset")
    dataset.add_argument("--config", required=True)
    dataset.add_argument("--output")
    enrichment = commands.add_parser("enrich-phase-e")
    enrichment.add_argument("--config", required=True)
    enrichment.add_argument("--output")
    leakage = commands.add_parser("detect-leakage")
    leakage.add_argument("--config", required=True)
    leakage.add_argument("--dataset")
    leakage.add_argument("--output")
    walk_forward = commands.add_parser("walk-forward")
    walk_forward.add_argument("--config", required=True)
    walk_forward.add_argument("--dataset")
    walk_forward.add_argument("--output")
    transformer_walk_forward = commands.add_parser("walk-forward-transformer")
    transformer_walk_forward.add_argument("--config", required=True)
    transformer_walk_forward.add_argument("--dataset")
    transformer_walk_forward.add_argument("--output")
    deep_baselines = commands.add_parser("walk-forward-deep-baselines")
    deep_baselines.add_argument("--config", required=True)
    deep_baselines.add_argument("--dataset")
    deep_baselines.add_argument("--output")
    ablation = commands.add_parser("ablate-features")
    ablation.add_argument("--config", required=True)
    ablation.add_argument("--dataset")
    ablation.add_argument("--output")
    transformer_ablation = commands.add_parser("ablate-transformer-features")
    transformer_ablation.add_argument("--config", required=True)
    transformer_ablation.add_argument("--dataset")
    transformer_ablation.add_argument("--output")
    benchmark = commands.add_parser("benchmark-models")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--dataset")
    benchmark.add_argument("--output")
    promotion = commands.add_parser("promotion-review")
    promotion.add_argument("--config", required=True)
    promotion.add_argument("--output")
    portfolio_benchmark = commands.add_parser("benchmark-portfolios-cpp")
    portfolio_benchmark.add_argument("--config", required=True)
    portfolio_benchmark.add_argument("--dataset")
    portfolio_benchmark.add_argument("--model-benchmark")
    portfolio_benchmark.add_argument("--bars")
    portfolio_benchmark.add_argument("--output")
    portfolio_ablation = commands.add_parser("benchmark-ablation-portfolios-cpp")
    portfolio_ablation.add_argument("--config", required=True)
    portfolio_ablation.add_argument("--dataset")
    portfolio_ablation.add_argument("--feature-ablation")
    portfolio_ablation.add_argument("--bars")
    portfolio_ablation.add_argument("--output")
    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--dataset")
    train.add_argument("--output")
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
    elif args.command == "validate-ort-cpp":
        print(_validate_ort_cpp(args.artifact, args.runner, args.output))
    elif args.command == "show-config":
        print(json.dumps(_load_config(args.config), ensure_ascii=False, sort_keys=True, indent=2))
    elif args.command == "build-dataset":
        print(_build_dataset(_load_config(args.config), args.output))
    elif args.command == "enrich-phase-e":
        print(_enrich_phase_e(_load_config(args.config), args.output))
    elif args.command == "detect-leakage":
        print(_detect_leakage(_load_config(args.config), args.dataset, args.output))
    elif args.command == "walk-forward":
        print(_walk_forward(_load_config(args.config), args.dataset, args.output))
    elif args.command == "walk-forward-transformer":
        print(_walk_forward_transformer(
            _load_config(args.config), args.dataset, args.output
        ))
    elif args.command == "walk-forward-deep-baselines":
        print(_walk_forward_deep_baselines(
            _load_config(args.config), args.dataset, args.output
        ))
    elif args.command == "ablate-features":
        print(_ablate_features(_load_config(args.config), args.dataset, args.output))
    elif args.command == "ablate-transformer-features":
        print(_ablate_transformer_features(
            _load_config(args.config), args.dataset, args.output
        ))
    elif args.command == "benchmark-models":
        print(_benchmark_models(_load_config(args.config), args.dataset, args.output))
    elif args.command == "promotion-review":
        print(_promotion_review(_load_config(args.config), args.output))
    elif args.command == "benchmark-portfolios-cpp":
        print(_portfolio_benchmark(
            _load_config(args.config), args.dataset, args.model_benchmark,
            args.bars, args.output,
        ))
    elif args.command == "benchmark-ablation-portfolios-cpp":
        print(_portfolio_ablation(
            _load_config(args.config), args.dataset, args.feature_ablation,
            args.bars, args.output,
        ))
    elif args.command == "train":
        print(_train(_load_config(args.config), args.dataset, args.output))
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

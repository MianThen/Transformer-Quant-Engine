from __future__ import annotations

import hashlib
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import BAR_V1_FEATURE_GROUPS


PREDICTION_FIELDS = (
    "expected_return", "expected_volatility", "direction_probability",
    "lower_quantile", "upper_quantile", "confidence",
)
FORMAL_MODELS = (
    "momentum_20", "reversal_5", "equal_weight", "cash",
    "ridge_multitask", "mlp", "tcn", "gru", "transformer_v1_1",
)


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.is_dir():
        import pyarrow.parquet as pq
        files = sorted(source.rglob("*.parquet"))
        if not files:
            raise ValueError(f"数据目录不含 Parquet: {source}")
        return pd.concat(
            [pq.ParquetFile(path).read().to_pandas() for path in files],
            ignore_index=True,
        )
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source, dtype={"symbol": "string"})
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    raise ValueError(f"仅支持 CSV/Parquet 表: {source}")


def _load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    required = {"timestamps", "symbols", *PREDICTION_FIELDS}
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError("预测文件缺少字段: " + ", ".join(missing))
    size = len(arrays["timestamps"])
    if size == 0 or any(len(arrays[name]) != size for name in required):
        raise ValueError("预测文件字段长度不一致或为空")
    keys = list(zip(arrays["timestamps"].tolist(), arrays["symbols"].astype(str).tolist()))
    if len(set(keys)) != len(keys) or keys != sorted(keys):
        raise ValueError("预测必须按 (timestamp, symbol) 唯一升序排列")
    for name in PREDICTION_FIELDS:
        values = np.asarray(arrays[name], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"预测字段 {name} 包含非有限值")
    return arrays


def _validate_bars(frame: pd.DataFrame, execution_mode: str) -> pd.DataFrame:
    required = {
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
        "is_listed", "is_suspended", "is_st", "industry", "industry_known_at", "universe_asof",
        "reference_data_known_at_max", "signal_open", "signal_high",
        "signal_low", "signal_close", "adjustment_factor",
    }
    execution_fields = {"upper_limit", "lower_limit", "lot_size", "min_buy_quantity"}
    if execution_mode == "required_for_promotion":
        required |= execution_fields
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("正式 C++ 组合回测行情缺少字段: " + ", ".join(missing))
    frame = frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    keys = frame[["timestamp", "symbol"]]
    if keys.duplicated().any():
        raise ValueError("行情 (timestamp, symbol) 不能重复")
    for name in ("industry_known_at", "universe_asof", "reference_data_known_at_max"):
        if (frame[name].to_numpy() > frame["timestamp"].to_numpy()).any():
            raise ValueError(f"{name} 晚于行情 timestamp，存在 PIT 泄漏")
    if execution_mode == "optional_for_model_evaluation":
        for name in ("upper_limit", "lower_limit"):
            if name not in frame:
                frame[name] = 0.0
            frame[name] = frame[name].fillna(0.0)
        for name in ("lot_size", "min_buy_quantity"):
            if name not in frame:
                frame[name] = 100
            frame[name] = frame[name].fillna(100)
    numeric = frame[[
        "open", "high", "low", "close", "volume", "lot_size", "min_buy_quantity",
        "signal_open", "signal_high", "signal_low", "signal_close",
        "adjustment_factor",
    ]]
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("行情 OHLCV/lot_size 包含非有限值")
    price_fields = [
        "open", "high", "low", "close", "signal_open", "signal_high",
        "signal_low", "signal_close", "adjustment_factor",
    ]
    if (frame[price_fields] <= 0).any().any():
        raise ValueError("原始/信号 OHLC 和 adjustment_factor 必须为正数")
    factor = frame["adjustment_factor"].to_numpy(dtype=np.float64)
    for name in ("open", "high", "low", "close"):
        expected = frame[name].to_numpy(dtype=np.float64) * factor
        actual = frame[f"signal_{name}"].to_numpy(dtype=np.float64)
        if not np.allclose(actual, expected, rtol=1e-6, atol=1e-8):
            raise ValueError(f"signal_{name} 与 raw {name} * adjustment_factor 不一致")
    if ((frame["volume"] < 0).any() or (frame["lot_size"] <= 0).any()
            or (frame["min_buy_quantity"] <= 0).any()):
        raise ValueError("volume/lot_size/min_buy_quantity 无效")
    return frame


def _execution_reference_mode(config: dict, dataset: dict) -> str:
    mode = config.get("execution_reference_mode", "required_for_promotion")
    if mode not in {"required_for_promotion", "optional_for_model_evaluation"}:
        raise ValueError("不支持的 execution_reference_mode")
    if "execution_reference_status" not in dataset:
        raise ValueError("数据集缺少 execution_reference_status")
    status = str(np.asarray(dataset["execution_reference_status"]).item())
    if mode == "required_for_promotion" and status != "READY":
        raise ValueError("正式组合回测要求 execution_reference_status=READY")
    if mode == "optional_for_model_evaluation":
        execution = config.get("execution", {})
        if execution.get("enforce_price_limits") is not False:
            raise ValueError("延期执行数据模式必须关闭 enforce_price_limits")
        if execution.get("enforce_board_lot") is not False:
            raise ValueError("延期执行数据模式必须关闭 enforce_board_lot")
    return mode


def _validate_actions(config: dict) -> pd.DataFrame:
    if config.get("point_in_time") is not True or not config.get("source"):
        raise ValueError("正式回测必须配置 point-in-time corporate_actions.source")
    frame = _read_table(config["source"])
    required = {
        "timestamp", "symbol", "cash_dividend_per_share",
        "share_multiplier", "known_at",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("corporate actions 缺少字段: " + ", ".join(missing))
    if len(frame) and (frame["known_at"].to_numpy() > frame["timestamp"].to_numpy()).any():
        raise ValueError("corporate actions known_at 晚于生效时间")
    return frame.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def _engine_fees(engine_module, engine, config: dict) -> None:
    if config.get("point_in_time") is not True:
        raise ValueError("费用配置必须声明 point_in_time=true")
    schedules = config.get("schedules", [])
    if not schedules:
        raise ValueError("费用配置必须包含至少一个生效区间")
    engine.set_fee_schedules([engine_module.FeeSchedule(
        int(item["effective_from"]), item.get("effective_to"),
        float(item["commission_rate"]), float(item["min_commission"]),
        float(item["stamp_tax_rate"]), float(item.get("transfer_fee_rate", 0.0)),
    ) for item in schedules])


def _frames(prediction: dict, bars: pd.DataFrame) -> list[dict]:
    lookup = {
        (timestamp, symbol): index
        for index, (timestamp, symbol) in enumerate(zip(
            prediction["timestamps"].tolist(), prediction["symbols"].astype(str).tolist()
        ))
    }
    result = []
    for timestamp in np.unique(prediction["timestamps"]):
        section = bars[bars["timestamp"] == timestamp]
        if section.empty:
            raise ValueError(f"预测 timestamp={timestamp} 在行情中不存在")
        values = {
            "timestamp": int(timestamp),
            "expected_return": [], "expected_volatility": [],
            "direction_probability": [], "lower_quantile": [],
            "upper_quantile": [], "confidence": [], "flags": [],
        }
        seen = set()
        for symbol in section["symbol"].astype(str):
            key = (timestamp, symbol)
            index = lookup.get(key)
            if index is None:
                values["expected_return"].append(-1.0)
                values["expected_volatility"].append(0.0)
                values["direction_probability"].append(0.0)
                values["lower_quantile"].append(-1.0)
                values["upper_quantile"].append(0.0)
                values["confidence"].append(1.0)
                values["flags"].append(1)
                continue
            seen.add(key)
            for name in PREDICTION_FIELDS:
                values[name].append(float(prediction[name][index]))
            values["flags"].append(1)
        expected = {key for key in lookup if key[0] == timestamp}
        missing = sorted(expected - seen)
        if missing:
            raise ValueError(
                f"预测 timestamp={timestamp} 包含行情外 symbol: "
                + ", ".join(symbol for _, symbol in missing[:5])
            )
        result.append(values)
    return result


def _market_snapshot(engine_module, row):
    market = engine_module.MarketSnapshot()
    market.timestamp = int(row.timestamp)
    market.symbol = str(row.symbol)
    for name in ("open", "high", "low", "close"):
        setattr(market, name, float(getattr(row, name)))
        signal_name = f"signal_{name}"
        setattr(market, signal_name, float(
            getattr(row, signal_name) if hasattr(row, signal_name) else getattr(row, name)
        ))
    market.volume = int(row.volume)
    market.is_listed = bool(row.is_listed)
    market.is_suspended = bool(row.is_suspended)
    market.is_st = bool(row.is_st)
    market.upper_limit = float(row.upper_limit)
    market.lower_limit = float(row.lower_limit)
    market.lot_size = int(row.lot_size)
    market.min_buy_quantity = int(
        row.min_buy_quantity if hasattr(row, "min_buy_quantity") else row.lot_size
    )
    market.industry = str(row.industry)
    return market


def _metrics(engine, replay: pd.DataFrame, initial_cash: float) -> dict:
    curve = list(engine.get_equity_curve())
    equity = np.asarray([float(point.equity) for point in curve], dtype=np.float64)
    returns = equity[1:] / equity[:-1] - 1.0 if len(equity) >= 2 else np.asarray([])
    count = max(1, int(np.ceil(len(returns) * 0.05))) if len(returns) else 0
    trades = list(engine.get_trade_history())
    average_equity = float(equity.mean()) if len(equity) else initial_cash
    turnover = sum(
        abs(float(trade.quantity) * float(trade.price)) for trade in trades
    ) / average_equity
    commission = {}
    for trade in trades:
        commission[trade.symbol] = commission.get(trade.symbol, 0.0) + float(trade.commission)
    last = replay.drop_duplicates("symbol", keep="last").set_index("symbol")
    symbol_contributions = {}
    for position in engine.get_positions():
        symbol = str(position.symbol)
        close = float(last.loc[symbol, "close"]) if symbol in last.index else 0.0
        pnl = float(position.realized_pnl) - commission.get(symbol, 0.0)
        if position.quantity and close > 0:
            pnl += float(position.quantity) * (close - float(position.avg_cost))
        symbol_contributions[symbol] = pnl / initial_cash
    if hasattr(engine, "get_corporate_action_history"):
        for action in engine.get_corporate_action_history():
            symbol = str(action.symbol)
            symbol_contributions[symbol] = (
                symbol_contributions.get(symbol, 0.0)
                + float(action.cash_dividend) / initial_cash
            )
    industry_contributions = {}
    for symbol, contribution in symbol_contributions.items():
        if symbol not in last.index:
            continue
        industry = str(last.loc[symbol, "industry"])
        industry_contributions[industry] = (
            industry_contributions.get(industry, 0.0) + contribution
        )
    return {
        "net_return": float(engine.get_total_return()),
        "sharpe": float(engine.get_sharpe_ratio()),
        "max_drawdown": float(engine.get_max_drawdown()),
        "turnover": float(turnover),
        "cvar_95": float(np.sort(returns)[:count].mean()) if count else 0.0,
        "symbol_contributions": symbol_contributions,
        "industry_contributions": industry_contributions,
        "orders": int(engine.get_order_count()),
        "trades": int(engine.get_trade_count()),
    }


def _run_one(
    engine_module, prediction: dict, bars: pd.DataFrame, actions: pd.DataFrame,
    config: dict, slippage_bps: float, replay_end, model_name: str,
) -> dict:
    initial_cash = float(config.get("initial_cash", 1_000_000.0))
    execution = engine_module.ExecutionConfig()
    execution_config = config["execution"]
    for name, value in execution_config.items():
        if name != "slippage_bps":
            setattr(execution, name, value)
    execution.slippage_bps = float(slippage_bps)
    engine = engine_module.BacktestEngine(
        initial_cash, engine_module.FillTiming.NEXT_OPEN, execution
    )
    _engine_fees(engine_module, engine, config["fees"])
    if hasattr(engine_module, "HistoryConfig"):
        history = engine_module.HistoryConfig()
        history.equity_sampling = engine_module.EquitySampling.DAILY
        engine.set_history_config(history)
    model_hash = int.from_bytes(
        hashlib.sha256(model_name.encode("utf-8")).digest()[:8], "big"
    ) or 1
    engine.set_precomputed_prediction_strategy(
        _frames(prediction, bars), config["policy"], config["risk"],
        {"max_order_intents": int(config["runtime"]["max_order_intents"]),
         "model_version_hash": model_hash},
    )
    start = prediction["timestamps"].min()
    replay = bars[(bars["timestamp"] >= start) & (bars["timestamp"] <= replay_end)]
    if replay.empty:
        raise ValueError("窗口没有可回放行情")
    if replay["timestamp"].max() != replay_end:
        raise ValueError("行情未覆盖窗口最后一个 label exit timestamp")
    actions_by_timestamp = {
        timestamp: group for timestamp, group in actions[
            (actions["timestamp"] >= start) & (actions["timestamp"] <= replay_end)
        ].groupby("timestamp", sort=True)
    }
    last_timestamp = None
    seen_symbols = set()
    for timestamp, section in replay.groupby("timestamp", sort=True):
        action_rows = (
            actions_by_timestamp.get(timestamp, pd.DataFrame())
            if timestamp > start else pd.DataFrame()
        )
        for action_row in action_rows.itertuples():
            if str(action_row.symbol) not in seen_symbols:
                continue
            action = engine_module.CorporateAction()
            action.timestamp = int(action_row.timestamp)
            action.symbol = str(action_row.symbol)
            action.cash_dividend_per_share = float(action_row.cash_dividend_per_share)
            action.share_multiplier = float(action_row.share_multiplier)
            action.description = "phase-e-point-in-time"
            engine.apply_corporate_action(action)
        engine.process_market_data_batch([
            _market_snapshot(engine_module, row) for row in section.itertuples()
        ])
        seen_symbols.update(section["symbol"].astype(str))
        last_timestamp = int(timestamp)
    engine.finalize(last_timestamp)
    return _metrics(engine, replay, initial_cash)


def run_cpp_portfolio_benchmark(
    config: dict,
    dataset_path: str | Path,
    benchmark_path: str | Path,
    bars_path: str | Path,
    output_path: str | Path,
    *,
    _engine_module=None,
) -> Path:
    if config.get("enabled") is not True:
        raise RuntimeError("C++ Portfolio Benchmark 必须显式设置 enabled=true")
    engine_module = _engine_module or importlib.import_module("cpp_engine")
    if getattr(engine_module, "__build_type__", None) != "Release":
        raise RuntimeError("正式组合 Benchmark 要求 Release C++ 扩展")
    if getattr(engine_module, "__lto_enabled__", None) is not True:
        raise RuntimeError("正式组合 Benchmark 要求 LTO C++ 扩展")
    if not hasattr(engine_module.BacktestEngine, "set_precomputed_prediction_strategy"):
        raise RuntimeError("C++ 扩展缺少 precomputed prediction runtime")
    with np.load(dataset_path, allow_pickle=False) as source:
        dataset = {name: source[name] for name in source.files}
    required_dataset = {
        "dataset_fingerprint", "timestamps", "symbols", "label_exit_timestamp",
        "frequency", "calendar_id",
        "price_adjustment_mode",
    }
    missing = sorted(required_dataset - set(dataset))
    if missing:
        raise ValueError("组合 Benchmark 数据集缺少字段: " + ", ".join(missing))
    if str(np.asarray(dataset["price_adjustment_mode"]).item()) != (
        "pit_adjusted_signal_raw_execution"
    ):
        raise ValueError("正式组合 Benchmark 要求 PIT 复权信号价与原始成交价分离")
    execution_mode = _execution_reference_mode(config, dataset)
    benchmark = Path(benchmark_path)
    prediction_manifest = json.loads(
        (benchmark / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    fingerprint = str(np.asarray(dataset["dataset_fingerprint"]).item())
    if prediction_manifest.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Prediction manifest 与数据集 fingerprint 不一致")
    bars = _validate_bars(_read_table(bars_path), execution_mode)
    actions = _validate_actions(config["corporate_actions"])
    required_models = list(config["required_models"])
    if tuple(required_models) != FORMAL_MODELS:
        raise ValueError("正式组合 Benchmark 的九个模型及顺序不能更改")
    if set(required_models) != set(prediction_manifest.get("models", {})):
        raise ValueError("required_models 必须与 prediction manifest 完全一致")
    scenarios = [float(value) for value in config["slippage_scenarios_bps"]]
    if scenarios != [0.0, 5.0, 10.0]:
        raise ValueError("正式组合 Benchmark 固定运行 0/5/10 bp")
    model_windows = prediction_manifest["models"]
    window_ids = [item["window_id"] for item in model_windows[required_models[0]]]
    if len(window_ids) < 3 or any(
        [item["window_id"] for item in model_windows[name]] != window_ids
        for name in required_models
    ):
        raise ValueError("所有模型必须包含相同的至少三个窗口")
    report_scenarios = []
    for slippage in scenarios:
        windows = []
        for window_id in window_ids:
            models = {}
            test_start = None
            test_end = None
            for model_name in required_models:
                item = next(
                    value for value in model_windows[model_name]
                    if value["window_id"] == window_id
                )
                prediction = _load_predictions(item["path"])
                starts = np.asarray(prediction["timestamps"])
                start, end = starts.min(), starts.max()
                if test_start is None:
                    test_start, test_end = start, end
                elif start != test_start or end != test_end:
                    raise ValueError("同一窗口各模型 timestamp 区间不一致")
                selected = np.isin(np.asarray(dataset["timestamps"]), starts)
                replay_end = np.asarray(dataset["label_exit_timestamp"])[selected].max()
                models[model_name] = _run_one(
                    engine_module, prediction, bars, actions, config,
                    slippage, replay_end, model_name,
                )
            windows.append({
                "window_id": window_id,
                "test_start": int(test_start), "test_end": int(test_end),
                "models": models,
            })
        report_scenarios.append({"slippage_bps": slippage, "windows": windows})
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "cpp", "net_of_cost": True,
        "execution_reference_mode": execution_mode,
        "promotion_eligible": execution_mode == "required_for_promotion",
        "limitations": [] if execution_mode == "required_for_promotion" else [
            "historical price-limit enforcement disabled",
            "historical board-lot enforcement disabled; placeholder lot size is 100",
            "research-only metrics are not promotion evidence",
        ],
        "dataset_fingerprint": fingerprint,
        "frequency": str(np.asarray(dataset["frequency"]).item()),
        "calendar_id": str(np.asarray(dataset["calendar_id"]).item()),
        "engine_build": {
            "build_type": engine_module.__build_type__,
            "lto_enabled": engine_module.__lto_enabled__,
            "compiler_id": getattr(engine_module, "__compiler_id__", "unknown"),
        },
        "cost_model": {
            "point_in_time_fees": True,
            "enforce_t_plus_one": config["execution"]["enforce_t_plus_one"],
            "enforce_price_limits": config["execution"]["enforce_price_limits"],
            "enforce_board_lot": config["execution"]["enforce_board_lot"],
            "max_volume_participation": config["execution"]["max_volume_participation"],
        },
        "scenarios": report_scenarios,
    }
    destination = output / "portfolio_backtest.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def run_cpp_portfolio_ablation(
    config: dict,
    dataset_path: str | Path,
    ablation_path: str | Path,
    bars_path: str | Path,
    output_path: str | Path,
    *,
    _engine_module=None,
) -> Path:
    if config.get("enabled") is not True:
        raise RuntimeError("C++ Portfolio Ablation 必须显式设置 enabled=true")
    engine_module = _engine_module or importlib.import_module("cpp_engine")
    if (getattr(engine_module, "__build_type__", None) != "Release" or
            getattr(engine_module, "__lto_enabled__", None) is not True):
        raise RuntimeError("正式 Portfolio Ablation 要求 Release/LTO C++ 扩展")
    with np.load(dataset_path, allow_pickle=False) as source:
        dataset = {name: source[name] for name in source.files}
    required_dataset = {
        "dataset_fingerprint", "timestamps", "label_exit_timestamp",
        "price_adjustment_mode",
    }
    missing = sorted(required_dataset - set(dataset))
    if missing:
        raise ValueError("Portfolio Ablation 数据集缺少字段: " + ", ".join(missing))
    if str(np.asarray(dataset["price_adjustment_mode"]).item()) != (
        "pit_adjusted_signal_raw_execution"
    ):
        raise ValueError("Portfolio Ablation 要求 PIT 复权信号价")
    execution_mode = _execution_reference_mode(config, dataset)
    fingerprint = str(np.asarray(dataset["dataset_fingerprint"]).item())
    root = Path(ablation_path)
    manifest = json.loads(
        (root / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    groups = list(manifest.get("groups", []))
    seeds = list(manifest.get("seeds", []))
    if groups != list(BAR_V1_FEATURE_GROUPS):
        raise ValueError("正式 Portfolio Ablation 必须按冻结顺序包含全部特征组")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("正式 Portfolio Ablation 至少需要三个不重复种子")
    if manifest.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Feature Ablation 与数据集 fingerprint 不一致")
    bars = _validate_bars(_read_table(bars_path), execution_mode)
    actions = _validate_actions(config["corporate_actions"])
    scenarios = [float(value) for value in config["slippage_scenarios_bps"]]
    if scenarios != [0.0, 5.0, 10.0]:
        raise ValueError("正式 Portfolio Ablation 固定运行 0/5/10 bp")
    results = []
    full_cache = {}
    prediction_cache = {}
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        full_root = seed_root / "baseline_full"
        window_ids = sorted(
            int(path.name.split("_")[1]) for path in full_root.glob("window_*")
            if (path / "predictions.npz").is_file()
        )
        if len(window_ids) < 3:
            raise ValueError(f"seed={seed} 的 full baseline 少于三个窗口")
        for group in groups:
            drop_root = seed_root / f"drop_{group}"
            paired_windows = []
            for slippage in scenarios:
                scenario_windows = []
                for window_id in window_ids:
                    full_path = full_root / f"window_{window_id:03d}" / "predictions.npz"
                    drop_path = drop_root / f"window_{window_id:03d}" / "predictions.npz"
                    if full_path not in prediction_cache:
                        prediction_cache[full_path] = _load_predictions(full_path)
                    if drop_path not in prediction_cache:
                        prediction_cache[drop_path] = _load_predictions(drop_path)
                    full_prediction = prediction_cache[full_path]
                    drop_prediction = prediction_cache[drop_path]
                    if not (
                        np.array_equal(full_prediction["timestamps"], drop_prediction["timestamps"])
                        and np.array_equal(full_prediction["symbols"], drop_prediction["symbols"])
                    ):
                        raise ValueError(
                            f"seed={seed} group={group} window={window_id} 样本不一致"
                        )
                    timestamps = np.asarray(full_prediction["timestamps"])
                    selected = np.isin(np.asarray(dataset["timestamps"]), timestamps)
                    replay_end = np.asarray(dataset["label_exit_timestamp"])[selected].max()
                    cache_key = (seed, window_id, slippage)
                    if cache_key not in full_cache:
                        full_cache[cache_key] = _run_one(
                            engine_module, full_prediction, bars, actions, config,
                            slippage, replay_end, f"seed_{seed}_full",
                        )
                    full_metrics = full_cache[cache_key]
                    drop_metrics = _run_one(
                        engine_module, drop_prediction, bars, actions, config,
                        slippage, replay_end, f"seed_{seed}_drop_{group}",
                    )
                    delta_fields = (
                        "net_return", "sharpe", "max_drawdown", "turnover", "cvar_95"
                    )
                    scenario_windows.append({
                        "window_id": window_id,
                        "full": full_metrics,
                        "drop": drop_metrics,
                        "delta_drop_minus_full": {
                            name: float(drop_metrics[name] - full_metrics[name])
                            for name in delta_fields
                        },
                    })
                paired_windows.append({
                    "slippage_bps": slippage, "windows": scenario_windows,
                })
            results.append({
                "seed": seed, "group": group, "scenarios": paired_windows,
            })
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "cpp", "net_of_cost": True,
        "execution_reference_mode": execution_mode,
        "promotion_eligible": execution_mode == "required_for_promotion",
        "limitations": [] if execution_mode == "required_for_promotion" else [
            "historical price-limit enforcement disabled",
            "historical board-lot enforcement disabled; placeholder lot size is 100",
            "research-only metrics are not promotion evidence",
        ],
        "dataset_fingerprint": fingerprint,
        "groups": groups, "seeds": seeds,
        "slippage_scenarios_bps": scenarios,
        "results": results,
    }
    destination = output / "portfolio_ablation.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination

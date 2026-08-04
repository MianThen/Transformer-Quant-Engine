#!/usr/bin/env python3
"""Generate a C++ Replay/CVaR research artifact from proxy market fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LIMITATIONS = [
    "REFERENCE_PRICE_PROXY_BAR_CLOSE",
    "NO_COMMISSION_OR_TAX_SCHEDULE",
    "NO_PRICE_LIMIT_FIELDS",
    "NO_CORPORATE_ACTION_ADJUSTMENT_FIELDS",
    "NO_BOARD_LOT_PROVENANCE",
    "NO_SLIPPAGE_PROVENANCE",
    "COMMON_SYMBOL_INTERSECTION_PROXY",
]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _finite_price(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("价格必须是有限正数")
    return result


def _batch(engine_module: Any, frame: pd.DataFrame) -> list[Any]:
    values = []
    for row in frame.itertuples(index=False):
        market = engine_module.MarketSnapshot()
        market.timestamp = int(row.timestamp)
        market.symbol = str(row.symbol)
        market.open = _finite_price(row.open)
        market.high = _finite_price(row.high)
        market.low = _finite_price(row.low)
        market.close = _finite_price(row.close)
        market.signal_open = market.open
        market.signal_high = market.high
        market.signal_low = market.low
        market.signal_close = market.close
        market.volume = int(max(0, int(row.volume)))
        market.is_listed = bool(row.is_listed)
        market.is_suspended = bool(row.is_suspended)
        market.is_st = bool(row.is_st)
        market.lot_size = 1
        market.min_buy_quantity = 1
        market.upper_limit = 0.0
        market.lower_limit = 0.0
        values.append(market)
    return values


def _make_rebalance_callback(
    engine_module: Any,
    engine: Any,
    scores_by_timestamp: dict[int, dict[str, float]],
    top_k: int,
):
    opened = False

    def callback(batch: list[Any]) -> list[Any]:
        nonlocal opened
        if not batch:
            return []
        timestamp = int(batch[0].timestamp)
        if not opened:
            return []
        scores = scores_by_timestamp.get(timestamp)
        if scores is None:
            raise ValueError(f"预测缺少 timestamp={timestamp}")
        prices = {
            market.symbol: _finite_price(market.close)
            for market in batch
            if market.is_listed and not market.is_suspended and not market.is_st
        }
        ranked = [
            symbol for symbol, _ in sorted(
                ((symbol, score) for symbol, score in scores.items() if symbol in prices),
                key=lambda item: (-item[1], item[0]),
            )[:top_k]
        ]
        if not ranked:
            return []
        positions = {
            position.symbol: int(position.quantity)
            for position in engine.get_positions()
            if int(position.quantity) > 0
        }
        equity = max(float(engine.get_equity()), 0.0)
        notional = equity * 0.95 / len(ranked)
        target = {
            symbol: int(notional // prices[symbol])
            for symbol in ranked
            if prices[symbol] > 0.0
        }
        orders = []
        for symbol, quantity in sorted(positions.items()):
            desired = target.get(symbol, 0)
            if quantity > desired:
                order = engine_module.Order()
                order.symbol = symbol
                order.side = engine_module.Side.SELL
                order.type = engine_module.OrderType.MARKET
                order.quantity = quantity - desired
                orders.append(order)
        for symbol in sorted(target):
            current = positions.get(symbol, 0)
            if target[symbol] > current:
                order = engine_module.Order()
                order.symbol = symbol
                order.side = engine_module.Side.BUY
                order.type = engine_module.OrderType.MARKET
                order.quantity = target[symbol] - current
                orders.append(order)
        return orders

    def mark_open() -> None:
        nonlocal opened
        opened = True

    return callback, mark_open


def run_proxy_replay(
    data_path: Path,
    prediction_path: Path,
    output_path: Path,
    *,
    top_k: int,
    confidence_level: float,
    initial_cash: float,
    calendar_id: str,
    periods_per_year: float,
    config_hash: int,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"输出文件不可覆盖: {output_path}")
    if top_k < 1 or not 0.0 < confidence_level < 1.0:
        raise ValueError("top_k/confidence_level 无效")
    if not math.isfinite(initial_cash) or initial_cash <= 0.0:
        raise ValueError("initial_cash 必须为有限正数")
    try:
        import cpp_engine
    except ImportError as exc:
        raise RuntimeError("需要带 performance analytics/portfolio math 的 cpp_engine") from exc
    if not hasattr(cpp_engine, "PeriodContributionReplaySink"):
        raise RuntimeError("cpp_engine 未启用 performance analytics")
    if not hasattr(cpp_engine, "estimate_empirical_cvar"):
        raise RuntimeError("cpp_engine 未启用 portfolio math")

    with np.load(prediction_path, allow_pickle=False) as prediction:
        timestamps = prediction["timestamps"].astype(np.int64, copy=False)
        symbols = prediction["symbols"].astype(str)
        scores = prediction["scores"].astype(np.float64, copy=False)
    if timestamps.size == 0 or timestamps.size != symbols.size or timestamps.size != scores.size:
        raise ValueError("预测文件字段为空或长度不一致")
    if not np.isfinite(scores).all():
        raise ValueError("预测 scores 必须全部有限")
    scores_by_timestamp: dict[int, dict[str, float]] = {}
    for timestamp, symbol, score in zip(timestamps, symbols, scores):
        bucket = scores_by_timestamp.setdefault(int(timestamp), {})
        if str(symbol) in bucket:
            raise ValueError("预测 timestamp/symbol 键重复")
        bucket[str(symbol)] = float(score)

    frame = pd.read_parquet(data_path)
    frame = frame[frame["timestamp"].isin(list(scores_by_timestamp))].copy()
    required = {
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
        "is_listed", "is_suspended", "is_st",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"行情缺少字段: {', '.join(missing)}")
    frame["symbol"] = frame["symbol"].astype(str)
    common_symbols: set[str] | None = None
    for _, current in frame.groupby("timestamp", sort=False):
        values = set(current["symbol"].astype(str))
        common_symbols = values if common_symbols is None else common_symbols & values
    if common_symbols is None or len(common_symbols) < top_k:
        raise ValueError("跨 proxy 窗口的共同股票数不足 top_k")
    frame = frame[frame["symbol"].isin(common_symbols)].copy()
    scores_by_timestamp = {
        timestamp: {
            symbol: score for symbol, score in values.items() if symbol in common_symbols
        }
        for timestamp, values in scores_by_timestamp.items()
    }
    expected_keys = set(
        (int(timestamp), str(symbol))
        for timestamp, values in scores_by_timestamp.items()
        for symbol in values
    )
    frame["_key"] = list(zip(frame["timestamp"].astype(int), frame["symbol"]))
    actual_keys = set(frame["_key"])
    frame = frame.drop(columns=["_key"])
    frame = frame.sort_values(["timestamp", "symbol"], kind="stable")
    if not expected_keys.issubset(actual_keys):
        raise ValueError("预测与行情的 timestamp/symbol 键集合不一致")

    engine_config = cpp_engine.ExecutionConfig()
    engine_config.enforce_t_plus_one = False
    engine_config.enforce_board_lot = False
    engine_config.enforce_price_limits = False
    engine_config.enforce_cash = True
    engine_config.slippage_bps = 0.0
    engine = cpp_engine.BacktestEngine(
        float(initial_cash), cpp_engine.FillTiming.CLOSE, engine_config
    )
    sink = cpp_engine.PeriodContributionReplaySink(
        calendar_id, float(periods_per_year), int(config_hash)
    )
    engine.set_replay_analytics_sink(sink)
    callback, mark_open = _make_rebalance_callback(
        cpp_engine, engine, scores_by_timestamp, top_k
    )
    engine.set_on_cross_section(callback)
    grouped = list(frame.groupby("timestamp", sort=True))
    if len(grouped) < 3:
        raise ValueError("proxy Replay 至少需要三个交易日")
    first_timestamp, first_frame = grouped[0]
    engine.process_market_data_batch(_batch(cpp_engine, first_frame))
    engine.open_performance_period(1, int(first_timestamp))
    mark_open()
    for index, (timestamp, current_frame) in enumerate(grouped[1:], start=1):
        engine.process_market_data_batch(_batch(cpp_engine, current_frame))
        engine.close_performance_period(index, int(timestamp))
        if index + 1 < len(grouped):
            engine.open_performance_period(index + 1, int(timestamp))
    engine.finalize(int(grouped[-1][0]))
    if sink.failed():
        raise RuntimeError("C++ PeriodContributionReplaySink failed")
    period_returns = list(sink.period_returns())
    period_end_timestamps = list(sink.period_end_timestamps())
    if len(period_returns) != len(period_end_timestamps) or len(period_returns) < 20:
        raise RuntimeError("C++ Replay period return 样本不足")
    cvar = cpp_engine.estimate_empirical_cvar(
        period_returns, period_end_timestamps, confidence_level, config_hash
    )
    source_fingerprint = _canonical_hash({
        "data_sha256": _file_hash(data_path),
        "prediction_sha256": _file_hash(prediction_path),
        "top_k": top_k,
        "confidence_level": confidence_level,
        "initial_cash": initial_cash,
    })
    dataset_fingerprint = _file_hash(data_path)
    ledger_artifact = json.loads(sink.serialize_return_ledger_artifact(
        source_fingerprint,
        dataset_fingerprint,
        "PROXY",
        False,
        LIMITATIONS,
    ))
    return_analysis = json.loads(sink.serialize_return_analysis_report(
        source_fingerprint,
        dataset_fingerprint,
        "PROXY",
        False,
        LIMITATIONS,
        cvar.get("var_loss"),
        cvar.get("expected_shortfall_loss"),
        cvar.get("return_cvar"),
    ))
    report = {
        "schema_version": 1,
        "status": "proxy_replay_complete",
        "reference_price_quality": "PROXY",
        "promotion_eligible": False,
        "cpp_replay_executed": True,
        "cvar_available": cvar.get("return_cvar") is not None,
        "data_path": str(data_path),
        "prediction_path": str(prediction_path),
        "data_sha256": _file_hash(data_path),
        "prediction_sha256": _file_hash(prediction_path),
        "source_replay_sha256": source_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "periods": len(period_returns),
        "first_timestamp": int(grouped[0][0]),
        "last_timestamp": int(grouped[-1][0]),
        "top_k": top_k,
        "initial_cash": initial_cash,
        "calendar_id": calendar_id,
        "cvar": cvar,
        "period_returns": period_returns,
        "period_end_timestamps": period_end_timestamps,
        "ledger_sha256": sink.ledger_sha256(),
        "ledger_artifact": ledger_artifact,
        "return_analysis_report": return_analysis,
        "limitations": LIMITATIONS,
        "promotion_block_reason": "PROXY reference/fill and missing economic provenance",
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--calendar-id", default="XSHG_TRADING_DAY_PROXY_V1")
    parser.add_argument("--periods-per-year", type=float, default=242.0)
    parser.add_argument("--config-hash", type=int, default=20260802)
    args = parser.parse_args()
    report = run_proxy_replay(
        Path(args.data).resolve(),
        Path(args.predictions).resolve(),
        Path(args.output).resolve(),
        top_k=args.top_k,
        confidence_level=args.confidence_level,
        initial_cash=args.initial_cash,
        calendar_id=args.calendar_id,
        periods_per_year=args.periods_per_year,
        config_hash=args.config_hash,
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "report_sha256": report["report_sha256"],
        "periods": report["periods"],
        "return_cvar": report["cvar"].get("return_cvar"),
        "promotion_eligible": report["promotion_eligible"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

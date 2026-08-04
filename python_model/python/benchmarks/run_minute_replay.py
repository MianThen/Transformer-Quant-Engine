from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="分钟线数据路径分阶段 benchmark")
    parser.add_argument("lake", help="MinuteBarDataLake 根目录")
    parser.add_argument("--symbol", action="append", dest="symbols")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--target-mb", type=int, default=256)
    parser.add_argument("--strategy-symbols", type=int, default=10)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--output", default="benchmark-report.json")
    parser.add_argument("--backend", choices=("cpp",), default="cpp")
    parser.add_argument(
        "--cache-state", choices=("cold", "warm", "uncontrolled"),
        default="uncontrolled",
    )
    parser.add_argument(
        "--cold-cache-command",
        help="非 Linux 平台执行 cold benchmark 前使用的清缓存命令",
    )
    args = parser.parse_args()
    if args.strategy_symbols <= 0:
        raise ValueError("strategy-symbols 必须为正数")
    if not 0.0 <= args.commission_rate < 1.0:
        raise ValueError("commission-rate 必须在 [0, 1) 内")

    os.environ["QBT_BACKEND"] = args.backend
    from python.benchmarks import MinuteReplayBenchmark
    from python.benchmarks.framework import validate_benchmark_backend
    from python.engine_api import (
        BacktestEngine, ExecutionConfig, Order, OrderType, Side,
    )
    from python.market_data import MinuteBarDataLake

    validate_benchmark_backend(BacktestEngine)

    benchmark = MinuteReplayBenchmark(
        MinuteBarDataLake(args.lake), target_bytes=args.target_mb * 1024 * 1024
    )
    def engine_factory():
        engine = BacktestEngine()
        config = ExecutionConfig()
        config.enforce_t_plus_one = False
        engine.set_execution_config(config)
        engine.set_commission_fn(
            lambda notional, _is_sell: notional * args.commission_rate
        )
        return engine

    callback_count = 0

    def round_trip_strategy(batch):
        nonlocal callback_count
        callback_count += 1
        if callback_count == 1:
            side = Side.BUY
        elif callback_count == 2:
            side = Side.SELL
        else:
            return []
        orders = []
        for snapshot in batch[:args.strategy_symbols]:
            order = Order()
            order.symbol = snapshot.symbol
            order.side = side
            order.type = OrderType.MARKET
            order.quantity = 1
            orders.append(order)
        return orders

    report = benchmark.run(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        engine_factory=engine_factory,
        strategy_callback=round_trip_strategy,
        strategy_name="two_batch_round_trip",
        strategy_parameters={
            "symbols": args.strategy_symbols,
            "quantity": 1,
            "commission_rate": args.commission_rate,
            "enforce_t_plus_one": False,
        },
        cache_state=args.cache_state,
        cold_cache_command=args.cold_cache_command,
    )
    report.save(args.output)
    for stage in report.stages:
        print(
            f"{stage.name:10s} {stage.seconds:9.4f}s "
            f"{stage.rows_per_second:12,.0f} rows/s "
            f"{stage.megabytes_per_second:9.1f} MB/s"
        )
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()

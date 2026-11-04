"""MA(10/30) 全市场双均线探针 —— 在同一 run_spec 下对旧湖/压实湖做逐位比对。

设计要点（全部确定性，无随机、无时间依赖）：
- 按 bar 的时间戳推进，逐 symbol 维护 10/30 周期收盘均线
- 上穿（MA10 由下向上穿过 MA30）买入，下穿卖出
- 固定下单数量 1 手（quantity=1），避免资金分配的浮点差异
- 手续费按成交额 * commission_rate，T+1 关闭（与既有基准一致，便于对齐）
- 同一批 bar 内先统一计算信号再统一下单，避免批边界影响结果

用法：
    python ma_dual_probe.py <lake_root> --start-ms ... --end-ms ... --output out.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict, deque


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lake")
    parser.add_argument("--start", type=int, default=None, help="起始时间戳（ns）")
    parser.add_argument("--end", type=int, default=None, help="结束时间戳（ns）")
    parser.add_argument("--fast", type=int, default=10)
    parser.add_argument("--slow", type=int, default=30)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--target-mb", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ["QBT_BACKEND"] = "cpp"

    from python.benchmarks import MinuteReplayBenchmark
    from python.engine_api import (
        BacktestEngine,
        ExecutionConfig,
        Order,
        OrderType,
        Side,
    )
    from python.market_data import MinuteBarDataLake

    # ---- 策略状态：逐 symbol 的收盘价环形缓冲 ----
    closes: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=args.slow))
    prev_diff: dict[str, float] = {}
    signal_count = {"buy": 0, "sell": 0}
    callback_count = 0

    def dual_ma_strategy(batch):
        nonlocal callback_count
        callback_count += 1
        orders = []
        # 先更新所有 symbol 的均线状态（同一批内信号互不影响，保证确定性）
        crosses = []
        for snapshot in batch:
            sym = snapshot.symbol
            buf = closes[sym]
            buf.append(float(snapshot.close))
            if len(buf) < args.slow:
                continue
            values = list(buf)
            fast_ma = sum(values[-args.fast:]) / args.fast
            slow_ma = sum(values) / args.slow
            diff = fast_ma - slow_ma
            prev = prev_diff.get(sym)
            if prev is not None:
                if prev <= 0.0 < diff:
                    crosses.append((sym, Side.BUY))
                elif prev >= 0.0 > diff:
                    crosses.append((sym, Side.SELL))
            prev_diff[sym] = diff
        # 再统一下单
        for sym, side in crosses:
            order = Order()
            order.symbol = sym
            order.side = side
            order.type = OrderType.MARKET
            order.quantity = args.quantity
            orders.append(order)
            signal_count["buy" if side == Side.BUY else "sell"] += 1
        return orders

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

    started = time.time()
    report = benchmark.run(
        symbols=None,
        start=args.start,
        end=args.end,
        engine_factory=engine_factory,
        strategy_callback=dual_ma_strategy,
        strategy_name="dual_ma_10_30",
        strategy_parameters={
            "fast": args.fast,
            "slow": args.slow,
            "quantity": args.quantity,
            "commission_rate": args.commission_rate,
            "enforce_t_plus_one": False,
        },
    )

    result = {}
    total_rows = 0
    modal = None
    for stage in report.stages:
        total_rows = max(total_rows, stage.rows or 0)
        if stage.name == "strategy_backtest":
            modal = stage
    if modal is not None:
        det = getattr(modal, "details", {}) or {}
        result = det.get("result", {})

    payload = {
        "lake": args.lake,
        "generation": report.lineage.get("catalog_generation"),
        "window": {"start": args.start, "end": args.end},
        "rows": total_rows,
        "callback_calls": callback_count,
        "signals": signal_count,
        "result": result,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

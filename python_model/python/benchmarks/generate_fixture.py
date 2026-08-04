from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成确定性的分钟线 benchmark Parquet")
    parser.add_argument("output")
    parser.add_argument("--symbols", type=int, default=1000)
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--minutes", type=int, default=240)
    args = parser.parse_args()
    if args.symbols <= 0 or args.days <= 0 or args.minutes <= 0:
        raise ValueError("symbols/days/minutes 必须为正数")

    import pyarrow as pa
    import pyarrow.parquet as parquet

    base = 1_767_225_600_000_000_000
    day_ns = 86_400_000_000_000
    minute_ns = 60_000_000_000
    timestamps = []
    symbols = []
    opens = []
    for day in range(args.days):
        for minute in range(args.minutes):
            timestamp = base + day * day_ns + minute * minute_ns
            for index in range(args.symbols):
                timestamps.append(timestamp)
                symbols.append(f"S{index:05d}")
                opens.append(10.0 + (index % 100) / 10 + minute / 10_000)
    prices = pa.array(opens, type=pa.float64())
    table = pa.table({
        "timestamp": pa.array(timestamps, type=pa.int64()),
        "symbol": pa.array(symbols, type=pa.string()),
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": pa.array([10_000] * len(timestamps), type=pa.int64()),
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, output, compression="zstd", row_group_size=128_000)
    print(f"rows={table.num_rows:,} output={output}")


if __name__ == "__main__":
    main()

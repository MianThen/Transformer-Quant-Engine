"""行情来源:scrapy baostock 日线(权威,止于爬虫最后更新)+ staging(akshare 回补)。

两种来源归一为同一列结构,主键 (timestamp, symbol);同键冲突时以 scrapy 为准。
"""
from __future__ import annotations

import datetime as dt
import glob
import sqlite3

import pandas as pd

from .config import SCRAPY_BARS_ROOT, SCRAPY_STATE_ROOT

NS_PER_DAY = 86_400_000_000_000

FRAME_COLUMNS = [
    "timestamp", "symbol", "open", "high", "low", "close", "volume",
    "is_listed", "is_suspended", "is_st", "source",
]


def date_to_ns(date: str) -> int:
    """交易日日期字符串 -> 该日 15:00 北京时间(07:00 UTC)收盘纳秒时间戳。"""
    year, month, day = (int(part) for part in date.split("-"))
    moment = dt.datetime(year, month, day, 7, tzinfo=dt.timezone.utc)
    return int(moment.timestamp()) * 1_000_000_000


def ns_to_date(ns: int) -> str:
    return dt.datetime.utcfromtimestamp(ns / 1e9 + 8 * 3600).strftime("%Y-%m-%d")


def _years_spanning(start_ns: int, end_ns: int) -> list[int]:
    first = dt.datetime.utcfromtimestamp(start_ns / 1e9).year
    last = dt.datetime.utcfromtimestamp(end_ns / 1e9).year
    return list(range(first, last + 1))


def _glob_files(root, symbols: list[str], years: list[int]) -> list[str]:
    files: list[str] = []
    for symbol in symbols:
        for year in years:
            files.extend(glob.glob(
                str(root / f"ticker={symbol}" / f"year={year}" / "*.parquet")
            ))
    return files


def load_scrapy_frame(
    symbols: list[str], start_ns: int, end_ns: int
) -> pd.DataFrame:
    """读取 baostock 日线并左连 security_state,返回按 (timestamp, symbol) 有序的表。"""
    years = _years_spanning(start_ns, end_ns)
    bar_files = _glob_files(SCRAPY_BARS_ROOT, symbols, years)
    if not bar_files:
        return pd.DataFrame(columns=FRAME_COLUMNS)

    import duckdb

    connection = duckdb.connect()
    bars = connection.execute(
        """
        SELECT timestamp, symbol, open, high, low, close, volume
        FROM read_parquet(?, hive_partitioning=true)
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp, symbol
        """,
        [bar_files, start_ns, end_ns],
    ).fetch_df()
    if bars.empty:
        return pd.DataFrame(columns=FRAME_COLUMNS)

    # 同一标的同一年份可能存在多个重载分片(如 part-20260101-20260722 与
    # part-20260101-20260921),按交集查询会返回重复交易日。引擎对同一截面内的
    # 重复 symbol 会直接抛错,故这里先按 (timestamp, symbol) 去重。
    before = len(bars)
    bars = (
        bars.drop_duplicates(subset=["timestamp", "symbol"], keep="first")
        .sort_values(["timestamp", "symbol"], kind="stable")
        .reset_index(drop=True)
    )
    if len(bars) != before:
        print(f"[warn] scrapy 分片重叠,已去重 {before - len(bars)} 行重复 bar")

    state_files = _glob_files(SCRAPY_STATE_ROOT, symbols, years)
    if state_files:
        states = connection.execute(
            """
            SELECT timestamp, symbol, is_listed, is_suspended, is_st
            FROM read_parquet(?, hive_partitioning=true)
            WHERE timestamp BETWEEN ? AND ?
            """,
            [state_files, start_ns, end_ns],
        ).fetch_df()
        states = states.drop_duplicates(
            subset=["timestamp", "symbol"], keep="first"
        )
        bars = bars.merge(states, on=["timestamp", "symbol"], how="left")
    else:
        for name in ("is_listed", "is_suspended", "is_st"):
            bars[name] = None

    # phase1e 数据合同要求状态全覆盖;此处兜底默认值并计数告警
    missing = int(bars["is_listed"].isna().sum())
    if missing:
        print(f"[warn] scrapy 行情有 {missing} 行缺 security_state,按默认状态处理")
    bars["is_listed"] = bars["is_listed"].fillna(True).astype(bool)
    bars["is_suspended"] = bars["is_suspended"].fillna(False).astype(bool)
    bars["is_st"] = bars["is_st"].fillna(False).astype(bool)
    bars["source"] = "scrapy"
    return bars[FRAME_COLUMNS]


def load_staging_frame(
    conn: sqlite3.Connection, start_ns: int, end_ns: int
) -> pd.DataFrame:
    """读取 paper.db bars_staging(akshare 回补段)。"""
    rows = conn.execute(
        """
        SELECT ts, symbol, open, high, low, close, volume
        FROM bars_staging WHERE ts BETWEEN ? AND ?
        ORDER BY ts, symbol
        """,
        (start_ns, end_ns),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=FRAME_COLUMNS)
    frame = pd.DataFrame(
        rows, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    )
    frame["is_listed"] = True
    frame["is_suspended"] = False
    frame["is_st"] = False
    frame["source"] = "staging"
    return frame[FRAME_COLUMNS]


def load_frame(
    symbols: list[str], start_ns: int, end_ns: int, conn: sqlite3.Connection
) -> pd.DataFrame:
    """合并 scrapy 与 staging,同 (timestamp, symbol) 冲突保留 scrapy。"""
    scrapy = load_scrapy_frame(symbols, start_ns, end_ns)
    staging = load_staging_frame(conn, start_ns, end_ns)
    if staging.empty:
        return scrapy
    if scrapy.empty:
        return staging
    merged = pd.concat([scrapy, staging], ignore_index=True)
    merged = merged.drop_duplicates(
        subset=["timestamp", "symbol"], keep="first"
    ).sort_values(["timestamp", "symbol"], kind="stable")
    return merged.reset_index(drop=True)


def _staging_watermark_ns(conn: sqlite3.Connection) -> int | None:
    """staging 侧最新已有 bar(回补续传水位)。"""
    row = conn.execute("SELECT max(ts) FROM bars_staging").fetchone()
    return int(row[0]) if row and row[0] is not None else None


def scrapy_boundary(symbols: list[str]) -> int | None:
    """scrapy 数据的真实边界时间戳;探测不到近期数据时回退到更早窗口。

    原实现只探 [-60d, now]:一旦 scrapy 超过 60 天没更新,探针落空即返回 None,
    回补整体中止("%s" 提示)。这里逐级放宽窗口,
    使边界始终等于数据真实末尾,而不是"最近 60 天是否有数据"。
    """
    now_ns = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1e9)
    for days in (60, 365, 1_460):
        start_ns = now_ns - days * NS_PER_DAY
        frame = load_scrapy_frame(symbols, start_ns, now_ns)
        if not frame.empty:
            return int(frame["timestamp"].max())
        print(f"[warn] scrapy 在最近 {days} 个日历日内没有行情,继续向前探测")
    return None


def source_boundary_ns(
    symbols: list[str], conn: sqlite3.Connection
) -> tuple[int | None, str]:
    """回补判定基准 = max(scrapy 边界, staging 水位),附带来源说明。"""
    scrapy_ns = scrapy_boundary(symbols)
    staging_ns = _staging_watermark_ns(conn)
    if scrapy_ns is None and staging_ns is None:
        return None, "无"
    if scrapy_ns is None:
        return staging_ns, "staging"
    if staging_ns is None:
        return scrapy_ns, "scrapy"
    return (
        (scrapy_ns, "scrapy")
        if scrapy_ns >= staging_ns
        else (staging_ns, "staging")
    )


def day_source_tags(frame: pd.DataFrame) -> dict[int, str]:
    """每个交易日 -> 'scrapy' | 'staging' | 'mixed'(按该日行情来源标注)。"""
    tags: dict[int, str] = {}
    for timestamp, group in frame.groupby("timestamp")["source"]:
        kinds = set(group.unique())
        tags[int(timestamp)] = (
            kinds.pop() if len(kinds) == 1 else "mixed"
        )
    return tags

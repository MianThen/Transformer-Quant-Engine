"""akshare(新浪源)日线回补:拉取缺失区间写入 paper.db 的 bars_staging。

不写入 scrapy 数据湖,保持上游血缘纯净。本机 eastmoney CDN 不可达,改用
stock_zh_a_daily(新浪)。成交量单位换算系数(手 vs 股)是跨源口径的一部分,
按 source 持久化在 staging_meta,保证同一 source 的整列单位恒一致。

续传水位取 staging 自身的最新 bar:scrapy 停更时不会退化成整段重拉。
"""
from __future__ import annotations

import datetime as dt
import os
import time

import pandas as pd

from . import sources, store
from .universe import load_universe

SOURCE_NAME = "sina"
_REQUEST_INTERVAL_SECONDS = 0.35
# 首次标定用的重叠对拍窗口(仅在尚无已存系数时需要)
_CALIBRATION_CALENDAR_DAYS = 10
# 已存系数被重叠段证伪的相对容差
_MULTIPLIER_TOLERANCE = 0.02


def _exchange_prefix(symbol: str) -> str:
    """A 股代码 -> 新浪源前缀:6 开头沪市,其余深市(本宇宙全部为 000xxx)。"""
    return "sh" if symbol.startswith("6") else "sz"


def _multiplier_key(source: str) -> str:
    return f"volume_multiplier:{source}"


def _calibrate(
    symbol: str, rows: list[dict], overlap: pd.DataFrame
) -> float | None:
    """用重叠日对拍收盘价与成交量比;价格不一致直接中止,量比返回换算系数。

    返回 None 表示该标的在本次拉取范围内没有可对拍的重叠日(样本缺失)。
    这是"无证据"而非"证据矛盾":调用方应降级为沿用已存系数,不能中止整轮回补——
    拉取起点晚于校准窗口时(例如只回补单日),零交集是正常情形。
    """
    fetched = pd.DataFrame(rows)[["ts", "close", "volume"]].rename(
        columns={"ts": "timestamp"}
    )
    base = overlap[overlap["symbol"] == symbol]
    merged = fetched.merge(base, on="timestamp", suffixes=("_ak", "_ref"))
    if merged.empty:
        return None
    for _, row in merged.iterrows():
        if abs(row["close_ak"] - row["close_ref"]) > 0.02:
            raise SystemExit(
                f"{symbol} 重叠日收盘价不一致"
                f"(参考 {row['close_ref']} vs 新浪 {row['close_ak']}),"
                "怀疑复权口径不匹配,中止回补"
            )
    ratios = [
        row["volume_ref"] / row["volume_ak"]
        for _, row in merged.iterrows()
        if row["volume_ak"] and row["volume_ak"] > 0 and row["volume_ref"] > 0
    ]
    if not ratios:
        raise SystemExit(f"{symbol} 重叠日成交量样本不足,无法校准,中止回补")
    median = sorted(ratios)[len(ratios) // 2]
    if abs(median - 100.0) / 100.0 < 0.2:
        return 100.0
    if abs(median - 1.0) < 0.2:
        return 1.0
    raise SystemExit(
        f"新浪与参考源成交量比中位数 {median:.2f} 异常"
        "(既非 1 也非 100),请人工核查数据口径后再回补"
    )


def _resolve_multiplier(
    conn, symbol: str, parsed: list[dict], overlap: pd.DataFrame | None,
    announce: bool = True,
) -> float:
    """确定并持久化成交量系数。

    已有时以库中值为准(保证整列单位一致),仅在重叠段能对拍时做一次证伪;
    没有任何重叠样本时直接沿用库中值,不再中止整轮回补。
    """
    key = _multiplier_key(SOURCE_NAME)
    stored = store.get_meta(conn, key)
    if overlap is None or overlap.empty:
        if stored is None:
            raise SystemExit(
                f"缺少 {SOURCE_NAME} 成交量系数且无重叠样本可标定,已中止(未写入任何数据)"
            )
        if announce:
            print(f"沿用已存成交量系数 ×{float(stored):g}(本轮无可对拍重叠段)")
        return float(stored)
    computed = _calibrate(symbol, parsed, overlap)
    if computed is None:
        # 该标的在拉取范围内没有重叠日(如单日回补):无证据可证伪,沿用库中值
        if announce:
            print(f"沿用已存成交量系数 ×{float(stored):g}"
                  f"({symbol} 在本次拉取范围内无可对拍重叠日)")
        return float(stored)
    if stored is None:
        store.set_meta(conn, key, float(computed))
        print(f"校准完成:成交量系数 ×{computed:g}(基于 {symbol} 重叠日,已持久化)")
        return float(computed)
    if abs(computed - float(stored)) / float(stored) > _MULTIPLIER_TOLERANCE:
        raise SystemExit(
            f"成交量系数与库中值不一致(本次 {computed:g} vs 已存 {float(stored):g});"
            "疑似上游单位变更,请人工核查后再回补"
        )
    if announce:
        print(f"成交量系数 ×{float(stored):g}(与库中值一致,已逐笔复核)")
    return float(stored)


def backfill_staging(
    ak,
    from_date: str | None,
    to_date: str | None,
    *,
    symbols: list[str] | None = None,
    conn=None,
) -> int:
    """回补缺失区间,返回写入行数(0 表示已是最新)。"""
    symbols = symbols if symbols is not None else load_universe()
    # 系统代理(macOS 配置)会拦截国内行情站,行情拉取全程直连
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"

    owns_conn = conn is None
    conn = conn if conn is not None else store.connect()
    try:
        return _backfill_locked(ak, from_date, to_date, symbols, conn)
    finally:
        if owns_conn:
            conn.close()


def _backfill_locked(ak, from_date, to_date, symbols, conn) -> int:
    if to_date is None:
        to_date = dt.date.today().strftime("%Y-%m-%d")
    to_ns = sources.date_to_ns(to_date)

    boundary_ns, basis = sources.source_boundary_ns(symbols, conn)
    if from_date is None:
        if boundary_ns is None:
            raise SystemExit(
                "探测不到近期行情(scrapy 与 staging 均为空),请显式传 --from"
            )
        from_date = sources.ns_to_date(boundary_ns + sources.NS_PER_DAY)
    from_ns = sources.date_to_ns(from_date)
    if from_ns > to_ns:
        print(f"回补区间为空:{from_date} 晚于 {to_date},无需回补")
        return 0

    key = _multiplier_key(SOURCE_NAME)
    stored = store.get_meta(conn, key)
    # 校准窗口 [boundary-10d, boundary] 必须先于回补段被拉到,否则 fetched 与
    # overlap 的日期交集恒为空(单日回补时必然触发"无重叠日")。故这里无条件
    # 把拉取起点前移到校准窗口起点;写库时仍按 from_ns 过滤,落盘范围不变。
    calib_start_ns = max(0, boundary_ns - _CALIBRATION_CALENDAR_DAYS * sources.NS_PER_DAY)
    fetch_from_ns = from_ns
    if boundary_ns is not None:
        fetch_from_ns = min(fetch_from_ns, calib_start_ns)
    fetch_from = sources.ns_to_date(fetch_from_ns)

    overlap = (
        sources.load_scrapy_frame(symbols, calib_start_ns, boundary_ns)
        if boundary_ns is not None
        else None
    )
    print(f"回补区间:{from_date} .. {to_date}  边界基准={basis}"
          f"(scrapy 止于 {sources.ns_to_date(boundary_ns) if boundary_ns else '未知'})"
          f"  拉取起点 {fetch_from}")

    compact_from = fetch_from.replace("-", "")
    compact_to = to_date.replace("-", "")
    total_rows = 0
    fetched_symbols = 0
    announced = False
    for index, symbol in enumerate(symbols):
        try:
            raw = ak.stock_zh_a_daily(
                symbol=f"{_exchange_prefix(symbol)}{symbol}",
                start_date=compact_from, end_date=compact_to, adjust="",
            )
        except Exception as exc:  # noqa: BLE001 - 单标的失败不阻断整体回补
            print(f"[warn] {symbol} 拉取失败:{exc}")
            time.sleep(_REQUEST_INTERVAL_SECONDS)
            continue
        if raw is None or raw.empty:
            continue
        parsed = [
            {
                "ts": sources.date_to_ns(str(record["date"])),
                "open": float(record["open"]),
                "high": float(record["high"]),
                "low": float(record["low"]),
                "close": float(record["close"]),
                "volume": float(record["volume"]),
            }
            for _, record in raw.iterrows()
        ]
        multiplier = _resolve_multiplier(
            conn, symbol, parsed, overlap, announce=not announced
        )
        announced = True
        rows = [
            {
                "source": SOURCE_NAME,
                "ts": row["ts"],
                "symbol": symbol,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": int(round(row["volume"] * multiplier)),
            }
            for row in parsed
            if from_ns <= row["ts"] <= to_ns
        ]
        if rows:
            store.upsert_many(conn, "bars_staging", rows)
            total_rows += len(rows)
            fetched_symbols += 1
        if (index + 1) % 20 == 0:
            conn.commit()
            print(f"  进度 {index + 1}/{len(symbols)} 标的,{total_rows} 行")
        time.sleep(_REQUEST_INTERVAL_SECONDS)

    conn.commit()
    if total_rows == 0:
        print(f"回补无新增:{from_date} .. {to_date} 上游暂无新交易日")
        return 0
    multiplier = store.get_meta(conn, key)
    store.log_run(
        conn, None, to_ns, True,
        f"backfill {from_date}..{to_date}: {fetched_symbols} 标的 {total_rows} 行 "
        f"volume_multiplier={float(multiplier):g} source={SOURCE_NAME} basis={basis}",
    )
    conn.commit()
    print(f"回补完成:{fetched_symbols}/{len(symbols)} 标的,共 {total_rows} 行")
    return total_rows

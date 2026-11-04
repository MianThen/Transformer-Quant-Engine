"""虚拟盘命令实现:init / sync / status / backfill。"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from . import costs, sources, store
from .config import (
    ARTIFACT_PATH,
    DEFAULT_INITIAL_CASH,
    DEFAULT_START_DATE,
    SCRAPY_BARS_ENTRY,
    SCRAPY_PYTHON,
    SCRAPY_STATE_ENTRY,
    SESSION_PRESETS,
    WARMUP_TRADING_DAYS,
)
from .engine_driver import load_engine_module, replay
from .universe import load_universe

# warmup 探测:170 交易日约 243 个日历日,再留 75% 余量
_WARMUP_PROBE_CALENDAR_DAYS = 420
# 日线 bar 落后于今天的容忍度(超过即提示数据未更新)
_FRESH_DAYS = 5


def _warmup_probe_days(warmup_days: int) -> int:
    """warmup 窗口向前多取的日历日余量(按交易日折算,替代固定 420 天)。"""
    return max(_WARMUP_PROBE_CALENDAR_DAYS, int(warmup_days * 1.75) + 30)


def _stamp_session(rows: list[dict], session_id: str) -> list[dict]:
    for row in rows:
        row["session_id"] = session_id
    return rows


def cmd_init(args: argparse.Namespace) -> None:
    manifest = json.loads((ARTIFACT_PATH / "manifest.json").read_text(encoding="utf-8"))
    start_ns = sources.date_to_ns(args.start_date)
    created = dt.datetime.now().isoformat(timespec="seconds")

    conn = store.connect()
    for spec in SESSION_PRESETS:
        store.upsert_session(conn, {
            "session_id": spec.session_id,
            "kind": spec.kind,
            "note": spec.note,
            "artifact_path": str(ARTIFACT_PATH.resolve()),
            "model_id": manifest["model_id"],
            "model_version": manifest["model_version"],
            "model_sha256": manifest["model_sha256"],
            "policy_json": json.dumps(spec.policy, ensure_ascii=False),
            "initial_cash": DEFAULT_INITIAL_CASH,
            "start_ns": start_ns,
            "warmup_days": WARMUP_TRADING_DAYS,
            "created_at": created,
            "cost_profile": spec.cost_profile,
        })
    conn.commit()

    symbols = load_universe()  # 首次会扫描 scrapy 2019 分区并缓存
    print(f"已初始化会话:{', '.join(s.session_id for s in SESSION_PRESETS)}")
    for spec in SESSION_PRESETS:
        profile = costs.get_profile(spec.cost_profile)
        print(f"  [{spec.session_id}] {costs.describe(profile)}")
    print(f"制品:{manifest['model_id']} @ {manifest['model_version']} "
          f"(sha {manifest['model_sha256'][:12]}…)")
    print(f"起始交易日:{args.start_date}  宇宙:{len(symbols)} 标的  "
          f"warmup:{WARMUP_TRADING_DAYS} 交易日")
    conn.close()


def _sync_window(conn, session, through_date: str | None):
    """确定回放窗口(warmup + 会话段),返回 (frame, through_ns, progressed)。"""
    symbols = load_universe()
    start_ns = int(session["start_ns"])
    warmup_days = int(session["warmup_days"])

    if through_date is not None:
        requested_ns = sources.date_to_ns(through_date)
        if requested_ns < start_ns:
            raise SystemExit(f"--date {through_date} 早于会话起始日")
        probe_end = requested_ns
    else:
        probe_end = int(
            (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)).timestamp()
            * 1e9
        )

    # 扩窗:上游数据陈旧时固定余量会取不满 warmup,这里按需翻倍最长到 6 年
    probe_days = _warmup_probe_days(warmup_days)
    frame = sources.load_frame(
        symbols, int(start_ns - probe_days * sources.NS_PER_DAY), probe_end, conn
    )
    while True:
        available = sorted(int(ts) for ts in frame["timestamp"].unique())
        if avail_warmup := len([ts for ts in available if ts < start_ns]):
            if avail_warmup >= warmup_days or probe_days >= 2_200:
                break
        if probe_days >= 2_200:
            break
        probe_days = min(probe_days * 2, 2_200)
        print(f"[warn] warmup 取样不足,向前扩窗到 {probe_days} 个日历日重取行情")
        frame = sources.load_frame(
            symbols, int(start_ns - probe_days * sources.NS_PER_DAY), probe_end, conn
        )
    if frame.empty:
        raise SystemExit("窗口内没有任何行情数据")

    available = sorted(int(ts) for ts in frame["timestamp"].unique())
    if through_date is not None:
        if requested_ns not in available:
            nearest = sources.ns_to_date(
                max(ts for ts in available if ts <= requested_ns)
                if any(ts <= requested_ns for ts in available) else available[0]
            )
            raise SystemExit(f"{through_date} 不是可用交易日(非交易日或无数据);最近可用:{nearest}")
        through_ns = requested_ns
    else:
        through_ns = available[-1]

    warmup_ts = [ts for ts in available if ts < start_ns][-warmup_days:]
    if len(warmup_ts) < warmup_days:
        print(f"[warn] warmup 窗口只有 {len(warmup_ts)} 个交易日(目标 {warmup_days})")
    window = frame[
        (frame["timestamp"].isin(warmup_ts))
        | ((frame["timestamp"] >= start_ns) & (frame["timestamp"] <= through_ns))
    ].sort_values(["timestamp", "symbol"], kind="stable")
    if through_date is None:
        last = store.last_trade_ns(conn, session["session_id"])
        progressed = last is None or through_ns > last
    else:
        progressed = True
    return window, through_ns, progressed


def cmd_sync(args: argparse.Namespace) -> None:
    conn = store.connect()
    session = store.get_session(conn, args.session)
    if session is None:
        raise SystemExit(f"会话未初始化:{args.session}(先运行 python -m python.paper init)")

    through_date = args.date
    window, through_ns, progressed = _sync_window(conn, session, through_date)

    if not progressed:
        print(f"[{args.session}] 已是最新:{sources.ns_to_date(through_ns)} 无新增交易日;"
              "如需按上游最新行情重算请传 --date")
        conn.close()
        return
    lag_days = (dt.datetime.now(dt.timezone.utc).date()
                - dt.date.fromisoformat(sources.ns_to_date(through_ns))).days
    if lag_days > _FRESH_DAYS:
        print(f"[warn] 行情只到 {sources.ns_to_date(through_ns)}(落后 {lag_days} 天),"
              "请先回补后再同步", file=sys.stderr)

    module = load_engine_module()
    profile = costs.get_profile(session["cost_profile"])
    result = replay(
        module,
        artifact_path=Path(session["artifact_path"]),
        policy=json.loads(session["policy_json"]),
        initial_cash=float(session["initial_cash"]),
        frame=window,
        cost_profile=profile,
    )

    # 虚拟账户自回放起点(warmup 首日)开业:模型在 warmup 后段即自然开始交易,
    # 全部订单/成交/权益都属于账户轨迹,不做起点切片。
    orders = _stamp_session(result["orders"], args.session)
    fills = _stamp_session(result["fills"], args.session)
    equity = _stamp_session(result["equity"], args.session)
    round_trips = _stamp_session(result["round_trips"], args.session)
    positions = _stamp_session(result["positions"], args.session)

    order_count_by_day: dict[int, int] = {}
    for order in orders:
        order_count_by_day[order["ts"]] = order_count_by_day.get(order["ts"], 0) + 1
    fill_count_by_day: dict[int, int] = {}
    for fill in fills:
        fill_count_by_day[fill["ts"]] = fill_count_by_day.get(fill["ts"], 0) + 1
    symbol_count_by_day = {
        int(ts): int(len(group)) for ts, group in window.groupby("timestamp")
    }
    source_tags = sources.day_source_tags(window)

    day_rows = []
    for point in equity:
        ts = point["ts"]
        day_rows.append({
            "session_id": args.session,
            "trade_ns": ts,
            "source": source_tags.get(ts, "unknown"),
            "n_symbols": symbol_count_by_day.get(ts, 0),
            "n_orders": order_count_by_day.get(ts, 0),
            "n_fills": fill_count_by_day.get(ts, 0),
            "equity": point["equity"],
            "cash": point["cash"],
        })

    store.upsert_many(conn, "orders", orders)
    store.upsert_many(conn, "fills", fills)
    store.upsert_many(conn, "equity_curve", equity)
    store.upsert_many(conn, "round_trips", round_trips)
    store.upsert_many(conn, "positions", positions)
    store.record_days(conn, day_rows)
    store.log_run(
        conn, args.session, through_ns, True,
        f"through={sources.ns_to_date(through_ns)} rows={len(window)} "
        f"orders={len(orders)} fills={len(fills)} "
        f"cost={profile.name} commission={result['total_commission']:.2f}",
    )
    conn.commit()

    first_day = sources.ns_to_date(int(window["timestamp"].iloc[0]))
    print(f"[{args.session}] 已推进至 {sources.ns_to_date(through_ns)}:"
          f" 账户起点 {first_day}  交易日 {len(day_rows)}  订单 {len(orders)}"
          f"  成交 {len(fills)}"
          f"  权益 {result['final_equity']:.2f}"
          f"  现金 {result['final_cash']:.2f}"
          f"  持仓 {len(positions)}"
          f"  成本 {profile.name}(累计费用 {result['total_commission']:.2f})")
    conn.close()


def cmd_status(args: argparse.Namespace) -> None:
    conn = store.connect()
    for spec in SESSION_PRESETS:
        try:
            summary = store.session_summary(conn, spec.session_id)
        except KeyError:
            print(f"[{spec.session_id}] 未初始化")
            continue
        last = (sources.ns_to_date(summary["last_trade_ns"])
                if summary["last_trade_ns"] else "-")
        equity = summary["last_equity"]
        ret = (equity / summary["initial_cash"] - 1.0) if equity else None
        profile = costs.get_profile(summary["cost_profile"])
        print(
            f"[{summary['session_id']}] {summary['kind']}  策略={summary['policy']}\n"
            f"    成本 {profile.name}  显性往返 {profile.round_trip_bps():.1f}bp"
            f"  滑点 {profile.slippage_bps:.1f}bp\n"
            f"    最近交易日 {last}  天数 {summary['days']}  "
            f"权益 {equity if equity is None else round(equity, 2)}  "
            f"收益 {'' if ret is None else f'{ret:.4%}'}\n"
            f"    订单 {summary['orders']}  成交 {summary['fills']}  "
            f"往返 {summary['round_trips']}  当前持仓 {summary['open_positions']}"
        )
    conn.close()


def _run_scrapy(entry: Path, symbols: list[str], extra: list[str]) -> None:
    """调用 scrapy 下载器(用其自身虚拟环境),失败即中止当日流程。"""
    if not entry.is_file():
        raise SystemExit(f"未找到上游下载器入口:{entry}(设 QBT_SCRAPY_ROOT 覆盖)")
    interpreter = SCRAPY_PYTHON if SCRAPY_PYTHON.is_file() else Path(sys.executable)
    command = [
        str(interpreter), str(entry),
        "--symbols", ",".join(symbols), *extra,
    ]
    print(f"$ {' '.join(command[:3])} --symbols({len(symbols)} 只) {' '.join(extra)}")
    completed = subprocess.run(command, cwd=str(entry.parent))
    if completed.returncode != 0:
        raise SystemExit(f"{entry.name} 退出码 {completed.returncode},中止当日流程")


def cmd_health(args: argparse.Namespace) -> int:
    """数据新鲜度自检:回答"行情到哪天、差在哪、谁能用"。"""
    conn = store.connect()
    symbols = load_universe()
    boundary_ns, basis = sources.source_boundary_ns(symbols, conn)
    today_ns = sources.date_to_ns(dt.date.today().isoformat())
    print(f"上游边界:{sources.ns_to_date(boundary_ns) if boundary_ns else '未知'}"
          f"(基准={basis}, scrapy 数据湖 vs staging)")

    scrapy_ns = sources.scrapy_boundary(symbols)
    print(f"scrapy 数据湖:止于 "
          f"{sources.ns_to_date(scrapy_ns) if scrapy_ns else '空(探测不到任何行情)'}")
    staging = conn.execute(
        "SELECT source, count(*), min(ts), max(ts), count(DISTINCT symbol) "
        "FROM bars_staging GROUP BY source"
    ).fetchall()
    if not staging:
        print("staging:空(尚未回补)")
    for row in staging:
        lag = (today_ns - int(row[3])) // sources.NS_PER_DAY
        print(f"staging[{row[0]}]:{sources.ns_to_date(int(row[2]))} .. "
              f"{sources.ns_to_date(int(row[3]))}  {row[1]} 行 / {row[4]} 标的 "
              f"(落后今天 {lag} 个日历日)")

    # 引擎回放窗口是否够 warmup
    probe_days = _warmup_probe_days(WARMUP_TRADING_DAYS)
    for spec in SESSION_PRESETS:
        session = store.get_session(conn, spec.session_id)
        if session is None:
            print(f"[{spec.session_id}] 未初始化")
            continue
        start_ns = int(session["start_ns"])
        warmup_frame = sources.load_frame(
            symbols, int(start_ns - probe_days * sources.NS_PER_DAY), start_ns,
            conn,
        )
        n_warmup = len(warmup_frame["timestamp"].unique()) if not warmup_frame.empty else 0
        last = store.last_trade_ns(conn, spec.session_id)
        flag = "OK" if n_warmup >= WARMUP_TRADING_DAYS else "不足"
        print(f"[{spec.session_id}] warmup {n_warmup}/{WARMUP_TRADING_DAYS} 交易日({flag})  "
              f"账户已到 {sources.ns_to_date(last) if last else '-'}  "
              f"可推进到 {sources.ns_to_date(boundary_ns) if boundary_ns else '-'}")

    # scrapy 覆盖区间内是否有 security_state 缺口(停牌/ST 判定依赖它)
    # 缺口只在 scrapy 侧有意义,故以 scrapy 自身末尾为探针右端,而非合并边界
    sample = symbols[:10]
    state_anchor = scrapy_ns or 0
    probe_start = max(0, state_anchor - 40 * sources.NS_PER_DAY)
    bars = sources.load_scrapy_frame(sample, probe_start, state_anchor)
    print(f"security_state:抽样 {len(sample)} 标的 "
          f"{'无缺口' if bars['is_listed'].notna().all() else '有缺口(见上方 warn)'}"
          ";staging 段无 PIT 状态,按 is_listed=True/非停牌 兜底")
    conn.close()
    scrapy_ok = (scrapy_ns is not None
                 and (today_ns - scrapy_ns) <= _FRESH_DAYS * sources.NS_PER_DAY)
    engine_ok = (boundary_ns is not None
                 and (today_ns - boundary_ns) <= _FRESH_DAYS * sources.NS_PER_DAY)
    if scrapy_ok and engine_ok:
        print("结论:数据可用")
    elif engine_ok:
        print("结论:引擎可推进(基准=staging),但 scrapy 数据湖已陈旧;"
              "daily 会自动刷新,失败时见上方退出码")
    else:
        print("结论:行情陈旧或探测不到,请执行 daily(或显式传 --from 回补)")
    return 0 if (scrapy_ok and engine_ok) else 1


def cmd_daily(args: argparse.Namespace) -> int:
    """一条命令完成每日推进:refresher -> 回补 -> 同步双会话 -> 概要。"""
    symbols = load_universe()
    if not args.skip_refresh:
        _run_scrapy(SCRAPY_BARS_ENTRY, symbols, ["--provider", "baostock",
                                                 "--request-interval", "0.3"])
        _run_scrapy(SCRAPY_STATE_ENTRY, symbols, ["--request-interval", "0.3"])
    if args.only_scrapy:
        return 0

    try:
        import akshare as ak
    except ImportError:
        raise SystemExit("akshare 未安装:.venv/bin/pip install akshare")
    from .backfill import backfill_staging

    backfill_staging(ak, None, args.to_date, symbols=symbols)
    # 成本对照臂(kind=comparison)不参与日常推进,按需用 sync 单独触发
    targets = [spec.session_id for spec in SESSION_PRESETS
               if spec.kind != "comparison"]
    if args.session:
        targets = [args.session]
    for session_id in targets:
        cmd_sync(argparse.Namespace(session=session_id, date=None, force=None))
    cmd_status(argparse.Namespace())
    return 0


def cmd_backfill(args: argparse.Namespace) -> None:
    """从 akshare 拉日线写入 bars_staging(不写入 scrapy 数据湖)。"""
    try:
        import akshare as ak
    except ImportError:
        raise SystemExit("akshare 未安装:.venv/bin/pip install akshare")

    from .backfill import backfill_staging

    backfill_staging(ak, args.from_date, args.to_date)

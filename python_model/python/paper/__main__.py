"""虚拟盘 CLI:python -m python.paper <command>。"""
from __future__ import annotations

import argparse

from . import runner
from .config import DEFAULT_START_DATE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python.paper", description="日频虚拟盘(paper trading)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="初始化 paper.db 与两个会话")
    init.add_argument("--start-date", default=DEFAULT_START_DATE)
    init.set_defaults(func=runner.cmd_init)

    sync = sub.add_parser("sync", help="推进会话到指定交易日(默认最新可用)")
    sync.add_argument("--session", required=True,
                      help="registered | observed")
    sync.add_argument("--date", default=None,
                      help="目标交易日 YYYY-MM-DD;缺省为最新可用交易日")
    sync.set_defaults(func=runner.cmd_sync)

    status = sub.add_parser("status", help="打印各会话概要")
    status.set_defaults(func=runner.cmd_status)

    health = sub.add_parser("health", help="数据新鲜度与 warmup 自检")
    health.set_defaults(func=runner.cmd_health)

    daily = sub.add_parser("daily", help="每日推进:刷新上游->回补->同步双会话")
    daily.add_argument("--skip-refresh", action="store_true",
                       help="跳过 scrapy 刷新(仅回补+同步)")
    daily.add_argument("--only-scrapy", action="store_true",
                       help="只刷新 scrapy 上游,不回补不同步")
    daily.add_argument("--only", action="store_true",
                       help="只推进一个会话(配合 --only-session)")
    daily.add_argument("--only-session", dest="session", default=None,
                       help="配合 --only 指定会话")
    daily.add_argument("--to-date", dest="to_date", default=None,
                       help="回补截止日 YYYY-MM-DD,缺省为今天")
    daily.set_defaults(func=runner.cmd_daily)

    backfill = sub.add_parser("backfill", help="akshare 日线回补进 bars_staging")
    backfill.add_argument("--from-date", default=None)
    backfill.add_argument("--to-date", default=None)
    backfill.set_defaults(func=runner.cmd_backfill)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

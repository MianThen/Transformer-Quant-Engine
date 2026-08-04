"""交易记录与净值的落库/查询。"""

from __future__ import annotations

import sqlite3
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.backtest_runner import BacktestResult

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
SCHEMA_VERSION = 3


class TradeStore:
    """封装 SQLite 连接,负责建表、写入回测结果、查询。"""

    def __init__(self, db_path: str | Path = "backtest.db"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        # SQLite 默认 foreign_keys=OFF,不开则 schema 里的 FK 约束形同虚设。
        # WAL 提升并发读写(Dashboard 读 + 回测写)与写入吞吐。
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        """执行基础 schema，并把旧数据库迁移到当前版本。"""
        version = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            self.conn.close()
            raise RuntimeError(
                f"数据库 schema 版本 {version} 高于程序支持的 {SCHEMA_VERSION}"
            )
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._migrate_schema(version)
        self.conn.commit()

    def _migrate_schema(self, version: int) -> None:
        if version < 1:
            self._migrate_to_v1()
            self.conn.execute("PRAGMA user_version = 1")
            version = 1
        if version < 2:
            self._migrate_to_v2()
            self.conn.execute("PRAGMA user_version = 2")
            version = 2
        if version < 3:
            self._migrate_to_v3()
            self.conn.execute("PRAGMA user_version = 3")

    def _migrate_to_v1(self) -> None:
        columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(backtest_runs)")
        }
        if "backend" not in columns:
            self.conn.execute(
                "ALTER TABLE backtest_runs "
                "ADD COLUMN backend TEXT NOT NULL DEFAULT ''"
            )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS run_config ("
            "run_id INTEGER PRIMARY KEY, spec_version INTEGER NOT NULL, "
            "backend TEXT NOT NULL, config_json TEXT NOT NULL, "
            "config_hash TEXT NOT NULL, "
            "FOREIGN KEY (run_id) REFERENCES backtest_runs(id))"
        )

    def _migrate_to_v2(self) -> None:
        metric_columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(performance_metrics)")
        }
        for name, definition in (
            ("total_round_trips", "INTEGER NOT NULL DEFAULT 0"),
            ("gross_pnl", "REAL NOT NULL DEFAULT 0.0"),
            ("net_pnl", "REAL NOT NULL DEFAULT 0.0"),
        ):
            if name not in metric_columns:
                self.conn.execute(
                    f"ALTER TABLE performance_metrics ADD COLUMN {name} {definition}"
                )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_run_config_hash "
            "ON run_config(config_hash)"
        )

    def _migrate_to_v3(self) -> None:
        columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(backtest_runs)")
        }
        for name, definition in (
            ("status", "TEXT NOT NULL DEFAULT 'SUCCEEDED'"),
            ("error_message", "TEXT NOT NULL DEFAULT ''"),
            ("completed_at", "TEXT"),
        ):
            if name not in columns:
                self.conn.execute(
                    f"ALTER TABLE backtest_runs ADD COLUMN {name} {definition}"
                )

    def begin_run(
        self,
        strategy_name: str,
        symbols: list[str],
        initial_cash: float,
        backend: str,
        run_spec=None,
    ) -> int:
        """先保存运行身份；即使后续失败，实验定义仍可审计。"""
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO backtest_runs "
                "(strategy_name, symbols, start_date, end_date, initial_cash, backend, status) "
                "VALUES (?, ?, '', '', ?, ?, 'RUNNING')",
                (strategy_name, ",".join(symbols), initial_cash, backend),
            )
            run_id = int(cursor.lastrowid)
            if run_spec is not None:
                self.conn.execute(
                    "INSERT INTO run_config "
                    "(run_id, spec_version, backend, config_json, config_hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id, run_spec.version, run_spec.backend,
                     run_spec.to_json(), run_spec.fingerprint()),
                )
        return run_id

    def mark_run_failed(self, run_id: int, error: BaseException) -> None:
        message = f"{type(error).__name__}: {error}"
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE backtest_runs SET status = 'FAILED', error_message = ?, "
                "completed_at = datetime('now') WHERE id = ? AND status = 'RUNNING'",
                (message[:4000], run_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"无法把 run_id={run_id} 标记为 FAILED")

    def create_run(self, strategy_name: str, symbols: list[str],
                   start_date: str, end_date: str, initial_cash: float,
                   backend: str = "") -> int:
        """插入一条回测运行记录,返回 run_id。"""
        cur = self.conn.execute(
            "INSERT INTO backtest_runs "
            "(strategy_name, symbols, start_date, end_date, initial_cash, backend) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (strategy_name, ",".join(symbols), start_date, end_date,
             initial_cash, backend),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def save_trades(self, run_id: int, trades: list) -> None:
        """批量写入逐笔成交。trades 为 TradeRecord 列表(C++ 或 Python 后端)。"""
        # side 用整数值判定,避免耦合 engine_api 的 Side 枚举:
        # 两后端的 Side 均为 BUY=0 / SELL=1,int(side) 通用。
        rows = [
            (
                run_id,
                t.symbol,
                "SELL" if int(t.side) == 1 else "BUY",
                t.quantity,
                t.price,
                t.timestamp,
                t.commission,
            )
            for t in trades
        ]
        self.conn.executemany(
            "INSERT INTO trades "
            "(run_id, symbol, side, quantity, price, timestamp, commission) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def save_equity_curve(self, run_id: int, equity_curve: list) -> None:
        """批量写入每日净值。equity_curve 为 C++ EquityPoint 列表。"""
        rows = self._daily_equity_rows(equity_curve)
        self.conn.execute("DELETE FROM daily_equity WHERE run_id = ?", (run_id,))
        self.conn.executemany(
            "INSERT INTO daily_equity "
            "(run_id, date, equity, cash, position_value) VALUES (?, ?, ?, ?, ?)",
            [(run_id, date, equity, cash, equity - cash)
             for date, equity, cash in rows],
        )
        self.conn.commit()

    def persist_result(
        self,
        strategy_name: str,
        symbols: list[str],
        result: "BacktestResult",
        *,
        run_id: int | None = None,
    ) -> int:
        """事务化保存一次完整回测，返回 run_id。"""
        equity_rows = self._daily_equity_rows(result.equity_curve)
        if equity_rows:
            start_date, end_date = equity_rows[0][0], equity_rows[-1][0]
        else:
            start_date = end_date = ""

        with self.conn:
            if run_id is None:
                cursor = self.conn.execute(
                    "INSERT INTO backtest_runs "
                    "(strategy_name, symbols, start_date, end_date, initial_cash, "
                    "backend, status, completed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'SUCCEEDED', datetime('now'))",
                    (strategy_name, ",".join(symbols), start_date, end_date,
                     getattr(result, "initial_cash", 0.0),
                     getattr(result, "backend", "")),
                )
                run_id = int(cursor.lastrowid)
            else:
                cursor = self.conn.execute(
                    "UPDATE backtest_runs SET strategy_name = ?, symbols = ?, "
                    "start_date = ?, end_date = ?, initial_cash = ?, backend = ?, "
                    "status = 'SUCCEEDED', error_message = '', "
                    "completed_at = datetime('now') "
                    "WHERE id = ? AND status = 'RUNNING'",
                    (strategy_name, ",".join(symbols), start_date, end_date,
                     getattr(result, "initial_cash", 0.0),
                     getattr(result, "backend", ""), run_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"无法完成 run_id={run_id}")
            self.conn.executemany(
                "INSERT INTO trades "
                "(run_id, symbol, side, quantity, price, timestamp, commission) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (run_id, trade.symbol,
                     "SELL" if int(trade.side) == 1 else "BUY",
                     trade.quantity, trade.price, trade.timestamp,
                     trade.commission)
                    for trade in result.trades
                ],
            )
            self.conn.executemany(
                "INSERT INTO daily_equity "
                "(run_id, date, equity, cash, position_value) VALUES (?, ?, ?, ?, ?)",
                [(run_id, date, equity, cash, equity - cash)
                 for date, equity, cash in equity_rows],
            )
            self.conn.executemany(
                "INSERT INTO round_trips "
                "(run_id, symbol, entry_side, quantity, entry_price, exit_price, "
                "opened_at, closed_at, gross_pnl, commission, net_pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id, item.symbol,
                        "SELL" if int(item.entry_side) == 1 else "BUY",
                        item.quantity, item.entry_price, item.exit_price,
                        item.opened_at, item.closed_at, item.gross_pnl,
                        item.commission, item.net_pnl,
                    )
                    for item in getattr(result, "round_trips", [])
                ],
            )
            round_trips = getattr(result, "round_trips", [])
            self.conn.execute(
                "INSERT INTO performance_metrics "
                "(run_id, total_return, annual_return, sharpe_ratio, max_drawdown, "
                "win_rate, total_trades, total_round_trips, gross_pnl, net_pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, result.total_return, result.annual_return,
                 result.sharpe_ratio, result.max_drawdown, result.win_rate,
                 len(result.trades), len(round_trips),
                 sum(item.gross_pnl for item in round_trips),
                 sum(item.net_pnl for item in round_trips)),
            )
            self.conn.executemany(
                "INSERT INTO orders "
                "(run_id, order_id, symbol, side, order_type, requested_quantity, "
                "filled_quantity, limit_price, avg_fill_price, status, reject_reason, "
                "created_timestamp, updated_timestamp, message) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id, record.order.id, record.order.symbol,
                        "SELL" if int(record.order.side) == 1 else "BUY",
                        "LIMIT" if int(record.order.type) == 1 else "MARKET",
                        record.order.quantity, record.filled_quantity,
                        record.order.limit_price, record.avg_fill_price,
                        _enum_name(record.status), _enum_name(record.reject_reason),
                        record.order.timestamp, record.updated_timestamp, record.message,
                    )
                    for record in getattr(result, "orders", [])
                ],
            )
            self.conn.executemany(
                "INSERT INTO corporate_actions "
                "(run_id, symbol, timestamp, cash_dividend, old_quantity, new_quantity) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (run_id, action.symbol, action.timestamp, action.cash_dividend,
                     action.old_quantity, action.new_quantity)
                    for action in getattr(result, "corporate_actions", [])
                ],
            )
            portfolio = getattr(result, "portfolio", None)
            if portfolio is not None:
                self.conn.execute(
                    "INSERT INTO portfolio_risk "
                    "(run_id, cash, equity, gross_exposure, net_exposure, "
                    "largest_position_weight, position_count, industry_exposure_json, "
                    "factor_exposure_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, portfolio.cash, portfolio.equity,
                     portfolio.gross_exposure, portfolio.net_exposure,
                     portfolio.largest_position_weight, portfolio.position_count,
                     json.dumps(dict(portfolio.industry_exposure), sort_keys=True),
                     json.dumps(dict(portfolio.factor_exposure), sort_keys=True)),
                )
            lineage = getattr(result, "data_lineage", None)
            if lineage is not None:
                self.conn.execute(
                    "INSERT INTO data_lineage "
                    "(run_id, catalog_generation, schema_hash, dataset_fingerprint, "
                    "source_fingerprints_json, query_fingerprint) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, lineage.catalog_generation, lineage.schema_hash,
                     lineage.dataset_fingerprint,
                     json.dumps(list(lineage.source_fingerprints), sort_keys=True),
                     lineage.query_fingerprint),
                )
            run_spec = getattr(result, "run_spec", None)
            if run_spec is not None:
                self.conn.execute(
                    "INSERT OR REPLACE INTO run_config "
                    "(run_id, spec_version, backend, config_json, config_hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id, run_spec.version, run_spec.backend,
                     run_spec.to_json(), run_spec.fingerprint()),
                )
        return run_id

    @staticmethod
    def _daily_equity_rows(equity_curve: list) -> list[tuple[str, float, float]]:
        """按上海时区聚合，每日保留最后一个权益点。"""
        shanghai = timezone(timedelta(hours=8))
        daily: dict[str, tuple[float, float]] = {}
        for point in equity_curve:
            timestamp = int(point.timestamp)
            date = datetime.fromtimestamp(timestamp // 1_000_000_000,
                                          tz=timezone.utc).astimezone(shanghai)
            daily[date.date().isoformat()] = (float(point.equity), float(point.cash))
        return [(date, equity, cash) for date, (equity, cash) in sorted(daily.items())]

    def get_trades(
        self, run_id: int, *, limit: int = 100, offset: int = 0,
    ) -> list[sqlite3.Row]:
        limit, offset = _validate_pagination(limit, offset)
        return self.conn.execute(
            "SELECT * FROM trades WHERE run_id = ? "
            "ORDER BY timestamp, id LIMIT ? OFFSET ?", (run_id, limit, offset)
        ).fetchall()

    def count_trades(self, run_id: int) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM trades WHERE run_id = ?", (run_id,)
        ).fetchone()[0])

    def get_trade_summary(self, run_id: int) -> sqlite3.Row:
        return self.conn.execute(
            "SELECT COUNT(*) AS total_trades, COUNT(DISTINCT symbol) AS symbols, "
            "COALESCE(SUM(quantity * price), 0.0) AS notional, "
            "COALESCE(SUM(commission), 0.0) AS commission "
            "FROM trades WHERE run_id = ?", (run_id,)
        ).fetchone()

    def get_trade_daily_summary(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT date(timestamp / 1000000000, 'unixepoch', '+8 hours') AS date, "
            "COUNT(*) AS trades FROM trades WHERE run_id = ? "
            "GROUP BY date ORDER BY date", (run_id,)
        ).fetchall()

    def get_trade_side_summary(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT side, SUM(quantity * price) AS notional "
            "FROM trades WHERE run_id = ? GROUP BY side ORDER BY side", (run_id,)
        ).fetchall()

    def get_round_trips(
        self, run_id: int, *, limit: int = 100, offset: int = 0,
    ) -> list[sqlite3.Row]:
        limit, offset = _validate_pagination(limit, offset)
        return self.conn.execute(
            "SELECT * FROM round_trips WHERE run_id = ? "
            "ORDER BY closed_at, id LIMIT ? OFFSET ?", (run_id, limit, offset)
        ).fetchall()

    def count_round_trips(self, run_id: int) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM round_trips WHERE run_id = ?", (run_id,)
        ).fetchone()[0])

    def get_equity_curve(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM daily_equity WHERE run_id = ? ORDER BY date", (run_id,)
        ).fetchall()

    def get_orders(
        self, run_id: int, *, limit: int = 100, offset: int = 0,
    ) -> list[sqlite3.Row]:
        limit, offset = _validate_pagination(limit, offset)
        return self.conn.execute(
            "SELECT * FROM orders WHERE run_id = ? "
            "ORDER BY order_id LIMIT ? OFFSET ?", (run_id, limit, offset)
        ).fetchall()

    def count_orders(self, run_id: int) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM orders WHERE run_id = ?", (run_id,)
        ).fetchone()[0])

    def get_order_status_summary(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT status, reject_reason, COUNT(*) AS orders, "
            "SUM(requested_quantity) AS requested_quantity, "
            "SUM(filled_quantity) AS filled_quantity "
            "FROM orders WHERE run_id = ? GROUP BY status, reject_reason "
            "ORDER BY orders DESC, status, reject_reason", (run_id,)
        ).fetchall()

    def get_corporate_actions(
        self, run_id: int, *, limit: int = 100, offset: int = 0,
    ) -> list[sqlite3.Row]:
        limit, offset = _validate_pagination(limit, offset)
        return self.conn.execute(
            "SELECT * FROM corporate_actions WHERE run_id = ? "
            "ORDER BY timestamp, symbol LIMIT ? OFFSET ?",
            (run_id, limit, offset),
        ).fetchall()

    def count_corporate_actions(self, run_id: int) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM corporate_actions WHERE run_id = ?", (run_id,)
        ).fetchone()[0])

    def get_portfolio_risk(self, run_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM portfolio_risk WHERE run_id = ?", (run_id,)
        ).fetchone()

    def get_data_lineage(self, run_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM data_lineage WHERE run_id = ?", (run_id,)
        ).fetchone()

    def get_run_config(self, run_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM run_config WHERE run_id = ?", (run_id,)
        ).fetchone()

    def list_runs(
        self, *, limit: int = 100, offset: int = 0,
    ) -> list[sqlite3.Row]:
        limit, offset = _validate_pagination(limit, offset)
        return self.conn.execute(
            "SELECT * FROM backtest_runs ORDER BY created_at DESC, id DESC "
            "LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()

    def count_runs(self) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM backtest_runs"
        ).fetchone()[0])

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM backtest_runs WHERE id = ?", (run_id,)
        ).fetchone()

    def close(self) -> None:
        self.conn.close()


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    return str(value).rsplit(".", 1)[-1]


def _validate_pagination(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or isinstance(offset, bool):
        raise ValueError("limit/offset 必须是整数")
    limit = int(limit)
    offset = int(offset)
    if not 1 <= limit <= 1_000 or offset < 0:
        raise ValueError("limit 必须在 [1, 1000]，offset 不能为负数")
    return limit, offset

"""paper.db 持久层:SQLite,全部幂等 upsert,独立于回测 backtest.db。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    note         TEXT,
    artifact_path TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    policy_json  TEXT NOT NULL,
    initial_cash REAL NOT NULL,
    start_ns     INTEGER NOT NULL,
    warmup_days  INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    cost_profile TEXT
);
CREATE TABLE IF NOT EXISTS days (
    session_id TEXT NOT NULL,
    trade_ns   INTEGER NOT NULL,
    source     TEXT NOT NULL,
    n_symbols  INTEGER NOT NULL,
    n_orders   INTEGER NOT NULL,
    n_fills    INTEGER NOT NULL,
    equity     REAL NOT NULL,
    cash       REAL NOT NULL,
    PRIMARY KEY (session_id, trade_ns)
);
CREATE TABLE IF NOT EXISTS orders (
    session_id TEXT NOT NULL,
    order_id   INTEGER NOT NULL,
    decision_id TEXT,
    ts          INTEGER NOT NULL,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    type        TEXT NOT NULL,
    quantity    INTEGER NOT NULL,
    status      TEXT NOT NULL,
    reject_reason TEXT NOT NULL,
    filled_quantity INTEGER NOT NULL,
    avg_fill_price  REAL NOT NULL,
    PRIMARY KEY (session_id, order_id)
);
CREATE TABLE IF NOT EXISTS fills (
    session_id TEXT NOT NULL,
    fill_index INTEGER NOT NULL,
    order_id   INTEGER NOT NULL,
    ts         INTEGER NOT NULL,
    symbol     TEXT NOT NULL,
    side       TEXT NOT NULL,
    quantity   INTEGER NOT NULL,
    price      REAL NOT NULL,
    commission REAL NOT NULL,
    PRIMARY KEY (session_id, fill_index)
);
CREATE TABLE IF NOT EXISTS equity_curve (
    session_id TEXT NOT NULL,
    ts         INTEGER NOT NULL,
    equity     REAL NOT NULL,
    cash       REAL NOT NULL,
    PRIMARY KEY (session_id, ts)
);
CREATE TABLE IF NOT EXISTS positions (
    session_id TEXT NOT NULL,
    asof_ns    INTEGER NOT NULL,
    symbol     TEXT NOT NULL,
    quantity   INTEGER NOT NULL,
    sellable   INTEGER NOT NULL,
    avg_cost   REAL NOT NULL,
    PRIMARY KEY (session_id, asof_ns, symbol)
);
CREATE TABLE IF NOT EXISTS round_trips (
    session_id TEXT NOT NULL,
    rt_index   INTEGER NOT NULL,
    symbol     TEXT NOT NULL,
    entry_side TEXT NOT NULL,
    quantity   INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_price  REAL NOT NULL,
    opened_ns  INTEGER NOT NULL,
    closed_ns  INTEGER NOT NULL,
    gross_pnl  REAL NOT NULL,
    commission REAL NOT NULL,
    net_pnl    REAL NOT NULL,
    PRIMARY KEY (session_id, rt_index)
);
CREATE TABLE IF NOT EXISTS bars_staging (
    source   TEXT NOT NULL,
    ts       INTEGER NOT NULL,
    symbol   TEXT NOT NULL,
    open     REAL NOT NULL,
    high     REAL NOT NULL,
    low      REAL NOT NULL,
    close    REAL NOT NULL,
    volume   INTEGER NOT NULL,
    PRIMARY KEY (source, ts, symbol)
);
CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    through_ns INTEGER,
    ok INTEGER NOT NULL,
    detail TEXT,
    at TEXT NOT NULL
);
-- 跨源口径标定:同一 source 的成交量换算系数必须全局唯一,否则相邻两次
-- 回补会写出不同单位的同一列(见 backfill.py 的混口径风险)
CREATE TABLE IF NOT EXISTS staging_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_session_ts ON orders(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_fills_session_ts ON fills(session_id, ts);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = DB_PATH if db_path is None else db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """就地补列(旧库没有 cost_profile;历史会话保持 NULL = 零成本口径)。"""
    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(sessions)")
    }
    if "cost_profile" not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN cost_profile TEXT")
        conn.commit()


def upsert_session(conn: sqlite3.Connection, record: dict) -> None:
    conn.execute(
        """
        INSERT INTO sessions VALUES (
            :session_id, :kind, :note, :artifact_path, :model_id,
            :model_version, :model_sha256, :policy_json, :initial_cash,
            :start_ns, :warmup_days, :created_at, :cost_profile
        )
        ON CONFLICT(session_id) DO UPDATE SET
            note=excluded.note, policy_json=excluded.policy_json,
            start_ns=excluded.start_ns, warmup_days=excluded.warmup_days,
            cost_profile=excluded.cost_profile
        """,
        record,
    )


def get_session(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()


def upsert_many(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ",".join(f":{name}" for name in columns)
    statement = (
        f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(statement, rows)


def get_meta(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute(
        "SELECT value FROM staging_meta WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return default
    return json.loads(row[0])


def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute(
        "INSERT INTO staging_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, json.dumps(value, ensure_ascii=False)),
    )


def record_days(conn: sqlite3.Connection, rows: list[dict]) -> None:
    upsert_many(conn, "days", rows)


def log_run(
    conn: sqlite3.Connection, session_id: str, through_ns: int,
    ok: bool, detail: str,
) -> None:
    import datetime as dt

    conn.execute(
        "INSERT INTO run_log (session_id, through_ns, ok, detail, at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            session_id, through_ns, int(ok), detail,
            dt.datetime.now().isoformat(timespec="seconds"),
        ),
    )


def last_trade_ns(conn: sqlite3.Connection, session_id: str) -> int | None:
    """会话已推进到的最后一个交易日(无记录返回 None)。"""
    row = conn.execute(
        "SELECT max(trade_ns) FROM days WHERE session_id = ?", (session_id,)
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def session_summary(conn: sqlite3.Connection, session_id: str) -> dict:
    """dashboard/status 共用的会话汇总。"""
    session = get_session(conn, session_id)
    if session is None:
        raise KeyError(f"会话不存在:{session_id}")
    last_day = conn.execute(
        "SELECT * FROM days WHERE session_id = ? ORDER BY trade_ns DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    counts = conn.execute(
        """
        SELECT
            (SELECT count(*) FROM days WHERE session_id = :s) AS days,
            (SELECT count(*) FROM orders WHERE session_id = :s) AS orders,
            (SELECT count(*) FROM fills WHERE session_id = :s) AS fills,
            (SELECT count(*) FROM round_trips WHERE session_id = :s) AS round_trips,
            (SELECT count(*) FROM positions WHERE session_id = :s
                AND asof_ns = (SELECT max(asof_ns) FROM positions
                               WHERE session_id = :s)
                AND quantity > 0) AS open_positions
        """,
        {"s": session_id},
    ).fetchone()
    return {
        "session_id": session_id,
        "kind": session["kind"],
        "note": session["note"],
        "policy": json.loads(session["policy_json"]),
        "cost_profile": session["cost_profile"],
        "initial_cash": session["initial_cash"],
        "start_ns": session["start_ns"],
        "days": counts["days"],
        "orders": counts["orders"],
        "fills": counts["fills"],
        "round_trips": counts["round_trips"],
        "open_positions": counts["open_positions"],
        "last_trade_ns": last_day["trade_ns"] if last_day else None,
        "last_equity": last_day["equity"] if last_day else None,
        "last_cash": last_day["cash"] if last_day else None,
    }

-- quant-backtester SQLite schema
-- 星型设计:以 backtest_runs 为中心,关联 trades / daily_equity / performance_metrics

-- 回测运行元数据
CREATE TABLE IF NOT EXISTS backtest_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,
    symbols       TEXT NOT NULL,            -- 逗号分隔的标的列表
    start_date    TEXT NOT NULL,
    end_date      TEXT NOT NULL,
    initial_cash  REAL NOT NULL,
    backend       TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'SUCCEEDED'
                  CHECK(status IN ('RUNNING', 'SUCCEEDED', 'FAILED')),
    error_message TEXT NOT NULL DEFAULT '',
    created_at    TEXT DEFAULT (datetime('now')),
    completed_at  TEXT
);

-- 逐笔成交
CREATE TABLE IF NOT EXISTS trades (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL,
    symbol     TEXT NOT NULL,
    side       TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
    quantity   INTEGER NOT NULL,
    price      REAL NOT NULL,
    timestamp  INTEGER NOT NULL,
    commission REAL DEFAULT 0.0,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
);

-- 每日净值
CREATE TABLE IF NOT EXISTS daily_equity (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL,
    date           TEXT NOT NULL,
    equity         REAL NOT NULL,
    cash           REAL NOT NULL,
    position_value REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
);

-- 绩效指标(每个 run 一行)
CREATE TABLE IF NOT EXISTS performance_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER UNIQUE NOT NULL,
    total_return  REAL,
    annual_return REAL,
    sharpe_ratio  REAL,
    max_drawdown  REAL,
    win_rate      REAL,
    total_trades  INTEGER,
    total_round_trips INTEGER NOT NULL DEFAULT 0,
    gross_pnl     REAL NOT NULL DEFAULT 0.0,
    net_pnl       REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
);

-- 匹配开仓与平仓后的净收益记录；胜率只使用该表，不按单边 fill 计算。
CREATE TABLE IF NOT EXISTS round_trips (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    symbol      TEXT NOT NULL,
    entry_side  TEXT NOT NULL CHECK(entry_side IN ('BUY', 'SELL')),
    quantity    INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_price  REAL NOT NULL,
    opened_at   INTEGER NOT NULL,
    closed_at   INTEGER NOT NULL,
    gross_pnl   REAL NOT NULL,
    commission  REAL NOT NULL,
    net_pnl     REAL NOT NULL,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
);

-- 委托最终状态：成交、拒绝、撤销和过期都保留，便于审计回测假设。
CREATE TABLE IF NOT EXISTS orders (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            INTEGER NOT NULL,
    order_id          INTEGER NOT NULL,
    symbol            TEXT NOT NULL,
    side              TEXT NOT NULL CHECK(side IN ('BUY', 'SELL')),
    order_type        TEXT NOT NULL CHECK(order_type IN ('MARKET', 'LIMIT')),
    requested_quantity INTEGER NOT NULL,
    filled_quantity   INTEGER NOT NULL,
    limit_price       REAL NOT NULL,
    avg_fill_price    REAL NOT NULL,
    status            TEXT NOT NULL,
    reject_reason     TEXT NOT NULL,
    created_timestamp INTEGER NOT NULL,
    updated_timestamp INTEGER NOT NULL,
    message           TEXT NOT NULL DEFAULT '',
    UNIQUE(run_id, order_id),
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    symbol          TEXT NOT NULL,
    timestamp       INTEGER NOT NULL,
    cash_dividend   REAL NOT NULL,
    old_quantity    INTEGER NOT NULL,
    new_quantity    INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
);

CREATE TABLE IF NOT EXISTS portfolio_risk (
    run_id                    INTEGER PRIMARY KEY,
    cash                      REAL NOT NULL,
    equity                    REAL NOT NULL,
    gross_exposure            REAL NOT NULL,
    net_exposure              REAL NOT NULL,
    largest_position_weight   REAL NOT NULL,
    position_count            INTEGER NOT NULL,
    industry_exposure_json    TEXT NOT NULL,
    factor_exposure_json      TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
);

CREATE TABLE IF NOT EXISTS data_lineage (
    run_id                 INTEGER PRIMARY KEY,
    catalog_generation     INTEGER NOT NULL,
    schema_hash            TEXT NOT NULL,
    dataset_fingerprint    TEXT NOT NULL,
    source_fingerprints_json TEXT NOT NULL,
    query_fingerprint      TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
);

-- 完整运行配置。config_hash 用于判断两次运行是否使用同一实验定义。
CREATE TABLE IF NOT EXISTS run_config (
    run_id       INTEGER PRIMARY KEY,
    spec_version INTEGER NOT NULL,
    backend      TEXT NOT NULL,
    config_json  TEXT NOT NULL,
    config_hash  TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_trades_run_time ON trades(run_id, timestamp, id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_round_trips_run_close ON round_trips(run_id, closed_at, id);
CREATE INDEX IF NOT EXISTS idx_equity_run ON daily_equity(run_id);
CREATE INDEX IF NOT EXISTS idx_orders_run_order ON orders(run_id, order_id);
CREATE INDEX IF NOT EXISTS idx_actions_run_time ON corporate_actions(run_id, timestamp, symbol);
CREATE INDEX IF NOT EXISTS idx_run_config_hash ON run_config(config_hash);

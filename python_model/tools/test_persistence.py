from __future__ import annotations

import json
import sqlite3

import pytest

from python.backtest_runner import BacktestRunner
from python.broker import Broker
from python.data_feed import DataFeed
from python.engine_api import ExecutionConfig, MarketSnapshot, Order, OrderType, Side
from python.market_data import DataLineage
from python.strategy import Strategy
from storage.metrics_store import MetricsStore
from storage.trade_store import TradeStore


DAY = 86_400_000_000_000


class ListFeed(DataFeed):
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def stream(self):
        yield from self.snapshots

    def lineage(self):
        return DataLineage(3, "schema", "dataset", ("source",), "query")


class BuyOnceStrategy(Strategy):
    def __init__(self):
        super().__init__(["000001"])
        self.ordered = False

    def on_market_data(self, snapshot):
        if self.ordered:
            return []
        self.ordered = True
        order = Order()
        order.symbol = snapshot.symbol
        order.side = Side.BUY
        order.type = OrderType.MARKET
        order.quantity = 10
        return [order]


class BuyThenSellStrategy(Strategy):
    def __init__(self):
        super().__init__(["000001"])
        self.calls = 0

    def on_market_data(self, snapshot):
        self.calls += 1
        if self.calls > 2:
            return []
        order = Order()
        order.symbol = snapshot.symbol
        order.side = Side.BUY if self.calls == 1 else Side.SELL
        order.type = OrderType.MARKET
        order.quantity = 10
        return [order]


def _bar(timestamp, open_price, close_price):
    snapshot = MarketSnapshot()
    snapshot.symbol = "000001"
    snapshot.timestamp = timestamp
    snapshot.open = open_price
    snapshot.high = max(open_price, close_price)
    snapshot.low = min(open_price, close_price)
    snapshot.close = close_price
    snapshot.volume = 1000
    return snapshot


def test_backtest_persists_complete_result_transactionally(tmp_path):
    store = TradeStore(tmp_path / "backtest.db")
    try:
        runner = BacktestRunner(
            BuyOnceStrategy(),
            ListFeed([_bar(DAY, 10.0, 10.0), _bar(2 * DAY, 11.0, 12.0)]),
            initial_cash=10_000.0,
            store=store,
        )
        result = runner.run()

        assert result.run_id is not None
        run = store.get_run(result.run_id)
        trades = store.get_trades(result.run_id)
        equity = store.get_equity_curve(result.run_id)
        metrics = MetricsStore(store.conn).get_metrics(result.run_id)
        orders = store.get_orders(result.run_id)
        portfolio = store.get_portfolio_risk(result.run_id)
        lineage = store.get_data_lineage(result.run_id)
        run_config = store.get_run_config(result.run_id)

        assert run["strategy_name"] == "BuyOnceStrategy"
        assert run["symbols"] == "000001"
        assert run["initial_cash"] == 10_000.0
        assert run["status"] == "SUCCEEDED"
        assert run["error_message"] == ""
        assert run["completed_at"] is not None
        assert store.count_runs() == 1
        assert store.list_runs(limit=1, offset=0)[0]["id"] == result.run_id
        assert len(trades) == 1
        assert trades[0]["side"] == "BUY"
        assert len(equity) == 2
        assert equity[-1]["equity"] == result.final_equity
        assert metrics["total_return"] == result.total_return
        assert metrics["total_trades"] == 1
        assert len(orders) == 1
        assert orders[0]["status"] == "FILLED"
        assert orders[0]["filled_quantity"] == 10
        assert portfolio["position_count"] == 1
        assert lineage["catalog_generation"] == 3
        assert lineage["dataset_fingerprint"] == "dataset"
        assert lineage["query_fingerprint"] == "query"
        assert run_config is not None
        assert run_config["backend"] == result.backend
        assert json.loads(run_config["config_json"])["data_lineage"][
            "catalog_generation"
        ] == 3
    finally:
        store.close()


def test_run_spec_and_legacy_schema_migration_are_persisted(tmp_path):
    database = tmp_path / "legacy.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE backtest_runs ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_name TEXT NOT NULL, "
        "symbols TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT NOT NULL, "
        "initial_cash REAL NOT NULL, created_at TEXT)"
    )
    connection.commit()
    connection.close()

    store = TradeStore(database)
    try:
        config = ExecutionConfig()
        config.slippage_bps = 12.5
        runner = BacktestRunner(
            BuyOnceStrategy(),
            ListFeed([_bar(DAY, 10.0, 10.0), _bar(2 * DAY, 11.0, 12.0)]),
            initial_cash=10_000.0,
            broker=Broker(min_commission=3.0, slippage=0.00125),
            execution_config=config,
            strategy_parameters={"lookback": 20},
            random_seed=7,
            store=store,
        )
        result = runner.run()
        run = store.get_run(result.run_id)
        saved = store.get_run_config(result.run_id)
        config_json = json.loads(saved["config_json"])

        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == 3
        assert run["backend"] == result.backend
        assert saved["config_hash"] == result.run_spec.fingerprint()
        assert config_json["strategy"]["parameters"] == {"lookback": 20}
        assert config_json["random_seed"] == 7
        assert config_json["execution_config"]["slippage_bps"] == 12.5
        assert config_json["broker"]["slippage"] == 0.00125
        assert config_json["broker"]["fee_schedules"][0]["min_commission"] == 3.0
        assert config_json["environment"]["backend_artifact_sha256"]
    finally:
        store.close()


def test_round_trips_are_persisted_and_large_tables_are_paginated(tmp_path):
    store = TradeStore(tmp_path / "round-trips.db")
    try:
        config = ExecutionConfig()
        config.enforce_t_plus_one = False
        runner = BacktestRunner(
            BuyThenSellStrategy(),
            ListFeed([
                _bar(DAY, 10.0, 10.0),
                _bar(2 * DAY, 10.0, 10.0),
                _bar(3 * DAY, 11.0, 11.0),
            ]),
            initial_cash=10_000.0,
            execution_config=config,
            broker=Broker(min_commission=1.0, commission_rate=0.0,
                          stamp_tax_rate=0.0),
            store=store,
        )
        result = runner.run()
        metrics = MetricsStore(store.conn).get_metrics(result.run_id)
        round_trips = store.get_round_trips(result.run_id)

        assert len(round_trips) == 1
        assert round_trips[0]["gross_pnl"] == 10.0
        assert round_trips[0]["commission"] == 2.0
        assert round_trips[0]["net_pnl"] == 8.0
        assert metrics["total_round_trips"] == 1
        assert metrics["win_rate"] == 1.0
        assert store.count_trades(result.run_id) == 2
        assert len(store.get_trades(result.run_id, limit=1, offset=0)) == 1
        assert len(store.get_trades(result.run_id, limit=1, offset=1)) == 1
        assert store.get_trade_summary(result.run_id)["commission"] == 2.0
        assert store.get_order_status_summary(result.run_id)[0]["orders"] == 2
        with pytest.raises(ValueError, match="limit"):
            store.get_trades(result.run_id, limit=0)
    finally:
        store.close()


def test_v1_schema_migrates_to_v3_and_future_versions_are_rejected(tmp_path):
    database = tmp_path / "v1.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE backtest_runs ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_name TEXT NOT NULL, "
        "symbols TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT NOT NULL, "
        "initial_cash REAL NOT NULL, backend TEXT NOT NULL DEFAULT '');"
        "CREATE TABLE performance_metrics ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER UNIQUE NOT NULL, "
        "total_return REAL, annual_return REAL, sharpe_ratio REAL, "
        "max_drawdown REAL, win_rate REAL, total_trades INTEGER);"
        "PRAGMA user_version = 1;"
    )
    connection.close()

    store = TradeStore(database)
    try:
        columns = {
            row["name"]
            for row in store.conn.execute("PRAGMA table_info(performance_metrics)")
        }
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == 3
        assert {"total_round_trips", "gross_pnl", "net_pnl"} <= columns
        run_columns = {
            row["name"]
            for row in store.conn.execute("PRAGMA table_info(backtest_runs)")
        }
        assert {"status", "error_message", "completed_at"} <= run_columns
        assert store.conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'round_trips'"
        ).fetchone() is not None
    finally:
        store.close()

    future = tmp_path / "future.db"
    connection = sqlite3.connect(future)
    connection.execute("PRAGMA user_version = 999")
    connection.close()
    with pytest.raises(RuntimeError, match="高于程序支持"):
        TradeStore(future)


def test_failed_run_keeps_frozen_config_and_error(tmp_path):
    class FailingStrategy(Strategy):
        def __init__(self):
            super().__init__(["000001"])

        def on_market_data(self, _snapshot):
            raise RuntimeError("strategy exploded")

    store = TradeStore(tmp_path / "failed.db")
    try:
        runner = BacktestRunner(
            FailingStrategy(),
            ListFeed([_bar(DAY, 10.0, 10.0)]),
            random_seed=17,
            strategy_parameters={"window": 5},
            store=store,
        )
        with pytest.raises(RuntimeError, match="strategy exploded"):
            runner.run()

        assert store.count_runs() == 1
        run = store.list_runs(limit=1)[0]
        assert run["status"] == "FAILED"
        assert run["completed_at"] is not None
        assert run["error_message"] == "RuntimeError: strategy exploded"
        saved = store.get_run_config(run["id"])
        config = json.loads(saved["config_json"])
        assert config["random_seed"] == 17
        assert config["strategy"]["parameters"] == {"window": 5}
        assert MetricsStore(store.conn).get_metrics(run["id"]) is None
    finally:
        store.close()

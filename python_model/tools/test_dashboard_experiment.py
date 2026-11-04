from __future__ import annotations

import csv
import threading
from datetime import date, time
from io import BytesIO
from types import SimpleNamespace

import pytest

from dashboard.experiment import (
    count_source_files,
    describe_exception,
    discover_source_files,
    ExperimentRequest,
    UPLOAD_CHUNK_BYTES,
    ingest_with_progress,
    local_datetime_to_ns,
    run_independent_experiments,
    run_experiment,
    save_uploaded_file,
    validate_source_path,
    zero_trade_diagnostic,
)


def test_describe_exception_keeps_empty_exception_type():
    assert describe_exception(AssertionError()) == "AssertionError: AssertionError()"


def test_zero_trade_diagnostic_explains_insufficient_cash():
    result = SimpleNamespace(
        trades=[],
        orders=[
            SimpleNamespace(reject_reason=SimpleNamespace(name="INSUFFICIENT_CASH"))
            for _ in range(3)
        ],
    )

    message = zero_trade_diagnostic(
        result, initial_cash=2_500.0, order_size=300,
    )

    assert "3 笔委托因资金不足" in message
    assert "每股可用资金约 8.33 元" in message


def test_ingest_progress_is_delivered_on_the_caller_thread(tmp_path):
    caller_thread = threading.get_ident()
    callback_threads = []

    class Lake:
        def ingest(self, source, progress):
            assert threading.get_ident() != caller_thread
            progress((source, 1))
            progress((source, 2))
            return "done"

    result = ingest_with_progress(
        Lake(), tmp_path / "bars.parquet",
        lambda value: callback_threads.append((threading.get_ident(), value)),
        poll_interval=0.001,
    )

    assert result == "done"
    assert [value for _, value in callback_threads] == [
        (tmp_path / "bars.parquet", 1),
        (tmp_path / "bars.parquet", 2),
    ]
    assert {thread_id for thread_id, _ in callback_threads} == {caller_thread}


def test_directory_discovery_is_recursive_filtered_and_deterministic(tmp_path):
    root = tmp_path / "bars"
    nested = root / "ticker=000001" / "year=2026"
    nested.mkdir(parents=True)
    (root / "ignored.sqlite3").write_bytes(b"state")
    (nested / "b.parquet").write_bytes(b"parquet")
    (nested / "a.csv").write_text("header\n", encoding="utf-8")

    files = list(discover_source_files(root))

    assert files == [nested / "a.csv", nested / "b.parquet"]
    assert count_source_files(root) == 2
    assert validate_source_path(root, allow_directory=True) == root.resolve()
    with pytest.raises(IsADirectoryError):
        validate_source_path(root)
from python.data_feed import DataLakeFeed
from python.market_data import DataLakeConfig, MinuteBarDataLake
from storage.trade_store import TradeStore


def test_uploaded_file_is_saved_in_fixed_size_chunks(tmp_path):
    class Upload(BytesIO):
        name = "bars.csv"

        def __init__(self, value):
            super().__init__(value)
            self.read_sizes = []

        def read(self, size=-1):
            self.read_sizes.append(size)
            return super().read(size)

    content = b"x" * (UPLOAD_CHUNK_BYTES + 17)
    uploaded = Upload(content)

    destination = save_uploaded_file(uploaded, tmp_path / "lake")

    assert destination.read_bytes() == content
    assert uploaded.read_sizes
    assert set(uploaded.read_sizes) == {UPLOAD_CHUNK_BYTES}
    assert validate_source_path(destination) == destination.resolve()


def test_experiment_runs_from_batch_feed_and_persists_result(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    source = tmp_path / "bars.csv"
    base = 1_704_159_060_000_000_000
    with source.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("timestamp", "symbol", "open", "high", "low", "close", "volume"))
        for index, price in enumerate((3.0, 2.0, 1.0, 2.0, 3.0, 4.0)):
            writer.writerow((
                base + index * 60_000_000_000,
                "000001", price, price, price, price, 10_000,
            ))
    lake = MinuteBarDataLake(DataLakeConfig(tmp_path / "lake", bucket_count=4))
    lake.ingest(source)

    def reject_object_stream(self):
        raise AssertionError("实验必须使用 stream_batches 快路径")

    monkeypatch.setattr(DataLakeFeed, "stream", reject_object_stream)
    database = tmp_path / "results.db"
    result = run_experiment(
        lake,
        database,
        ExperimentRequest(
            strategy="双均线",
            symbol="000001",
            initial_cash=10_000.0,
            order_size=1,
            short_window=2,
            long_window=3,
        ),
    )

    assert result.run_id == 1
    assert len(result.trades) == 1
    store = TradeStore(database)
    try:
        assert store.get_run(1)["status"] == "SUCCEEDED"
        assert store.count_trades(1) == 1
    finally:
        store.close()


def test_independent_and_portfolio_multi_symbol_experiments(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    source = tmp_path / "multi-symbol.csv"
    base = 1_704_159_060_000_000_000
    with source.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("timestamp", "symbol", "open", "high", "low", "close", "volume"))
        for index, price in enumerate((3.0, 2.0, 1.0, 2.0, 3.0, 3.0)):
            for symbol in ("000001", "600000"):
                writer.writerow((
                    base + index * 60_000_000_000,
                    symbol, price, price, price, price, 100_000,
                ))
    lake = MinuteBarDataLake(DataLakeConfig(tmp_path / "lake", bucket_count=4))
    lake.ingest(source)
    database = tmp_path / "results.db"

    independent = run_independent_experiments(
        lake,
        database,
        ExperimentRequest(
            strategy="双均线",
            symbol="000001",
            symbols=("000001", "600000"),
            mode="independent",
            initial_cash=10_000.0,
            order_size=100,
            short_window=2,
            long_window=3,
        ),
    )
    portfolio = run_experiment(
        lake,
        database,
        ExperimentRequest(
            strategy="双均线",
            symbol="000001",
            symbols=("000001", "600000"),
            mode="portfolio",
            initial_cash=10_000.0,
            order_size=100,
            short_window=2,
            long_window=3,
            capital_utilization=0.95,
        ),
    )

    assert [item.symbol for item in independent] == ["000001", "600000"]
    assert all(item.result is not None and not item.error for item in independent)
    assert all(len(item.result.trades) == 1 for item in independent)
    assert len({item.result.run_id for item in independent}) == 2
    assert {trade.symbol for trade in portfolio.trades} == {"000001", "600000"}
    assert portfolio.run_id not in {item.result.run_id for item in independent}
    store = TradeStore(database)
    try:
        assert store.get_run(portfolio.run_id)["symbols"] == "000001,600000"
        assert store.count_runs() == 3
    finally:
        store.close()


def test_experiment_request_and_local_time_validation(tmp_path):
    with pytest.raises(ValueError, match="short_window"):
        ExperimentRequest(
            strategy="双均线", symbol="X", initial_cash=1.0,
            order_size=1, short_window=20, long_window=5,
        )
    with pytest.raises(FileNotFoundError):
        validate_source_path(tmp_path / "missing.csv")
    assert local_datetime_to_ns(date(2026, 1, 5), time(9, 30)) > 0


def test_trade_analysis_handles_successful_run_without_trades(tmp_path):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    database = tmp_path / "empty-trades.db"
    store = TradeStore(database)
    try:
        run_id = store.create_run(
            "MACrossStrategy", ["000001"], "2026-01-01", "2026-01-31",
            2_000.0, backend="cpp",
        )
    finally:
        store.close()

    page = AppTest.from_file("dashboard/pages/2_trade_analysis.py")
    page.session_state["db_path"] = str(database)
    page.session_state["run_id"] = run_id
    page.run(timeout=10)

    assert not page.exception
    assert [item.value for item in page.info] == [
        "该回测运行成功，但没有产生成交。"
        "请检查初始资金、目标持仓数量、策略信号和交易约束。"
    ]
    assert [(item.label, item.value) for item in page.metric] == [
        ("成交笔数", "0"), ("涉及标的", "0"), ("累计手续费", "0.00"),
        ("平仓轮次", "0"), ("平仓胜率", "-"),
    ]
    assert not page.get("plotly_chart")


def test_data_experiment_page_offers_multi_symbol_modes(tmp_path):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    source = tmp_path / "symbols.csv"
    with source.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("timestamp", "symbol", "open", "high", "low", "close", "volume"))
        for symbol in ("000001", "600000"):
            writer.writerow((
                1_704_159_060_000_000_000,
                symbol, 10.0, 10.0, 10.0, 10.0, 10_000,
            ))
    lake_root = tmp_path / "lake"
    MinuteBarDataLake(DataLakeConfig(lake_root, bucket_count=64)).ingest(source)

    page = AppTest.from_file("dashboard/pages/0_data_experiment.py")
    page.session_state["lake_root"] = str(lake_root)
    page.session_state["db_path"] = str(tmp_path / "results.db")
    page.session_state["batch_comparison"] = [{
        "标的": "000001", "状态": "完成", "运行编号": 1,
        "期末权益": 10_100.0, "总收益": 0.01, "Sharpe": 1.2,
        "最大回撤": -0.02, "成交笔数": 2, "说明": "",
    }]
    page.run(timeout=10)

    assert not page.exception
    mode = page.get("button_group")[0]
    assert [option.content for option in mode.options] == [
        "独立批量", "共享资金组合",
    ]
    assert page.multiselect[0].options == ["000001", "600000"]
    assert len(page.dataframe) == 1

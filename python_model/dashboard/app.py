"""回测结果 Dashboard 主入口。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.utils import database_path, page_window
from storage.metrics_store import MetricsStore
from storage.trade_store import TradeStore


def main():
    st.set_page_config(page_title="quant-backtester", layout="wide")
    st.title("量化回测结果")

    default_path = st.session_state.get("db_path", database_path())
    db_path = st.sidebar.text_input("数据库", value=default_path)
    st.session_state["db_path"] = db_path
    store = TradeStore(db_path)
    try:
        total_runs = store.count_runs()
        if total_runs == 0:
            st.info("暂无回测记录。先运行一个示例策略。")
            return

        run_page_size = st.sidebar.selectbox("每页运行", (25, 50, 100), index=1)
        run_pages = max(1, (total_runs + run_page_size - 1) // run_page_size)
        run_page = st.sidebar.number_input(
            "运行页", min_value=1, max_value=run_pages, value=1
        )
        _, _, run_offset = page_window(total_runs, run_page, run_page_size)
        runs = store.list_runs(limit=run_page_size, offset=run_offset)

        labels = {
            f"#{row['id']} [{row['status']}] {row['strategy_name']} "
            f"({row['created_at']})": row["id"]
            for row in runs
        }
        selected = st.sidebar.selectbox("回测运行", list(labels))
        run_id = labels[selected]
        st.session_state["run_id"] = run_id
        run = store.get_run(run_id)
        metrics = MetricsStore(store.conn).get_metrics(run_id)

        if run["status"] == "FAILED":
            st.error(run["error_message"] or "回测失败")
        elif run["status"] == "RUNNING":
            st.warning("回测仍在运行")
        st.caption(
            f"状态: {run['status']} | 策略: {run['strategy_name']} | "
            f"后端: {run['backend']} | 标的: {run['symbols']} | "
            f"区间: {run['start_date']} 至 {run['end_date']}"
        )
        columns = st.columns(6)
        columns[0].metric("总收益", _percent(metrics, "total_return"))
        columns[1].metric("年化收益", _percent(metrics, "annual_return"))
        columns[2].metric("Sharpe", _number(metrics, "sharpe_ratio"))
        columns[3].metric("最大回撤", _percent(metrics, "max_drawdown"))
        columns[4].metric("成交笔数", int(metrics["total_trades"]) if metrics else 0)
        columns[5].metric(
            "平仓轮次", int(metrics["total_round_trips"]) if metrics else 0
        )
    finally:
        store.close()


def _percent(row, key: str) -> str:
    return "-" if row is None or row[key] is None else f"{row[key]:.2%}"


def _number(row, key: str) -> str:
    return "-" if row is None or row[key] is None else f"{row[key]:.2f}"


if __name__ == "__main__":
    main()

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.utils import database_path, page_window, resolve_run_id, rows_frame
from storage.metrics_store import MetricsStore
from storage.trade_store import TradeStore

st.title("交易分析")
store = TradeStore(st.session_state.get("db_path", database_path()))
try:
    run_id = resolve_run_id(store, st.session_state)
    if run_id is None:
        st.info("还没有可分析的回测记录。")
        st.stop()

    run = store.get_run(run_id)
    summary = store.get_trade_summary(run_id)
    metrics = MetricsStore(store.conn).get_metrics(run_id)
    total_trades = store.count_trades(run_id)
    if total_trades:
        daily = rows_frame(store.get_trade_daily_summary(run_id))
        sides = rows_frame(store.get_trade_side_summary(run_id))

        page_size = st.selectbox("每页成交", (50, 100, 250, 500), index=1)
        pages = max(1, (total_trades + page_size - 1) // page_size)
        page = st.number_input("成交页", min_value=1, max_value=pages, value=1)
        _, _, offset = page_window(total_trades, page, page_size)
        fills = rows_frame(store.get_trades(run_id, limit=page_size, offset=offset))

        total_round_trips = store.count_round_trips(run_id)
        round_trip_page_size = st.selectbox(
            "每页平仓轮次", (50, 100, 250, 500), index=1
        )
        round_trip_pages = max(
            1, (total_round_trips + round_trip_page_size - 1) // round_trip_page_size
        )
        round_trip_page = st.number_input(
            "平仓轮次页", min_value=1, max_value=round_trip_pages, value=1
        )
        _, _, round_trip_offset = page_window(
            total_round_trips, round_trip_page, round_trip_page_size
        )
        round_trips = rows_frame(
            store.get_round_trips(
                run_id, limit=round_trip_page_size, offset=round_trip_offset
            )
        ) if total_round_trips else rows_frame([])
finally:
    store.close()

cards = st.columns(5)
cards[0].metric("成交笔数", int(summary["total_trades"]))
cards[1].metric("涉及标的", int(summary["symbols"]))
cards[2].metric("累计手续费", f"{summary['commission']:,.2f}")
cards[3].metric("平仓轮次", int(metrics["total_round_trips"]) if metrics else 0)
cards[4].metric("平仓胜率", "-" if metrics is None else f"{metrics['win_rate']:.2%}")

if total_trades == 0:
    if run["status"] == "FAILED":
        st.error(run["error_message"] or "该回测运行失败，未产生成交。")
    elif run["status"] == "RUNNING":
        st.info("该回测仍在运行，暂时没有成交记录。")
    else:
        st.info(
            "该回测运行成功，但没有产生成交。"
            "请检查初始资金、目标持仓数量、策略信号和交易约束。"
        )
    st.stop()

left, right = st.columns(2)
with left:
    st.plotly_chart(px.bar(daily, x="date", y="trades", title="每日成交笔数"),
                    width="stretch")
with right:
    st.plotly_chart(px.pie(sides, names="side", values="notional", title="买卖成交额"),
                    width="stretch")

fill_tab, round_trip_tab = st.tabs(("逐笔成交", "平仓轮次"))
with fill_tab:
    if fills.empty:
        st.info("该页没有成交记录。")
    else:
        fills["datetime"] = pd.to_datetime(
            fills["timestamp"], unit="ns", utc=True
        ).dt.tz_convert("Asia/Shanghai")
        fills["notional"] = fills["quantity"] * fills["price"]
        st.dataframe(
            fills[["datetime", "symbol", "side", "quantity", "price",
                   "notional", "commission"]],
            width="stretch", hide_index=True,
        )
with round_trip_tab:
    if round_trips.empty:
        st.info("该回测没有已完成的平仓轮次。")
    else:
        round_trips["opened_at"] = pd.to_datetime(
            round_trips["opened_at"], unit="ns", utc=True
        ).dt.tz_convert("Asia/Shanghai")
        round_trips["closed_at"] = pd.to_datetime(
            round_trips["closed_at"], unit="ns", utc=True
        ).dt.tz_convert("Asia/Shanghai")
        st.dataframe(round_trips, width="stretch", hide_index=True)

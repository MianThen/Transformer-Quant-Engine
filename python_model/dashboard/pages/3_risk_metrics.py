from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils import (database_path, equity_frame, max_drawdown_duration,
                             resolve_run_id, rolling_sharpe)
from storage.trade_store import TradeStore

st.title("风险指标")
store = TradeStore(st.session_state.get("db_path", database_path()))
try:
    run_id = resolve_run_id(store, st.session_state)
    run = store.get_run(run_id) if run_id is not None else None
    frame = (
        equity_frame(store.get_equity_curve(run_id), run["initial_cash"])
        if run_id is not None else None
    )
finally:
    store.close()
if frame is None or len(frame) < 2:
    st.info("权益数据不足，无法计算风险指标。")
    st.stop()

frame["rolling_sharpe"] = rolling_sharpe(frame["return"])
var_95 = frame["return"].quantile(0.05)
cards = st.columns(4)
cards[0].metric("日波动率", f"{frame['return'].std(ddof=1):.2%}")
cards[1].metric("年化波动率", f"{frame['return'].std(ddof=1) * (252 ** 0.5):.2%}")
cards[2].metric("历史 VaR 95%", f"{var_95:.2%}")
cards[3].metric("最长回撤", f"{max_drawdown_duration(frame['drawdown'])} 天")

sharpe_figure = go.Figure(go.Scatter(x=frame["date"], y=frame["rolling_sharpe"],
                                     line=dict(color="#1677ff")))
sharpe_figure.update_layout(title="20 日滚动 Sharpe", height=340,
                            margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(sharpe_figure, width="stretch")

left, right = st.columns(2)
with left:
    st.plotly_chart(px.histogram(frame, x="return", nbins=30, title="日收益分布"),
                    width="stretch")
with right:
    underwater = go.Figure(go.Scatter(x=frame["date"], y=frame["drawdown"],
                                      fill="tozeroy", line=dict(color="#d14343")))
    underwater.update_layout(title="水下曲线", yaxis_tickformat=".1%")
    st.plotly_chart(underwater, width="stretch")

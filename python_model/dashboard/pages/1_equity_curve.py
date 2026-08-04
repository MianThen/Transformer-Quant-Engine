from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.utils import database_path, equity_frame, resolve_run_id
from storage.trade_store import TradeStore

st.title("权益与回撤")
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
if frame is None or frame.empty:
    st.info("该回测没有权益数据。")
    st.stop()

figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       row_heights=[0.7, 0.3], vertical_spacing=0.06)
figure.add_trace(go.Scatter(x=frame["date"], y=frame["equity"],
                            name="权益", line=dict(color="#1677ff")), row=1, col=1)
figure.add_trace(go.Scatter(x=frame["date"], y=frame["cash"],
                            name="现金", line=dict(color="#6b7280")), row=1, col=1)
figure.add_trace(go.Scatter(x=frame["date"], y=frame["drawdown"],
                            name="回撤", fill="tozeroy",
                            line=dict(color="#d14343")), row=2, col=1)
figure.update_yaxes(tickformat=".1%", row=2, col=1)
figure.update_layout(height=650, margin=dict(l=20, r=20, t=30, b=20),
                     hovermode="x unified")
st.plotly_chart(figure, width="stretch")

summary = st.columns(3)
summary[0].metric("期末权益", f"{frame['equity'].iloc[-1]:,.2f}")
summary[1].metric("最低权益", f"{frame['equity'].min():,.2f}")
summary[2].metric("最大回撤", f"{frame['drawdown'].min():.2%}")

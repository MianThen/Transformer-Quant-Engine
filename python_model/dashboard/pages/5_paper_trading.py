"""虚拟盘(paper trading)监控页:双会话权益对照、成交流水、持仓与往返统计。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from python.paper.config import DB_PATH, SESSION_PRESETS
from python.paper.sources import ns_to_date


@st.cache_data(ttl=30)
def _query(db_path: str, sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


st.title("虚拟盘(Paper Trading)")
st.caption(
    "冠军制品 champion-legacy-fold3-safe 的日频虚拟盘:收盘截面决策、次日开盘成交。"
    "纯观测环境——制品 promotion_eligible=false,结果不构成晋级证据;"
    "特征尺度问题(d06436b)未闭环,registered 会话零订单属登记口径的忠实表现。"
)

db_path = str(DB_PATH)
if not Path(db_path).exists():
    st.info("尚未初始化虚拟盘;运行 python -m python.paper init")
    st.stop()

# ---- 会话概要 ----
metrics = st.columns(len(SESSION_PRESETS))
session_ids = []
for column, spec in zip(metrics, SESSION_PRESETS):
    row = _query(db_path, """
        SELECT
            (SELECT count(*) FROM days WHERE session_id = ?) AS days,
            (SELECT equity FROM days WHERE session_id = ?
                ORDER BY trade_ns DESC LIMIT 1) AS equity,
            (SELECT count(*) FROM orders WHERE session_id = ?) AS orders,
            (SELECT count(*) FROM fills WHERE session_id = ?) AS fills,
            (SELECT count(*) FROM positions WHERE session_id = ?
                AND asof_ns = (SELECT max(asof_ns) FROM positions
                               WHERE session_id = ?)
                AND quantity > 0) AS open_positions,
            (SELECT initial_cash FROM sessions WHERE session_id = ?) AS initial_cash
    """, (spec.session_id,) * 7)
    session_ids.append(spec.session_id)
    if row.empty or row["days"].iloc[0] == 0:
        column.metric(spec.session_id, "未推进")
        continue
    equity = float(row["equity"].iloc[0])
    initial = float(row["initial_cash"].iloc[0])
    column.metric(
        f"{spec.session_id}({spec.kind})",
        f"{equity:,.0f}",
        f"{equity / initial - 1:+.2%}",
        help=spec.note,
    )
    column.caption(
        f"天数 {row['days'].iloc[0]}  订单 {row['orders'].iloc[0]}  "
        f"成交 {row['fills'].iloc[0]}  持仓 {row['open_positions'].iloc[0]}"
    )

# ---- 权益对照 ----
st.subheader("权益曲线对照")
figure = go.Figure()
for spec in SESSION_PRESETS:
    curve = _query(db_path, """
        SELECT ts, equity, cash FROM equity_curve
        WHERE session_id = ? ORDER BY ts
    """, (spec.session_id,))
    if curve.empty:
        continue
    initial = _query(db_path, """
        SELECT initial_cash FROM sessions WHERE session_id = ?
    """, (spec.session_id,))["initial_cash"].iloc[0]
    dates = [ns_to_date(int(ts)) for ts in curve["ts"]]
    figure.add_trace(go.Scatter(
        x=dates, y=curve["equity"] / initial, name=spec.session_id,
    ))
figure.update_yaxes(tickformat=".1%")
figure.update_layout(
    height=420, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified",
    yaxis_title="相对初始资金",
)
st.plotly_chart(figure, width="stretch")

# ---- 会话明细 ----
selected = st.radio("会话", session_ids, horizontal=True)
latest = _query(db_path, """
    SELECT max(asof_ns) AS m FROM positions WHERE session_id = ?
""", (selected,))["m"].iloc[0]

left, right = st.columns(2)
with left:
    st.markdown("**当前持仓**")
    if latest is not None:
        positions = _query(db_path, """
            SELECT symbol, quantity, sellable, avg_cost FROM positions
            WHERE session_id = ? AND asof_ns = ? AND quantity > 0
            ORDER BY quantity * avg_cost DESC
        """, (selected, int(latest)))
        if positions.empty:
            st.caption("空仓")
        else:
            st.dataframe(positions, width="stretch", hide_index=True)
    else:
        st.caption("未推进")

with right:
    st.markdown("**往返交易统计**")
    trips = _query(db_path, """
        SELECT count(*) AS n, sum(net_pnl) AS net,
               sum(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) AS winners,
               avg(quantity * entry_price) AS avg_notional
        FROM round_trips WHERE session_id = ?
    """, (selected,))
    if trips["n"].iloc[0]:
        n = int(trips["n"].iloc[0])
        st.caption(
            f"{n} 笔往返  净盈亏 {trips['net'].iloc[0]:,.0f}  "
            f"胜率 {trips['winners'].iloc[0] / n:.1%}  "
            f"平均名义 {trips['avg_notional'].iloc[0]:,.0f}"
        )
    else:
        st.caption("无往返交易")

st.subheader("成交流水(最近 40 笔)")
fills = _query(db_path, """
    SELECT ts, symbol, side, quantity, price, commission FROM fills
    WHERE session_id = ? ORDER BY ts DESC, fill_index DESC LIMIT 40
""", (selected,))
if fills.empty:
    st.caption("无成交(registered 会话按登记口径 minimum_confidence=0.55 过滤,零订单属预期)")
else:
    fills["日期"] = [ns_to_date(int(ts)) for ts in fills["ts"]]
    fills = fills.drop(columns=["ts"]).rename(columns={
        "symbol": "标的", "side": "方向", "quantity": "数量",
        "price": "价格", "commission": "费用",
    })
    st.dataframe(fills, width="stretch", hide_index=True)

st.subheader("每日摘要(最近 15 个交易日)")
days = _query(db_path, """
    SELECT trade_ns, source, n_symbols, n_orders, n_fills, equity, cash
    FROM days WHERE session_id = ? ORDER BY trade_ns DESC LIMIT 15
""", (selected,))
if not days.empty:
    days["日期"] = [ns_to_date(int(ts)) for ts in days["trade_ns"]]
    days = days.drop(columns=["trade_ns"]).rename(columns={
        "source": "数据源", "n_symbols": "截面标的", "n_orders": "新订单",
        "n_fills": "成交", "equity": "权益", "cash": "现金",
    })
    st.dataframe(days, width="stretch", hide_index=True)

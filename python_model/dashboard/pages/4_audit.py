from __future__ import annotations

import json

import streamlit as st

from dashboard.utils import database_path, page_window, resolve_run_id, rows_frame
from storage.trade_store import TradeStore

st.title("运行审计")
store = TradeStore(st.session_state.get("db_path", database_path()))
try:
    run_id = resolve_run_id(store, st.session_state)
    if run_id is None:
        st.info("暂无回测记录。")
        st.stop()

    total_orders = store.count_orders(run_id)
    page_size = st.selectbox("每页委托", (50, 100, 250, 500), index=1)
    pages = max(1, (total_orders + page_size - 1) // page_size)
    page = st.number_input("委托页", min_value=1, max_value=pages, value=1)
    _, _, offset = page_window(total_orders, page, page_size)

    orders = rows_frame(store.get_orders(run_id, limit=page_size, offset=offset))
    order_summary = rows_frame(store.get_order_status_summary(run_id))
    total_actions = store.count_corporate_actions(run_id)
    action_page_size = st.selectbox("每页公司行动", (50, 100, 250, 500), index=1)
    action_pages = max(1, (total_actions + action_page_size - 1) // action_page_size)
    action_page = st.number_input(
        "公司行动页", min_value=1, max_value=action_pages, value=1
    )
    _, _, action_offset = page_window(
        total_actions, action_page, action_page_size
    )
    actions = rows_frame(
        store.get_corporate_actions(
            run_id, limit=action_page_size, offset=action_offset
        )
    ) if total_actions else rows_frame([])
    portfolio = store.get_portfolio_risk(run_id)
    lineage = store.get_data_lineage(run_id)
    run_config = store.get_run_config(run_id)
finally:
    store.close()

orders_tab, actions_tab, risk_tab, lineage_tab, config_tab = st.tabs(
    ("委托", "公司行动", "组合风险", "数据血缘", "运行配置")
)
with orders_tab:
    st.dataframe(order_summary, width="stretch", hide_index=True)
    st.dataframe(orders, width="stretch", hide_index=True)
with actions_tab:
    if actions.empty:
        st.info("该回测没有公司行动。")
    else:
        st.dataframe(actions, width="stretch", hide_index=True)
with risk_tab:
    if portfolio is None:
        st.info("该回测没有组合风险快照。")
    else:
        value = dict(portfolio)
        value["industry_exposure"] = json.loads(value.pop("industry_exposure_json"))
        value["factor_exposure"] = json.loads(value.pop("factor_exposure_json"))
        st.json(value)
with lineage_tab:
    if lineage is None:
        st.info("该回测没有数据血缘。")
    else:
        value = dict(lineage)
        value["source_fingerprints"] = json.loads(
            value.pop("source_fingerprints_json")
        )
        st.json(value)
with config_tab:
    if run_config is None:
        st.info("该回测没有 RunSpec。")
    else:
        st.code(run_config["config_hash"], language=None)
        st.json(json.loads(run_config["config_json"]))

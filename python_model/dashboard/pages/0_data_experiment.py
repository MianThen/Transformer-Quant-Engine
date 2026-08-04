from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from dashboard.experiment import (
    describe_exception,
    ExperimentRequest,
    format_bytes,
    count_source_files,
    ingest_directory_with_progress,
    ingest_with_progress,
    local_datetime_to_ns,
    ns_to_local_datetime,
    run_independent_experiments,
    run_experiment,
    save_uploaded_file,
    validate_source_path,
    zero_trade_diagnostic,
)
from dashboard.utils import database_path
from python.engine_api import BACKEND
from python.market_data import MinuteBarDataLake


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAKE = os.getenv("QBT_DATA_LAKE_PATH", str(PROJECT_ROOT / "data" / "lake"))
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


st.set_page_config(page_title="数据实验", layout="wide")
st.title("数据实验")

lake_root = st.sidebar.text_input(
    "数据湖目录",
    value=st.session_state.get("lake_root", DEFAULT_LAKE),
)
st.session_state["lake_root"] = lake_root
db_path = st.sidebar.text_input(
    "结果数据库",
    value=st.session_state.get("db_path", database_path()),
)
st.session_state["db_path"] = db_path
st.sidebar.caption(f"执行后端: {BACKEND}")

try:
    lake = MinuteBarDataLake(lake_root)
except Exception as error:
    st.error(f"无法打开数据湖: {error}")
    st.stop()

summary = lake.summary()
summary_columns = st.columns(4)
summary_columns[0].metric("数据行数", f"{summary.rows:,}")
summary_columns[1].metric("分区文件", f"{summary.files:,}")
summary_columns[2].metric("Catalog 版本", summary.generation)
summary_columns[3].metric(
    "数据区间",
    "-" if summary.min_timestamp is None else
    f"{ns_to_local_datetime(summary.min_timestamp):%Y-%m-%d} 至 "
    f"{ns_to_local_datetime(summary.max_timestamp):%Y-%m-%d}",
)

import_tab, run_tab = st.tabs(("导入行情", "运行回测"))

with import_tab:
    source_mode = st.radio(
        "数据来源",
        ("本机文件路径", "小文件上传"),
        horizontal=True,
        help="大文件请选择本机路径，浏览器上传会占用额外内存。",
    )
    source_path = None
    uploaded = None
    source_file_count = 0
    if source_mode == "本机文件路径":
        path_value = st.text_input(
            "CSV 或 Parquet 路径",
            placeholder="/Users/name/data/minute-bars.parquet",
        )
        if path_value:
            try:
                source_path = validate_source_path(path_value, allow_directory=True)
                if source_path.is_dir():
                    provider_dirs = sorted(
                        child.name for child in source_path.iterdir()
                        if child.is_dir() and child.name.startswith("provider=")
                    )
                    if len(provider_dirs) > 1:
                        raise ValueError(
                            "目录包含多个数据源，请选择一个 provider 子目录: "
                            + ", ".join(provider_dirs)
                        )
                    file_count = count_source_files(source_path)
                    source_file_count = file_count
                    st.caption(f"目录 | {file_count:,} 个 CSV/Parquet 文件，将递归批量导入")
                    if file_count == 0:
                        st.error("目录中没有 CSV、Parquet 或 PQ 文件")
                else:
                    st.caption(
                        f"{source_path.name} | {format_bytes(source_path.stat().st_size)}"
                    )
            except Exception as error:
                st.error(str(error))
    else:
        st.caption("仅支持 100 MiB 以内文件；更大文件请使用本机文件路径。")
        uploaded = st.file_uploader(
            "选择 CSV 或 Parquet",
            type=("csv", "parquet", "pq"),
            help="仅用于 100 MiB 以内文件；更大文件请使用本机路径。",
        )
        if uploaded is not None:
            st.caption(f"{uploaded.name} | {format_bytes(uploaded.size)}")
            if uploaded.size > MAX_UPLOAD_BYTES:
                st.error("文件超过 100 MiB，请改用本机文件路径。")

    preview_column, import_column = st.columns(2)
    if preview_column.button(
        "预览前 20 行",
        disabled=source_path is None or source_path.is_dir(),
        width="stretch",
    ):
        try:
            preview = lake.preview(source_path, limit=20)
            st.dataframe(preview.to_pandas(), width="stretch", hide_index=True)
        except Exception as error:
            st.error(f"预览失败: {error}")

    can_import = (
        source_path is not None
        and (not source_path.is_dir() or source_file_count > 0)
    ) or (
        uploaded is not None and uploaded.size <= MAX_UPLOAD_BYTES
    )
    if import_column.button(
        "导入数据湖",
        type="primary",
        disabled=not can_import,
        width="stretch",
    ):
        progress_text = st.empty()

        def report(progress):
            if hasattr(progress, "files_completed"):
                progress_text.info(
                    f"已处理 {progress.files_completed:,}/{progress.files_total:,} 个文件，"
                    f"累计 {progress.rows:,} 行"
                )
            elif progress.batches == 1 or progress.batches % 10 == 0:
                progress_text.info(
                    f"已校验 {progress.rows:,} 行，{progress.batches:,} 个批次"
                )

        try:
            selected = source_path
            if uploaded is not None:
                selected = save_uploaded_file(uploaded, lake_root)
            with st.spinner("正在流式校验并写入分区数据湖..."):
                if selected.is_dir():
                    result = ingest_directory_with_progress(lake, selected, report)
                else:
                    result = ingest_with_progress(lake, selected, report)
            progress_text.empty()
            if hasattr(result, "files_imported"):
                st.success(
                    f"批量导入完成：{result.files_imported:,} 个新文件，"
                    f"跳过 {result.files_skipped:,} 个，{result.rows:,} 行，"
                    f"{result.fragments:,} 个分区文件。"
                )
                st.rerun()
            elif result.skipped:
                st.info("相同内容已经导入，无需重复处理。")
            else:
                st.success(
                    f"导入完成：{result.rows:,} 行，{result.files:,} 个分区文件。"
                )
                st.rerun()
        except Exception as error:
            progress_text.empty()
            st.error(f"导入失败: {describe_exception(error)}")

with run_tab:
    if summary.rows == 0:
        st.info("请先导入行情数据。")
    else:
        @st.cache_data(show_spinner=False)
        def load_symbols(root: str, generation: int):
            return MinuteBarDataLake(root).list_symbols()

        symbols = load_symbols(str(Path(lake_root).expanduser().resolve()), summary.generation)
        if not symbols:
            st.warning("数据湖中没有可用标的。")
            st.stop()

        left, right = st.columns(2)
        with left:
            mode_label = st.segmented_control(
                "回测模式",
                ("独立批量", "共享资金组合"),
                default="独立批量",
            )
            mode = "portfolio" if mode_label == "共享资金组合" else "independent"
            strategy_name = st.selectbox("策略", ("双均线", "均值回归"))
            selected_symbols = st.multiselect(
                "标的",
                symbols,
                default=[symbols[0]],
                max_selections=50,
                placeholder="选择最多 50 个标的",
            )
            initial_cash = st.number_input(
                "每个标的初始资金" if mode == "independent" else "组合初始资金",
                min_value=0.0, value=1_000_000.0, step=100_000.0,
            )
            if mode == "independent":
                order_size = int(st.number_input(
                    "每个标的目标持仓数量（100 股整数倍）",
                    min_value=100, value=100, step=100,
                ))
                capital_utilization = 0.95
                st.caption(
                    f"按当前本金，每股可用资金约 "
                    f"{initial_cash / order_size:,.2f} 元（未计手续费）"
                )
            else:
                order_size = 100
                capital_utilization = st.slider(
                    "组合资金利用率",
                    min_value=50, max_value=99, value=95, step=1,
                    format="%d%%",
                ) / 100.0
                if selected_symbols:
                    budget = initial_cash * capital_utilization / len(selected_symbols)
                    st.caption(
                        f"等权分配：{len(selected_symbols)} 个标的，"
                        f"每个预算约 {budget:,.2f} 元"
                    )
        with right:
            if strategy_name == "双均线":
                short_window = int(st.number_input(
                    "短均线窗口", min_value=1, value=5, step=1
                ))
                long_window = int(st.number_input(
                    "长均线窗口", min_value=2, value=20, step=1
                ))
                window = 20
                num_std = 2.0
            else:
                window = int(st.number_input(
                    "统计窗口", min_value=2, value=20, step=1
                ))
                num_std = float(st.number_input(
                    "标准差倍数", min_value=0.1, value=2.0, step=0.1
                ))
                short_window = 5
                long_window = 20
            full_range = st.checkbox("使用完整数据区间", value=True)
            start = end = None
            if not full_range:
                minimum = ns_to_local_datetime(summary.min_timestamp)
                maximum = ns_to_local_datetime(summary.max_timestamp)
                start_day = st.date_input(
                    "开始日期",
                    value=minimum.date(),
                    min_value=minimum.date(),
                    max_value=maximum.date(),
                )
                start_time = st.time_input("开始时间", value=minimum.time())
                end_day = st.date_input(
                    "结束日期",
                    value=maximum.date(),
                    min_value=minimum.date(),
                    max_value=maximum.date(),
                )
                end_time = st.time_input("结束时间", value=maximum.time())
                start = local_datetime_to_ns(start_day, start_time)
                end = local_datetime_to_ns(end_day, end_time)

        if st.button(
            "开始回测", type="primary", width="stretch",
            disabled=not selected_symbols,
        ):
            try:
                request = ExperimentRequest(
                    strategy=strategy_name,
                    symbol=selected_symbols[0],
                    symbols=tuple(selected_symbols),
                    mode=mode,
                    initial_cash=float(initial_cash),
                    order_size=order_size,
                    capital_utilization=float(capital_utilization),
                    start=start,
                    end=end,
                    short_window=short_window,
                    long_window=long_window,
                    window=window,
                    num_std=num_std,
                )
                if mode == "independent":
                    progress_bar = st.progress(0.0, text="准备独立批量回测...")

                    def report_batch(progress):
                        progress_bar.progress(
                            progress.completed / progress.total,
                            text=(
                                f"已完成 {progress.completed}/{progress.total}："
                                f"{progress.symbol}"
                            ),
                        )

                    outcomes = run_independent_experiments(
                        lake, db_path, request, progress=report_batch,
                    )
                    progress_bar.empty()
                    comparison = []
                    successful = []
                    for outcome in outcomes:
                        if outcome.result is None:
                            comparison.append({
                                "标的": outcome.symbol,
                                "状态": "失败",
                                "错误": outcome.error,
                            })
                            continue
                        result = outcome.result
                        successful.append(result)
                        diagnostic = zero_trade_diagnostic(
                            result,
                            initial_cash=float(initial_cash),
                            order_size=order_size,
                        )
                        comparison.append({
                            "标的": outcome.symbol,
                            "状态": "零成交" if diagnostic else "完成",
                            "运行编号": result.run_id,
                            "期末权益": result.final_equity,
                            "总收益": result.total_return,
                            "Sharpe": result.sharpe_ratio,
                            "最大回撤": result.max_drawdown,
                            "成交笔数": len(result.trades),
                            "说明": diagnostic or "",
                        })
                    st.session_state["batch_comparison"] = comparison
                    if successful:
                        st.session_state["run_id"] = successful[-1].run_id
                    failures = len(outcomes) - len(successful)
                    if failures:
                        st.warning(
                            f"独立批量完成：{len(successful)} 个成功，"
                            f"{failures} 个失败。"
                        )
                    else:
                        st.success(f"独立批量完成：{len(successful)} 个标的。")
                else:
                    with st.spinner("正在流式回放组合行情并执行策略..."):
                        result = run_experiment(lake, db_path, request)
                    st.session_state["run_id"] = result.run_id
                    st.session_state.pop("batch_comparison", None)
                    diagnostic = zero_trade_diagnostic(
                        result,
                        initial_cash=float(initial_cash),
                        order_size=None,
                    )
                    if diagnostic:
                        st.warning(f"{diagnostic}运行编号 #{result.run_id}。")
                    else:
                        st.success(
                            f"组合回测完成：期末权益 "
                            f"{result.final_equity:,.2f}，总收益 "
                            f"{result.total_return:.2%}，成交 "
                            f"{len(result.trades):,} 笔，运行编号 #{result.run_id}。"
                        )
            except Exception as error:
                st.error(f"回测失败: {error}")

        comparison = st.session_state.get("batch_comparison")
        if comparison:
            st.subheader("独立批量对比")
            st.dataframe(
                comparison,
                column_config={
                    "总收益": st.column_config.NumberColumn(format="percent"),
                    "最大回撤": st.column_config.NumberColumn(format="percent"),
                    "期末权益": st.column_config.NumberColumn(format="%.2f"),
                    "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                },
                width="stretch",
                hide_index=True,
            )

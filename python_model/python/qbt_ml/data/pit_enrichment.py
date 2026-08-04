from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


OUTPUT_SCHEMA = pa.schema([
    ("timestamp", pa.int64()), ("symbol", pa.string()),
    ("open", pa.float64()), ("high", pa.float64()),
    ("low", pa.float64()), ("close", pa.float64()), ("volume", pa.int64()),
    ("bar_observed", pa.bool_()),
    ("signal_open", pa.float64()), ("signal_high", pa.float64()),
    ("signal_low", pa.float64()), ("signal_close", pa.float64()),
    ("adjustment_factor", pa.float64()), ("adjustment_known_at", pa.int64()),
    ("is_listed", pa.bool_()), ("is_suspended", pa.bool_()),
    ("is_st", pa.bool_()), ("is_tradable", pa.bool_()),
    ("upper_limit", pa.float64()), ("lower_limit", pa.float64()),
    ("lot_size", pa.int64()), ("min_buy_quantity", pa.int64()),
    ("industry", pa.string()), ("industry_known_at", pa.int64()),
    ("universe_asof", pa.int64()),
    ("reference_data_known_at_max", pa.int64()),
    ("frequency", pa.string()), ("calendar_id", pa.string()),
], metadata={
    b"qbt.schema": b"PHASE_E_DAILY_ENRICHED_V1",
    b"qbt.price_adjustment_mode": b"pit_adjusted_signal_raw_execution",
})

BAR_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
STATE_COLUMNS = (
    "timestamp", "symbol", "is_listed", "is_suspended", "is_st", "is_tradable",
    "universe_asof", "reference_data_known_at_max",
)
ADJUSTMENT_COLUMNS = (
    "timestamp", "symbol", "back_adjust_factor", "known_at",
)
INDUSTRY_COLUMNS = (
    "timestamp", "symbol", "industry", "classification", "known_at", "snapshot_asof",
)
RULE_COLUMNS = (
    "timestamp", "symbol", "upper_limit", "lower_limit", "lot_size",
    "min_buy_quantity", "known_at",
)


def _parquet_files(root: str | Path | None, symbol: str | None = None) -> list[Path]:
    if root is None:
        return []
    path = Path(root).expanduser().resolve()
    if symbol is not None:
        path = path / f"ticker={symbol}"
    return sorted(path.rglob("*.parquet")) if path.is_dir() else []


def _read_files(files: list[Path], columns: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for path in files:
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        missing = sorted(set(columns) - schema_names)
        if missing:
            raise ValueError(f"{path} 缺少字段: {', '.join(missing)}")
        frames.append(pq.ParquetFile(path).read(columns=list(columns)).to_pandas())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _require_unique(table: pd.DataFrame, keys: list[str], label: str) -> None:
    if table.duplicated(keys).any():
        sample = table.loc[table.duplicated(keys, keep=False), keys].head().to_dict("records")
        raise ValueError(f"{label} 主键重复: {sample}")


def _asof_adjustment(rows: pd.DataFrame, adjustments: pd.DataFrame) -> pd.DataFrame:
    events = adjustments.rename(columns={
        "back_adjust_factor": "adjustment_factor",
        "known_at": "adjustment_known_at",
    }).drop(columns=["symbol"])
    if not events.empty:
        _require_unique(events, ["timestamp"], "adjustment_factor")
        if (events["adjustment_known_at"] > events["timestamp"]).any():
            raise ValueError("adjustment factor 含未来 known_at")
        rows = pd.merge_asof(
            rows.sort_values("timestamp"), events.sort_values("timestamp"),
            on="timestamp", direction="backward", allow_exact_matches=True,
        )
    else:
        rows["adjustment_factor"] = np.nan
        rows["adjustment_known_at"] = np.nan
    rows["adjustment_factor"] = rows["adjustment_factor"].fillna(1.0)
    rows["adjustment_known_at"] = rows["adjustment_known_at"].fillna(0).astype("int64")
    factors = rows["adjustment_factor"].to_numpy(dtype=np.float64)
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("adjustment factor 必须为有限正数")
    return rows


def _asof_industry(rows: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    events = industry.drop(columns=["symbol"]).copy()
    if events.empty:
        rows["industry"] = None
        rows["industry_known_at"] = 0
        return rows
    if (events["known_at"] > events["snapshot_asof"]).any():
        raise ValueError("industry snapshot 含未来 known_at")
    distinct = events.drop_duplicates(["snapshot_asof", "industry", "classification"])
    if distinct.duplicated("snapshot_asof", keep=False).any():
        raise ValueError("同一 industry snapshot 存在多个分类结果，必须先冻结分类标准")
    # snapshot_asof is the first time this provider snapshot was actually available.
    events["industry_known_at"] = events[["known_at", "snapshot_asof"]].max(axis=1)
    events = events.sort_values(
        ["snapshot_asof", "known_at", "timestamp", "classification"], kind="stable"
    ).drop_duplicates("snapshot_asof", keep="last")
    events = events[["snapshot_asof", "industry", "industry_known_at"]]
    rows = pd.merge_asof(
        rows.sort_values("timestamp"), events.sort_values("snapshot_asof"),
        left_on="timestamp", right_on="snapshot_asof", direction="backward",
        allow_exact_matches=True,
    ).drop(columns=["snapshot_asof"])
    rows["industry_known_at"] = rows["industry_known_at"].fillna(0).astype("int64")
    return rows


def _join_execution_rules(rows: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    if rules.empty:
        for name in ("upper_limit", "lower_limit", "lot_size", "min_buy_quantity"):
            rows[name] = np.nan
        rows["rule_known_at"] = 0
        return rows
    rules = rules.rename(columns={"known_at": "rule_known_at"}).drop(columns=["symbol"])
    _require_unique(rules, ["timestamp"], "execution rule")
    if (rules["rule_known_at"] > rules["timestamp"]).any():
        raise ValueError("execution rule 含未来 known_at")
    return rows.merge(rules, on="timestamp", how="left", validate="one_to_one")


def _enrich_symbol(
    symbol: str,
    bars: pd.DataFrame,
    states: pd.DataFrame,
    adjustments: pd.DataFrame,
    industry: pd.DataFrame,
    rules: pd.DataFrame,
    *,
    frequency: str,
    calendar_id: str,
) -> pd.DataFrame:
    _require_unique(bars, ["timestamp", "symbol"], "bar")
    _require_unique(states, ["timestamp", "symbol"], "security_state")
    unexpected = bars.merge(
        states[["timestamp", "symbol"]], on=["timestamp", "symbol"], how="left",
        indicator=True,
    )
    if (unexpected["_merge"] == "left_only").any():
        raise ValueError(f"{symbol} 存在没有 PIT 状态骨架的 Bar")
    rows = states.merge(
        bars, on=["timestamp", "symbol"], how="left", validate="one_to_one",
        indicator="_bar_merge",
    ).sort_values("timestamp", kind="stable").reset_index(drop=True)
    rows["bar_observed"] = rows.pop("_bar_merge").eq("both")
    missing = ~rows["bar_observed"]
    illegal_missing = missing & rows["is_listed"].astype(bool) & ~rows["is_suspended"].astype(bool)
    if illegal_missing.any():
        timestamps = rows.loc[illegal_missing, "timestamp"].head().tolist()
        raise ValueError(f"{symbol} 非停牌上市日缺少原始 Bar: {timestamps}")
    previous_close = rows["close"].ffill()
    if (missing & previous_close.isna()).any():
        raise ValueError(f"{symbol} 首个停牌状态之前没有可沿用的原始收盘价")
    for name in ("open", "high", "low", "close"):
        rows.loc[missing, name] = previous_close[missing]
    rows.loc[missing, "volume"] = 0
    rows["volume"] = rows["volume"].astype("int64")
    rows = _asof_adjustment(rows, adjustments)
    rows = _asof_industry(rows, industry)
    rows = _join_execution_rules(rows, rules)
    for name in ("open", "high", "low", "close"):
        rows[f"signal_{name}"] = rows[name] * rows["adjustment_factor"]
    expected_tradable = (
        rows["is_listed"].astype(bool) & ~rows["is_suspended"].astype(bool)
    )
    if (rows["is_tradable"].astype(bool) != expected_tradable).any():
        raise ValueError(f"{symbol} is_tradable 与上市/停牌状态不一致")
    known_columns = [
        "reference_data_known_at_max", "adjustment_known_at",
        "industry_known_at", "rule_known_at",
    ]
    rows["reference_data_known_at_max"] = rows[known_columns].fillna(0).max(axis=1).astype("int64")
    if (rows["reference_data_known_at_max"] > rows["timestamp"]).any():
        raise ValueError(f"{symbol} 富化结果包含未来参考数据")
    rows["frequency"] = frequency
    rows["calendar_id"] = calendar_id
    return rows[list(OUTPUT_SCHEMA.names)]


def _fingerprint(files: list[Path]) -> str:
    file_digests = []
    for path in sorted(set(files)):
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(block)
        file_digests.append(file_digest.digest())
    digest = hashlib.sha256()
    for value in sorted(file_digests):
        digest.update(value)
    return digest.hexdigest()


def _validate_corporate_actions(files: list[Path]) -> int:
    required = {
        "timestamp", "symbol", "cash_dividend_per_share",
        "stock_dividend_per_share", "reserve_to_stock_per_share",
        "share_multiplier", "known_at",
    }
    keys = set()
    count = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        missing = sorted(required - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"{path} 企业行动表缺少字段: {', '.join(missing)}")
        table = parquet.read(columns=sorted(required)).to_pandas()
        if (table["known_at"] > table["timestamp"]).any():
            raise ValueError(f"{path} 企业行动表含未来 known_at")
        if (table["share_multiplier"] < 1).any():
            raise ValueError(f"{path} 企业行动 share_multiplier 小于 1")
        for row in table.itertuples(index=False):
            key = (
                int(row.timestamp), str(row.symbol),
                float(row.cash_dividend_per_share),
                float(row.stock_dividend_per_share),
                float(row.reserve_to_stock_per_share), float(row.share_multiplier),
            )
            if key in keys:
                raise ValueError(f"企业行动生效主键重复: {key}")
            keys.add(key)
        count += len(table)
    return count


def _validate_formal_rows(rows: pd.DataFrame, symbol: str) -> None:
    required = (
        "industry", "upper_limit", "lower_limit", "lot_size", "min_buy_quantity",
    )
    missing = [name for name in required if rows[name].isna().any()]
    if missing:
        raise ValueError(f"{symbol} 正式富化字段存在空值: {', '.join(missing)}")
    prices = rows[["upper_limit", "lower_limit"]].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"{symbol} 涨跌停价必须为有限正数")
    if (rows["lower_limit"] > rows["upper_limit"]).any():
        raise ValueError(f"{symbol} lower_limit 高于 upper_limit")
    for name in ("lot_size", "min_buy_quantity"):
        values = rows[name].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values <= 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{symbol} {name} 必须为正整数")


def enrich_phase_e(config: dict, output_override: str | Path | None = None) -> Path:
    required = ("bars_root", "state_root", "adjustment_root", "industry_root")
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError("Phase E 富化配置缺少: " + ", ".join(missing))
    frequency = str(config.get("frequency", ""))
    calendar_id = str(config.get("calendar_id", ""))
    if frequency != "1d" or not calendar_id or "CONFIGURE_" in calendar_id:
        raise ValueError("当前富化器要求 frequency=1d 和已冻结的 calendar_id")
    legacy_formal = config.get("formal")
    execution_mode = config.get(
        "execution_reference_mode",
        "required_for_promotion" if legacy_formal is True
        else "optional_for_model_evaluation",
    )
    allowed_modes = {"optional_for_model_evaluation", "required_for_promotion"}
    if execution_mode not in allowed_modes:
        raise ValueError("execution_reference_mode 仅支持 optional_for_model_evaluation/required_for_promotion")
    output_value = output_override or config.get("output")
    if not output_value:
        raise ValueError("Phase E 富化配置缺少 output")
    output = Path(output_value)
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"富化数据集不可覆盖已有目录: {output}")

    state_root = Path(config["state_root"]).expanduser().resolve()
    symbol_dirs = sorted(path for path in state_root.glob("ticker=*") if path.is_dir())
    if not symbol_dirs:
        raise ValueError("security_state 没有 ticker 分区")
    industry_files = _parquet_files(config["industry_root"])
    industry = _read_files(industry_files, INDUSTRY_COLUMNS)
    corporate_files = _parquet_files(config.get("corporate_action_root"))
    corporate_action_rows = _validate_corporate_actions(corporate_files)
    rule_root = config.get("execution_rules_root")
    model_blockers = []
    execution_blockers = []
    if not bool(config.get("adjustment_history_complete")):
        model_blockers.append("adjustment_history_complete 未经数据审计确认")
    if not industry_files:
        execution_blockers.append("缺少 PIT industry snapshot")
    if not corporate_files:
        execution_blockers.append("缺少独立 PIT corporate_action 表")
    if not rule_root:
        execution_blockers.append("缺少历史 upper/lower limit 与 PIT lot/min_buy_quantity 表")
    if execution_mode == "required_for_promotion" and (model_blockers or execution_blockers):
        raise ValueError(
            "正式 Phase E 富化被阻断: "
            + "; ".join(model_blockers + execution_blockers)
        )

    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    source_files = list(industry_files) + list(corporate_files)
    rows_written = 0
    symbols_written = 0
    incomplete_symbols = []
    try:
        for state_dir in symbol_dirs:
            symbol = state_dir.name.split("=", 1)[1]
            state_files = _parquet_files(state_root, symbol)
            bar_files = _parquet_files(config["bars_root"], symbol)
            adjustment_files = _parquet_files(config["adjustment_root"], symbol)
            rule_files = _parquet_files(rule_root, symbol) if rule_root else []
            if not bar_files:
                raise ValueError(f"{symbol} 没有原始 Bar")
            states = _read_files(state_files, STATE_COLUMNS)
            bars = _read_files(bar_files, BAR_COLUMNS)
            adjustments = _read_files(adjustment_files, ADJUSTMENT_COLUMNS)
            rules = _read_files(rule_files, RULE_COLUMNS)
            symbol_industry = industry.loc[industry["symbol"].astype(str) == symbol].copy()
            enriched = _enrich_symbol(
                symbol, bars, states, adjustments, symbol_industry, rules,
                frequency=frequency, calendar_id=calendar_id,
            )
            if execution_mode == "required_for_promotion":
                _validate_formal_rows(enriched, symbol)
            else:
                try:
                    _validate_formal_rows(enriched, symbol)
                except ValueError as error:
                    if len(incomplete_symbols) < 20:
                        incomplete_symbols.append(str(error))
            years = pd.to_datetime(enriched["timestamp"], unit="ns", utc=True).dt.year
            for year, positions in years.groupby(years).groups.items():
                destination = staging / f"ticker={symbol}" / f"year={int(year)}" / "part.parquet"
                destination.parent.mkdir(parents=True, exist_ok=True)
                table = pa.Table.from_pandas(
                    enriched.loc[positions].reset_index(drop=True),
                    schema=OUTPUT_SCHEMA, preserve_index=False,
                )
                pq.write_table(table, destination, compression="zstd", row_group_size=65_536)
            source_files.extend(state_files + bar_files + adjustment_files + rule_files)
            rows_written += len(enriched)
            symbols_written += 1
        if incomplete_symbols:
            execution_blockers.append("富化行仍缺正式行业或执行规则字段")
        model_status = "BLOCKED" if model_blockers else "READY"
        execution_status = "READY" if not execution_blockers else "DEFERRED"
        if model_status == "BLOCKED":
            status = "INSUFFICIENT_EVIDENCE"
        elif execution_status == "DEFERRED":
            status = "MODEL_READY_EXECUTION_DEFERRED"
        else:
            status = "READY_FOR_PROMOTION"
        report = {
            "schema": "PHASE_E_DAILY_ENRICHED_V1",
            "status": status,
            "model_evaluation_status": model_status,
            "execution_promotion_status": execution_status,
            "execution_reference_mode": execution_mode,
            "promotion_eligible": model_status == "READY" and execution_status == "READY",
            "price_adjustment_mode": "pit_adjusted_signal_raw_execution",
            "frequency": frequency,
            "calendar_id": calendar_id,
            "rows": rows_written,
            "symbols": symbols_written,
            "model_blockers": model_blockers,
            "execution_blockers": execution_blockers,
            "incomplete_symbol_samples": incomplete_symbols,
            "source_file_count": len(set(source_files)),
            "source_fingerprint_sha256": _fingerprint(source_files),
            "corporate_action_rows": corporate_action_rows,
            "suspended_slot_policy": "previous_raw_close_ohlc_zero_volume_bar_observed_false",
            "industry_availability": "max(provider_known_at,snapshot_asof)",
        }
        (staging / "enrichment_report.json").write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output

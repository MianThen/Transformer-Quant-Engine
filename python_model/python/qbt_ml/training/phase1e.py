from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ablation import _file_hash, _run_subprocess_jobs, implementation_hash
from .walk_forward import TimestampSplit


@dataclass(frozen=True)
class Phase1EDataSpecV1:
    bars_root: str
    security_state_root: str
    selection_year: int = 2019
    minimum_selection_sessions: int = 220
    universe_size: int = 120
    source_start_year: int = 2020
    phase1b_last_timestamp: int = 1719385200000000000

    def __post_init__(self) -> None:
        if (self.selection_year >= self.source_start_year or
                self.minimum_selection_sessions < 1 or self.universe_size < 80 or
                self.phase1b_last_timestamp <= 0):
            raise ValueError("Phase 1E 数据规格无效")


def _sql_path(path: str | Path) -> str:
    value = str(Path(path).resolve())
    if "'" in value:
        raise ValueError("数据路径不能包含单引号")
    return value


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_phase1e_source(
    spec: Phase1EDataSpecV1,
    output_path: str | Path,
    audit_path: str | Path,
) -> Path:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Phase 1E 数据构建需要 DuckDB") from exc
    output = Path(output_path)
    audit = Path(audit_path)
    if output.exists() or audit.exists():
        raise FileExistsError("Phase 1E 数据与审计文件不可覆盖")
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    bars = _sql_path(spec.bars_root)
    states = _sql_path(spec.security_state_root)
    connection = duckdb.connect()
    connection.execute(f"""
        CREATE TEMP TABLE selected AS
        SELECT symbol
        FROM read_parquet(
            '{bars}/ticker=*/year={spec.selection_year}/*.parquet',
            hive_partitioning=true
        )
        GROUP BY symbol
        HAVING count(DISTINCT timestamp) >= {spec.minimum_selection_sessions}
        ORDER BY symbol
        LIMIT {spec.universe_size}
    """)
    selected = [row[0] for row in connection.execute(
        "SELECT symbol FROM selected ORDER BY symbol"
    ).fetchall()]
    if len(selected) != spec.universe_size:
        raise RuntimeError("满足预注册历史覆盖条件的股票不足")
    connection.execute(f"""
        CREATE TEMP TABLE bars AS
        SELECT timestamp, symbol, open, high, low, close, volume
        FROM read_parquet(
            '{bars}/ticker=*/year=20*/*.parquet', hive_partitioning=true
        )
        WHERE year >= {spec.source_start_year}
          AND symbol IN (SELECT symbol FROM selected)
    """)
    connection.execute(f"""
        CREATE TEMP TABLE states AS
        SELECT timestamp, symbol, is_listed, is_suspended, is_st, is_tradable,
               universe_asof, reference_data_known_at_max
        FROM read_parquet(
            '{states}/ticker=*/year=20*/*.parquet', hive_partitioning=true
        )
        WHERE year >= {spec.source_start_year}
          AND symbol IN (SELECT symbol FROM selected)
    """)
    duplicate_states = connection.execute("""
        SELECT count(*) FROM (
            SELECT timestamp, symbol FROM states
            GROUP BY timestamp, symbol HAVING count(*) != 1
        )
    """).fetchone()[0]
    missing_states = connection.execute("""
        SELECT count(*) FROM bars
        LEFT JOIN states USING (timestamp, symbol)
        WHERE states.symbol IS NULL
    """).fetchone()[0]
    pit_violations = connection.execute("""
        SELECT count(*) FROM states
        WHERE universe_asof > timestamp OR reference_data_known_at_max > timestamp
    """).fetchone()[0]
    if duplicate_states or missing_states or pit_violations:
        raise RuntimeError(
            "security_state 未通过唯一键、全覆盖或 available-at 校验"
        )
    output_sql = _sql_path(output)
    connection.execute(f"""
        COPY (
            SELECT b.timestamp, b.symbol, b.open, b.high, b.low, b.close, b.volume,
                   s.is_listed, s.is_suspended, s.is_st, s.is_tradable,
                   s.universe_asof, s.reference_data_known_at_max
            FROM bars b JOIN states s USING (timestamp, symbol)
            ORDER BY b.timestamp, b.symbol
        ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    stats = connection.execute("""
        SELECT count(*), count(DISTINCT timestamp), count(DISTINCT symbol),
               min(timestamp), max(timestamp), count_if(is_suspended), count_if(is_st)
        FROM read_parquet(?)
    """, [output_sql]).fetchone()
    new_sizes = connection.execute("""
        SELECT count(DISTINCT symbol) AS n
        FROM read_parquet(?) WHERE timestamp > ? GROUP BY timestamp
    """, [output_sql, spec.phase1b_last_timestamp]).fetchnumpy()["n"]
    if new_sizes.size < 378 or float(np.median(new_sizes)) < 80:
        raise RuntimeError("新 OOS 不满足三个窗口或 median N >= 4K")
    contract = {
        "schema_version": 1,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "source_adjustment": "none",
        "available_at_semantics": "observable_by_daily_close_for_next_open",
        "selection": {
            "selection_year": spec.selection_year,
            "minimum_sessions": spec.minimum_selection_sessions,
            "ordering": "symbol_ascending",
            "universe_size": spec.universe_size,
            "selected_symbols_sha256": _canonical_hash(selected),
        },
        "phase1b_last_timestamp": spec.phase1b_last_timestamp,
        "rows": stats[0],
        "timestamps": stats[1],
        "symbols": stats[2],
        "first_timestamp": stats[3],
        "last_timestamp": stats[4],
        "suspended_rows": stats[5],
        "st_rows": stats[6],
        "new_oos_timestamps": int(new_sizes.size),
        "new_oos_cross_section_min_median_max": [
            int(new_sizes.min()), float(np.median(new_sizes)), int(new_sizes.max())
        ],
        "pit_violations": pit_violations,
        "missing_security_state_rows": missing_states,
        "source_sha256": _file_hash(output),
        "remaining_economic_inputs": [
            "price_limits", "corporate_action_adjustment",
            "commission_and_tax", "slippage_or_reference_price_cost",
            "cpp_replay_return_series_for_cvar",
        ],
    }
    audit.write_text(
        json.dumps({**contract, "contract_sha256": _canonical_hash(contract)},
                   ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def phase1e_oos_splits(
    timestamps,
    *,
    phase1b_last_timestamp: int,
    train_size: int = 504,
    validation_size: int = 126,
    test_size: int = 126,
    purge_timestamps: int = 6,
    embargo_timestamps: int = 5,
    window_count: int = 3,
) -> tuple[TimestampSplit, ...]:
    values = np.asarray(timestamps)
    unique = np.unique(values)
    new_times = unique[unique > phase1b_last_timestamp]
    if new_times.size < test_size * window_count:
        raise ValueError("Phase 1E 新 OOS 时间不足三个窗口")
    folds = []
    for fold_index in range(window_count):
        test_times = new_times[fold_index * test_size:(fold_index + 1) * test_size]
        test_start = int(np.searchsorted(unique, test_times[0]))
        validation_end = test_start - purge_timestamps - embargo_timestamps
        validation_start = validation_end - validation_size
        train_end = validation_start - purge_timestamps
        train_start = train_end - train_size
        if train_start < 0:
            raise ValueError("Phase 1E 历史训练窗口不足")
        train_times = unique[train_start:train_end]
        validation_times = unique[validation_start:validation_end]
        folds.append(TimestampSplit(
            train=np.flatnonzero(np.isin(values, train_times)),
            validation=np.flatnonzero(np.isin(values, validation_times)),
            test=np.flatnonzero(np.isin(values, test_times)),
            train_timestamps=train_times,
            validation_timestamps=validation_times,
            test_timestamps=test_times,
        ))
    return tuple(folds)


def run_phase1e_diagnostics(config: dict, dataset_path: str | Path,
                            audit_path: str | Path, output: str | Path, train_fn) -> Path:
    settings = config.get("phase1e_diagnostics", {})
    if settings.get("enabled") is not True:
        raise RuntimeError("Phase 1E diagnostics 默认关闭")
    dataset_path = Path(dataset_path)
    audit_path = Path(audit_path)
    data_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cutoff = int(data_audit["phase1b_last_timestamp"])
    with np.load(dataset_path, allow_pickle=False) as data:
        folds = phase1e_oos_splits(
            data["timestamps"], phase1b_last_timestamp=cutoff,
            train_size=int(settings.get("train_timestamps", 504)),
            validation_size=int(settings.get("validation_timestamps", 126)),
            test_size=int(settings.get("test_timestamps", 126)),
            purge_timestamps=int(settings.get("purge_timestamps", 6)),
            embargo_timestamps=int(settings.get("embargo_timestamps", 5)),
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    code_hash = implementation_hash()
    preregistered = {
        "schema_version": 1,
        "registered_before_training": True,
        "diagnostic_only": True,
        "promotion_allowed": False,
        "dataset_sha256": _file_hash(dataset_path),
        "data_audit_sha256": _file_hash(audit_path),
        "implementation_sha256": code_hash,
        "phase1b_last_timestamp": cutoff,
        "window_count": 3,
        "cadence_steps": int(settings.get("cadence_steps", 32)),
        "sampling_seed": int(settings.get("sampling_seed", 20260801)),
        "folds": [{
            "fold": index + 1,
            "train_first": int(fold.train_timestamps[0]),
            "train_last": int(fold.train_timestamps[-1]),
            "validation_first": int(fold.validation_timestamps[0]),
            "validation_last": int(fold.validation_timestamps[-1]),
            "test_first": int(fold.test_timestamps[0]),
            "test_last": int(fold.test_timestamps[-1]),
        } for index, fold in enumerate(folds)],
    }
    preregistered_hash = _canonical_hash(preregistered)
    (output / "preregistered_contract.json").write_text(
        json.dumps({**preregistered, "contract_sha256": preregistered_hash},
                   ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    jobs = []
    for fold_index, fold in enumerate(folds):
        for mode in ("none", "diagnostics"):
            run_config = copy.deepcopy(config)
            run_config["gradient_optimization"] = {
                "mode": mode,
                "cadence_steps": int(settings.get("cadence_steps", 32)),
                "seed": int(settings.get("sampling_seed", 20260801)),
                "fold": fold_index + 1,
                "regime": "ALL",
            } if mode == "diagnostics" else {"mode": "none"}
            jobs.append((
                train_fn, run_config, str(dataset_path),
                str(output / f"fold-{fold_index + 1}" / mode), fold,
            ))
    _run_subprocess_jobs(
        jobs, output / "jobs", int(settings.get("parallel_workers", 6)), code_hash
    )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Phase 1E 结果校验需要 PyTorch") from exc
    fold_reports = []
    for fold_index, fold in enumerate(folds):
        baseline_path = output / f"fold-{fold_index + 1}" / "none"
        diagnostic_path = output / f"fold-{fold_index + 1}" / "diagnostics"
        baseline = torch.load(
            baseline_path / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        diagnostic = torch.load(
            diagnostic_path / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        parity = baseline["history"] == diagnostic["history"] and all(
            torch.equal(baseline["model_state_dict"][name], value)
            for name, value in diagnostic["model_state_dict"].items()
        )
        artifact = json.loads((
            diagnostic_path / "gradient_conflict_artifact.json"
        ).read_text(encoding="utf-8"))
        if not parity:
            raise RuntimeError("真实训练中 diagnostics 改变了 baseline 数值")
        fold_reports.append({
            "fold": fold_index + 1,
            "baseline_diagnostic_exact_parity": parity,
            "gradient_report_sha256": artifact["report_sha256"],
            "sample_count": len(artifact["samples"]),
            "split": {
                "train_first": int(fold.train_timestamps[0]),
                "train_last": int(fold.train_timestamps[-1]),
                "validation_first": int(fold.validation_timestamps[0]),
                "validation_last": int(fold.validation_timestamps[-1]),
                "test_first": int(fold.test_timestamps[0]),
                "test_last": int(fold.test_timestamps[-1]),
            },
        })
    contract = {
        "schema_version": 1,
        "status": "diagnostics_only_no_promotion",
        "dataset_sha256": _file_hash(dataset_path),
        "data_audit_sha256": _file_hash(audit_path),
        "implementation_sha256": code_hash,
        "preregistered_contract_sha256": preregistered_hash,
        "phase1b_last_timestamp": cutoff,
        "window_count": 3,
        "cadence_steps": int(settings.get("cadence_steps", 32)),
        "sampling_seed": int(settings.get("sampling_seed", 20260801)),
        "folds": fold_reports,
    }
    (output / "diagnostic_report.json").write_text(
        json.dumps({**contract, "report_sha256": _canonical_hash(contract)},
                   ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output

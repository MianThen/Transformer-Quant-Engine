from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet


def _load_audit_module():
    path = Path(__file__).parents[2] / "tools" / "audit_phase3a_exposure_source.py"
    spec = importlib.util.spec_from_file_location("phase3a_source_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_parquet(path: Path, with_exposure: bool) -> None:
    columns = {
        "timestamp": pa.array([1, 2], type=pa.int64()),
        "symbol": pa.array(["000001", "000001"]),
        "universe_asof": pa.array([1, 2], type=pa.int64()),
        "reference_data_known_at_max": pa.array([1, 2], type=pa.int64()),
    }
    if with_exposure:
        columns.update({
            "industry_id": pa.array([10, 10], type=pa.int64()),
            "beta": pa.array([0.8, 0.9], type=pa.float64()),
            "momentum": pa.array([0.1, 0.2], type=pa.float64()),
            "residual_volatility": pa.array([0.2, 0.3], type=pa.float64()),
            "liquidity": pa.array([1.0, 1.1], type=pa.float64()),
            "size": pa.array([9.0, 9.1], type=pa.float64()),
            "available_at": pa.array([1, 2], type=pa.int64()),
        })
    table = pa.table(columns)
    table = table.replace_schema_metadata({
        b"qbt.schema": b"TEST_PIT_EXPOSURE_V1",
        b"qbt.asof_semantics": b"available_at",
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, path)


def test_baostock_like_state_schema_fails_closed(tmp_path: Path) -> None:
    module = _load_audit_module()
    _write_parquet(tmp_path / "state.parquet", with_exposure=False)
    report = module.audit(tmp_path, 0)
    assert report["status"] == "SOURCE_UNAVAILABLE"
    assert report["phase_exit_eligible"] is False
    assert "NO_INDUSTRY_EXPOSURE_FIELDS" in report["failure_reasons"]
    assert "NO_STYLE_EXPOSURE_FIELDS" in report["failure_reasons"]
    assert "NO_EXPLICIT_EXPOSURE_AVAILABLE_AT_FIELDS" in report["failure_reasons"]


def test_complete_pit_exposure_schema_is_compatible(tmp_path: Path) -> None:
    module = _load_audit_module()
    _write_parquet(tmp_path / "exposure.parquet", with_exposure=True)
    report = module.audit(tmp_path, 0)
    assert report["status"] == "SOURCE_COMPATIBLE"
    assert report["source_schema_compatible"] is True
    assert report["source_schema_hash"]
    payload = dict(report)
    claimed = payload.pop("report_sha256")
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert claimed == hashlib.sha256(encoded).hexdigest()

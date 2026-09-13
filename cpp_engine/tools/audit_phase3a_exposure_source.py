#!/usr/bin/env python3
"""Audit whether a parquet source can satisfy the Phase 3A PIT exposure contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as parquet
except ImportError as exc:
    raise SystemExit("pyarrow is required to audit parquet schemas") from exc


BASE_FIELDS = {
    "timestamp",
    "symbol",
    "universe_asof",
    "reference_data_known_at_max",
}
INDUSTRY_FIELDS = {
    "industry",
    "industry_id",
    "pit_industry_id",
    "industry_code",
}
STYLE_FIELDS = {
    "beta",
    "momentum",
    "residual_volatility",
    "liquidity",
    "size",
}
PIT_FIELDS = {"available_at", "decision_at", "asof", "as_of"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("report_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def metadata_map(file: parquet.ParquetFile) -> dict[str, str]:
    metadata = file.schema_arrow.metadata or {}
    return {
        key.decode(errors="replace"): value.decode(errors="replace")
        for key, value in metadata.items()
    }


def audit(root: Path, max_files: int) -> dict[str, Any]:
    paths = sorted(root.rglob("*.parquet"))
    discovered_count = len(paths)
    if max_files > 0:
        paths = paths[:max_files]
    schema_counts: dict[str, int] = {}
    metadata_values: dict[str, set[str]] = {}
    union_fields: set[str] = set()
    scanned = 0
    failures: list[str] = []
    for path in paths:
        try:
            file = parquet.ParquetFile(path)
            fields = set(file.schema_arrow.names)
            metadata = metadata_map(file)
        except Exception as exc:
            failures.append(f"{path.relative_to(root)}:{type(exc).__name__}:{exc}")
            continue
        scanned += 1
        union_fields.update(fields)
        schema_key = ",".join(sorted(fields))
        schema_counts[schema_key] = schema_counts.get(schema_key, 0) + 1
        for key in ("qbt.schema", "qbt.asof_semantics", "qbt.timezone"):
            if key in metadata:
                metadata_values.setdefault(key, set()).add(metadata[key])

    industry_fields = sorted(
        field for field in union_fields
        if field in INDUSTRY_FIELDS or field.startswith("industry_")
    )
    style_fields = sorted(
        field for field in union_fields
        if field in STYLE_FIELDS or field.startswith("style_")
    )
    pit_fields = sorted(
        field for field in union_fields
        if field in PIT_FIELDS or field.startswith("available_at_")
    )
    base_fields = sorted(union_fields & BASE_FIELDS)
    reasons = list(failures[:20])
    if not paths:
        reasons.append("NO_PARQUET_FILES")
    if not industry_fields:
        reasons.append("NO_INDUSTRY_EXPOSURE_FIELDS")
    if not style_fields:
        reasons.append("NO_STYLE_EXPOSURE_FIELDS")
    if not pit_fields:
        reasons.append("NO_EXPLICIT_EXPOSURE_AVAILABLE_AT_FIELDS")
    if not BASE_FIELDS.issubset(union_fields):
        reasons.append("BASE_PIT_FIELDS_INCOMPLETE")
    compatible = not reasons
    normalized_metadata = {
        key: sorted(values) for key, values in sorted(metadata_values.items())
    }
    source_schema_hash = canonical_hash({
        "schema_variant_counts": dict(sorted(schema_counts.items())),
        "metadata_values": normalized_metadata,
    })
    report: dict[str, Any] = {
        "schema_version": 1,
        "role": "phase3a_pit_exposure_source_audit_v1",
        "status": "SOURCE_COMPATIBLE" if compatible else "SOURCE_UNAVAILABLE",
        "source_root": str(root.resolve()),
        "scan_mode": "ALL_FILES" if max_files <= 0 else "PREFIX_SAMPLE",
        "file_count_discovered": discovered_count,
        "file_count_scanned": scanned,
        "max_files": max_files,
        "schema_variant_count": len(schema_counts),
        "schema_variant_counts": dict(sorted(schema_counts.items())),
        "metadata_values": normalized_metadata,
        "source_schema_hash": source_schema_hash,
        "base_fields_present": base_fields,
        "industry_exposure_fields_present": industry_fields,
        "style_exposure_fields_present": style_fields,
        "pit_exposure_availability_fields_present": pit_fields,
        "point_in_time_exposure_source": compatible,
        "source_schema_compatible": compatible,
        "failure_reasons": reasons,
        "phase_exit_eligible": False,
        "promotion_eligible": False,
        "evidence_level": "SOURCE_AUDIT_ONLY",
    }
    report["report_sha256"] = canonical_hash(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Users/Zhuanz/PycharmProjects/scrapy/data/security_state/provider=baostock"),
    )
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/phase3a-exposure-source-audit/phase3a_exposure_source_audit.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"source root does not exist: {root}")
    report = audit(root, args.max_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    file_hash = sha256_bytes(encoded.encode())
    args.output.with_name(args.output.name + ".sha256").write_text(
        f"{file_hash}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "status": report["status"],
        "file_count_scanned": report["file_count_scanned"],
        "schema_variant_count": report["schema_variant_count"],
        "failure_reasons": report["failure_reasons"],
        "report_sha256": report["report_sha256"],
        "file_sha256": file_hash,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

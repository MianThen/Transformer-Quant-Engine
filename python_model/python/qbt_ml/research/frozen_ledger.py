"""Read-only bridge for C++ Return Analysis V1 ledger artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any


class FrozenLedgerValidationError(ValueError):
    """Raised when a C++ ledger artifact is not immutable and self-consistent."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _reject_constant(token: str) -> None:
    raise ValueError(token)


@dataclass(frozen=True)
class FrozenCppLedger:
    source_path: Path
    schema_version: int
    ledger_sha256: str
    records: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "FrozenCppLedger":
        source_path = Path(path)
        try:
            value = json.loads(source_path.read_text(encoding="utf-8"),
                               parse_constant=_reject_constant)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise FrozenLedgerValidationError("无法读取有效的 C++ ledger JSON") from exc
        if (not isinstance(value, dict) or isinstance(value.get("schema_version"), bool)
                or value.get("schema_version") != 1):
            raise FrozenLedgerValidationError("C++ ledger schema_version 必须为 1")
        records = value.get("records")
        if not isinstance(records, list):
            raise FrozenLedgerValidationError("C++ ledger 缺少 records 数组")
        digest = value.get("ledger_sha256")
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(char not in "0123456789abcdef" for char in digest)):
            raise FrozenLedgerValidationError("C++ ledger_sha256 格式无效")
        if hashlib.sha256(_canonical_json(records)).hexdigest() != digest:
            raise FrozenLedgerValidationError("C++ ledger_sha256 校验失败")

        previous: tuple[int, int] | None = None
        frozen: list[Mapping[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                raise FrozenLedgerValidationError("ledger record 必须是对象")
            start, end = record.get("period_start"), record.get("period_end")
            if (not isinstance(start, int) or isinstance(start, bool) or start <= 0 or
                    not isinstance(end, int) or isinstance(end, bool) or end <= start):
                raise FrozenLedgerValidationError("ledger period 边界无效")
            if previous is not None:
                if (start, end) <= previous:
                    raise FrozenLedgerValidationError("ledger period 必须严格递增")
                if start < previous[1]:
                    raise FrozenLedgerValidationError("ledger period 不得重叠")
            previous = (start, end)
            for key in ("period_return", "starting_equity", "ending_equity"):
                if not _finite_number(record.get(key)):
                    raise FrozenLedgerValidationError(f"ledger 缺少有限数值字段: {key}")
            if record["starting_equity"] <= 0 or record["ending_equity"] < 0:
                raise FrozenLedgerValidationError("ledger equity 必须满足非负/正起始约束")
            if record["period_return"] <= -1:
                raise FrozenLedgerValidationError("ledger period_return 不得低于 -100%")
            frozen.append(MappingProxyType(dict(record)))
        manifest = value.get("manifest", {})
        if not isinstance(manifest, dict):
            raise FrozenLedgerValidationError("manifest 必须是对象")
        for key in ("promotion_eligible", "benchmark_available"):
            if key in manifest and not isinstance(manifest[key], bool):
                raise FrozenLedgerValidationError(f"manifest {key} 必须是布尔值")
        limitations = manifest.get("limitations")
        if limitations is not None and (not isinstance(limitations, list) or
                                         any(not isinstance(item, str) for item in limitations)):
            raise FrozenLedgerValidationError("manifest limitations 必须是字符串数组")
        return cls(source_path, 1, digest, tuple(frozen), MappingProxyType(dict(manifest)))

    def period_returns(self) -> tuple[float, ...]:
        return tuple(float(record["period_return"]) for record in self.records)

    def benchmark_returns(self) -> tuple[float, ...] | None:
        values = tuple(record.get("benchmark_return") for record in self.records)
        if not values or any(value is None for value in values):
            return None
        if any(not _finite_number(value) for value in values):
            raise FrozenLedgerValidationError("benchmark_return 必须是有限数值")
        return tuple(float(value) for value in values)

    @property
    def promotion_eligible(self) -> bool:
        if self.manifest.get("reference_price_quality") in {"PROXY", "ARRIVAL_PROXY"}:
            return False
        return (self.manifest.get("promotion_eligible", False) is True and
                self.manifest.get("benchmark_available", False) is True)

    @property
    def limitations(self) -> tuple[str, ...]:
        values = self.manifest.get("limitations", ())
        return tuple(values) if isinstance(values, list) and all(
            isinstance(value, str) for value in values) else ("INVALID_LIMITATIONS_FIELD",)


@dataclass(frozen=True)
class FrozenReturnAnalysisReport:
    source_path: Path
    schema_version: int
    report_sha256: str
    ledger_hash: int
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "FrozenReturnAnalysisReport":
        source_path = Path(path)
        try:
            raw = source_path.read_text(encoding="utf-8")
            value = json.loads(raw, parse_constant=_reject_constant)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise FrozenLedgerValidationError("无法读取有效的 C++ Return Analysis JSON") from exc
        if (not isinstance(value, dict) or value.get("schema_version") != 1 or
                value.get("role") != "return_analysis_v1"):
            raise FrozenLedgerValidationError("Return Analysis schema 不受支持")
        digest = value.get("report_sha256")
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(char not in "0123456789abcdef" for char in digest)):
            raise FrozenLedgerValidationError("report_sha256 格式无效")
        marker = ',"report_sha256":"'
        marker_start = raw.rfind(marker)
        if marker_start < 0 or not raw.endswith("}\n") and not raw.endswith("}"):
            raise FrozenLedgerValidationError("Return Analysis report hash 尾部无效")
        unsigned = raw[:marker_start] + "}"
        if hashlib.sha256(unsigned.encode("utf-8")).hexdigest() != digest:
            raise FrozenLedgerValidationError("Return Analysis report SHA-256 校验失败")
        manifest = value.get("manifest")
        metrics = value.get("metrics")
        if not isinstance(manifest, dict) or not isinstance(metrics, dict):
            raise FrozenLedgerValidationError("Return Analysis 缺少 manifest/metrics")
        ledger_hash = value.get("ledger_hash")
        if not isinstance(ledger_hash, int) or isinstance(ledger_hash, bool) or ledger_hash == 0:
            raise FrozenLedgerValidationError("Return Analysis ledger_hash 无效")
        observations = metrics.get("observations")
        if (not isinstance(observations, int) or isinstance(observations, bool) or
                observations < 0):
            raise FrozenLedgerValidationError("Return Analysis observations 无效")
        for key, number in metrics.items():
            if key in {"status", "observations", "maximum_drawdown_duration", "artifact_hash"}:
                continue
            if number is not None and not _finite_number(number):
                raise FrozenLedgerValidationError(f"Return Analysis 指标不是有限数值: {key}")
        promotion = manifest.get("promotion_eligible", False)
        quality = manifest.get("reference_price_quality", "")
        if (promotion is True and
                (not isinstance(quality, str) or quality in
                 {"", "UNKNOWN", "UNAVAILABLE", "PROXY", "ARRIVAL_PROXY"})):
            raise FrozenLedgerValidationError("代理或缺失 reference price 不得晋级")
        return cls(source_path, 1, digest, ledger_hash,
                   MappingProxyType(dict(manifest)), MappingProxyType(dict(metrics)))

    @property
    def promotion_eligible(self) -> bool:
        return self.manifest.get("promotion_eligible", False) is True

    @property
    def limitations(self) -> tuple[str, ...]:
        values = self.manifest.get("limitations", ())
        return tuple(values) if isinstance(values, list) and all(
            isinstance(value, str) for value in values) else ("INVALID_LIMITATIONS_FIELD",)


@dataclass(frozen=True)
class FrozenDriftSnapshot:
    """Read-only validator for a C++ DriftSnapshotContract V0 artifact."""

    source_path: Path
    schema_version: int
    report_sha256: str
    labels_mature: bool
    available_at_utc: str
    hashes: Mapping[str, str]

    @classmethod
    def load(cls, path: str | Path) -> "FrozenDriftSnapshot":
        source_path = Path(path)
        try:
            raw = source_path.read_text(encoding="utf-8")
            value = json.loads(raw, parse_constant=_reject_constant)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise FrozenLedgerValidationError("无法读取有效的 Drift Snapshot JSON") from exc
        if (not isinstance(value, dict) or value.get("schema_version") != 0 or
                value.get("role") != "drift_snapshot_v0"):
            raise FrozenLedgerValidationError("Drift Snapshot schema 不受支持")
        labels_mature = value.get("labels_mature")
        if not isinstance(labels_mature, bool):
            raise FrozenLedgerValidationError("Drift Snapshot labels_mature 必须是布尔值")
        available_at_utc = value.get("available_at_utc")
        if not isinstance(available_at_utc, str) or not available_at_utc.endswith("Z") or "T" not in available_at_utc:
            raise FrozenLedgerValidationError("Drift Snapshot available_at_utc 必须是 UTC 时间")
        try:
            parsed_available_at = datetime.fromisoformat(
                available_at_utc[:-1] + "+00:00")
        except ValueError as exc:
            raise FrozenLedgerValidationError("Drift Snapshot available_at_utc 格式无效") from exc
        if parsed_available_at.tzinfo != timezone.utc:
            raise FrozenLedgerValidationError("Drift Snapshot available_at_utc 必须带 UTC 时区")

        hash_fields = (
            "model_manifest_sha256", "raw_schema_hash",
            "preprocessing_spec_sha256", "feature_schema_hash",
            "prediction_schema_hash", "raw_fields_sha256",
            "preprocessed_features_sha256", "prediction_values_sha256",
            "embedding_values_sha256", "source_snapshot_set_sha256",
            "ledger_schema_hash",
        )
        optional_hash_fields = ("label_spec_sha256", "matured_labels_sha256")
        hashes: dict[str, str] = {}
        for key in hash_fields:
            digest = value.get(key)
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(char not in "0123456789abcdef" for char in digest)):
                raise FrozenLedgerValidationError(f"Drift Snapshot {key} 格式无效")
            hashes[key] = digest
        for key in optional_hash_fields:
            digest = value.get(key, "")
            if digest == "" and not labels_mature:
                hashes[key] = digest
                continue
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(char not in "0123456789abcdef" for char in digest)):
                raise FrozenLedgerValidationError(f"Drift Snapshot {key} 格式无效")
            hashes[key] = digest

        digest = value.get("report_sha256")
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(char not in "0123456789abcdef" for char in digest)):
            raise FrozenLedgerValidationError("Drift Snapshot report_sha256 格式无效")
        marker = ',"report_sha256":"'
        marker_start = raw.rfind(marker)
        if (marker_start < 0 or not raw.endswith("}") and not raw.endswith("}\n")):
            raise FrozenLedgerValidationError("Drift Snapshot report hash 尾部无效")
        unsigned = raw[:marker_start] + "}"
        if hashlib.sha256(unsigned.encode("utf-8")).hexdigest() != digest:
            raise FrozenLedgerValidationError("Drift Snapshot report SHA-256 校验失败")
        return cls(source_path, 0, digest, labels_mature, available_at_utc,
                   MappingProxyType(dict(hashes)))

    def hash(self, name: str) -> str:
        try:
            return self.hashes[name]
        except KeyError as exc:
            raise KeyError(f"unknown Drift Snapshot hash: {name}") from exc

    @property
    def ledger_schema_hash(self) -> str:
        return self.hashes["ledger_schema_hash"]

    @property
    def promotion_eligible(self) -> bool:
        return False

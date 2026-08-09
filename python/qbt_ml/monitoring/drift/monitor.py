"""Deterministic, observation-only drift statistics for frozen snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any


class DriftArtifactValidationError(ValueError):
    """Raised when a drift specification or artifact is not self-consistent."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_like(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _valid_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z") or "T" not in value:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo == timezone.utc


def _utc_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _finite_values(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values if _finite(value))


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _median(values: Sequence[float]) -> float | None:
    return _quantile(values, 0.5)


def _moment_summary(values: Sequence[Any], quantiles: Sequence[float]) -> dict[str, Any]:
    total = len(values)
    finite_values = _finite_values(values)
    missing = total - len(finite_values)
    mean = sum(finite_values) / len(finite_values) if finite_values else None
    variance = (
        sum((value - mean) ** 2 for value in finite_values) / len(finite_values)
        if finite_values
        else None
    )
    median = _median(finite_values)
    mad = (
        _median(tuple(abs(value - median) for value in finite_values))
        if median is not None
        else None
    )
    return {
        "count": total,
        "finite_count": len(finite_values),
        "missing_count": missing,
        "missing_rate": missing / total if total else None,
        "mean": mean,
        "variance": variance,
        "median": median,
        "mad": mad,
        "quantiles": {
            str(probability): _quantile(finite_values, probability)
            for probability in quantiles
        },
    }


def _bucket_counts(values: Sequence[Any], edges: Sequence[float]) -> list[int]:
    counts = [0] * (len(edges) + 2)
    for value in values:
        if not _finite(value):
            counts[0] += 1
            continue
        numeric = float(value)
        if numeric < edges[0]:
            counts[1] += 1
            continue
        bucket = len(edges) + 1
        for index in range(len(edges) - 1):
            if numeric < edges[index + 1]:
                bucket = index + 2
                break
        counts[bucket] += 1
    return counts


def _psi(reference_counts: Sequence[int], current_counts: Sequence[int], floor: float) -> float | None:
    reference_total = sum(reference_counts)
    current_total = sum(current_counts)
    if reference_total == 0 or current_total == 0:
        return None
    reference_shares = [
        max(reference_count / reference_total, floor)
        for reference_count in reference_counts
    ]
    current_shares = [
        max(current_count / current_total, floor)
        for current_count in current_counts
    ]
    reference_normalizer = sum(reference_shares)
    current_normalizer = sum(current_shares)
    value = 0.0
    for reference_share, current_share in zip(
        (share / reference_normalizer for share in reference_shares),
        (share / current_normalizer for share in current_shares),
    ):
        value += (current_share - reference_share) * math.log(
            current_share / reference_share
        )
    return value


def _ks_d(reference: Sequence[Any], current: Sequence[Any]) -> float | None:
    reference_values = sorted(_finite_values(reference))
    current_values = sorted(_finite_values(current))
    if not reference_values or not current_values:
        return None
    points = sorted(set(reference_values + current_values))
    reference_index = 0
    current_index = 0
    maximum = 0.0
    for point in points:
        while reference_index < len(reference_values) and reference_values[reference_index] <= point:
            reference_index += 1
        while current_index < len(current_values) and current_values[current_index] <= point:
            current_index += 1
        maximum = max(
            maximum,
            abs(
                reference_index / len(reference_values)
                - current_index / len(current_values)
            ),
        )
    return maximum


def compare_scalar_drift(
    reference: Sequence[Any],
    current: Sequence[Any],
    *,
    bin_edges: Sequence[float],
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    psi_probability_floor: float = 1e-12,
) -> dict[str, Any]:
    edges = tuple(float(edge) for edge in bin_edges)
    if len(edges) < 2 or any(not math.isfinite(edge) for edge in edges):
        raise DriftArtifactValidationError("PSI bin_edges 必须至少包含两个有限值")
    if any(left >= right for left, right in zip(edges, edges[1:])):
        raise DriftArtifactValidationError("PSI bin_edges 必须严格递增")
    if any(probability < 0.0 or probability > 1.0 for probability in quantiles):
        raise DriftArtifactValidationError("quantiles 必须位于 [0, 1]")
    if not math.isfinite(psi_probability_floor) or not 0.0 < psi_probability_floor < 1.0:
        raise DriftArtifactValidationError("PSI probability floor 必须位于 (0, 1)")
    reference_counts = _bucket_counts(reference, edges)
    current_counts = _bucket_counts(current, edges)
    return {
        "reference": _moment_summary(reference, quantiles),
        "current": _moment_summary(current, quantiles),
        "reference_bucket_counts": reference_counts,
        "current_bucket_counts": current_counts,
        "psi": _psi(reference_counts, current_counts, psi_probability_floor),
        "ks_d": _ks_d(reference, current),
        "status": "OK" if len(reference) > 0 and len(current) > 0 else "INSUFFICIENT_DATA",
    }


@dataclass(frozen=True)
class DriftMonitorSpecV1:
    schema_version: int = 1
    reference_mode: str = "training_static"
    frequency: str = ""
    calendar_id: str = ""
    reference_window_id: str = ""
    current_window_id: str = ""
    psi_bin_edges: tuple[float, ...] = ()
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    psi_probability_floor: float = 1e-12
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 1
    mean_block_length: float = 10.0
    fdr_family: str = ""
    fast_window_sessions: int = 1
    confirm_window_sessions: int = 3
    persistence_required: int = 2
    prediction_top_k: int = 20
    psi_warn_threshold: float | None = None
    psi_critical_threshold: float | None = None
    ks_warn_threshold: float | None = None
    ks_critical_threshold: float | None = None
    embedding_checkpoint_id: str = ""
    embedding_layer: str = ""
    embedding_pooling: str = ""
    embedding_dimension: int = 0
    embedding_anchor_sha256: str = ""
    embedding_mmd_bandwidth: float = 1.0
    diagnostic_only: bool = True
    available_at_utc: str = ""

    def validate(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise DriftArtifactValidationError("DriftMonitorSpec schema_version 必须为 1")
        if self.reference_mode not in {"training_static", "rolling_recent"}:
            raise DriftArtifactValidationError("reference_mode 不受支持")
        if not all(isinstance(value, str) and value for value in (
            self.frequency,
            self.calendar_id,
            self.reference_window_id,
            self.current_window_id,
            self.fdr_family,
            self.available_at_utc,
        )):
            raise DriftArtifactValidationError("DriftMonitorSpec 的窗口和日历字段不能为空")
        if not _valid_utc_timestamp(self.available_at_utc):
            raise DriftArtifactValidationError("DriftMonitorSpec available_at_utc 无效")
        if len(self.psi_bin_edges) < 2 or any(
            not math.isfinite(edge) for edge in self.psi_bin_edges
        ) or any(left >= right for left, right in zip(self.psi_bin_edges, self.psi_bin_edges[1:])):
            raise DriftArtifactValidationError("psi_bin_edges 必须严格递增且有限")
        if not self.quantiles or any(
            not math.isfinite(probability) or probability < 0.0 or probability > 1.0
            for probability in self.quantiles
        ) or any(left >= right for left, right in zip(self.quantiles, self.quantiles[1:])):
            raise DriftArtifactValidationError("quantiles 必须严格递增且位于 [0, 1]")
        if not 0.0 < self.psi_probability_floor < 1.0:
            raise DriftArtifactValidationError("psi_probability_floor 必须位于 (0, 1)")
        if self.bootstrap_replicates <= 0 or self.bootstrap_seed <= 0:
            raise DriftArtifactValidationError("bootstrap replicates/seed 必须为正")
        if not math.isfinite(self.mean_block_length) or self.mean_block_length < 1.0:
            raise DriftArtifactValidationError("mean_block_length 必须不小于 1")
        if self.fast_window_sessions <= 0 or self.confirm_window_sessions < self.fast_window_sessions:
            raise DriftArtifactValidationError("fast/confirm window 顺序无效")
        if self.persistence_required <= 0:
            raise DriftArtifactValidationError("persistence_required 必须为正")
        if isinstance(self.prediction_top_k, bool) or self.prediction_top_k <= 0:
            raise DriftArtifactValidationError("prediction_top_k 必须为正")
        thresholds = (
            self.psi_warn_threshold,
            self.psi_critical_threshold,
            self.ks_warn_threshold,
            self.ks_critical_threshold,
        )
        if any(value is not None and (not math.isfinite(value) or value < 0.0) for value in thresholds):
            raise DriftArtifactValidationError("漂移阈值必须为非负有限值")
        if (self.psi_warn_threshold is not None and self.psi_critical_threshold is not None and
                self.psi_warn_threshold > self.psi_critical_threshold):
            raise DriftArtifactValidationError("PSI warn threshold 不得高于 critical")
        if (self.ks_warn_threshold is not None and self.ks_critical_threshold is not None and
                self.ks_warn_threshold > self.ks_critical_threshold):
            raise DriftArtifactValidationError("KS warn threshold 不得高于 critical")
        if self.embedding_dimension < 0:
            raise DriftArtifactValidationError("embedding_dimension 不得为负")
        if self.embedding_dimension > 0 and not all(isinstance(value, str) and value for value in (
            self.embedding_checkpoint_id,
            self.embedding_layer,
            self.embedding_pooling,
        )):
            raise DriftArtifactValidationError("embedding 元数据不完整")
        if self.embedding_dimension > 0 and not _digest_like(self.embedding_anchor_sha256):
            raise DriftArtifactValidationError("embedding_anchor_sha256 格式无效")
        if not math.isfinite(self.embedding_mmd_bandwidth) or self.embedding_mmd_bandwidth <= 0.0:
            raise DriftArtifactValidationError("embedding_mmd_bandwidth 必须为正")
        if not isinstance(self.diagnostic_only, bool) or not self.diagnostic_only:
            raise DriftArtifactValidationError("Phase 1D V1 embedding 必须保持 diagnostic_only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_mode": self.reference_mode,
            "frequency": self.frequency,
            "calendar_id": self.calendar_id,
            "reference_window_id": self.reference_window_id,
            "current_window_id": self.current_window_id,
            "psi_bin_edges": list(self.psi_bin_edges),
            "quantiles": list(self.quantiles),
            "psi_probability_floor": self.psi_probability_floor,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "mean_block_length": self.mean_block_length,
            "fdr_family": self.fdr_family,
            "fast_window_sessions": self.fast_window_sessions,
            "confirm_window_sessions": self.confirm_window_sessions,
            "persistence_required": self.persistence_required,
            "prediction_top_k": self.prediction_top_k,
            "psi_warn_threshold": self.psi_warn_threshold,
            "psi_critical_threshold": self.psi_critical_threshold,
            "ks_warn_threshold": self.ks_warn_threshold,
            "ks_critical_threshold": self.ks_critical_threshold,
            "embedding_checkpoint_id": self.embedding_checkpoint_id,
            "embedding_layer": self.embedding_layer,
            "embedding_pooling": self.embedding_pooling,
            "embedding_dimension": self.embedding_dimension,
            "embedding_anchor_sha256": self.embedding_anchor_sha256,
            "embedding_mmd_bandwidth": self.embedding_mmd_bandwidth,
            "diagnostic_only": self.diagnostic_only,
            "available_at_utc": self.available_at_utc,
        }

    @property
    def spec_sha256(self) -> str:
        self.validate()
        return _sha256_text(_canonical_json(self.to_dict()))


@dataclass(frozen=True)
class RawDataQualityWindow:
    window_id: str
    schema_hash: str
    required_fields: tuple[str, ...]
    observed_fields: tuple[str, ...]
    row_count: int
    expected_row_count: int
    stale_row_count: int
    duplicate_row_count: int
    invalid_timestamp_count: int
    universe: tuple[str, ...]
    corporate_action_state: str
    adjustment_state: str
    available_at_utc: str

    def validate(self) -> None:
        if not self.window_id or not _digest_like(self.schema_hash):
            raise DriftArtifactValidationError("RawDataQualityWindow id/schema hash 无效")
        for fields in (self.required_fields, self.observed_fields):
            if any(not isinstance(field, str) or not field for field in fields):
                raise DriftArtifactValidationError("RawDataQualityWindow 字段名无效")
            if len(fields) != len(set(fields)):
                raise DriftArtifactValidationError("RawDataQualityWindow 字段名不得重复")
        if any(not isinstance(symbol, str) or not symbol for symbol in self.universe):
            raise DriftArtifactValidationError("RawDataQualityWindow universe 无效")
        if len(self.universe) != len(set(self.universe)):
            raise DriftArtifactValidationError("RawDataQualityWindow universe 不得重复")
        counts = (
            self.row_count,
            self.expected_row_count,
            self.stale_row_count,
            self.duplicate_row_count,
            self.invalid_timestamp_count,
        )
        if any(isinstance(count, bool) or count < 0 for count in counts):
            raise DriftArtifactValidationError("RawDataQualityWindow 行计数不得为负")
        if any(count > self.row_count for count in counts[2:]):
            raise DriftArtifactValidationError("RawDataQualityWindow 行计数超过 row_count")
        if self.corporate_action_state not in {"KNOWN", "UNKNOWN", "UNAVAILABLE", "PROXY"}:
            raise DriftArtifactValidationError("corporate_action_state 不受支持")
        if self.adjustment_state not in {"KNOWN", "UNKNOWN", "UNAVAILABLE", "PROXY"}:
            raise DriftArtifactValidationError("adjustment_state 不受支持")
        if not _valid_utc_timestamp(self.available_at_utc):
            raise DriftArtifactValidationError("RawDataQualityWindow available_at_utc 无效")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "window_id": self.window_id,
            "schema_hash": self.schema_hash,
            "required_fields": list(self.required_fields),
            "observed_fields": list(self.observed_fields),
            "row_count": self.row_count,
            "expected_row_count": self.expected_row_count,
            "stale_row_count": self.stale_row_count,
            "duplicate_row_count": self.duplicate_row_count,
            "invalid_timestamp_count": self.invalid_timestamp_count,
            "universe": list(self.universe),
            "corporate_action_state": self.corporate_action_state,
            "adjustment_state": self.adjustment_state,
            "available_at_utc": self.available_at_utc,
        }


def compare_raw_data_quality(
    reference: RawDataQualityWindow,
    current: RawDataQualityWindow,
) -> dict[str, Any]:
    reference.validate()
    current.validate()
    reference_fields = set(reference.required_fields)
    current_fields = set(current.observed_fields)
    reference_universe = set(reference.universe)
    current_universe = set(current.universe)
    union = reference_universe | current_universe
    overlap = reference_universe & current_universe
    missing_reference = sorted(reference_fields - set(reference.observed_fields))
    missing_current = sorted(reference_fields - current_fields)
    hard_failures: list[str] = []
    warnings: list[str] = []
    if reference.schema_hash != current.schema_hash:
        hard_failures.append("SCHEMA_HASH_MISMATCH")
    if missing_reference:
        hard_failures.append("REFERENCE_FIELDS_MISSING")
    if missing_current:
        hard_failures.append("CURRENT_FIELDS_MISSING")
    if current.row_count == 0:
        hard_failures.append("CURRENT_EMPTY")
    if current.invalid_timestamp_count:
        hard_failures.append("INVALID_TIMESTAMP")
    if current.duplicate_row_count:
        hard_failures.append("DUPLICATE_ROWS")
    if current.stale_row_count:
        warnings.append("STALE_ROWS")
    if current.corporate_action_state in {"UNKNOWN", "UNAVAILABLE", "PROXY"}:
        warnings.append("CORPORATE_ACTION_PROVENANCE_INCOMPLETE")
    if current.adjustment_state in {"UNKNOWN", "UNAVAILABLE", "PROXY"}:
        warnings.append("ADJUSTMENT_PROVENANCE_INCOMPLETE")
    if _utc_datetime(current.available_at_utc) < _utc_datetime(reference.available_at_utc):
        hard_failures.append("AVAILABLE_AT_REGRESSION")
    if hard_failures:
        status = "HARD_FAILURE"
    elif warnings:
        status = "WARN"
    else:
        status = "OK"
    return {
        "status": status,
        "hard_failures": sorted(set(hard_failures)),
        "warnings": sorted(set(warnings)),
        "reference_window_id": reference.window_id,
        "current_window_id": current.window_id,
        "schema_compatible": reference.schema_hash == current.schema_hash,
        "missing_fields": {
            "reference": missing_reference,
            "current": missing_current,
        },
        "coverage": {
            "reference": reference.row_count / reference.expected_row_count
            if reference.expected_row_count else None,
            "current": current.row_count / current.expected_row_count
            if current.expected_row_count else None,
        },
        "stale_rate": current.stale_row_count / current.row_count
        if current.row_count else None,
        "duplicate_rate": current.duplicate_row_count / current.row_count
        if current.row_count else None,
        "invalid_timestamp_rate": current.invalid_timestamp_count / current.row_count
        if current.row_count else None,
        "universe": {
            "reference_count": len(reference_universe),
            "current_count": len(current_universe),
            "overlap_count": len(overlap),
            "added": sorted(current_universe - reference_universe),
            "removed": sorted(reference_universe - current_universe),
            "jaccard": len(overlap) / len(union) if union else None,
        },
        "corporate_action_state": {
            "reference": reference.corporate_action_state,
            "current": current.corporate_action_state,
        },
        "adjustment_state": {
            "reference": reference.adjustment_state,
            "current": current.adjustment_state,
        },
        "available_at_utc": {
            "reference": reference.available_at_utc,
            "current": current.available_at_utc,
        },
    }


def _family_summary(
    reference: Mapping[str, Sequence[Any]] | None,
    current: Mapping[str, Sequence[Any]] | None,
    spec: DriftMonitorSpecV1,
) -> dict[str, Any]:
    reference = reference or {}
    current = current or {}
    names = sorted(set(reference) | set(current))
    result: dict[str, Any] = {}
    for name in names:
        result[name] = compare_scalar_drift(
            reference.get(name, ()),
            current.get(name, ()),
            bin_edges=spec.psi_bin_edges,
            quantiles=spec.quantiles,
            psi_probability_floor=spec.psi_probability_floor,
        )
    return result


def _summary_sha256(summary: Mapping[str, Any]) -> str:
    return _sha256_text(_canonical_json(summary))


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, (value ^ (value >> 31)) & ((1 << 64) - 1)


def _uniform_53(state: int) -> tuple[int, float]:
    state, value = _splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


def _stationary_indices(length: int, mean_block_length: float, state: int) -> tuple[list[int], int]:
    state, value = _splitmix64(state)
    index = value % length
    indices: list[int] = []
    for position in range(length):
        indices.append(index)
        if position + 1 == length:
            break
        state, uniform = _uniform_53(state)
        if uniform < 1.0 / mean_block_length:
            state, value = _splitmix64(state)
            index = value % length
        else:
            index = (index + 1) % length
    return indices, state


def stationary_bootstrap_mean_difference(
    reference: Sequence[Any],
    current: Sequence[Any],
    *,
    replicates: int,
    seed: int,
    mean_block_length: float,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if len(reference) != len(current) or not reference:
        raise DriftArtifactValidationError("stationary bootstrap 输入必须等长且非空")
    if isinstance(replicates, bool) or replicates <= 0 or isinstance(seed, bool) or seed <= 0:
        raise DriftArtifactValidationError("stationary bootstrap replicates/seed 必须为正")
    if not math.isfinite(mean_block_length) or mean_block_length < 1.0:
        raise DriftArtifactValidationError("stationary bootstrap block length 无效")
    if not 0.0 < confidence_level < 1.0:
        raise DriftArtifactValidationError("confidence_level 必须位于 (0, 1)")
    differences = [
        float(current_value) - float(reference_value)
        for reference_value, current_value in zip(reference, current)
        if _finite(reference_value) and _finite(current_value)
    ]
    if not differences:
        return {
            "status": "INSUFFICIENT_DATA",
            "observations": 0,
            "replicates": replicates,
            "seed": seed,
            "mean_block_length": mean_block_length,
        }
    observed = sum(differences) / len(differences)
    bootstrap_values: list[float] = []
    state = seed
    for _ in range(replicates):
        indices, state = _stationary_indices(len(differences), mean_block_length, state)
        bootstrap_values.append(sum(differences[index] for index in indices) / len(indices))
    extreme = sum(abs(value) >= abs(observed) for value in bootstrap_values)
    result: dict[str, Any] = {
        "status": "OK",
        "observations": len(differences),
        "replicates": replicates,
        "seed": seed,
        "mean_block_length": mean_block_length,
        "observed_difference": observed,
        "bootstrap_mean": sum(bootstrap_values) / len(bootstrap_values),
        "p_value": (extreme + 1) / (replicates + 1.0),
        "confidence_level": confidence_level,
        "confidence_lower": _quantile(bootstrap_values, (1.0 - confidence_level) / 2.0),
        "confidence_upper": _quantile(bootstrap_values, 1.0 - (1.0 - confidence_level) / 2.0),
    }
    result["artifact_hash"] = _sha256_text(_canonical_json(result))
    return result


def benjamini_hochberg(
    p_values: Mapping[str, Any],
    *,
    alpha: float,
    family: str,
    method: str = "BH",
) -> dict[str, Any]:
    if not family or not 0.0 < alpha <= 1.0:
        raise DriftArtifactValidationError("FDR family/alpha 无效")
    if method not in {"BH", "BY"}:
        raise DriftArtifactValidationError("FDR method 只支持 BH 或 BY")
    entries = []
    for name, probability in p_values.items():
        if not isinstance(name, str) or not name or not _finite(probability) or not 0.0 <= probability <= 1.0:
            raise DriftArtifactValidationError("FDR p-value 必须是 [0, 1] 有限值")
        entries.append((name, float(probability)))
    entries.sort(key=lambda item: (item[1], item[0]))
    count = len(entries)
    correction = sum(1.0 / index for index in range(1, count + 1)) if method == "BY" else 1.0
    q_values: dict[str, float] = {}
    running = 1.0
    for rank in range(count, 0, -1):
        name, probability = entries[rank - 1]
        candidate = probability * count * correction / rank
        running = min(running, candidate)
        q_values[name] = min(running, 1.0)
    rejected = {name: q_values[name] <= alpha for name, _ in entries}
    result: dict[str, Any] = {
        "status": "OK",
        "family": family,
        "method": method,
        "alpha": alpha,
        "p_values": {name: probability for name, probability in sorted(entries)},
        "q_values": {name: q_values[name] for name in sorted(q_values)},
        "rejected": {name: rejected[name] for name in sorted(rejected)},
    }
    result["artifact_hash"] = _sha256_text(_canonical_json(result))
    return result


def _matrix_from_mapping(values: Mapping[str, Sequence[Any]]) -> tuple[tuple[str, ...], list[list[Any]]]:
    names = tuple(sorted(values))
    if not names:
        return names, []
    lengths = {len(values[name]) for name in names}
    if len(lengths) != 1:
        raise DriftArtifactValidationError("correlation 输入字段长度必须一致")
    return names, [[values[name][row] for name in names] for row in range(lengths.pop())]


def _correlation_matrix(rows: Sequence[Sequence[Any]], dimension: int) -> list[list[float]]:
    matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    for left in range(dimension):
        for right in range(left, dimension):
            pairs = [
                (float(row[left]), float(row[right]))
                for row in rows
                if len(row) == dimension and _finite(row[left]) and _finite(row[right])
            ]
            if left == right:
                matrix[left][right] = 1.0 if len(pairs) >= 2 else 0.0
                continue
            if len(pairs) < 2:
                correlation = 0.0
            else:
                left_mean = sum(pair[0] for pair in pairs) / len(pairs)
                right_mean = sum(pair[1] for pair in pairs) / len(pairs)
                left_variance = sum((pair[0] - left_mean) ** 2 for pair in pairs)
                right_variance = sum((pair[1] - right_mean) ** 2 for pair in pairs)
                denominator = math.sqrt(left_variance * right_variance)
                correlation = (
                    sum((pair[0] - left_mean) * (pair[1] - right_mean) for pair in pairs)
                    / denominator
                    if denominator > 0.0
                    else 0.0
                )
            matrix[left][right] = correlation
            matrix[right][left] = correlation
    return matrix


def _jacobi_eigenvalues(matrix: Sequence[Sequence[float]]) -> list[float]:
    dimension = len(matrix)
    values = [list(row) for row in matrix]
    for row in values:
        if len(row) != dimension:
            raise DriftArtifactValidationError("eigenvalue 输入矩阵必须方阵")
    for _ in range(max(1, dimension * dimension * 32)):
        pivot = (0, 0, 0.0)
        for left in range(dimension):
            for right in range(left + 1, dimension):
                candidate = abs(values[left][right])
                if candidate > pivot[2]:
                    pivot = (left, right, candidate)
        left, right, magnitude = pivot
        if magnitude <= 1e-12:
            break
        angle = 0.5 * math.atan2(
            2.0 * values[left][right], values[right][right] - values[left][left]
        )
        cosine = math.cos(angle)
        sine = math.sin(angle)
        for index in range(dimension):
            if index in {left, right}:
                continue
            left_value = values[index][left]
            right_value = values[index][right]
            values[index][left] = values[left][index] = cosine * left_value - sine * right_value
            values[index][right] = values[right][index] = sine * left_value + cosine * right_value
        left_diagonal = values[left][left]
        right_diagonal = values[right][right]
        cross = values[left][right]
        values[left][left] = cosine * cosine * left_diagonal - 2.0 * sine * cosine * cross + sine * sine * right_diagonal
        values[right][right] = sine * sine * left_diagonal + 2.0 * sine * cosine * cross + cosine * cosine * right_diagonal
        values[left][right] = values[right][left] = 0.0
    return sorted((values[index][index] for index in range(dimension)), reverse=True)


def _effective_rank(eigenvalues: Sequence[float]) -> float | None:
    positive = [max(value, 0.0) for value in eigenvalues]
    total = sum(positive)
    if total <= 0.0:
        return None
    probabilities = [value / total for value in positive if value > 0.0]
    return math.exp(-sum(probability * math.log(probability) for probability in probabilities))


def compare_correlation_drift(
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    reference_names = tuple(sorted(reference))
    current_names = tuple(sorted(current))
    if reference_names != current_names:
        return {
            "status": "SCHEMA_MISMATCH",
            "reference_features": list(reference_names),
            "current_features": list(current_names),
            "added": sorted(set(current_names) - set(reference_names)),
            "removed": sorted(set(reference_names) - set(current_names)),
        }
    names, reference_rows = _matrix_from_mapping(reference)
    _, current_rows = _matrix_from_mapping(current)
    if not names or not reference_rows or not current_rows:
        return {
            "status": "INSUFFICIENT_DATA",
            "features": list(names),
            "reference_count": len(reference_rows),
            "current_count": len(current_rows),
        }
    reference_matrix = _correlation_matrix(reference_rows, len(names))
    current_matrix = _correlation_matrix(current_rows, len(names))
    difference = [
        [current_matrix[row][column] - reference_matrix[row][column] for column in range(len(names))]
        for row in range(len(names))
    ]
    frobenius = math.sqrt(sum(value * value for row in difference for value in row))
    eigenvalues = _jacobi_eigenvalues(difference)
    operator_distance = max((abs(value) for value in eigenvalues), default=0.0)
    result: dict[str, Any] = {
        "status": "OK",
        "features": list(names),
        "reference_count": len(reference_rows),
        "current_count": len(current_rows),
        "reference_correlation": reference_matrix,
        "current_correlation": current_matrix,
        "reference_eigenvalues": _jacobi_eigenvalues(reference_matrix),
        "current_eigenvalues": _jacobi_eigenvalues(current_matrix),
        "difference_eigenvalues": eigenvalues,
        "reference_effective_rank": _effective_rank(_jacobi_eigenvalues(reference_matrix)),
        "current_effective_rank": _effective_rank(_jacobi_eigenvalues(current_matrix)),
        "frobenius_distance": frobenius,
        "operator_distance": operator_distance,
    }
    result["artifact_hash"] = _sha256_text(_canonical_json(result))
    return result


def compare_rbf_mmd(
    reference: Sequence[Sequence[Any]],
    current: Sequence[Sequence[Any]],
    *,
    bandwidth: float,
    projection: Sequence[int] | None = None,
) -> dict[str, Any]:
    if not math.isfinite(bandwidth) or bandwidth <= 0.0:
        raise DriftArtifactValidationError("MMD RBF bandwidth 必须为正")
    reference_rows = [
        tuple(float(value) if _finite(value) else math.nan for value in row)
        for row in reference
    ]
    current_rows = [
        tuple(float(value) if _finite(value) else math.nan for value in row)
        for row in current
    ]
    if not reference_rows or not current_rows:
        return {
            "status": "INSUFFICIENT_DATA",
            "reference_count": 0,
            "current_count": 0,
            "dimension": 0,
            "kernel_spec": {
                "kernel": "RBF",
                "estimator": "biased",
                "bandwidth": bandwidth,
                "projection": list(projection) if projection is not None else None,
            },
        }
    dimensions = {len(row) for row in reference_rows + current_rows}
    if len(dimensions) != 1 or not dimensions:
        raise DriftArtifactValidationError("MMD 输入行维度必须一致且非空")
    dimension = dimensions.pop()
    selected = tuple(range(dimension)) if projection is None else tuple(projection)
    if not selected or any(isinstance(index, bool) or index < 0 or index >= dimension for index in selected):
        raise DriftArtifactValidationError("MMD projection 索引无效")
    if len(set(selected)) != len(selected):
        raise DriftArtifactValidationError("MMD projection 索引不得重复")
    filtered_reference = [row for row in reference_rows if all(math.isfinite(row[index]) for index in selected)]
    filtered_current = [row for row in current_rows if all(math.isfinite(row[index]) for index in selected)]
    spec_payload = {
        "kernel": "RBF",
        "estimator": "biased",
        "bandwidth": bandwidth,
        "projection": list(selected),
    }
    kernel_spec_sha256 = _sha256_text(_canonical_json(spec_payload))
    if not filtered_reference or not filtered_current:
        return {
            "status": "INSUFFICIENT_DATA",
            "reference_count": len(filtered_reference),
            "current_count": len(filtered_current),
            "dimension": len(selected),
            "kernel_spec": spec_payload,
            "kernel_spec_sha256": kernel_spec_sha256,
        }
    reference_points = [tuple(row[index] for index in selected) for row in filtered_reference]
    current_points = [tuple(row[index] for index in selected) for row in filtered_current]

    def kernel(left: Sequence[float], right: Sequence[float]) -> float:
        squared_distance = sum((left[index] - right[index]) ** 2 for index in range(len(selected)))
        return math.exp(-squared_distance / (2.0 * bandwidth * bandwidth))

    reference_term = sum(kernel(left, right) for left in reference_points for right in reference_points) / (len(reference_points) ** 2)
    current_term = sum(kernel(left, right) for left in current_points for right in current_points) / (len(current_points) ** 2)
    cross_term = sum(kernel(left, right) for left in reference_points for right in current_points) / (len(reference_points) * len(current_points))
    mmd_squared = max(reference_term + current_term - 2.0 * cross_term, 0.0)
    result = {
        "status": "OK",
        "reference_count": len(reference_points),
        "current_count": len(current_points),
        "dimension": len(selected),
        "kernel_spec": spec_payload,
        "kernel_spec_sha256": kernel_spec_sha256,
        "mmd_squared": mmd_squared,
        "mmd": math.sqrt(mmd_squared),
    }
    result["artifact_hash"] = _sha256_text(_canonical_json(result))
    return result


def _ranked_symbols(scores: Mapping[str, Any], top_k: int) -> list[tuple[str, float]]:
    normalized: list[tuple[str, float]] = []
    for symbol, score in scores.items():
        if not isinstance(symbol, str) or not symbol or not _finite(score):
            raise DriftArtifactValidationError("RankingScore symbol/score 无效")
        normalized.append((symbol, float(score)))
    normalized.sort(key=lambda item: (-item[1], item[0]))
    return normalized[: min(top_k, len(normalized))]


def _top_k_concentration(ranked: Sequence[tuple[str, float]], all_scores: Mapping[str, Any]) -> float | None:
    denominator = sum(abs(float(score)) for score in all_scores.values() if _finite(score))
    if denominator <= 0.0:
        return None
    return sum(abs(score) for _, score in ranked) / denominator


def compare_prediction_drift(
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
    *,
    bin_edges: Sequence[float],
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ranking_reference: Mapping[str, Any] | None = None,
    ranking_current: Mapping[str, Any] | None = None,
    confidence_reference: Sequence[Any] | None = None,
    confidence_current: Sequence[Any] | None = None,
    interval_width_reference: Sequence[Any] | None = None,
    interval_width_current: Sequence[Any] | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    if isinstance(top_k, bool) or top_k <= 0:
        raise DriftArtifactValidationError("prediction top_k 必须为正")
    outputs = {
        name: compare_scalar_drift(
            reference.get(name, ()),
            current.get(name, ()),
            bin_edges=bin_edges,
            quantiles=quantiles,
        )
        for name in sorted(set(reference) | set(current))
    }
    if confidence_reference is not None or confidence_current is not None:
        if confidence_reference is None or confidence_current is None:
            raise DriftArtifactValidationError("confidence reference/current 必须同时提供")
        outputs["confidence"] = compare_scalar_drift(
            confidence_reference,
            confidence_current,
            bin_edges=bin_edges,
            quantiles=quantiles,
        )
    if interval_width_reference is not None or interval_width_current is not None:
        if interval_width_reference is None or interval_width_current is None:
            raise DriftArtifactValidationError("interval width reference/current 必须同时提供")
        outputs["interval_width"] = compare_scalar_drift(
            interval_width_reference,
            interval_width_current,
            bin_edges=bin_edges,
            quantiles=quantiles,
        )
    if ranking_reference is None and ranking_current is None:
        ranking: dict[str, Any] | str = "UNAVAILABLE"
    elif ranking_reference is None or ranking_current is None:
        raise DriftArtifactValidationError("RankingScore reference/current 必须同时提供")
    else:
        reference_ranked = _ranked_symbols(ranking_reference, top_k)
        current_ranked = _ranked_symbols(ranking_current, top_k)
        reference_top = {symbol for symbol, _ in reference_ranked}
        current_top = {symbol for symbol, _ in current_ranked}
        denominator = min(len(reference_top), len(current_top))
        overlap = len(reference_top & current_top) / denominator if denominator else None
        ranking = {
            "status": "OK" if reference_ranked and current_ranked else "INSUFFICIENT_DATA",
            "top_k": top_k,
            "reference_top_k": [symbol for symbol, _ in reference_ranked],
            "current_top_k": [symbol for symbol, _ in current_ranked],
            "top_k_overlap": overlap,
            "top_k_turnover": 1.0 - overlap if overlap is not None else None,
            "reference_concentration": _top_k_concentration(reference_ranked, ranking_reference),
            "current_concentration": _top_k_concentration(current_ranked, ranking_current),
            "reference_universe_count": len(ranking_reference),
            "current_universe_count": len(ranking_current),
        }
    result: dict[str, Any] = {
        "status": "OK" if outputs or isinstance(ranking, dict) else "INSUFFICIENT_DATA",
        "outputs": outputs,
        "ranking": ranking,
    }
    result["artifact_hash"] = _sha256_text(_canonical_json(result))
    return result


def _pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    return (
        sum(left_value * right_value for left_value, right_value in zip(left_centered, right_centered))
        / denominator
        if denominator > 0.0
        else None
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2.0 + 1.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _ndcg_at_k(scores: Sequence[float], labels: Sequence[float], top_k: int) -> float | None:
    if not scores or len(scores) != len(labels):
        return None
    selected = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]
    ideal = sorted(range(len(labels)), key=lambda index: (-labels[index], index))[:top_k]
    predicted_gain = sum(
        (2.0 ** max(labels[index], 0.0) - 1.0) / math.log2(rank + 2.0)
        for rank, index in enumerate(selected)
    )
    ideal_gain = sum(
        (2.0 ** max(labels[index], 0.0) - 1.0) / math.log2(rank + 2.0)
        for rank, index in enumerate(ideal)
    )
    return predicted_gain / ideal_gain if ideal_gain > 0.0 else None


def _utility_spread(scores: Sequence[float], labels: Sequence[float], top_k: int) -> float | None:
    if not scores or len(scores) != len(labels):
        return None
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    count = min(top_k, len(order))
    if count == 0:
        return None
    top = order[:count]
    bottom = order[-count:]
    return sum(labels[index] for index in top) / count - sum(labels[index] for index in bottom) / count


def _label_window_metrics(scores: Sequence[Any], labels: Sequence[Any], top_k: int) -> dict[str, Any]:
    pairs = [
        (float(score), float(label))
        for score, label in zip(scores, labels)
        if _finite(score) and _finite(label)
    ]
    if not pairs:
        return {"status": "INSUFFICIENT_DATA", "observations": 0}
    finite_scores = [pair[0] for pair in pairs]
    finite_labels = [pair[1] for pair in pairs]
    rank_ic = _pearson_correlation(_average_ranks(finite_scores), _average_ranks(finite_labels))
    brier = (
        sum((score - label) ** 2 for score, label in pairs) / len(pairs)
        if all(0.0 <= score <= 1.0 and label in {0.0, 1.0} for score, label in pairs)
        else None
    )
    return {
        "status": "OK",
        "observations": len(pairs),
        "pearson_ic": _pearson_correlation(finite_scores, finite_labels),
        "rank_ic": rank_ic,
        "ndcg_at_k": _ndcg_at_k(finite_scores, finite_labels, top_k),
        "utility_spread": _utility_spread(finite_scores, finite_labels, top_k),
        "mae": sum(abs(score - label) for score, label in pairs) / len(pairs),
        "brier": brier,
    }


def compare_label_concept_drift(
    reference_labels: Sequence[Any],
    current_labels: Sequence[Any],
    reference_scores: Sequence[Any],
    current_scores: Sequence[Any],
    *,
    top_k: int = 20,
    reference_session_ics: Sequence[Any] | None = None,
    current_session_ics: Sequence[Any] | None = None,
) -> dict[str, Any]:
    if isinstance(top_k, bool) or top_k <= 0:
        raise DriftArtifactValidationError("label top_k 必须为正")
    if len(reference_labels) != len(reference_scores) or len(current_labels) != len(current_scores):
        raise DriftArtifactValidationError("label/score reference/current 必须分别等长")
    reference = _label_window_metrics(reference_scores, reference_labels, top_k)
    current = _label_window_metrics(current_scores, current_labels, top_k)
    session_metrics: dict[str, Any] = {}
    for name, values in (("reference", reference_session_ics), ("current", current_session_ics)):
        if values is None:
            session_metrics[name] = {"status": "INSUFFICIENT_SESSIONS", "observations": 0, "icir": None}
            continue
        finite_values = [float(value) for value in values if _finite(value)]
        if len(finite_values) < 2:
            session_metrics[name] = {
                "status": "INSUFFICIENT_SESSIONS",
                "observations": len(finite_values),
                "icir": None,
            }
            continue
        mean = sum(finite_values) / len(finite_values)
        variance = sum((value - mean) ** 2 for value in finite_values) / (len(finite_values) - 1)
        session_metrics[name] = {
            "status": "OK",
            "observations": len(finite_values),
            "mean_ic": mean,
            "icir": mean / math.sqrt(variance) if variance > 0.0 else None,
        }
    result: dict[str, Any] = {
        "status": "OK" if reference["status"] == "OK" and current["status"] == "OK" else "INSUFFICIENT_DATA",
        "top_k": top_k,
        "reference": reference,
        "current": current,
        "session_ic": session_metrics,
        "delta": {
            metric: current[metric] - reference[metric]
            if _finite(current.get(metric)) and _finite(reference.get(metric))
            else None
            for metric in ("pearson_ic", "rank_ic", "ndcg_at_k", "utility_spread", "mae", "brier")
        },
    }
    result["artifact_hash"] = _sha256_text(_canonical_json(result))
    return result


def _embedding_covariance(rows: Sequence[Sequence[float]], dimension: int) -> tuple[list[float], list[list[float]]]:
    centroid = [sum(row[index] for row in rows) / len(rows) for index in range(dimension)]
    covariance = [
        [
            sum((row[left] - centroid[left]) * (row[right] - centroid[right]) for row in rows)
            / len(rows)
            for right in range(dimension)
        ]
        for left in range(dimension)
    ]
    return centroid, covariance


def compare_embedding_drift(
    reference: Sequence[Sequence[Any]],
    current: Sequence[Sequence[Any]],
    *,
    reference_metadata: Mapping[str, Any],
    current_metadata: Mapping[str, Any],
    mmd_bandwidth: float,
) -> dict[str, Any]:
    if not _finite(mmd_bandwidth) or float(mmd_bandwidth) <= 0.0:
        raise DriftArtifactValidationError("embedding MMD bandwidth 必须为正有限值")
    metadata_keys = ("checkpoint_id", "layer", "pooling", "dimension", "anchor_sha256")
    reference_view = {key: reference_metadata.get(key) for key in metadata_keys}
    current_view = {key: current_metadata.get(key) for key in metadata_keys}
    incompatible = [
        key for key in metadata_keys if reference_view[key] != current_view[key]
    ]
    if not _digest_like(reference_view.get("anchor_sha256")) or not _digest_like(current_view.get("anchor_sha256")):
        incompatible.append("anchor_sha256")
    if incompatible:
        return {
            "status": "INCOMPATIBLE",
            "diagnostic_only": True,
            "incompatibility_reasons": sorted(set(incompatible)),
            "reference_metadata": reference_view,
            "current_metadata": current_view,
        }
    dimension = reference_view["dimension"]
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise DriftArtifactValidationError("embedding dimension 必须为正整数")
    reference_rows = [
        tuple(float(value) if _finite(value) else math.nan for value in row)
        for row in reference
    ]
    current_rows = [
        tuple(float(value) if _finite(value) else math.nan for value in row)
        for row in current
    ]
    if any(len(row) != dimension for row in reference_rows + current_rows):
        raise DriftArtifactValidationError("embedding 行维度与 metadata 不一致")
    reference_rows = [row for row in reference_rows if all(math.isfinite(value) for value in row)]
    current_rows = [row for row in current_rows if all(math.isfinite(value) for value in row)]
    if not reference_rows or not current_rows:
        return {
            "status": "INSUFFICIENT_DATA",
            "diagnostic_only": True,
            "reference_metadata": reference_view,
            "current_metadata": current_view,
            "reference_count": len(reference_rows),
            "current_count": len(current_rows),
        }
    reference_centroid, reference_covariance = _embedding_covariance(reference_rows, dimension)
    current_centroid, current_covariance = _embedding_covariance(current_rows, dimension)
    covariance_difference = [
        [current_covariance[left][right] - reference_covariance[left][right] for right in range(dimension)]
        for left in range(dimension)
    ]
    covariance_distance = math.sqrt(
        sum(value * value for row in covariance_difference for value in row)
    )
    reference_eigenvalues = _jacobi_eigenvalues(reference_covariance)
    current_eigenvalues = _jacobi_eigenvalues(current_covariance)
    mmd = compare_rbf_mmd(
        reference_rows,
        current_rows,
        bandwidth=mmd_bandwidth,
    )
    result: dict[str, Any] = {
        "status": "OK",
        "diagnostic_only": True,
        "reference_metadata": reference_view,
        "current_metadata": current_view,
        "reference_count": len(reference_rows),
        "current_count": len(current_rows),
        "reference_centroid": reference_centroid,
        "current_centroid": current_centroid,
        "centroid_distance": math.sqrt(
            sum((current_centroid[index] - reference_centroid[index]) ** 2 for index in range(dimension))
        ),
        "reference_covariance": reference_covariance,
        "current_covariance": current_covariance,
        "covariance_frobenius_distance": covariance_distance,
        "reference_eigenvalues": reference_eigenvalues,
        "current_eigenvalues": current_eigenvalues,
        "reference_effective_rank": _effective_rank(reference_eigenvalues),
        "current_effective_rank": _effective_rank(current_eigenvalues),
        "mmd": mmd,
    }
    result["artifact_hash"] = _sha256_text(_canonical_json(result))
    return result


def _signal_state(summary: Mapping[str, Any], spec: DriftMonitorSpecV1) -> tuple[str, list[str]]:
    state = "INFO"
    reasons: list[str] = []
    for name, values in summary.items():
        psi = values.get("psi")
        ks_d = values.get("ks_d")
        if spec.psi_critical_threshold is not None and psi is not None and psi >= spec.psi_critical_threshold:
            state = "CRITICAL"
            reasons.append(f"{name}:PSI_CRITICAL")
        elif spec.psi_warn_threshold is not None and psi is not None and psi >= spec.psi_warn_threshold:
            if state != "CRITICAL":
                state = "WARN"
            reasons.append(f"{name}:PSI_WARN")
        if spec.ks_critical_threshold is not None and ks_d is not None and ks_d >= spec.ks_critical_threshold:
            state = "CRITICAL"
            reasons.append(f"{name}:KS_CRITICAL")
        elif spec.ks_warn_threshold is not None and ks_d is not None and ks_d >= spec.ks_warn_threshold:
            if state != "CRITICAL":
                state = "WARN"
            reasons.append(f"{name}:KS_WARN")
    return state, sorted(set(reasons))


def _concept_signal(summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(summary, Mapping) or summary.get("status") != "OK":
        return False
    reference = summary.get("reference")
    current = summary.get("current")
    if not isinstance(reference, Mapping) or not isinstance(current, Mapping):
        return False
    deteriorations = 0
    for metric in ("pearson_ic", "rank_ic", "ndcg_at_k", "utility_spread"):
        if _finite(reference.get(metric)) and _finite(current.get(metric)):
            deteriorations += int(float(current[metric]) < float(reference[metric]))
    for metric in ("mae", "brier"):
        if _finite(reference.get(metric)) and _finite(current.get(metric)):
            deteriorations += int(float(current[metric]) > float(reference[metric]))
    return deteriorations >= 2


def transition_alert_state(
    previous_state: str,
    observed_state: str,
    persistence_count: int,
    persistence_required: int,
    hard_failure: bool,
    *,
    marginal_signal: bool | None = None,
    concept_signal: bool | None = None,
) -> tuple[str, list[str]]:
    """Apply the observation-only alert transition contract.

    When ``marginal_signal`` and ``concept_signal`` are supplied, a non-hard
    ``CRITICAL`` observation is promoted only when both signals are present.
    Omitting them preserves the basic INFO/WARN/CRITICAL transition contract.
    """
    valid_states = {"INFO", "WARN", "CRITICAL"}
    if previous_state not in valid_states or observed_state not in valid_states:
        raise DriftArtifactValidationError("alert state 必须为 INFO/WARN/CRITICAL")
    if isinstance(persistence_count, bool) or persistence_count < 0:
        raise DriftArtifactValidationError("persistence_count 不得为负")
    if isinstance(persistence_required, bool) or persistence_required <= 0:
        raise DriftArtifactValidationError("persistence_required 必须为正")
    if not isinstance(hard_failure, bool):
        raise DriftArtifactValidationError("hard_failure 必须为布尔值")
    if marginal_signal is not None and not isinstance(marginal_signal, bool):
        raise DriftArtifactValidationError("marginal_signal 必须为布尔值")
    if concept_signal is not None and not isinstance(concept_signal, bool):
        raise DriftArtifactValidationError("concept_signal 必须为布尔值")
    reasons: list[str] = []
    if hard_failure:
        return "CRITICAL", ["HARD_FAILURE"]
    effective_observed = observed_state
    if (
        observed_state == "CRITICAL"
        and marginal_signal is not None
        and concept_signal is not None
        and not (marginal_signal and concept_signal)
    ):
        effective_observed = "WARN"
        reasons.append("CRITICAL_REQUIRES_JOINT_SIGNAL")
    if effective_observed == "INFO":
        if previous_state != "INFO":
            reasons.append("RECOVERED")
        return "INFO", sorted(set(reasons))
    if persistence_count < persistence_required:
        reasons.append("PERSISTENCE_PENDING")
        return "INFO", sorted(set(reasons))
    return effective_observed, sorted(set(reasons))


def build_drift_artifact(
    spec: DriftMonitorSpecV1,
    *,
    source_snapshot_set_sha256: str,
    raw_reference: Mapping[str, Sequence[Any]] | None = None,
    raw_current: Mapping[str, Sequence[Any]] | None = None,
    raw_quality_reference: RawDataQualityWindow | None = None,
    raw_quality_current: RawDataQualityWindow | None = None,
    feature_reference: Mapping[str, Sequence[Any]] | None = None,
    feature_current: Mapping[str, Sequence[Any]] | None = None,
    correlation_reference: Mapping[str, Sequence[Any]] | None = None,
    correlation_current: Mapping[str, Sequence[Any]] | None = None,
    mmd_reference: Sequence[Sequence[Any]] | None = None,
    mmd_current: Sequence[Sequence[Any]] | None = None,
    mmd_bandwidth: float | None = None,
    mmd_projection: Sequence[int] | None = None,
    embedding_reference: Sequence[Sequence[Any]] | None = None,
    embedding_current: Sequence[Sequence[Any]] | None = None,
    embedding_reference_metadata: Mapping[str, Any] | None = None,
    embedding_current_metadata: Mapping[str, Any] | None = None,
    prediction_reference: Mapping[str, Sequence[Any]] | None = None,
    prediction_current: Mapping[str, Sequence[Any]] | None = None,
    prediction_ranking_reference: Mapping[str, Any] | None = None,
    prediction_ranking_current: Mapping[str, Any] | None = None,
    prediction_confidence_reference: Sequence[Any] | None = None,
    prediction_confidence_current: Sequence[Any] | None = None,
    prediction_interval_width_reference: Sequence[Any] | None = None,
    prediction_interval_width_current: Sequence[Any] | None = None,
    label_reference: Mapping[str, Sequence[Any]] | None = None,
    label_current: Mapping[str, Sequence[Any]] | None = None,
    labels_mature: bool = False,
    label_available_at_utc: str | None = None,
    label_scores_reference: Sequence[Any] | None = None,
    label_scores_current: Sequence[Any] | None = None,
    label_session_ics_reference: Sequence[Any] | None = None,
    label_session_ics_current: Sequence[Any] | None = None,
    persistence_count: int = 0,
    bootstrap_results: Mapping[str, Mapping[str, Any]] | None = None,
    fdr_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    spec.validate()
    if not _digest_like(source_snapshot_set_sha256):
        raise DriftArtifactValidationError("source_snapshot_set_sha256 格式无效")
    if not isinstance(labels_mature, bool):
        raise DriftArtifactValidationError("labels_mature 必须是布尔值")
    if isinstance(persistence_count, bool) or persistence_count < 0:
        raise DriftArtifactValidationError("persistence_count 不得为负")
    if labels_mature and not _valid_utc_timestamp(label_available_at_utc):
        raise DriftArtifactValidationError("成熟标签必须提供 label_available_at_utc")
    if labels_mature and _utc_datetime(label_available_at_utc) > _utc_datetime(spec.available_at_utc):
        raise DriftArtifactValidationError("label_available_at_utc 晚于 artifact available_at_utc")
    raw_summary = _family_summary(raw_reference, raw_current, spec)
    raw_quality = (
        compare_raw_data_quality(raw_quality_reference, raw_quality_current)
        if raw_quality_reference is not None and raw_quality_current is not None
        else "UNAVAILABLE"
    )
    if isinstance(raw_quality, dict) and _utc_datetime(raw_quality["available_at_utc"]["current"]) > _utc_datetime(spec.available_at_utc):
        raw_quality["status"] = "HARD_FAILURE"
        raw_quality["hard_failures"] = sorted(set(raw_quality["hard_failures"] + ["AVAILABLE_AT_AFTER_ARTIFACT"]))
    feature_summary = _family_summary(feature_reference, feature_current, spec)
    correlation_summary = (
        compare_correlation_drift(correlation_reference, correlation_current)
        if correlation_reference is not None and correlation_current is not None
        else "UNAVAILABLE"
    )
    mmd_summary = (
        compare_rbf_mmd(
            mmd_reference,
            mmd_current,
            bandwidth=mmd_bandwidth,
            projection=mmd_projection,
        )
        if mmd_reference is not None and mmd_current is not None and mmd_bandwidth is not None
        else "UNAVAILABLE"
    )
    embedding_summary: dict[str, Any] | str = "INCOMPATIBLE"
    if embedding_reference is not None or embedding_current is not None:
        if embedding_reference is None or embedding_current is None:
            raise DriftArtifactValidationError("embedding reference/current 必须同时提供")
        default_embedding_metadata = {
            "checkpoint_id": spec.embedding_checkpoint_id,
            "layer": spec.embedding_layer,
            "pooling": spec.embedding_pooling,
            "dimension": spec.embedding_dimension,
            "anchor_sha256": spec.embedding_anchor_sha256,
        }
        embedding_summary = compare_embedding_drift(
            embedding_reference,
            embedding_current,
            reference_metadata=embedding_reference_metadata or default_embedding_metadata,
            current_metadata=embedding_current_metadata or default_embedding_metadata,
            mmd_bandwidth=spec.embedding_mmd_bandwidth,
        )
    prediction_summary = _family_summary(prediction_reference, prediction_current, spec)
    prediction_relation_summary = (
        compare_prediction_drift(
            prediction_reference or {},
            prediction_current or {},
            bin_edges=spec.psi_bin_edges,
            quantiles=spec.quantiles,
            ranking_reference=prediction_ranking_reference,
            ranking_current=prediction_ranking_current,
            confidence_reference=prediction_confidence_reference,
            confidence_current=prediction_confidence_current,
            interval_width_reference=prediction_interval_width_reference,
            interval_width_current=prediction_interval_width_current,
            top_k=spec.prediction_top_k,
        )
        if any(value is not None for value in (
            prediction_reference,
            prediction_current,
            prediction_ranking_reference,
            prediction_ranking_current,
            prediction_confidence_reference,
            prediction_confidence_current,
            prediction_interval_width_reference,
            prediction_interval_width_current,
        ))
        else "UNAVAILABLE"
    )
    label_summary = _family_summary(label_reference, label_current, spec) if labels_mature else None
    label_concept_summary = (
        compare_label_concept_drift(
            next(iter(label_reference.values()), ()) if label_reference else (),
            next(iter(label_current.values()), ()) if label_current else (),
            label_scores_reference or (),
            label_scores_current or (),
            top_k=spec.prediction_top_k,
            reference_session_ics=label_session_ics_reference,
            current_session_ics=label_session_ics_current,
        )
        if labels_mature and label_scores_reference is not None and label_scores_current is not None
        and label_reference is not None and label_current is not None
        else None
    )
    states = [_signal_state(summary, spec) for summary in (
        raw_summary,
        feature_summary,
        prediction_summary,
        label_summary or {},
    )]
    signal_state = "INFO"
    reasons: list[str] = []
    for state, state_reasons in states:
        if state == "CRITICAL":
            signal_state = "CRITICAL"
        elif state == "WARN" and signal_state != "CRITICAL":
            signal_state = "WARN"
        reasons.extend(state_reasons)
    raw_hard_failure = isinstance(raw_quality, dict) and raw_quality["status"] == "HARD_FAILURE"
    raw_warning = isinstance(raw_quality, dict) and raw_quality["status"] == "WARN"
    if raw_hard_failure:
        signal_state = "CRITICAL"
        reasons.append("RAW_DATA_HARD_FAILURE")
    elif raw_warning and signal_state == "INFO":
        signal_state = "WARN"
        reasons.append("RAW_DATA_WARNING")
    alert_state, transition_reasons = transition_alert_state(
        "INFO",
        signal_state,
        persistence_count,
        spec.persistence_required,
        raw_hard_failure,
        marginal_signal=signal_state != "INFO",
        concept_signal=_concept_signal(label_concept_summary),
    )
    reasons.extend(transition_reasons)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": "drift_artifact_v1",
        "spec_sha256": spec.spec_sha256,
        "reference_mode": spec.reference_mode,
        "reference_window_id": spec.reference_window_id,
        "current_window_id": spec.current_window_id,
        "available_at_utc": spec.available_at_utc,
        "source_snapshot_set_sha256": source_snapshot_set_sha256,
        "data_status": (
            raw_quality["status"]
            if isinstance(raw_quality, dict) and raw_quality["status"] != "OK"
            else "OK" if raw_summary or feature_summary or prediction_summary
            else "UNAVAILABLE"
        ),
        "raw_data_quality": raw_quality,
        "raw_data_quality_summary_sha256": (
            _summary_sha256(raw_quality) if isinstance(raw_quality, dict) else "UNAVAILABLE"
        ),
        "raw_drift": raw_summary,
        "raw_drift_summary_sha256": _summary_sha256(raw_summary),
        "feature_drift": feature_summary,
        "feature_drift_summary_sha256": _summary_sha256(feature_summary),
        "correlation_drift": correlation_summary,
        "correlation_drift_summary_sha256": (
            _summary_sha256(correlation_summary)
            if isinstance(correlation_summary, dict)
            else "UNAVAILABLE"
        ),
        "mmd_drift": mmd_summary,
        "mmd_drift_summary_sha256": (
            _summary_sha256(mmd_summary) if isinstance(mmd_summary, dict) else "UNAVAILABLE"
        ),
        "prediction_drift": prediction_summary,
        "prediction_drift_summary_sha256": _summary_sha256(prediction_summary),
        "prediction_relation_drift": prediction_relation_summary,
        "prediction_relation_drift_summary_sha256": (
            _summary_sha256(prediction_relation_summary)
            if isinstance(prediction_relation_summary, dict)
            else "UNAVAILABLE"
        ),
        "label_drift": label_summary if label_summary is not None else "PENDING_LABELS",
        "label_drift_summary_sha256": (
            _summary_sha256(label_summary) if label_summary is not None else "PENDING_LABELS"
        ),
        "concept_performance": (
            label_concept_summary
            if label_concept_summary is not None
            else "PENDING_LABELS" if not labels_mature else "INSUFFICIENT_DATA"
        ),
        "concept_performance_summary_sha256": (
            _summary_sha256(label_concept_summary)
            if label_concept_summary is not None
            else "PENDING_LABELS" if not labels_mature else "INSUFFICIENT_DATA"
        ),
        "embedding_drift": embedding_summary,
        "embedding_drift_summary_sha256": (
            _summary_sha256(embedding_summary)
            if isinstance(embedding_summary, dict)
            else embedding_summary
        ),
        "diagnostic_only": True,
        "alert_state": alert_state,
        "alert_reasons": sorted(set(reasons)),
        "persistence_count": persistence_count,
        "retraining_review_recommended": alert_state == "CRITICAL",
        "label_available_at_utc": label_available_at_utc if labels_mature else None,
        "inference": {
            "bootstrap": dict(sorted((bootstrap_results or {}).items())),
            "fdr": dict(sorted((fdr_results or {}).items())),
        },
    }
    unsigned = _canonical_json(payload)
    report_sha256 = _sha256_text(unsigned)
    return unsigned[:-1] + ',"report_sha256":"' + report_sha256 + '"}'


def validate_drift_artifact(value: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Path):
        try:
            raw = Path(value).read_text(encoding="utf-8")
        except OSError as exc:
            raise DriftArtifactValidationError("无法读取 DriftArtifact") from exc
    elif isinstance(value, str):
        if value.lstrip().startswith(("{", "[")):
            raw = value
        else:
            try:
                candidate = Path(value)
                raw = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
            except OSError:
                raw = value
    else:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    try:
        parsed = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DriftArtifactValidationError("DriftArtifact JSON 无效") from exc
    if not isinstance(parsed, dict) or parsed.get("schema_version") != 1 or parsed.get("role") != "drift_artifact_v1":
        raise DriftArtifactValidationError("DriftArtifact schema 不受支持")
    report_sha256 = parsed.get("report_sha256")
    if not _digest_like(report_sha256):
        raise DriftArtifactValidationError("DriftArtifact report_sha256 格式无效")
    marker = ',"report_sha256":"'
    marker_start = raw.rfind(marker)
    if marker_start < 0 or not raw.endswith("}"):
        raise DriftArtifactValidationError("DriftArtifact report hash 尾部无效")
    unsigned = raw[:marker_start] + "}"
    if _sha256_text(unsigned) != report_sha256:
        raise DriftArtifactValidationError("DriftArtifact report SHA-256 校验失败")
    if parsed.get("diagnostic_only") is not True:
        raise DriftArtifactValidationError("DriftArtifact 必须是 diagnostic_only")
    if parsed.get("data_status") not in {"OK", "WARN", "HARD_FAILURE", "UNAVAILABLE"}:
        raise DriftArtifactValidationError("DriftArtifact data_status 无效")
    if not _digest_like(parsed.get("spec_sha256")) or not _digest_like(
        parsed.get("source_snapshot_set_sha256")
    ):
        raise DriftArtifactValidationError("DriftArtifact spec/source hash 格式无效")
    if not _valid_utc_timestamp(parsed.get("available_at_utc")):
        raise DriftArtifactValidationError("DriftArtifact available_at_utc 无效")
    raw_quality = parsed.get("raw_data_quality")
    if raw_quality is not None:
        raw_quality_hash = parsed.get("raw_data_quality_summary_sha256")
        if raw_quality == "UNAVAILABLE":
            if raw_quality_hash != "UNAVAILABLE":
                raise DriftArtifactValidationError("DriftArtifact raw quality hash 无效")
        elif isinstance(raw_quality, dict):
            if not _digest_like(raw_quality_hash) or _summary_sha256(raw_quality) != raw_quality_hash:
                raise DriftArtifactValidationError("DriftArtifact raw quality hash 校验失败")
            if raw_quality.get("status") not in {"OK", "WARN", "HARD_FAILURE"}:
                raise DriftArtifactValidationError("DriftArtifact raw quality status 无效")
        else:
            raise DriftArtifactValidationError("DriftArtifact raw_data_quality 无效")
    if not isinstance(parsed.get("inference", {}), dict):
        raise DriftArtifactValidationError("DriftArtifact inference 必须是对象")
    for relationship in ("correlation", "mmd", "prediction_relation", "embedding"):
        summary = parsed.get(f"{relationship}_drift")
        summary_hash = parsed.get(f"{relationship}_drift_summary_sha256")
        if summary is None:
            continue
        if isinstance(summary, str) and summary in {"UNAVAILABLE", "INCOMPATIBLE"}:
            if summary_hash != summary:
                raise DriftArtifactValidationError(f"DriftArtifact {relationship} hash 无效")
        elif isinstance(summary, dict):
            if not _digest_like(summary_hash) or _summary_sha256(summary) != summary_hash:
                raise DriftArtifactValidationError(f"DriftArtifact {relationship} summary hash 校验失败")
            if relationship == "embedding":
                if summary.get("diagnostic_only") is not True:
                    raise DriftArtifactValidationError("DriftArtifact embedding 必须是 diagnostic_only")
                if summary.get("status") not in {"OK", "INSUFFICIENT_DATA", "INCOMPATIBLE"}:
                    raise DriftArtifactValidationError("DriftArtifact embedding status 无效")
        else:
            raise DriftArtifactValidationError(f"DriftArtifact {relationship} summary 无效")
    for family in ("raw", "feature", "prediction"):
        summary = parsed.get(f"{family}_drift")
        summary_hash = parsed.get(f"{family}_drift_summary_sha256")
        if not isinstance(summary, dict) or not _digest_like(summary_hash):
            raise DriftArtifactValidationError(f"DriftArtifact {family} summary 无效")
        if _summary_sha256(summary) != summary_hash:
            raise DriftArtifactValidationError(f"DriftArtifact {family} summary hash 校验失败")
    label_summary = parsed.get("label_drift")
    label_summary_hash = parsed.get("label_drift_summary_sha256")
    concept_summary = parsed.get("concept_performance")
    concept_summary_hash = parsed.get("concept_performance_summary_sha256")
    if label_summary == "PENDING_LABELS":
        if (label_summary_hash != "PENDING_LABELS" or concept_summary != "PENDING_LABELS" or
                concept_summary_hash != "PENDING_LABELS"):
            raise DriftArtifactValidationError("标签未成熟时 label/concept summary 必须 pending")
    elif isinstance(label_summary, dict):
        if not _digest_like(label_summary_hash) or _summary_sha256(label_summary) != label_summary_hash:
            raise DriftArtifactValidationError("DriftArtifact label summary hash 校验失败")
    else:
        raise DriftArtifactValidationError("DriftArtifact label_drift 无效")
    if concept_summary == "PENDING_LABELS":
        if concept_summary_hash != "PENDING_LABELS":
            raise DriftArtifactValidationError("concept summary pending hash 无效")
    elif concept_summary == "INSUFFICIENT_DATA":
        if concept_summary_hash != "INSUFFICIENT_DATA":
            raise DriftArtifactValidationError("concept summary insufficient hash 无效")
    elif isinstance(concept_summary, dict):
        if (not _digest_like(concept_summary_hash) or
                _summary_sha256(concept_summary) != concept_summary_hash):
            raise DriftArtifactValidationError("concept summary hash 校验失败")
    else:
        raise DriftArtifactValidationError("concept_performance summary 无效")
    if parsed.get("alert_state") not in {"INFO", "WARN", "CRITICAL"}:
        raise DriftArtifactValidationError("DriftArtifact alert_state 无效")
    return MappingProxyType(dict(parsed))

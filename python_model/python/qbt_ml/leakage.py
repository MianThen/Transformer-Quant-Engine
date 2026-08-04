from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class LeakageViolation:
    code: str
    severity: str
    message: str
    count: int
    sample_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class LeakageReport:
    detector_version: str
    status: str
    dataset_fingerprint: str
    sample_count: int
    checks: dict[str, dict]
    violations: tuple[LeakageViolation, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["violations"] = [asdict(item) for item in self.violations]
        return value

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    def require_pass(self) -> None:
        if self.status != "PASS":
            codes = ", ".join(item.code for item in self.violations)
            raise ValueError(f"Leakage Detection 未通过: {codes}")


PROVENANCE_FIELDS = (
    "signal_asof",
    "feature_source_max_timestamp",
    "label_entry_timestamp",
    "label_exit_timestamp",
    "universe_asof",
    "reference_data_known_at_max",
)


def dataset_fingerprint(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    excluded = {"dataset_fingerprint", "leakage_report_sha256"}
    for name in sorted(set(arrays) - excluded):
        value = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        if value.dtype.kind in {"U", "S"}:
            for item in value.reshape(-1):
                encoded = str(item).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def audit_feature_time_invariance(source, build_features, *, cutoff=None) -> dict[str, bool | int]:
    timestamps = np.sort(source["timestamp"].unique())
    if timestamps.size < 2:
        raise ValueError("时间不变性检查至少需要两个时间点")
    cutoff = timestamps[timestamps.size // 2] if cutoff is None else cutoff
    prefix_source = source[source["timestamp"] <= cutoff].copy()
    full = build_features(source)
    prefix = build_features(prefix_source)
    full_prefix = full.table["timestamp"].to_numpy() <= cutoff
    prefix_pass = (
        np.array_equal(full.table.loc[full_prefix].to_numpy(), prefix.table.to_numpy())
        and np.array_equal(full.values[full_prefix], prefix.values)
        and np.array_equal(full.valid_mask[full_prefix], prefix.valid_mask)
    )

    mutated = source.copy()
    future = mutated["timestamp"] > cutoff
    for name in ("open", "high", "low", "close"):
        mutated.loc[future, name] = mutated.loc[future, name] * 1.37
    mutated.loc[future, "volume"] = mutated.loc[future, "volume"] + 7919
    changed = build_features(mutated)
    mutation_pass = (
        np.array_equal(full.values[full_prefix], changed.values[full_prefix])
        and np.array_equal(full.valid_mask[full_prefix], changed.valid_mask[full_prefix])
    )
    return {
        "cutoff": int(cutoff),
        "prefix_invariance_pass": bool(prefix_pass),
        "future_mutation_pass": bool(mutation_pass),
    }


def audit_dataset(
    arrays: Mapping[str, np.ndarray],
    *,
    split=None,
    normalizer_fit_end=None,
) -> LeakageReport:
    violations: list[LeakageViolation] = []
    checks: dict[str, dict] = {}
    sample_count = int(np.asarray(arrays.get("timestamps", [])).size)

    def add_check(code: str, passed: bool, message: str, indices=()) -> None:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        checks[code] = {"status": "PASS" if passed else "FAIL", "checked": sample_count}
        if not passed:
            violations.append(LeakageViolation(
                code, "CRITICAL", message, int(indices.size),
                tuple(int(value) for value in indices[:20]),
            ))

    missing = [name for name in PROVENANCE_FIELDS if name not in arrays]
    add_check(
        "PROVENANCE_REQUIRED", not missing,
        "数据集缺少 provenance 字段: " + ", ".join(missing),
        np.arange(len(missing)),
    )
    if not missing:
        signal = np.asarray(arrays["signal_asof"])
        feature_max = np.asarray(arrays["feature_source_max_timestamp"])
        entry = np.asarray(arrays["label_entry_timestamp"])
        exit_time = np.asarray(arrays["label_exit_timestamp"])
        reference_max = np.asarray(arrays["reference_data_known_at_max"])
        universe_asof = np.asarray(arrays["universe_asof"])
        same_length = all(np.asarray(arrays[name]).size == sample_count for name in PROVENANCE_FIELDS)
        add_check("PROVENANCE_LENGTH", same_length, "provenance 长度与样本数不一致")
        if same_length:
            bad = np.flatnonzero(feature_max > signal)
            add_check("FEATURE_TIME", bad.size == 0, "特征使用了 signal_asof 之后的数据", bad)
            bad = np.flatnonzero(reference_max > signal)
            add_check("REFERENCE_TIME", bad.size == 0, "参考数据在 signal_asof 时尚不可知", bad)
            bad = np.flatnonzero(universe_asof > signal)
            add_check("UNIVERSE_TIME", bad.size == 0, "股票池使用了未来时点", bad)
            bad = np.flatnonzero((entry <= signal) | (exit_time < entry))
            add_check("LABEL_INTERVAL", bad.size == 0, "标签 entry/exit 时间区间无效", bad)

    if "timestamps" in arrays and "symbols" in arrays:
        timestamps = np.asarray(arrays["timestamps"])
        symbols = np.asarray(arrays["symbols"]).astype(str)
        keys = np.char.add(np.char.add(timestamps.astype(str), "\x1f"), symbols)
        _, counts = np.unique(keys, return_counts=True)
        add_check("SPLIT_KEY_UNIQUENESS", bool((counts == 1).all()), "存在重复 timestamp/symbol")

    for name in ("prefix_invariance_pass", "future_mutation_pass"):
        passed = name in arrays and bool(np.asarray(arrays[name]).item())
        add_check(name.upper(), passed, f"{name} 未通过或未执行")
    point_in_time = "point_in_time_required" in arrays and bool(
        np.asarray(arrays["point_in_time_required"]).item()
    )
    add_check(
        "POINT_IN_TIME_DECLARATION", point_in_time,
        "数据集未声明 point-in-time 数据契约",
    )
    lineage_present = all(
        name in arrays and str(np.asarray(arrays[name]).item()).strip()
        for name in (
            "calendar_id", "universe_id", "feature_code_sha256",
            "label_spec_json", "label_spec_sha256",
        )
    )
    add_check("LINEAGE_REQUIRED", lineage_present, "数据集缺少日历、股票池或 LabelSpec lineage")
    if lineage_present:
        label_json = str(np.asarray(arrays["label_spec_json"]).item())
        label_hash = hashlib.sha256(label_json.encode("utf-8")).hexdigest()
        declared_label_hash = str(np.asarray(arrays["label_spec_sha256"]).item())
        add_check("LABEL_SPEC_HASH", label_hash == declared_label_hash, "LabelSpec hash 不匹配")

    if split is not None and not missing:
        partitions = {
            "train": np.asarray(split.train, dtype=np.int64),
            "validation": np.asarray(split.validation, dtype=np.int64),
            "test": np.asarray(split.test, dtype=np.int64),
        }
        disjoint = not (
            np.intersect1d(partitions["train"], partitions["validation"]).size
            or np.intersect1d(partitions["train"], partitions["test"]).size
            or np.intersect1d(partitions["validation"], partitions["test"]).size
        )
        add_check("SPLIT_INDEX_DISJOINT", disjoint, "train/validation/test 索引重叠")
        signal = np.asarray(arrays["signal_asof"])
        exit_time = np.asarray(arrays["label_exit_timestamp"])
        train_val = exit_time[partitions["train"]].max() < signal[partitions["validation"]].min()
        val_test = exit_time[partitions["validation"]].max() < signal[partitions["test"]].min()
        add_check("TRAIN_VALIDATION_PURGE", bool(train_val), "train 标签区间延伸到 validation")
        add_check("VALIDATION_TEST_PURGE", bool(val_test), "validation 标签区间延伸到 test")
        if normalizer_fit_end is not None:
            fit_pass = normalizer_fit_end <= signal[partitions["train"]].max()
            add_check("NORMALIZER_FIT_SCOPE", bool(fit_pass), "normalizer 使用了训练期之后的数据")

    actual_fingerprint = dataset_fingerprint(arrays)
    declared = str(np.asarray(arrays.get("dataset_fingerprint", actual_fingerprint)).item())
    add_check("DATASET_FINGERPRINT", declared == actual_fingerprint, "数据集 fingerprint 不匹配")
    status = "FAIL" if violations else "PASS"
    return LeakageReport(
        detector_version="1.0.0",
        status=status,
        dataset_fingerprint=actual_fingerprint,
        sample_count=sample_count,
        checks=checks,
        violations=tuple(violations),
    )

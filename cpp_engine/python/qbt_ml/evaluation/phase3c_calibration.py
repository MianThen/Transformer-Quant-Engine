"""Phase 3C evaluation on frozen three-window prediction artifacts.

This module is deliberately evaluation-only.  It aligns frozen predictions to
the PIT dataset by the exact ``(timestamp, symbol)`` key, fits probability and
CQR calibrators on each validation split, and applies the frozen artifacts to
the corresponding test split without updating them from test observations.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from python.qbt_ml.calibration.conformal import (
    RollingCqrSpec,
    apply_cqr_calibrator,
    fit_cqr_calibrator,
)
from python.qbt_ml.calibration.probability import (
    ProbabilityCalibrationValidationError,
    calibration_metrics,
    fit_weighted_isotonic,
    fit_weighted_platt,
)


PREDICTION_OUTPUTS = (
    "expected_return",
    "expected_volatility",
    "direction_probability",
    "lower_quantile",
    "upper_quantile",
    "confidence",
)
REQUIRED_PREDICTION_FIELDS = ("timestamps", "symbols", *PREDICTION_OUTPUTS)
DEFAULT_FOLDS = (1, 2, 3)
DEFAULT_TOP_K = 20
DEFAULT_ECE_BINS = 10
CQR_TARGET_COVERAGE = 0.80
CQR_CONFIG_HASH = 308001
CQR_MAXIMUM_CALIBRATION_OBSERVATIONS = 100_000


class Phase3CCalibrationError(ValueError):
    """Raised when a Phase 3C input or provenance check fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase3CCalibrationError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"缺少 JSON 文件: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise Phase3CCalibrationError(f"JSON 无效: {path}") from exc
    _require(isinstance(value, dict), f"JSON 根节点必须是对象: {path}")
    return value


def _utc_from_ns(timestamp_ns: int) -> str:
    _require(isinstance(timestamp_ns, int) and timestamp_ns > 0, "timestamp 必须为正整数")
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    _require(nanoseconds % 1_000 == 0, "校准审计时间必须至少可精确表示到微秒")
    value = datetime.fromtimestamp(
        seconds + nanoseconds / 1_000_000_000,
        tz=timezone.utc,
    )
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _key_sha256(timestamps: np.ndarray, symbols: np.ndarray) -> str:
    return _canonical_sha256(
        [[int(timestamp), str(symbol)] for timestamp, symbol in zip(timestamps, symbols)]
    )


def _float_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


@dataclass(frozen=True)
class DatasetTargets:
    timestamps: np.ndarray
    symbols: np.ndarray
    expected_return: np.ndarray
    soft_direction: np.ndarray
    key_to_index: Mapping[tuple[int, str], int]
    unique_timestamps: np.ndarray
    label_spec: Mapping[str, Any]
    label_spec_json_sha256: str


@dataclass(frozen=True)
class PredictionSplit:
    timestamps: np.ndarray
    symbols: np.ndarray
    expected_return: np.ndarray
    expected_volatility: np.ndarray
    direction_probability: np.ndarray
    lower_quantile: np.ndarray
    upper_quantile: np.ndarray
    confidence: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.timestamps.size)


@dataclass(frozen=True)
class AlignedTargets:
    expected_return: np.ndarray
    hard_direction: np.ndarray
    soft_direction: np.ndarray


def _load_dataset_targets(path: Path) -> DatasetTargets:
    _require(path.is_file(), f"缺少 PIT dataset: {path}")
    with np.load(path, allow_pickle=False) as values:
        required = {
            "timestamps",
            "symbols",
            "expected_return",
            "direction",
            "label_spec_json",
            "label_spec_sha256",
        }
        missing = sorted(required - set(values.files))
        _require(not missing, "PIT dataset 缺少字段: " + ", ".join(missing))
        timestamps = np.asarray(values["timestamps"], dtype=np.int64)
        symbols = np.asarray(values["symbols"]).astype(str)
        expected_return = np.asarray(values["expected_return"], dtype=np.float64)
        soft_direction = np.asarray(values["direction"], dtype=np.float64)
        label_spec_json = str(np.asarray(values["label_spec_json"]).item())
        supplied_label_spec_sha256 = str(
            np.asarray(values["label_spec_sha256"]).item()
        )

    size = timestamps.size
    _require(size > 0, "PIT dataset 不得为空")
    _require(
        timestamps.ndim == symbols.ndim == expected_return.ndim == soft_direction.ndim == 1,
        "PIT dataset 对齐字段必须是一维",
    )
    _require(
        symbols.size == expected_return.size == soft_direction.size == size,
        "PIT dataset 对齐字段长度不一致",
    )
    _require(np.all(timestamps > 0), "PIT dataset timestamp 必须为正")
    _require(np.all(np.isfinite(expected_return)), "expected_return 含非有限值")
    _require(
        np.all(np.isfinite(soft_direction))
        and np.all((soft_direction >= 0.0) & (soft_direction <= 1.0)),
        "训练 direction soft label 必须位于 [0,1]",
    )
    actual_label_spec_sha256 = hashlib.sha256(
        label_spec_json.encode("utf-8")
    ).hexdigest()
    _require(
        actual_label_spec_sha256 == supplied_label_spec_sha256,
        "dataset label_spec_sha256 校验失败",
    )
    try:
        label_spec = json.loads(label_spec_json)
    except json.JSONDecodeError as exc:
        raise Phase3CCalibrationError("dataset label_spec_json 无效") from exc
    _require(isinstance(label_spec, dict), "dataset label_spec_json 必须是对象")

    keys = [(int(timestamp), str(symbol)) for timestamp, symbol in zip(timestamps, symbols)]
    key_to_index = {key: index for index, key in enumerate(keys)}
    _require(len(key_to_index) == size, "PIT dataset 存在重复 timestamp+symbol")
    return DatasetTargets(
        timestamps=timestamps,
        symbols=symbols,
        expected_return=expected_return,
        soft_direction=soft_direction,
        key_to_index=key_to_index,
        unique_timestamps=np.unique(timestamps),
        label_spec=label_spec,
        label_spec_json_sha256=actual_label_spec_sha256,
    )


def _load_prediction(path: Path) -> PredictionSplit:
    _require(path.is_file(), f"缺少冻结预测: {path}")
    with np.load(path, allow_pickle=False) as values:
        missing = sorted(set(REQUIRED_PREDICTION_FIELDS) - set(values.files))
        _require(not missing, "冻结预测缺少字段: " + ", ".join(missing))
        arrays = {
            "timestamps": np.asarray(values["timestamps"], dtype=np.int64),
            "symbols": np.asarray(values["symbols"]).astype(str),
            **{
                name: np.asarray(values[name], dtype=np.float64)
                for name in PREDICTION_OUTPUTS
            },
        }

    size = arrays["timestamps"].size
    _require(size > 0, f"冻结预测不得为空: {path}")
    for name, array in arrays.items():
        _require(array.ndim == 1 and array.size == size, f"{path}: {name} 未一维对齐")
        if name not in {"timestamps", "symbols"}:
            _require(np.all(np.isfinite(array)), f"{path}: {name} 含非有限值")
    _require(np.all(arrays["timestamps"] > 0), f"{path}: timestamp 必须为正")
    probabilities = arrays["direction_probability"]
    _require(
        np.all((probabilities >= 0.0) & (probabilities <= 1.0)),
        f"{path}: direction_probability 超出 [0,1]",
    )
    _require(
        np.all(arrays["lower_quantile"] <= arrays["upper_quantile"]),
        f"{path}: q10/q90 crossing",
    )
    keys = list(zip(arrays["timestamps"].tolist(), arrays["symbols"].tolist()))
    _require(len(set(keys)) == size, f"{path}: 存在重复 timestamp+symbol")
    return PredictionSplit(**arrays)


def _align_targets(dataset: DatasetTargets, prediction: PredictionSplit) -> AlignedTargets:
    indices: list[int] = []
    missing: list[tuple[int, str]] = []
    for timestamp, symbol in zip(prediction.timestamps, prediction.symbols):
        key = (int(timestamp), str(symbol))
        index = dataset.key_to_index.get(key)
        if index is None:
            missing.append(key)
        else:
            indices.append(index)
    _require(
        not missing,
        "冻结预测无法按 timestamp+symbol 对齐 PIT dataset；首个缺失键="
        + repr(missing[0]) if missing else "",
    )
    aligned_indices = np.asarray(indices, dtype=np.int64)
    expected_return = dataset.expected_return[aligned_indices]
    soft_direction = dataset.soft_direction[aligned_indices]
    hard_direction = (expected_return > 0.0).astype(np.float64)
    return AlignedTargets(
        expected_return=expected_return,
        hard_direction=hard_direction,
        soft_direction=soft_direction,
    )


def _validate_contract(path: Path, dataset_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _load_json_object(path)
    supplied_hash = contract.get("contract_sha256")
    unsigned = dict(contract)
    unsigned.pop("contract_sha256", None)
    _require(
        isinstance(supplied_hash, str)
        and supplied_hash == _canonical_sha256(unsigned),
        "Phase 1E preregistered contract hash 校验失败",
    )
    _require(contract.get("baseline_run") == "none", "Phase 3C 必须消费冻结 none baseline")
    _require(
        contract.get("dataset_sha256") == dataset_sha256,
        "PIT dataset SHA 与 Phase 1E 冻结合同不一致",
    )
    folds = contract.get("folds")
    _require(isinstance(folds, list) and folds, "Phase 1E 合同缺少 folds")
    fold_by_number: dict[str, Any] = {}
    for item in folds:
        _require(isinstance(item, dict) and isinstance(item.get("fold"), int), "合同 fold 无效")
        fold_by_number[str(item["fold"])] = item
    return contract, fold_by_number


def _find_job_manifest(run_root: Path, fold_directory: Path) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((run_root / "jobs").glob("job-*.json")):
        value = _load_json_object(path)
        output = value.get("output")
        if isinstance(output, str) and Path(output).resolve() == fold_directory.resolve():
            matches.append((path, value))
    _require(len(matches) == 1, f"{fold_directory}: 必须唯一匹配一个 job manifest")
    return matches[0]


def _manifest_prediction(
    fold_directory: Path,
    metrics: Mapping[str, Any],
    *,
    split_role: str,
) -> tuple[Path, str, Mapping[str, Any]]:
    manifest_key = (
        "validation_prediction_artifact"
        if split_role == "validation"
        else "prediction_artifact"
    )
    artifact = metrics.get(manifest_key)
    _require(isinstance(artifact, dict), f"metrics.json 缺少 {manifest_key}")
    expected_name = (
        "validation_predictions.npz"
        if split_role == "validation"
        else "test_predictions.npz"
    )
    _require(artifact.get("path") == expected_name, f"{manifest_key}.path 未冻结为 {expected_name}")
    outputs = artifact.get("outputs")
    _require(
        isinstance(outputs, list) and set(PREDICTION_OUTPUTS).issubset(outputs),
        f"{manifest_key}.outputs 缺少六输出",
    )
    path = fold_directory / expected_name
    actual_sha256 = _file_sha256(path)
    _require(
        artifact.get("sha256") == actual_sha256,
        f"{path}: SHA 与 metrics manifest 不一致",
    )
    return path, actual_sha256, artifact


def _validate_window(
    prediction: PredictionSplit,
    metrics: Mapping[str, Any],
    contract_fold: Mapping[str, Any],
    *,
    split_role: str,
) -> None:
    sample_key = "validation_samples" if split_role == "validation" else "test_samples"
    first_key = f"{split_role}_first"
    last_key = f"{split_role}_last"
    split = metrics.get("split")
    _require(isinstance(split, dict), "metrics.json 缺少 split")
    first = int(np.min(prediction.timestamps))
    last = int(np.max(prediction.timestamps))
    _require(metrics.get(sample_key) == prediction.sample_count, f"{sample_key} 与预测不一致")
    _require(split.get(first_key) == first and split.get(last_key) == last, f"metrics {split_role} 边界不一致")
    _require(
        contract_fold.get(first_key) == first and contract_fold.get(last_key) == last,
        f"合同 {split_role} 边界与预测不一致",
    )


def _interval_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    _require(lower.size == upper.size == target.size and lower.size > 0, "区间评估输入未对齐")
    _require(np.all(lower <= upper), "区间评估发生 crossing")
    covered = (target >= lower) & (target <= upper)
    lower_miss = target < lower
    upper_miss = target > upper
    widths = upper - lower
    return {
        "sample_count": int(target.size),
        "empirical_coverage": float(np.mean(covered)),
        "lower_miss_rate": float(np.mean(lower_miss)),
        "upper_miss_rate": float(np.mean(upper_miss)),
        "mean_interval_width": float(np.mean(widths)),
        "median_interval_width": float(np.median(widths)),
    }


def _top_k_stability(
    timestamps: np.ndarray,
    symbols: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    top_k: int,
) -> dict[str, Any]:
    _require(isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0, "top_k 必须为正整数")
    _require(
        timestamps.size == symbols.size == reference.size == candidate.size,
        "top-k 输入未对齐",
    )
    overlaps: list[float] = []
    jaccards: list[float] = []
    exact_membership = 0
    exact_order = 0
    universe_sizes: list[int] = []
    for timestamp in np.unique(timestamps):
        indices = np.flatnonzero(timestamps == timestamp)
        universe_sizes.append(int(indices.size))
        count = min(top_k, int(indices.size))
        base_order = np.lexsort((symbols[indices], -reference[indices]))[:count]
        candidate_order = np.lexsort((symbols[indices], -candidate[indices]))[:count]
        base_symbols = tuple(str(value) for value in symbols[indices][base_order])
        candidate_symbols = tuple(str(value) for value in symbols[indices][candidate_order])
        base_set = set(base_symbols)
        candidate_set = set(candidate_symbols)
        intersection = len(base_set & candidate_set)
        union = len(base_set | candidate_set)
        overlaps.append(intersection / count)
        jaccards.append(intersection / union)
        exact_membership += int(base_set == candidate_set)
        exact_order += int(base_symbols == candidate_symbols)
    cross_sections = len(overlaps)
    _require(cross_sections > 0, "top-k 没有可评估截面")
    return {
        "definition": "PER_CROSS_SECTION_CALIBRATED_VS_UNCALIBRATED_MEMBERSHIP_V1",
        "score_order": "PROBABILITY_DESC_SYMBOL_ASCENDING",
        "requested_top_k": top_k,
        "cross_section_count": cross_sections,
        "minimum_universe_size": min(universe_sizes),
        "mean_membership_overlap": float(np.mean(overlaps)),
        "minimum_membership_overlap": float(np.min(overlaps)),
        "mean_jaccard": float(np.mean(jaccards)),
        "exact_membership_match_rate": exact_membership / cross_sections,
        "exact_order_match_rate": exact_order / cross_sections,
    }


def _fold_evaluation(
    validation: PredictionSplit,
    validation_target: AlignedTargets,
    test: PredictionSplit,
    test_target: AlignedTargets,
    *,
    top_k: int,
    ece_bins: int,
) -> dict[str, Any]:
    validation_times_utc = [_utc_from_ns(int(value)) for value in validation.timestamps]
    fit_start_utc = _utc_from_ns(int(np.min(validation.timestamps)))
    fit_end_utc = _utc_from_ns(int(np.max(validation.timestamps)))
    first_test_utc = _utc_from_ns(int(np.min(test.timestamps)))
    fit_arguments = {
        "split_role": "validation",
        "fit_start_utc": fit_start_utc,
        "fit_end_utc": fit_end_utc,
        "available_at_utc": first_test_utc,
        "observation_at_utc": validation_times_utc,
        "label_available_at_utc": None,
        "minimum_sample_count": 20,
        "ece_bins": ece_bins,
    }
    fitted_methods: dict[str, dict[str, Any]] = {}
    calibrated_test: dict[str, np.ndarray | None] = {}
    for method, fitter in (
        ("platt", fit_weighted_platt),
        ("isotonic", fit_weighted_isotonic),
    ):
        try:
            artifact = fitter(
                validation.direction_probability,
                validation_target.hard_direction,
                **fit_arguments,
            )
        except ProbabilityCalibrationValidationError as exc:
            fitted_methods[method] = {
                "status": "UNAVAILABLE_FAIL_CLOSED",
                "validation_error": str(exc),
                "runtime_fallback_used": False,
            }
            calibrated_test[method] = None
            continue
        fit_payload = artifact.to_dict()
        fit_payload.update({
            "status": "AVAILABLE",
            "runtime_fallback_used": False,
        })
        fitted_methods[method] = fit_payload
        calibrated_test[method] = artifact.predict(
            test.direction_probability,
            decision_at_utc=first_test_utc,
        )
    probability_test_metrics = {
        "uncalibrated": calibration_metrics(
            test.direction_probability,
            test_target.hard_direction,
            ece_bins=ece_bins,
        ).to_dict(),
    }
    top_k_stability: dict[str, Any] = {}
    for method in ("platt", "isotonic"):
        prediction = calibrated_test[method]
        probability_test_metrics[method] = (
            None
            if prediction is None
            else calibration_metrics(
                prediction,
                test_target.hard_direction,
                ece_bins=ece_bins,
            ).to_dict()
        )
        top_k_stability[f"{method}_vs_uncalibrated"] = (
            None
            if prediction is None
            else _top_k_stability(
                test.timestamps,
                test.symbols,
                test.direction_probability,
                prediction,
                top_k=top_k,
            )
        )

    cqr_spec = RollingCqrSpec(
        target_coverage=CQR_TARGET_COVERAGE,
        minimum_calibration_observations=20,
        maximum_calibration_observations=CQR_MAXIMUM_CALIBRATION_OBSERVATIONS,
        exponential_decay=1.0,
        config_hash=CQR_CONFIG_HASH,
    )
    cqr_available_at = int(np.min(test.timestamps)) - 1
    cqr = fit_cqr_calibrator(
        validation.lower_quantile,
        validation.upper_quantile,
        validation_target.expected_return,
        validation.timestamps,
        available_at=cqr_available_at,
        spec=cqr_spec,
    )
    calibrated_lower, calibrated_upper = apply_cqr_calibrator(
        cqr,
        test.lower_quantile,
        test.upper_quantile,
        test.timestamps,
        forecast_available_at=cqr_available_at,
    )
    return {
        "validation_fit": {
            "split_role": "validation",
            "test_feedback_used": False,
            "row_wise_label_available_at_supplied": False,
            "platt": fitted_methods["platt"],
            "isotonic": fitted_methods["isotonic"],
            "cqr": cqr.to_dict(),
        },
        "test_probability_calibration": {
            "direction_probability_semantics": "P(expected_return > 0)",
            "evaluation_label": "1{PIT_dataset.expected_return > 0}",
            "method_status": {
                method: fitted_methods[method]["status"]
                for method in ("platt", "isotonic")
            },
            "metrics": probability_test_metrics,
            "top_k_stability": top_k_stability,
        },
        "test_cqr": {
            "method": "FIXED_VALIDATION_CQR_UNIFORM_FINITE_SAMPLE_HIGHER_V1",
            "nominal_interval": "q10/q90",
            "target_coverage": CQR_TARGET_COVERAGE,
            "q_hat": cqr.q_hat,
            "raw_q_hat": cqr.raw_q_hat,
            "raw": _interval_metrics(
                test.lower_quantile,
                test.upper_quantile,
                test_target.expected_return,
            ),
            "calibrated": _interval_metrics(
                calibrated_lower,
                calibrated_upper,
                test_target.expected_return,
            ),
            "test_feedback_used": False,
            "formal_distribution_free_claim": False,
        },
        "aligned_labels": {
            "validation_key_sha256": _key_sha256(
                validation.timestamps, validation.symbols
            ),
            "test_key_sha256": _key_sha256(test.timestamps, test.symbols),
            "validation_hard_positive_rate": float(
                np.mean(validation_target.hard_direction)
            ),
            "test_hard_positive_rate": float(np.mean(test_target.hard_direction)),
            "validation_training_soft_direction_summary": _float_summary(
                validation_target.soft_direction
            ),
            "test_training_soft_direction_summary": _float_summary(
                test_target.soft_direction
            ),
        },
    }


def _descriptive_summary(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probability: dict[str, Any] = {}
    for method in ("uncalibrated", "platt", "isotonic"):
        available = [
            fold["test_probability_calibration"]["metrics"][method]
            for fold in folds
            if fold["test_probability_calibration"]["metrics"][method]
            is not None
        ]
        probability[method] = {
            "available_fold_count": len(available),
            "unavailable_fold_count": len(folds) - len(available),
            **{
                metric: (
                    float(np.mean([item[metric] for item in available]))
                    if available
                    else None
                )
                for metric in ("brier", "nll", "ece")
            },
        }
    return {
        "aggregation": "UNWEIGHTED_FOLD_MACRO_DESCRIPTIVE_ONLY",
        "probability_metrics": probability,
        "cqr": {
            "raw_mean_empirical_coverage": float(
                np.mean([fold["test_cqr"]["raw"]["empirical_coverage"] for fold in folds])
            ),
            "calibrated_mean_empirical_coverage": float(
                np.mean(
                    [fold["test_cqr"]["calibrated"]["empirical_coverage"] for fold in folds]
                )
            ),
            "raw_mean_interval_width": float(
                np.mean([fold["test_cqr"]["raw"]["mean_interval_width"] for fold in folds])
            ),
            "calibrated_mean_interval_width": float(
                np.mean(
                    [fold["test_cqr"]["calibrated"]["mean_interval_width"] for fold in folds]
                )
            ),
            "mean_q_hat": float(np.mean([fold["test_cqr"]["q_hat"] for fold in folds])),
        },
    }


def run_phase3c_calibration(
    dataset_path: Path,
    run_root: Path,
    output_path: Path,
    *,
    folds: Sequence[int] = DEFAULT_FOLDS,
    top_k: int = DEFAULT_TOP_K,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, Any]:
    """Run and persist the deterministic Phase 3C calibration report."""

    dataset_path = dataset_path.resolve()
    run_root = run_root.resolve()
    output_path = output_path.resolve()
    normalized_folds = tuple(int(value) for value in folds)
    _require(normalized_folds and len(set(normalized_folds)) == len(normalized_folds), "folds 必须非空且不重复")
    _require(all(value > 0 for value in normalized_folds), "fold 编号必须为正")
    _require(isinstance(ece_bins, int) and ece_bins >= 2, "ece_bins 必须至少为 2")

    dataset_sha256 = _file_sha256(dataset_path)
    dataset = _load_dataset_targets(dataset_path)
    contract_path = run_root / "preregistered_contract.json"
    contract, contract_folds = _validate_contract(contract_path, dataset_sha256)
    _require(
        set(normalized_folds).issubset({int(value) for value in contract_folds}),
        "请求 fold 不在冻结合同中",
    )

    fold_reports: list[dict[str, Any]] = []
    provenance_folds: list[dict[str, Any]] = []
    for fold_number in normalized_folds:
        contract_fold = contract_folds[str(fold_number)]
        fold_directory = run_root / f"fold-{fold_number}" / "none"
        metrics_path = fold_directory / "metrics.json"
        metrics = _load_json_object(metrics_path)
        validation_path, validation_sha256, validation_manifest = _manifest_prediction(
            fold_directory,
            metrics,
            split_role="validation",
        )
        test_path, test_sha256, test_manifest = _manifest_prediction(
            fold_directory,
            metrics,
            split_role="test",
        )
        validation = _load_prediction(validation_path)
        test = _load_prediction(test_path)
        _validate_window(
            validation,
            metrics,
            contract_fold,
            split_role="validation",
        )
        _validate_window(test, metrics, contract_fold, split_role="test")
        _require(
            int(np.max(validation.timestamps)) < int(np.min(test.timestamps)),
            f"fold-{fold_number}: validation/test 时间窗口重叠",
        )

        job_path, job = _find_job_manifest(run_root, fold_directory)
        config = job.get("config")
        _require(isinstance(config, dict), f"{job_path}: config 缺失")
        label_v2 = config.get("label_v2")
        split_config = config.get("split")
        _require(isinstance(label_v2, dict), f"{job_path}: label_v2 缺失")
        _require(isinstance(split_config, dict), f"{job_path}: split 缺失")
        horizon_bars = label_v2.get("horizon_bars")
        purge_timestamps = split_config.get("purge_timestamps")
        _require(horizon_bars == 5, f"{job_path}: horizon_bars 未冻结为 5")
        _require(purge_timestamps == 6, f"{job_path}: purge_timestamps 未冻结为 6")
        intervening_timestamps = int(
            np.count_nonzero(
                (dataset.unique_timestamps > int(np.max(validation.timestamps)))
                & (dataset.unique_timestamps < int(np.min(test.timestamps)))
            )
        )
        _require(
            intervening_timestamps >= purge_timestamps,
            f"fold-{fold_number}: validation/test 间隔不足 purge=6",
        )

        validation_target = _align_targets(dataset, validation)
        test_target = _align_targets(dataset, test)
        evaluation = _fold_evaluation(
            validation,
            validation_target,
            test,
            test_target,
            top_k=top_k,
            ece_bins=ece_bins,
        )
        fold_reports.append(
            {
                "fold": fold_number,
                "validation_sample_count": validation.sample_count,
                "test_sample_count": test.sample_count,
                "validation_first_timestamp": int(np.min(validation.timestamps)),
                "validation_last_timestamp": int(np.max(validation.timestamps)),
                "test_first_timestamp": int(np.min(test.timestamps)),
                "test_last_timestamp": int(np.max(test.timestamps)),
                "label_maturity_support": {
                    "horizon_bars": horizon_bars,
                    "purge_timestamps": purge_timestamps,
                    "embargo_timestamps": split_config.get("embargo_timestamps"),
                    "intervening_dataset_timestamps": intervening_timestamps,
                    "row_wise_label_available_at_present": False,
                    "claim": "WINDOW_LEVEL_SUPPORT_ONLY_NOT_ROW_WISE_PROOF",
                },
                **evaluation,
            }
        )
        provenance_folds.append(
            {
                "fold": fold_number,
                "metrics_manifest": {
                    "path": str(metrics_path),
                    "file_sha256": _file_sha256(metrics_path),
                },
                "job_manifest": {
                    "path": str(job_path),
                    "file_sha256": _file_sha256(job_path),
                },
                "validation_prediction": {
                    "path": str(validation_path),
                    "file_sha256": validation_sha256,
                    "metrics_manifest_sha256": validation_manifest["sha256"],
                    "sha_matches_metrics_manifest": True,
                },
                "test_prediction": {
                    "path": str(test_path),
                    "file_sha256": test_sha256,
                    "metrics_manifest_sha256": test_manifest["sha256"],
                    "sha_matches_metrics_manifest": True,
                },
            }
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "role": "phase3c_real_three_window_calibration_report",
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "evidence_tier": "REAL_FROZEN_OOS_DIAGNOSTIC",
        "diagnostic_only": True,
        "phase_exit_eligible": False,
        "promotion_eligible": False,
        "winner_selected": False,
        "post_hoc_winner_selection_allowed": False,
        "input_validation": {
            "all_checks_passed": True,
            "dataset": {
                "path": str(dataset_path),
                "file_sha256": dataset_sha256,
                "contract_declared_sha256": contract["dataset_sha256"],
                "sha_matches_preregistered_contract": True,
                "label_spec_sha256": dataset.label_spec_json_sha256,
            },
            "preregistered_contract": {
                "path": str(contract_path),
                "file_sha256": _file_sha256(contract_path),
                "canonical_contract_sha256": contract["contract_sha256"],
                "canonical_hash_valid": True,
            },
            "folds": provenance_folds,
        },
        "evaluation_contract": {
            "folds": list(normalized_folds),
            "model_run": "none",
            "alignment_key": ["timestamp", "symbol"],
            "alignment_policy": "EXACT_ONE_TO_ONE_FAIL_CLOSED",
            "direction_probability_semantics": "P(expected_return > 0)",
            "evaluation_direction_label": {
                "kind": "HARD_EVENT",
                "definition": "1{PIT_dataset.expected_return > 0}",
            },
            "original_training_direction_label": {
                "kind": "SOFT_LABEL",
                "dataset_field": "direction",
                "direction_temperature": dataset.label_spec.get("direction_temperature"),
                "direction_threshold": dataset.label_spec.get("direction_threshold"),
                "label_spec_sha256": dataset.label_spec_json_sha256,
            },
            "probability_calibration_fit_split": "validation",
            "probability_calibration_apply_split": "test",
            "test_feedback_into_calibrator": False,
            "ece_bins": ece_bins,
            "top_k": top_k,
            "cqr": {
                "fit_split": "validation",
                "apply_split": "test",
                "target_coverage": CQR_TARGET_COVERAGE,
                "nominal_quantiles": [0.10, 0.90],
                "calibration_weighting": "UNIFORM",
                "finite_sample_quantile": "HIGHER",
                "test_feedback_into_calibrator": False,
                "formal_distribution_free_claim": False,
            },
            "label_maturity": {
                "horizon_bars": 5,
                "purge_timestamps": 6,
                "row_wise_label_available_at_present": False,
                "evidence_scope": "JOB_CONFIG_AND_WINDOW_SEPARATION_ONLY",
            },
            "numeric_exit_gate_preregistered": False,
            "gate_consequence": "NO_PHASE_EXIT_AND_NO_POST_HOC_WINNER",
        },
        "folds": fold_reports,
        "descriptive_summary": _descriptive_summary(fold_reports),
        "limitations": [
            "NO_PREREGISTERED_NUMERIC_PHASE3C_EXIT_GATE",
            "NO_ROW_WISE_LABEL_AVAILABLE_AT_PROVENANCE",
            "LABEL_MATURITY_SUPPORTED_ONLY_BY_HORIZON_5_PURGE_6_AND_WINDOW_GAP",
            "FINANCIAL_TIME_SERIES_SHIFT_INVALIDATES_UNCONDITIONAL_DISTRIBUTION_FREE_CLAIM",
            "DESCRIPTIVE_COMPARISON_ONLY_NO_POST_HOC_CALIBRATOR_WINNER",
        ],
    }
    report["report_sha256"] = _canonical_sha256(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "Phase3CCalibrationError",
    "run_phase3c_calibration",
]

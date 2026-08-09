from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from python.qbt_ml.evaluation.phase3c_calibration import (
    Phase3CCalibrationError,
    run_phase3c_calibration,
)


DAY_NS = 86_400_000_000_000
BASE_NS = 1_700_000_000_000_000_000


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _prediction_payload(
    timestamps: np.ndarray,
    symbols: np.ndarray,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    symbol_index = np.asarray([int(value[1:]) for value in symbols], dtype=np.float64)
    time_offset = (timestamps - int(np.min(timestamps))) / DAY_NS
    probability = 0.08 + 0.07 * symbol_index + 0.002 * time_offset
    return {
        "timestamps": timestamps,
        "symbols": symbols,
        "scores": -0.02 + 0.004 * symbol_index,
        "utility": target,
        "relevance": np.clip(0.1 + 0.07 * symbol_index, 0.0, 1.0),
        "expected_return": -0.02 + 0.004 * symbol_index,
        "expected_volatility": 0.02 + 0.0002 * symbol_index,
        "direction_probability": probability,
        "lower_quantile": np.full(target.size, -0.005),
        "upper_quantile": np.full(target.size, 0.005),
        "confidence": np.full(target.size, 0.65),
    }


def _write_case(tmp_path: Path) -> tuple[Path, Path]:
    dataset_path = tmp_path / "phase1e_pit_dataset.npz"
    run_root = tmp_path / "run"
    fold_directory = run_root / "fold-1" / "none"
    fold_directory.mkdir(parents=True)

    unique_timestamps = BASE_NS + np.arange(10, dtype=np.int64) * DAY_NS
    symbols = np.asarray([f"S{index:02d}" for index in range(12)])
    timestamps = np.repeat(unique_timestamps, symbols.size)
    tiled_symbols = np.tile(symbols, unique_timestamps.size)
    return_pattern = np.asarray(
        [-0.010, -0.008, -0.006, 0.002, -0.004, -0.002,
         0.004, -0.001, 0.006, 0.008, -0.0005, 0.010],
        dtype=np.float64,
    )
    expected_return = np.tile(return_pattern, unique_timestamps.size)
    soft_direction = 1.0 / (1.0 + np.exp(-expected_return / 0.0025))
    label_spec_json = json.dumps(
        {
            "direction_temperature": 0.0025,
            "direction_threshold": 0.0,
            "execution_lag_bars": 1,
            "horizon_bars": 5,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    np.savez(
        dataset_path,
        timestamps=timestamps,
        symbols=tiled_symbols,
        expected_return=expected_return,
        direction=soft_direction,
        label_spec_json=np.asarray(label_spec_json),
        label_spec_sha256=np.asarray(
            hashlib.sha256(label_spec_json.encode("utf-8")).hexdigest()
        ),
    )

    validation_selected = np.isin(timestamps, unique_timestamps[:2])
    test_selected = np.isin(timestamps, unique_timestamps[8:])
    validation_payload = _prediction_payload(
        timestamps[validation_selected],
        tiled_symbols[validation_selected],
        expected_return[validation_selected],
    )
    test_payload = _prediction_payload(
        timestamps[test_selected],
        tiled_symbols[test_selected],
        expected_return[test_selected],
    )
    validation_path = fold_directory / "validation_predictions.npz"
    test_path = fold_directory / "test_predictions.npz"
    np.savez(validation_path, **validation_payload)
    np.savez(test_path, **test_payload)

    split = {
        "validation_first": int(unique_timestamps[0]),
        "validation_last": int(unique_timestamps[1]),
        "test_first": int(unique_timestamps[8]),
        "test_last": int(unique_timestamps[9]),
    }
    outputs = [
        "expected_return",
        "expected_volatility",
        "direction_probability",
        "lower_quantile",
        "upper_quantile",
        "confidence",
    ]
    metrics = {
        "split": split,
        "validation_samples": int(validation_payload["timestamps"].size),
        "test_samples": int(test_payload["timestamps"].size),
        "validation_prediction_artifact": {
            "path": validation_path.name,
            "outputs": outputs,
            "sha256": _file_hash(validation_path),
        },
        "prediction_artifact": {
            "path": test_path.name,
            "outputs": outputs,
            "sha256": _file_hash(test_path),
        },
    }
    _write_json(fold_directory / "metrics.json", metrics)

    contract = {
        "baseline_run": "none",
        "dataset_sha256": _file_hash(dataset_path),
        "folds": [{"fold": 1, **split}],
        "window_count": 1,
    }
    contract["contract_sha256"] = _canonical_hash(contract)
    _write_json(run_root / "preregistered_contract.json", contract)
    _write_json(
        run_root / "jobs" / "job-1.json",
        {
            "config": {
                "label_v2": {"horizon_bars": 5},
                "split": {"purge_timestamps": 6, "embargo_timestamps": 5},
            },
            "dataset": str(dataset_path.resolve()),
            "output": str(fold_directory.resolve()),
            "split": split,
        },
    )
    return dataset_path, run_root


def _refresh_contract_dataset_hash(dataset_path: Path, run_root: Path) -> None:
    path = run_root / "preregistered_contract.json"
    contract = json.loads(path.read_text(encoding="utf-8"))
    contract["dataset_sha256"] = _file_hash(dataset_path)
    contract.pop("contract_sha256")
    contract["contract_sha256"] = _canonical_hash(contract)
    _write_json(path, contract)


def test_real_report_contract_metrics_and_test_only_application(tmp_path: Path) -> None:
    dataset_path, run_root = _write_case(tmp_path)
    output = tmp_path / "report.json"
    first = run_phase3c_calibration(
        dataset_path,
        run_root,
        output,
        folds=(1,),
        top_k=5,
        ece_bins=5,
    )
    replay = run_phase3c_calibration(
        dataset_path,
        run_root,
        tmp_path / "replay.json",
        folds=(1,),
        top_k=5,
        ece_bins=5,
    )

    assert first["report_sha256"] == replay["report_sha256"]
    assert first["input_validation"]["all_checks_passed"] is True
    assert first["phase_exit_eligible"] is False
    assert first["winner_selected"] is False
    fold = first["folds"][0]
    assert fold["validation_fit"]["test_feedback_used"] is False
    assert fold["test_cqr"]["target_coverage"] == pytest.approx(0.8)
    assert fold["test_cqr"]["formal_distribution_free_claim"] is False
    assert set(fold["test_probability_calibration"]["metrics"]) == {
        "uncalibrated",
        "platt",
        "isotonic",
    }
    assert fold["test_probability_calibration"]["top_k_stability"][
        "platt_vs_uncalibrated"
    ]["mean_membership_overlap"] == pytest.approx(1.0)
    assert fold["label_maturity_support"]["intervening_dataset_timestamps"] == 6
    assert first["evaluation_contract"]["original_training_direction_label"][
        "kind"
    ] == "SOFT_LABEL"


def test_test_label_mutation_does_not_refit_validation_calibrators(tmp_path: Path) -> None:
    dataset_path, run_root = _write_case(tmp_path)
    first = run_phase3c_calibration(
        dataset_path,
        run_root,
        tmp_path / "first.json",
        folds=(1,),
        top_k=5,
        ece_bins=5,
    )
    with np.load(dataset_path, allow_pickle=False) as values:
        dataset = {name: np.asarray(values[name]) for name in values.files}
    test_start = BASE_NS + 8 * DAY_NS
    selected = dataset["timestamps"] >= test_start
    dataset["expected_return"] = dataset["expected_return"].copy()
    dataset["direction"] = dataset["direction"].copy()
    dataset["expected_return"][selected] *= -1.0
    dataset["direction"][selected] = 1.0 - dataset["direction"][selected]
    np.savez(dataset_path, **dataset)
    _refresh_contract_dataset_hash(dataset_path, run_root)

    changed = run_phase3c_calibration(
        dataset_path,
        run_root,
        tmp_path / "changed.json",
        folds=(1,),
        top_k=5,
        ece_bins=5,
    )
    first_fold = first["folds"][0]
    changed_fold = changed["folds"][0]
    for method in ("platt", "isotonic", "cqr"):
        assert first_fold["validation_fit"][method]["artifact_sha256"] == (
            changed_fold["validation_fit"][method]["artifact_sha256"]
        )
    assert first_fold["test_probability_calibration"]["metrics"]["uncalibrated"][
        "brier"
    ] != changed_fold["test_probability_calibration"]["metrics"]["uncalibrated"][
        "brier"
    ]


def test_prediction_sha_must_match_metrics_manifest(tmp_path: Path) -> None:
    dataset_path, run_root = _write_case(tmp_path)
    metrics_path = run_root / "fold-1" / "none" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["prediction_artifact"]["sha256"] = "0" * 64
    _write_json(metrics_path, metrics)

    with pytest.raises(Phase3CCalibrationError, match="SHA 与 metrics manifest"):
        run_phase3c_calibration(
            dataset_path,
            run_root,
            tmp_path / "report.json",
            folds=(1,),
            top_k=5,
            ece_bins=5,
        )


def test_reverse_calibrators_fail_closed_without_aborting_report(
    tmp_path: Path,
) -> None:
    dataset_path, run_root = _write_case(tmp_path)
    prediction_path = (
        run_root / "fold-1" / "none" / "validation_predictions.npz"
    )
    with np.load(prediction_path, allow_pickle=False) as values:
        prediction = {name: np.asarray(values[name]) for name in values.files}
    target_by_symbol = {
        f"S{index:02d}": value
        for index, value in enumerate(
            [-1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1]
        )
    }
    prediction["direction_probability"] = np.asarray([
        0.1 if target_by_symbol[str(symbol)] > 0 else 0.9
        for symbol in prediction["symbols"]
    ])
    np.savez(prediction_path, **prediction)
    metrics_path = run_root / "fold-1" / "none" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["validation_prediction_artifact"]["sha256"] = _file_hash(
        prediction_path
    )
    _write_json(metrics_path, metrics)

    report = run_phase3c_calibration(
        dataset_path,
        run_root,
        tmp_path / "report.json",
        folds=(1,),
        top_k=5,
        ece_bins=5,
    )

    fold = report["folds"][0]
    assert fold["validation_fit"]["platt"]["status"] == (
        "UNAVAILABLE_FAIL_CLOSED"
    )
    assert fold["validation_fit"]["isotonic"]["status"] == (
        "UNAVAILABLE_FAIL_CLOSED"
    )
    assert fold["test_probability_calibration"]["metrics"]["platt"] is None
    assert fold["test_probability_calibration"]["top_k_stability"][
        "isotonic_vs_uncalibrated"
    ] is None
    assert fold["validation_fit"]["platt"]["runtime_fallback_used"] is False

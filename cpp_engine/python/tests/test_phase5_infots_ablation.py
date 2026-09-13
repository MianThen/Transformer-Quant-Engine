from __future__ import annotations

import copy

import pytest

from python.qbt_ml.evaluation.infots_ablation import (
    FOLD_IDS,
    GROUP_IDS,
    METRIC_NAMES,
    PREDICTION_OUTPUTS,
    InfoTSAblationValidationError,
    InfoTSBudgetContractV1,
    build_infots_ablation_report,
    validate_infots_ablation_report,
    validate_infots_fold_artifact,
)


def _contract() -> InfoTSBudgetContractV1:
    return InfoTSBudgetContractV1(supervised_budget=50, gate_spec_sha256="a" * 64)


def _artifact(group_id: str, fold_id: int, *, metric_shift: float = 0.0) -> dict:
    contract = _contract()
    return {
        "schema_version": 1,
        "group_id": group_id,
        "fold_id": fold_id,
        "supervised_budget": contract.supervised_budget,
        "gate_spec_sha256": contract.gate_spec_sha256,
        "test_blind": True,
        "split": {
            "train_end": 100,
            "validation_end": 120,
            "test_start": 125,
            "test_end": 140,
            "purge_gap": 4,
        },
        "prediction_artifact": {
            "path": f"fold-{fold_id}/{group_id}/test_predictions.npz",
            "sha256": "b" * 64,
            "outputs": list(PREDICTION_OUTPUTS),
            "row_count": 10,
        },
        "embedding_snapshots": [
            {"role": "validation", "sha256": "c" * 64, "row_count": 10, "dimension": 16},
            {"role": "test", "sha256": "d" * 64, "row_count": 10, "dimension": 16},
        ],
        "metrics": {
            "composite_error": 0.4 + metric_shift,
            "return_mae": 0.1 + metric_shift,
            "direction_brier": 0.2 + metric_shift,
            "volatility_mae": 0.03 + metric_shift,
            "ndcg_at_20": 0.5 - metric_shift,
            "rank_ic": 0.1 - metric_shift,
        },
        "gates": {
            "clean_non_degraded": True,
            "stress_non_degraded": True,
            "three_window_consistent": True,
            "quality_gate_passed": True,
            "stability_gate_passed": True,
        },
    }


def test_three_group_equal_budget_report_is_deterministic_and_never_selects_winner():
    contract = _contract()
    artifacts = [_artifact(group_id, fold_id, metric_shift=0.01 if group_id == "infots" else 0.0) for group_id in GROUP_IDS for fold_id in FOLD_IDS]
    report = build_infots_ablation_report(artifacts, contract)
    validate_infots_ablation_report(report)
    repeated = build_infots_ablation_report(artifacts, contract)
    assert report == repeated
    assert report["test_blind"] is True
    assert report["purged_oos_guard_passed"] is True
    assert report["all_gates_passed"] is True
    assert report["phase_exit_eligible"] is False
    assert report["winner_selected"] is False


def test_ablation_rejects_future_leakage_duplicate_or_budget_mismatch():
    contract = _contract()
    baseline = _artifact("from_scratch", 1)
    future = copy.deepcopy(baseline)
    future["split"]["test_start"] = 123
    with pytest.raises(InfoTSAblationValidationError):
        validate_infots_fold_artifact(future, contract)
    duplicate = [_artifact(group_id, fold_id) for group_id in GROUP_IDS for fold_id in FOLD_IDS]
    duplicate[-1] = copy.deepcopy(duplicate[-2])
    with pytest.raises(InfoTSAblationValidationError):
        build_infots_ablation_report(duplicate, contract)
    wrong_budget = _artifact("from_scratch", 1)
    wrong_budget["supervised_budget"] = 49
    with pytest.raises(InfoTSAblationValidationError):
        validate_infots_fold_artifact(wrong_budget, contract)


def test_ablation_rejects_six_output_or_embedding_provenance_break():
    contract = _contract()
    broken = _artifact("infots", 1)
    broken["prediction_artifact"]["outputs"] = list(PREDICTION_OUTPUTS[:-1])
    with pytest.raises(InfoTSAblationValidationError):
        validate_infots_fold_artifact(broken, contract)
    broken_embedding = _artifact("infots", 1)
    broken_embedding["embedding_snapshots"][1]["sha256"] = "not-a-hash"
    with pytest.raises(InfoTSAblationValidationError):
        validate_infots_fold_artifact(broken_embedding, contract)

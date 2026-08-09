from __future__ import annotations

import copy

import pytest

from python.qbt_ml.evaluation.infots_ablation import (
    GROUP_IDS,
    PREDICTION_OUTPUTS,
    FOLD_IDS,
    InfoTSBudgetContractV1,
)
from python.qbt_ml.evaluation.infots_replay import (
    InfoTSReplayRowV1,
    InfoTSReplaySpecV1,
    InfoTSReplayValidationError,
    build_infots_joint_replay_report,
    compute_infots_cpp_report_sha256,
    compute_infots_replay_input_sha256,
    validate_infots_cpp_replay_artifact,
    validate_infots_joint_replay_report,
)


def _contract() -> InfoTSBudgetContractV1:
    return InfoTSBudgetContractV1(supervised_budget=50, gate_spec_sha256="a" * 64)


def _fold_artifact(group_id: str, fold_id: int) -> dict:
    contract = _contract()
    return {
        "schema_version": 1,
        "group_id": group_id,
        "fold_id": fold_id,
        "supervised_budget": contract.supervised_budget,
        "gate_spec_sha256": contract.gate_spec_sha256,
        "test_blind": True,
        "split": {"train_end": 100, "validation_end": 120, "test_start": 125, "test_end": 140, "purge_gap": 4},
        "prediction_artifact": {"path": f"fold-{fold_id}/{group_id}/test_predictions.npz", "sha256": ("c" * 64), "outputs": list(PREDICTION_OUTPUTS), "row_count": 24},
        "embedding_snapshots": [
            {"role": "validation", "sha256": "d" * 64, "row_count": 10, "dimension": 16},
            {"role": "test", "sha256": "e" * 64, "row_count": 24, "dimension": 16},
        ],
        "metrics": {"composite_error": 0.4, "return_mae": 0.1, "direction_brier": 0.2, "volatility_mae": 0.03, "ndcg_at_20": 0.5, "rank_ic": 0.1},
        "gates": {"clean_non_degraded": True, "stress_non_degraded": True, "three_window_consistent": True, "quality_gate_passed": True, "stability_gate_passed": True},
    }


def _spec(contract: InfoTSBudgetContractV1, artifact: dict) -> InfoTSReplaySpecV1:
    return InfoTSReplaySpecV1(
        schema_version=1,
        group_id=artifact["group_id"],
        fold_id=artifact["fold_id"],
        policy_id="INFOTS-PRECOMPUTED-PROXY-V1",
        contract_sha256=contract.contract_sha256,
        dataset_fingerprint="b" * 64,
        prediction_artifact_sha256=artifact["prediction_artifact"]["sha256"],
        validation_embedding_sha256="d" * 64,
        test_embedding_sha256="e" * 64,
        source_snapshot_set_sha256="f" * 64,
        config_hash=20260809,
    )


def _rows() -> list[InfoTSReplayRowV1]:
    return [
        InfoTSReplayRowV1(
            session_id=index + 1,
            prediction_available_at=100 + index * 10,
            decision_at=100 + index * 10,
            realized_at=110 + index * 10,
            realized_proxy_return=(0.01 if index % 4 == 0 else (-0.006 if index % 4 == 1 else (0.004 if index % 4 == 2 else -0.002))),
            prediction_outputs=(0.003, 0.02, 0.55, -0.01, 0.02, 0.8),
        )
        for index in range(24)
    ]


def _cpp_report(spec: InfoTSReplaySpecV1, rows: list[InfoTSReplayRowV1]) -> dict:
    source_hash = compute_infots_replay_input_sha256(spec, rows)
    report = {
        "schema_version": 1,
        "role": "phase5_infots_precomputed_replay",
        "status": "OK",
        "evidence_level": "RESEARCH_PROXY",
        "test_blind": True,
        "purged_oos_guard_passed": True,
        "group_id": spec.group_id,
        "fold_id": spec.fold_id,
        "minimum_tail_observations": spec.minimum_tail_observations,
        "policy_id": spec.policy_id,
        "prediction_outputs": list(PREDICTION_OUTPUTS),
        "row_count": len(rows),
        "observations": len(rows),
        "source_replay_sha256": source_hash,
        "ledger_sha256": "1" * 64,
        "ledger_hash": 987654,
        "dataset_fingerprint": spec.dataset_fingerprint,
        "contract_sha256": spec.contract_sha256,
        "prediction_artifact_sha256": spec.prediction_artifact_sha256,
        "validation_embedding_sha256": spec.validation_embedding_sha256,
        "test_embedding_sha256": spec.test_embedding_sha256,
        "source_snapshot_set_sha256": spec.source_snapshot_set_sha256,
        "metrics": {"cumulative_return": 0.04, "sharpe": 1.1, "maximum_drawdown": -0.02, "var_loss": 0.006, "expected_shortfall_loss": 0.007, "return_cvar": -0.007},
        "claim_scope": "RESEARCH_PROXY",
        "reference_price_quality": "PROXY",
        "execution_data_state": "UNAVAILABLE",
        "corporate_action_state": "UNAVAILABLE",
        "bar_reference_policy": "PROXY_BAR_REFERENCE",
        "action_policy": "NO_ACTION",
        "lot_policy": "LOT_1",
        "limit_policy": "DISABLED",
        "fee_policy": "ASSUMED_ZERO",
        "slippage_policy": "UNAVAILABLE",
        "research_comparison_eligible": True,
        "phase_exit_eligible": False,
        "promotion_eligible": False,
    }
    report["artifact_sha256"] = compute_infots_cpp_report_sha256(report)
    return report


def test_cross_language_report_hash_and_future_guard():
    contract = _contract()
    artifact = _fold_artifact("infots", 1)
    spec = _spec(contract, artifact)
    rows = _rows()
    report = _cpp_report(spec, rows)
    checked = validate_infots_cpp_replay_artifact(report, spec, len(rows), report["source_replay_sha256"])
    assert checked["artifact_sha256"] == compute_infots_cpp_report_sha256(report)
    future = copy.deepcopy(rows)
    future[2] = InfoTSReplayRowV1(**{**future[2].__dict__, "prediction_available_at": future[2].decision_at + 1})
    with pytest.raises(InfoTSReplayValidationError):
        compute_infots_replay_input_sha256(spec, future)
    with pytest.raises(InfoTSReplayValidationError):
        compute_infots_replay_input_sha256(spec, rows[:2])


def test_joint_acceptance_requires_all_nine_and_links_hashes():
    contract = _contract()
    artifacts = [_fold_artifact(group, fold) for group in GROUP_IDS for fold in FOLD_IDS]
    reports = [_cpp_report(_spec(contract, artifact), _rows()) for artifact in artifacts]
    joint = build_infots_joint_replay_report(artifacts, reports, contract)
    validate_infots_joint_replay_report(joint)
    assert joint["cpp_replay_count"] == 9
    assert joint["all_cpp_replays_passed"] is True
    assert joint["promotion_eligible"] is False

    broken = copy.deepcopy(reports)
    broken[0]["prediction_artifact_sha256"] = "0" * 64
    broken[0]["artifact_sha256"] = compute_infots_cpp_report_sha256(broken[0])
    with pytest.raises(InfoTSReplayValidationError):
        build_infots_joint_replay_report(artifacts, broken, contract)

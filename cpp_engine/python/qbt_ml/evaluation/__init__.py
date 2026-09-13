"""Deterministic evaluation runners for frozen research artifacts."""

from .phase3c_calibration import (
    Phase3CCalibrationError,
    run_phase3c_calibration,
)
from .infots_ablation import (
    InfoTSAblationValidationError,
    InfoTSBudgetContractV1,
    build_infots_ablation_report,
    validate_infots_ablation_report,
    validate_infots_fold_artifact,
)
from .infots_replay import (
    InfoTSReplayRowV1,
    InfoTSReplaySpecV1,
    InfoTSReplayValidationError,
    build_infots_joint_replay_report,
    compute_infots_cpp_report_sha256,
    compute_infots_replay_input_sha256,
    validate_infots_cpp_replay_artifact,
    validate_infots_joint_replay_report,
)

__all__ = [
    "Phase3CCalibrationError",
    "run_phase3c_calibration",
    "InfoTSAblationValidationError",
    "InfoTSBudgetContractV1",
    "build_infots_ablation_report",
    "validate_infots_ablation_report",
    "validate_infots_fold_artifact",
    "InfoTSReplayRowV1",
    "InfoTSReplaySpecV1",
    "InfoTSReplayValidationError",
    "build_infots_joint_replay_report",
    "compute_infots_cpp_report_sha256",
    "compute_infots_replay_input_sha256",
    "validate_infots_cpp_replay_artifact",
    "validate_infots_joint_replay_report",
]

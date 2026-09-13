"""Cross-language contract for Phase 5 precomputed research replay."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

from .infots_ablation import (
    GROUP_IDS,
    PREDICTION_OUTPUTS,
    InfoTSBudgetContractV1,
    InfoTSAblationValidationError,
    validate_infots_fold_artifact,
)


class InfoTSReplayValidationError(ValueError):
    """Raised when a precomputed C++ replay or its provenance is invalid."""


REPLAY_MAGIC = b"QBT-PHASE5-INFOTS-REPLAY-V1\x00"
REPORT_MAGIC = b"QBT-PHASE5-INFOTS-REPLAY-REPORT-V1\x00"
PROXY_FIELDS = {
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
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise InfoTSReplayValidationError(f"{name} 必须是小写 SHA-256")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise InfoTSReplayValidationError(f"{name} 必须是有限数值")
    return float(value)


def _pack_u32(value: int) -> bytes:
    return struct.pack(">I", int(value))


def _pack_u64(value: int) -> bytes:
    return struct.pack(">Q", int(value))


def _pack_i64(value: int) -> bytes:
    return struct.pack(">q", int(value))


def _pack_double(value: float) -> bytes:
    return struct.pack(">d", 0.0 if value == 0.0 else float(value))


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _pack_u64(len(encoded)) + encoded


@dataclass(frozen=True)
class InfoTSReplaySpecV1:
    schema_version: int
    group_id: str
    fold_id: int
    policy_id: str
    contract_sha256: str
    dataset_fingerprint: str
    prediction_artifact_sha256: str
    validation_embedding_sha256: str
    test_embedding_sha256: str
    source_snapshot_set_sha256: str
    calendar_id: str = "CN-EQUITY-RESEARCH"
    calendar_periods_per_year: float = 252.0
    config_hash: int = 1
    minimum_tail_observations: int = 20
    initial_equity: float = 1.0
    claim_scope: str = "RESEARCH_PROXY"
    reference_price_quality: str = "PROXY"
    execution_data_state: str = "UNAVAILABLE"
    corporate_action_state: str = "UNAVAILABLE"
    bar_reference_policy: str = "PROXY_BAR_REFERENCE"
    action_policy: str = "NO_ACTION"
    lot_policy: str = "LOT_1"
    limit_policy: str = "DISABLED"
    fee_policy: str = "ASSUMED_ZERO"
    slippage_policy: str = "UNAVAILABLE"

    def validate(self) -> None:
        if self.schema_version != 1 or self.group_id not in GROUP_IDS:
            raise InfoTSReplayValidationError("replay spec schema/group 无效")
        if isinstance(self.fold_id, bool) or not isinstance(self.fold_id, Integral) or self.fold_id not in (1, 2, 3):
            raise InfoTSReplayValidationError("fold_id 必须是 1/2/3")
        if not self.policy_id or isinstance(self.config_hash, bool) or self.config_hash == 0:
            raise InfoTSReplayValidationError("policy/config hash 无效")
        for name in (
            "contract_sha256", "dataset_fingerprint", "prediction_artifact_sha256",
            "validation_embedding_sha256", "test_embedding_sha256", "source_snapshot_set_sha256",
        ):
            _digest(getattr(self, name), name)
        if not self.calendar_id or not math.isfinite(float(self.calendar_periods_per_year)) or self.calendar_periods_per_year <= 0:
            raise InfoTSReplayValidationError("calendar spec 无效")
        if isinstance(self.minimum_tail_observations, bool) or self.minimum_tail_observations < 2:
            raise InfoTSReplayValidationError("minimum_tail_observations 必须至少为 2")
        if not math.isfinite(float(self.initial_equity)) or self.initial_equity <= 0:
            raise InfoTSReplayValidationError("initial_equity 必须为正有限值")
        for name, expected in PROXY_FIELDS.items():
            if getattr(self, name) != expected:
                raise InfoTSReplayValidationError(f"{name} 必须冻结为 {expected}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class InfoTSReplayRowV1:
    session_id: int
    prediction_available_at: int
    decision_at: int
    realized_at: int
    realized_proxy_return: float
    prediction_outputs: tuple[float, ...]

    def validate(self) -> None:
        if isinstance(self.session_id, bool) or self.session_id <= 0:
            raise InfoTSReplayValidationError("session_id 必须为正整数")
        if min(self.prediction_available_at, self.decision_at, self.realized_at) <= 0:
            raise InfoTSReplayValidationError("replay timestamp 必须为正整数")
        if self.prediction_available_at > self.decision_at or self.realized_at <= self.decision_at:
            raise InfoTSReplayValidationError("replay row 存在未来信息")
        realized = _finite(self.realized_proxy_return, "realized_proxy_return")
        if realized <= -1.0 or len(self.prediction_outputs) != len(PREDICTION_OUTPUTS):
            raise InfoTSReplayValidationError("proxy return 或六输出无效")
        outputs = tuple(_finite(value, f"prediction_outputs[{index}]") for index, value in enumerate(self.prediction_outputs))
        if outputs[1] < 0.0 or not 0.0 <= outputs[2] <= 1.0 or outputs[3] > outputs[4] or not 0.0 <= outputs[5] <= 1.0:
            raise InfoTSReplayValidationError("六输出范围或分位数顺序无效")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InfoTSReplayRowV1":
        row = cls(
            session_id=int(value["session_id"]),
            prediction_available_at=int(value["prediction_available_at"]),
            decision_at=int(value["decision_at"]),
            realized_at=int(value["realized_at"]),
            realized_proxy_return=float(value["realized_proxy_return"]),
            prediction_outputs=tuple(float(item) for item in value["prediction_outputs"]),
        )
        row.validate()
        return row

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "session_id": self.session_id,
            "prediction_available_at": self.prediction_available_at,
            "decision_at": self.decision_at,
            "realized_at": self.realized_at,
            "realized_proxy_return": self.realized_proxy_return,
            "prediction_outputs": list(self.prediction_outputs),
        }


def _append_spec(payload: bytearray, spec: InfoTSReplaySpecV1) -> None:
    payload.extend(_pack_u32(spec.schema_version))
    payload.extend(_pack_string(spec.group_id))
    payload.extend(_pack_u32(spec.fold_id))
    payload.extend(_pack_string(spec.policy_id))
    payload.extend(_pack_string(spec.contract_sha256))
    payload.extend(_pack_string(spec.dataset_fingerprint))
    payload.extend(_pack_string(spec.prediction_artifact_sha256))
    payload.extend(_pack_string(spec.validation_embedding_sha256))
    payload.extend(_pack_string(spec.test_embedding_sha256))
    payload.extend(_pack_string(spec.source_snapshot_set_sha256))
    payload.extend(_pack_string(spec.calendar_id))
    payload.extend(_pack_double(spec.calendar_periods_per_year))
    payload.extend(_pack_u64(spec.config_hash))
    payload.extend(_pack_u32(spec.minimum_tail_observations))
    payload.extend(_pack_double(spec.initial_equity))
    for name in PROXY_FIELDS:
        payload.extend(_pack_string(getattr(spec, name)))


def _append_row(payload: bytearray, row: InfoTSReplayRowV1) -> None:
    payload.extend(_pack_u64(row.session_id))
    payload.extend(_pack_i64(row.prediction_available_at))
    payload.extend(_pack_i64(row.decision_at))
    payload.extend(_pack_i64(row.realized_at))
    payload.extend(_pack_double(row.realized_proxy_return))
    for value in row.prediction_outputs:
        payload.extend(_pack_double(value))


def compute_infots_replay_input_sha256(spec: InfoTSReplaySpecV1, rows: Sequence[InfoTSReplayRowV1]) -> str:
    spec.validate()
    normalized = tuple(rows)
    if len(normalized) < max(2, spec.minimum_tail_observations):
        raise InfoTSReplayValidationError("replay rows 不足以输出经验 CVaR")
    previous: InfoTSReplayRowV1 | None = None
    for row in normalized:
        row.validate()
        if previous is not None and (
            row.session_id <= previous.session_id or
            row.prediction_available_at < previous.prediction_available_at or
            row.decision_at < previous.decision_at or
            row.decision_at < previous.realized_at or
            row.realized_at <= previous.realized_at
        ):
            raise InfoTSReplayValidationError("replay rows 时间或 session 未单调")
        previous = row
    payload = bytearray(REPLAY_MAGIC)
    _append_spec(payload, spec)
    payload.extend(_pack_u64(len(normalized)))
    for row in normalized:
        _append_row(payload, row)
    return hashlib.sha256(payload).hexdigest()


def _report_hash_payload(report: Mapping[str, Any]) -> bytes:
    payload = bytearray(REPORT_MAGIC)
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise InfoTSReplayValidationError("C++ report 缺少 metrics")
    for name in ("status", "group_id"):
        value = report.get(name)
        if not isinstance(value, str):
            raise InfoTSReplayValidationError(f"report.{name} 无效")
        payload.extend(_pack_string(value))
    payload.extend(_pack_u32(report.get("fold_id")))
    payload.extend(_pack_u32(report.get("minimum_tail_observations")))
    policy_id = report.get("policy_id")
    if not isinstance(policy_id, str):
        raise InfoTSReplayValidationError("report.policy_id 无效")
    payload.extend(_pack_string(policy_id))
    payload.extend(_pack_u64(report.get("row_count")))
    payload.extend(_pack_u64(report.get("observations")))
    for name in ("source_replay_sha256", "ledger_sha256"):
        value = report.get(name)
        if not isinstance(value, str):
            raise InfoTSReplayValidationError(f"report.{name} 无效")
        payload.extend(_pack_string(value))
    payload.extend(_pack_u64(report.get("ledger_hash")))
    for name in (
        "dataset_fingerprint", "contract_sha256", "prediction_artifact_sha256",
        "validation_embedding_sha256", "test_embedding_sha256", "source_snapshot_set_sha256",
    ):
        value = report.get(name)
        if not isinstance(value, str):
            raise InfoTSReplayValidationError(f"report.{name} 无效")
        payload.extend(_pack_string(value))
    for name in (
        "cumulative_return", "sharpe", "maximum_drawdown", "var_loss",
        "expected_shortfall_loss", "return_cvar",
    ):
        payload.extend(_pack_double(_finite(metrics.get(name), f"metrics.{name}")))
    for name in ("research_comparison_eligible", "phase_exit_eligible", "promotion_eligible"):
        value = report.get(name)
        if not isinstance(value, bool):
            raise InfoTSReplayValidationError(f"report.{name} 无效")
        payload.extend(_pack_u32(1 if value else 0))
    payload.extend(_pack_u32(1))
    payload.extend(_pack_u32(1))
    return bytes(payload)


def compute_infots_cpp_report_sha256(report: Mapping[str, Any]) -> str:
    return hashlib.sha256(_report_hash_payload(report)).hexdigest()


def validate_infots_cpp_replay_artifact(
    report: Mapping[str, Any],
    expected_spec: InfoTSReplaySpecV1 | None = None,
    expected_row_count: int | None = None,
    expected_input_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(report, Mapping) or report.get("schema_version") != 1 or report.get("role") != "phase5_infots_precomputed_replay":
        raise InfoTSReplayValidationError("C++ InfoTS replay report schema 无效")
    if report.get("status") != "OK" or report.get("evidence_level") != "RESEARCH_PROXY":
        raise InfoTSReplayValidationError("C++ replay 必须是成功的 RESEARCH_PROXY")
    if tuple(report.get("prediction_outputs", ())) != PREDICTION_OUTPUTS:
        raise InfoTSReplayValidationError("C++ replay 六输出语义不匹配")
    if expected_spec is not None:
        expected_spec.validate()
        for name in ("group_id", "fold_id", "policy_id", "contract_sha256", "dataset_fingerprint", "prediction_artifact_sha256", "validation_embedding_sha256", "test_embedding_sha256", "source_snapshot_set_sha256"):
            if report.get(name) != getattr(expected_spec, name):
                raise InfoTSReplayValidationError(f"C++ replay {name} provenance 不匹配")
    for name, expected in PROXY_FIELDS.items():
        if report.get(name) != expected:
            raise InfoTSReplayValidationError(f"C++ replay {name} 状态不匹配")
    for name in ("source_replay_sha256", "ledger_sha256", "artifact_sha256"):
        _digest(report.get(name), name)
    if expected_row_count is not None and report.get("row_count") != expected_row_count:
        raise InfoTSReplayValidationError("C++ replay row_count 与 prediction artifact 不匹配")
    minimum_tail = report.get("minimum_tail_observations")
    if isinstance(minimum_tail, bool) or not isinstance(minimum_tail, Integral) or minimum_tail < 2:
        raise InfoTSReplayValidationError("C++ replay minimum_tail_observations 无效")
    if report.get("observations") != report.get("row_count") or report.get("row_count", 0) < minimum_tail:
        raise InfoTSReplayValidationError("C++ replay observations 不足以输出 CVaR")
    if expected_input_sha256 is not None and report.get("source_replay_sha256") != expected_input_sha256:
        raise InfoTSReplayValidationError("C++ replay source hash 不匹配")
    if report.get("test_blind") is not True or report.get("purged_oos_guard_passed") is not True:
        raise InfoTSReplayValidationError("C++ replay test-blind/purge gate 不安全")
    if report.get("research_comparison_eligible") is not True or report.get("phase_exit_eligible") is not False or report.get("promotion_eligible") is not False:
        raise InfoTSReplayValidationError("C++ replay 晋级状态不安全")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise InfoTSReplayValidationError("C++ replay 缺少 metrics")
    for name in ("cumulative_return", "sharpe", "maximum_drawdown", "var_loss", "expected_shortfall_loss", "return_cvar"):
        if metrics.get(name) is None:
            raise InfoTSReplayValidationError(f"C++ replay 缺少 {name}")
        _finite(metrics[name], f"metrics.{name}")
    if report.get("artifact_sha256") != compute_infots_cpp_report_sha256(report):
        raise InfoTSReplayValidationError("C++ replay artifact_sha256 不匹配")
    return dict(report)


def build_infots_joint_replay_report(
    artifacts: Sequence[Mapping[str, Any]],
    cpp_reports: Sequence[Mapping[str, Any]],
    contract: InfoTSBudgetContractV1,
) -> dict[str, Any]:
    contract.validate()
    expected_count = len(contract.group_ids) * len(contract.fold_ids)
    if len(artifacts) != expected_count or len(cpp_reports) != expected_count:
        raise InfoTSReplayValidationError(f"必须提供 {expected_count} 个 Python/C++ group/fold artifact")
    normalized = [validate_infots_fold_artifact(artifact, contract) for artifact in artifacts]
    by_key = {(item["group_id"], item["fold_id"]): item for item in normalized}
    reports: dict[tuple[str, int], dict[str, Any]] = {}
    for report in cpp_reports:
        key = (report.get("group_id"), report.get("fold_id"))
        if key in reports or key not in by_key:
            raise InfoTSReplayValidationError("C++ replay group/fold 缺失或重复")
        fold = by_key[key]
        checked = validate_infots_cpp_replay_artifact(
            report,
            expected_row_count=fold["prediction_artifact"]["row_count"],
        )
        if checked["contract_sha256"] != contract.contract_sha256:
            raise InfoTSReplayValidationError("C++ replay contract hash 不匹配")
        if checked["prediction_artifact_sha256"] != fold["prediction_artifact"]["sha256"]:
            raise InfoTSReplayValidationError("C++ replay prediction hash 未链接 Python artifact")
        embedding_hashes = {item["role"]: item["sha256"] for item in fold["embedding_snapshots"]}
        if checked["validation_embedding_sha256"] != embedding_hashes["validation"] or checked["test_embedding_sha256"] != embedding_hashes["test"]:
            raise InfoTSReplayValidationError("C++ replay embedding hash 未链接 Python snapshots")
        reports[key] = checked
    expected_keys = {(group, fold) for group in contract.group_ids for fold in contract.fold_ids}
    if set(reports) != expected_keys:
        raise InfoTSReplayValidationError("Python/C++ group/fold 集合不一致")
    ordered = [reports[(group, fold)] for group in contract.group_ids for fold in contract.fold_ids]
    report: dict[str, Any] = {
        "schema_version": 1,
        "role": "phase5_infots_python_cpp_joint_replay_acceptance",
        "contract_sha256": contract.contract_sha256,
        "evidence_level": "RESEARCH_PROXY",
        "group_ids": list(contract.group_ids),
        "fold_ids": list(contract.fold_ids),
        "test_blind": True,
        "purged_oos_guard_passed": True,
        "cpp_replay_count": len(ordered),
        "all_cpp_replays_passed": True,
        "source_replay_sha256": _sha256_json([item["source_replay_sha256"] for item in ordered]),
        "ledger_sha256": _sha256_json([item["ledger_sha256"] for item in ordered]),
        "reports": [
            {
                "group_id": item["group_id"],
                "fold_id": item["fold_id"],
                "row_count": item["row_count"],
                "source_replay_sha256": item["source_replay_sha256"],
                "ledger_sha256": item["ledger_sha256"],
                "artifact_sha256": item["artifact_sha256"],
                "return_cvar": item["metrics"]["return_cvar"],
            }
            for item in ordered
        ],
        "phase_exit_eligible": False,
        "promotion_eligible": False,
        "limitations": [
            "joint acceptance is research-proxy only",
            "does not replace real CUDA training or formal OOS evidence",
            "does not permit production promotion",
        ],
    }
    report["report_sha256"] = hashlib.sha256(
        _canonical_json(report).encode("utf-8")
    ).hexdigest()
    return report


def validate_infots_joint_replay_report(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping) or report.get("schema_version") != 1 or report.get("role") != "phase5_infots_python_cpp_joint_replay_acceptance":
        raise InfoTSReplayValidationError("joint replay report schema 无效")
    stored = _digest(report.get("report_sha256"), "report_sha256")
    payload = dict(report)
    payload.pop("report_sha256", None)
    if hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest() != stored:
        raise InfoTSReplayValidationError("joint replay report hash 不匹配")
    if report.get("all_cpp_replays_passed") is not True or report.get("phase_exit_eligible") is not False or report.get("promotion_eligible") is not False:
        raise InfoTSReplayValidationError("joint replay gate 状态不安全")

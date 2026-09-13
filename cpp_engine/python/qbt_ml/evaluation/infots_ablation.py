"""Fail-closed Phase 5 supervised-budget and purged-OOS artifact evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence


class InfoTSAblationValidationError(ValueError):
    """Raised when a Phase 5 ablation artifact violates its frozen contract."""


GROUP_IDS = ("from_scratch", "fixed_augmentation", "infots")
FOLD_IDS = (1, 2, 3)
PREDICTION_OUTPUTS = (
    "expected_return",
    "expected_volatility",
    "direction_probability",
    "lower_quantile",
    "upper_quantile",
    "confidence",
)
METRIC_NAMES = (
    "composite_error",
    "return_mae",
    "direction_brier",
    "volatility_mae",
    "ndcg_at_20",
    "rank_ic",
)
GATE_NAMES = (
    "clean_non_degraded",
    "stress_non_degraded",
    "three_window_consistent",
    "quality_gate_passed",
    "stability_gate_passed",
)
LOWER_IS_BETTER = frozenset(METRIC_NAMES[:4])


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest_like(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise InfoTSAblationValidationError(f"{name} 必须为正整数")
    return int(value)


@dataclass(frozen=True)
class InfoTSBudgetContractV1:
    schema_version: int = 1
    hypothesis_id: str = "INFOTS-SUPERVISED-BUDGET-ABLATION-V1"
    group_ids: tuple[str, ...] = GROUP_IDS
    fold_ids: tuple[int, ...] = FOLD_IDS
    supervised_budget: int = 50
    gate_spec_sha256: str = ""
    prediction_outputs: tuple[str, ...] = PREDICTION_OUTPUTS
    embedding_snapshot_roles: tuple[str, ...] = ("validation", "test")
    production_eval: bool = False

    def validate(self) -> None:
        if self.schema_version != 1 or not self.hypothesis_id:
            raise InfoTSAblationValidationError("budget contract schema/id 无效")
        if tuple(self.group_ids) != GROUP_IDS:
            raise InfoTSAblationValidationError("group_ids 必须固定为三组预算对照")
        if len(self.fold_ids) < 3 or len(set(self.fold_ids)) != len(self.fold_ids) or any(
            isinstance(fold, bool) or not isinstance(fold, Integral) or fold <= 0 for fold in self.fold_ids
        ):
            raise InfoTSAblationValidationError("fold_ids 必须是至少三个唯一正整数")
        _positive_integer(self.supervised_budget, "supervised_budget")
        if not _digest_like(self.gate_spec_sha256):
            raise InfoTSAblationValidationError("gate_spec_sha256 无效")
        if tuple(self.prediction_outputs) != PREDICTION_OUTPUTS:
            raise InfoTSAblationValidationError("六输出语义或顺序不匹配")
        if tuple(self.embedding_snapshot_roles) != ("validation", "test"):
            raise InfoTSAblationValidationError("embedding snapshot roles 必须是 validation/test")
        if self.production_eval is not False:
            raise InfoTSAblationValidationError("Phase 5 ablation 禁止 production_eval")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "hypothesis_id": self.hypothesis_id,
            "group_ids": list(self.group_ids),
            "fold_ids": list(self.fold_ids),
            "supervised_budget": self.supervised_budget,
            "gate_spec_sha256": self.gate_spec_sha256,
            "prediction_outputs": list(self.prediction_outputs),
            "embedding_snapshot_roles": list(self.embedding_snapshot_roles),
            "production_eval": self.production_eval,
        }

    @property
    def contract_sha256(self) -> str:
        return _sha256_json(self.to_dict())


def _validate_relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise InfoTSAblationValidationError(f"{name} 必须是相对路径")
    parts = value.replace("\\", "/").split("/")
    if ".." in parts or any(not part for part in parts):
        raise InfoTSAblationValidationError(f"{name} 含非法路径段")
    return value


def _validate_split(split: Any) -> dict[str, int]:
    if not isinstance(split, Mapping):
        raise InfoTSAblationValidationError("split 必须为对象")
    names = ("train_end", "validation_end", "test_start", "test_end", "purge_gap")
    values = {name: _positive_integer(split.get(name), f"split.{name}") for name in names if name != "purge_gap"}
    purge_gap = split.get("purge_gap")
    if isinstance(purge_gap, bool) or not isinstance(purge_gap, Integral) or purge_gap < 0:
        raise InfoTSAblationValidationError("split.purge_gap 必须为非负整数")
    values["purge_gap"] = int(purge_gap)
    if not values["train_end"] < values["validation_end"] < values["test_start"] <= values["test_end"]:
        raise InfoTSAblationValidationError("split 时间边界未严格递增")
    if values["test_start"] <= values["validation_end"] + values["purge_gap"]:
        raise InfoTSAblationValidationError("test_start 未越过 purge gap")
    return values


def validate_infots_fold_artifact(
    artifact: Mapping[str, Any],
    contract: InfoTSBudgetContractV1,
) -> dict[str, Any]:
    contract.validate()
    if not isinstance(artifact, Mapping) or artifact.get("schema_version") != 1:
        raise InfoTSAblationValidationError("fold artifact schema 无效")
    group_id = artifact.get("group_id")
    if group_id not in contract.group_ids:
        raise InfoTSAblationValidationError("group_id 不在三组预算合同中")
    fold_id = artifact.get("fold_id")
    if fold_id not in contract.fold_ids:
        raise InfoTSAblationValidationError("fold_id 不在合同中")
    if artifact.get("supervised_budget") != contract.supervised_budget:
        raise InfoTSAblationValidationError("三组 supervised budget 必须完全一致")
    if artifact.get("test_blind") is not True:
        raise InfoTSAblationValidationError("artifact 必须声明 test_blind=true")
    if artifact.get("gate_spec_sha256") != contract.gate_spec_sha256:
        raise InfoTSAblationValidationError("gate_spec_sha256 与预注册合同不一致")
    split = _validate_split(artifact.get("split"))
    prediction = artifact.get("prediction_artifact")
    if not isinstance(prediction, Mapping):
        raise InfoTSAblationValidationError("缺少 prediction_artifact")
    _validate_relative_path(prediction.get("path"), "prediction_artifact.path")
    if not _digest_like(prediction.get("sha256")):
        raise InfoTSAblationValidationError("prediction_artifact.sha256 无效")
    if tuple(prediction.get("outputs", ())) != PREDICTION_OUTPUTS:
        raise InfoTSAblationValidationError("prediction artifact 六输出不完整或顺序错误")
    _positive_integer(prediction.get("row_count"), "prediction_artifact.row_count")
    embeddings = artifact.get("embedding_snapshots")
    if not isinstance(embeddings, Sequence) or isinstance(embeddings, (str, bytes)) or len(embeddings) != 2:
        raise InfoTSAblationValidationError("必须同时提供 validation/test 两个 embedding snapshot")
    roles = set()
    for embedding in embeddings:
        if not isinstance(embedding, Mapping):
            raise InfoTSAblationValidationError("embedding snapshot 必须为对象")
        role = embedding.get("role")
        if role not in contract.embedding_snapshot_roles or role in roles:
            raise InfoTSAblationValidationError("embedding snapshot role 重复或无效")
        roles.add(role)
        if not _digest_like(embedding.get("sha256")):
            raise InfoTSAblationValidationError("embedding snapshot sha256 无效")
        _positive_integer(embedding.get("row_count"), "embedding.row_count")
        _positive_integer(embedding.get("dimension"), "embedding.dimension")
    if roles != set(contract.embedding_snapshot_roles):
        raise InfoTSAblationValidationError("embedding snapshot 必须覆盖 validation/test")
    metrics = artifact.get("metrics")
    if not isinstance(metrics, Mapping) or any(not _finite(metrics.get(name)) for name in METRIC_NAMES):
        raise InfoTSAblationValidationError("metrics 缺少有限的六项评估指标")
    gates = artifact.get("gates")
    if not isinstance(gates, Mapping) or any(not isinstance(gates.get(name), bool) for name in GATE_NAMES):
        raise InfoTSAblationValidationError("gates 缺少完整布尔门槛")
    return {
        "schema_version": 1,
        "group_id": group_id,
        "fold_id": int(fold_id),
        "supervised_budget": int(artifact["supervised_budget"]),
        "gate_spec_sha256": contract.gate_spec_sha256,
        "test_blind": True,
        "split": split,
        "prediction_artifact": dict(prediction),
        "embedding_snapshots": [dict(embedding) for embedding in embeddings],
        "metrics": {name: float(metrics[name]) for name in METRIC_NAMES},
        "gates": {name: bool(gates[name]) for name in GATE_NAMES},
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise InfoTSAblationValidationError("聚合 values 为空")
    return float(sum(values) / len(values))


def build_infots_ablation_report(
    artifacts: Sequence[Mapping[str, Any]],
    contract: InfoTSBudgetContractV1,
    *,
    evidence_level: str = "REFERENCE_ONLY",
) -> dict[str, Any]:
    contract.validate()
    if evidence_level not in {"REFERENCE_ONLY", "FORMAL_OOS"}:
        raise InfoTSAblationValidationError("evidence_level 无效")
    expected_count = len(contract.group_ids) * len(contract.fold_ids)
    if len(artifacts) != expected_count:
        raise InfoTSAblationValidationError(f"必须提供 {expected_count} 个 group/fold artifact")
    normalized = [validate_infots_fold_artifact(artifact, contract) for artifact in artifacts]
    keys = {(artifact["group_id"], artifact["fold_id"]) for artifact in normalized}
    expected_keys = {(group_id, fold_id) for group_id in contract.group_ids for fold_id in contract.fold_ids}
    if keys != expected_keys:
        raise InfoTSAblationValidationError("group/fold artifact 缺失或重复")
    by_group = {group_id: [artifact for artifact in normalized if artifact["group_id"] == group_id] for group_id in contract.group_ids}
    summary: dict[str, Any] = {}
    for group_id, group_artifacts in by_group.items():
        summary[group_id] = {
            "folds": {
                str(artifact["fold_id"]): {
                    "metrics": artifact["metrics"],
                    "gates": artifact["gates"],
                    "split": artifact["split"],
                }
                for artifact in sorted(group_artifacts, key=lambda item: item["fold_id"])
            },
            "mean_metrics": {
                name: _mean([artifact["metrics"][name] for artifact in group_artifacts])
                for name in METRIC_NAMES
            },
            "all_gates_passed": all(all(artifact["gates"].values()) for artifact in group_artifacts),
        }
    baseline = summary["from_scratch"]["mean_metrics"]
    for group_id in contract.group_ids:
        summary[group_id]["mean_delta_vs_from_scratch"] = {
            name: summary[group_id]["mean_metrics"][name] - baseline[name]
            for name in METRIC_NAMES
        }
    all_test_blind = all(artifact["test_blind"] for artifact in normalized)
    all_gates_passed = all(summary[group_id]["all_gates_passed"] for group_id in contract.group_ids)
    report = {
        "schema_version": 1,
        "role": "phase5_infots_supervised_budget_ablation",
        "hypothesis_id": contract.hypothesis_id,
        "contract_sha256": contract.contract_sha256,
        "evidence_level": evidence_level,
        "group_ids": list(contract.group_ids),
        "fold_ids": list(contract.fold_ids),
        "supervised_budget": contract.supervised_budget,
        "test_blind": all_test_blind,
        "purged_oos_guard_passed": all(
            artifact["split"]["test_start"] > artifact["split"]["validation_end"] + artifact["split"]["purge_gap"]
            for artifact in normalized
        ),
        "summary": summary,
        "all_gates_passed": all_gates_passed,
        "winner_selected": False,
        "research_gate_passed": False,
        "phase_exit_eligible": False,
        "promotion_eligible": False,
        "limitations": [
            "CPU evaluator only",
            "no post-hoc winner selection",
            "formal OOS evidence must be independently registered",
        ],
    }
    report["report_sha256"] = _sha256_json(report)
    return report


def validate_infots_ablation_report(report: Mapping[str, Any]) -> None:
    if not isinstance(report, Mapping) or report.get("schema_version") != 1:
        raise InfoTSAblationValidationError("ablation report schema 无效")
    stored_hash = report.get("report_sha256")
    if not _digest_like(stored_hash):
        raise InfoTSAblationValidationError("ablation report hash 无效")
    payload = dict(report)
    payload.pop("report_sha256", None)
    if _sha256_json(payload) != stored_hash:
        raise InfoTSAblationValidationError("ablation report hash 不匹配")
    if report.get("winner_selected") is not False or report.get("promotion_eligible") is not False:
        raise InfoTSAblationValidationError("ablation report 不得自动选 winner 或晋级")

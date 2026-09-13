"""Deterministic CPU contracts for the Phase 5 InfoTS pretraining scaffold."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np


class InfoTSArtifactValidationError(ValueError):
    """Raised when an InfoTS contract, augmentation, or artifact is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _digest_like(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _finite_real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_features(features: Any) -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] == 0 or array.shape[1] == 0 or array.shape[2] == 0:
        raise InfoTSArtifactValidationError("features 必须为非空 [N,T,F]")
    if not np.isfinite(array).all():
        raise InfoTSArtifactValidationError("features 必须全部为有限值")
    return array


def _as_valid_mask(valid_mask: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(valid_mask, dtype=bool)
    if array.shape != shape:
        raise InfoTSArtifactValidationError("valid_mask 必须为 [N,T] 并与 features 对齐")
    if not array.any(axis=1).all():
        raise InfoTSArtifactValidationError("每个样本至少需要一个有效时间点")
    return array


def _as_timestamps(timestamps: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(timestamps)
    if array.ndim == 1 and array.shape[0] == shape[1]:
        array = np.broadcast_to(array, shape).copy()
    if array.shape != shape:
        raise InfoTSArtifactValidationError("timestamps 必须为 [T] 或 [N,T]")
    if np.issubdtype(array.dtype, np.number):
        if not np.isfinite(array.astype(np.float64)).all():
            raise InfoTSArtifactValidationError("timestamps 必须全部为有限值")
    for row_index, row in enumerate(array):
        valid_values = row
        if len(valid_values) > 1 and any(left >= right for left, right in zip(valid_values, valid_values[1:])):
            raise InfoTSArtifactValidationError(f"样本 {row_index} 的 timestamps 必须严格递增")
    return array


def _array_hash(name: str, array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(_canonical_json(list(contiguous.shape)).encode("ascii"))
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True)
class InfoTSAugmentationSpecV1:
    schema_version: int = 1
    augmentation_id: str = "INFOTS-CAUSAL-BASELINE-V1"
    seed: int = 1
    crop_min_length: int = 2
    crop_max_length: int = 0
    noise_std: float = 0.01
    time_mask_probability: float = 0.10
    continuous_feature_indices: tuple[int, ...] = ()
    state_feature_indices: tuple[int, ...] = ()
    padding_value: float = 0.0
    future_guard: bool = True

    def validate(self, feature_count: int | None = None, time_steps: int | None = None) -> None:
        if self.schema_version != 1 or not isinstance(self.augmentation_id, str) or not self.augmentation_id:
            raise InfoTSArtifactValidationError("augmentation spec schema/id 无效")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed <= 0:
            raise InfoTSArtifactValidationError("seed 必须为正整数")
        if isinstance(self.crop_min_length, bool) or self.crop_min_length < 1:
            raise InfoTSArtifactValidationError("crop_min_length 必须为正整数")
        if isinstance(self.crop_max_length, bool) or self.crop_max_length < 0:
            raise InfoTSArtifactValidationError("crop_max_length 必须为非负整数")
        if self.crop_max_length and self.crop_max_length < self.crop_min_length:
            raise InfoTSArtifactValidationError("crop_max_length 不得小于 crop_min_length")
        if not _finite_real(self.noise_std) or self.noise_std < 0.0:
            raise InfoTSArtifactValidationError("noise_std 必须为非负有限值")
        if not _finite_real(self.time_mask_probability) or not 0.0 <= self.time_mask_probability < 1.0:
            raise InfoTSArtifactValidationError("time_mask_probability 必须位于 [0,1)")
        if not _finite_real(self.padding_value):
            raise InfoTSArtifactValidationError("padding_value 必须为有限值")
        if self.future_guard is not True:
            raise InfoTSArtifactValidationError("future_guard 必须开启")
        if any(not isinstance(index, Integral) or isinstance(index, bool) or index < 0 for index in self.continuous_feature_indices):
            raise InfoTSArtifactValidationError("continuous_feature_indices 无效")
        if any(not isinstance(index, Integral) or isinstance(index, bool) or index < 0 for index in self.state_feature_indices):
            raise InfoTSArtifactValidationError("state_feature_indices 无效")
        if len(set(self.continuous_feature_indices)) != len(self.continuous_feature_indices):
            raise InfoTSArtifactValidationError("continuous_feature_indices 不得重复")
        if len(set(self.state_feature_indices)) != len(self.state_feature_indices):
            raise InfoTSArtifactValidationError("state_feature_indices 不得重复")
        if set(self.continuous_feature_indices) & set(self.state_feature_indices):
            raise InfoTSArtifactValidationError("continuous/state feature 不得重叠")
        if feature_count is not None:
            all_indices = (*self.continuous_feature_indices, *self.state_feature_indices)
            if any(index >= feature_count for index in all_indices):
                raise InfoTSArtifactValidationError("feature index 超出输入维度")
        if time_steps is not None and self.crop_min_length > time_steps:
            raise InfoTSArtifactValidationError("crop_min_length 超出时间维度")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "augmentation_id": self.augmentation_id,
            "seed": self.seed,
            "crop_min_length": self.crop_min_length,
            "crop_max_length": self.crop_max_length,
            "noise_std": self.noise_std,
            "time_mask_probability": self.time_mask_probability,
            "continuous_feature_indices": list(self.continuous_feature_indices),
            "state_feature_indices": list(self.state_feature_indices),
            "padding_value": self.padding_value,
            "future_guard": self.future_guard,
        }

    @property
    def spec_sha256(self) -> str:
        return _sha256_json(self.to_dict())


def _masked_padding(view: np.ndarray, valid_mask: np.ndarray, padding_value: float) -> None:
    invalid = ~valid_mask
    if invalid.any():
        view[invalid] = padding_value


def _suffix_crop_mask(valid_mask: np.ndarray, rng: np.random.Generator, spec: InfoTSAugmentationSpecV1) -> tuple[np.ndarray, list[dict[str, int]]]:
    cropped = np.zeros_like(valid_mask, dtype=bool)
    metadata: list[dict[str, int]] = []
    for row_index, row in enumerate(valid_mask):
        valid_indices = np.flatnonzero(row)
        valid_count = int(valid_indices.size)
        maximum = spec.crop_max_length or valid_count
        maximum = min(maximum, valid_count)
        minimum = min(spec.crop_min_length, maximum)
        length = int(rng.integers(minimum, maximum + 1))
        end_index = int(valid_indices[-1])
        start_position = valid_count - length
        start_index = int(valid_indices[start_position])
        cropped[row_index, start_index:end_index + 1] = row[start_index:end_index + 1]
        metadata.append({"start_index": start_index, "end_index": end_index, "length": length})
    return cropped, metadata


def apply_causal_augmentations(
    features: Any,
    valid_mask: Any,
    timestamps: Any,
    spec: InfoTSAugmentationSpecV1,
) -> dict[str, Any]:
    """Create deterministic global/local views without reading future or padding values."""
    source = _as_features(features)
    mask = _as_valid_mask(valid_mask, source.shape[:2])
    timestamp_array = _as_timestamps(timestamps, source.shape[:2])
    spec.validate(feature_count=source.shape[2], time_steps=source.shape[1])
    rng = np.random.default_rng(spec.seed)
    continuous = np.asarray(spec.continuous_feature_indices, dtype=np.int64)
    state = np.asarray(spec.state_feature_indices, dtype=np.int64)
    if continuous.size == 0:
        continuous = np.asarray([index for index in range(source.shape[2]) if index not in set(state.tolist())], dtype=np.int64)
    global_view = source.copy()
    local_view = source.copy()
    if continuous.size and spec.noise_std > 0.0:
        noise = rng.normal(0.0, spec.noise_std, size=(source.shape[0], source.shape[1], continuous.size))
        global_view[:, :, continuous] += noise
        local_view[:, :, continuous] += noise
    global_time_mask = np.zeros(mask.shape, dtype=bool)
    for row_index, row in enumerate(mask):
        valid_indices = np.flatnonzero(row)
        selected = rng.random(valid_indices.size) < spec.time_mask_probability
        if selected.all() and selected.size:
            selected[int(np.argmax(rng.random(selected.size)))] = False
        global_time_mask[row_index, valid_indices[selected]] = True
    if continuous.size:
        global_view[global_time_mask[:, :, None] & np.isin(np.arange(source.shape[2]), continuous)[None, None, :]] = spec.padding_value
    local_mask, crop_metadata = _suffix_crop_mask(mask, rng, spec)
    local_time_mask = np.zeros(mask.shape, dtype=bool)
    for row_index, row in enumerate(local_mask):
        valid_indices = np.flatnonzero(row)
        selected = rng.random(valid_indices.size) < spec.time_mask_probability
        if selected.all() and selected.size:
            selected[int(np.argmax(rng.random(selected.size)))] = False
        local_time_mask[row_index, valid_indices[selected]] = True
    if continuous.size:
        local_view[local_time_mask[:, :, None] & np.isin(np.arange(source.shape[2]), continuous)[None, None, :]] = spec.padding_value
    _masked_padding(global_view, mask, spec.padding_value)
    _masked_padding(local_view, local_mask, spec.padding_value)
    if state.size:
        if not np.array_equal(global_view[:, :, state][mask], source[:, :, state][mask]):
            raise InfoTSArtifactValidationError("global view 改变了 state feature")
        if not np.array_equal(local_view[:, :, state][local_mask], source[:, :, state][local_mask]):
            raise InfoTSArtifactValidationError("local view 改变了 state feature")
    if np.any(global_time_mask & ~mask) or np.any(local_time_mask & ~local_mask):
        raise InfoTSArtifactValidationError("time-mask 触及 padding")
    valid_timestamp_rows = [timestamp_array[index][mask[index]].tolist() for index in range(mask.shape[0])]
    local_timestamp_rows = [timestamp_array[index][local_mask[index]].tolist() for index in range(mask.shape[0])]
    if any(row and local_row[-1] > row[-1] for row, local_row in zip(valid_timestamp_rows, local_timestamp_rows)):
        raise InfoTSArtifactValidationError("local crop 引入未来 timestamp")
    metadata = {
        "schema_version": 1,
        "augmentation_id": spec.augmentation_id,
        "spec_sha256": spec.spec_sha256,
        "seed": spec.seed,
        "source_features_sha256": _array_hash("features", source),
        "source_valid_mask_sha256": _array_hash("valid_mask", mask),
        "source_timestamps_sha256": _array_hash("timestamps", timestamp_array),
        "global_time_mask_sha256": _array_hash("global_time_mask", global_time_mask),
        "local_time_mask_sha256": _array_hash("local_time_mask", local_time_mask),
        "crop": crop_metadata,
        "future_guard_passed": True,
        "state_guard_passed": True,
        "padding_guard_passed": True,
    }
    metadata["augmentation_sha256"] = _sha256_json(metadata)
    return {
        "global_view": global_view,
        "local_view": local_view,
        "global_valid_mask": mask.copy(),
        "local_valid_mask": local_mask,
        "global_time_mask": global_time_mask,
        "local_time_mask": local_time_mask,
        "timestamps": timestamp_array.copy(),
        "metadata": metadata,
    }


def _as_embedding_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise InfoTSArtifactValidationError(f"{name} 必须为至少 2 行的二维 embedding")
    if not np.isfinite(array).all():
        raise InfoTSArtifactValidationError(f"{name} 必须全部为有限值")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0.0):
        raise InfoTSArtifactValidationError(f"{name} 不得包含零向量")
    return array / norms[:, None]


def info_nce_loss(
    global_embeddings: Any,
    local_embeddings: Any,
    *,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Compute symmetric in-batch InfoNCE with deterministic NumPy math."""
    if not _finite_real(temperature) or temperature <= 0.0:
        raise InfoTSArtifactValidationError("temperature 必须为正有限值")
    global_normalized = _as_embedding_matrix(global_embeddings, "global_embeddings")
    local_normalized = _as_embedding_matrix(local_embeddings, "local_embeddings")
    if global_normalized.shape != local_normalized.shape:
        raise InfoTSArtifactValidationError("global/local embedding 形状必须一致")
    logits = global_normalized @ local_normalized.T / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    log_prob_global = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    reverse_logits = local_normalized @ global_normalized.T / float(temperature)
    reverse_logits -= np.max(reverse_logits, axis=1, keepdims=True)
    log_prob_local = reverse_logits - np.log(np.exp(reverse_logits).sum(axis=1, keepdims=True))
    diagonal = np.arange(global_normalized.shape[0])
    loss = float(-0.5 * (log_prob_global[diagonal, diagonal].mean() + log_prob_local[diagonal, diagonal].mean()))
    cosine = global_normalized @ local_normalized.T
    negative = cosine.copy()
    negative[diagonal, diagonal] = -np.inf
    result = {
        "schema_version": 1,
        "temperature": float(temperature),
        "batch_size": int(global_normalized.shape[0]),
        "embedding_dimension": int(global_normalized.shape[1]),
        "loss": loss,
        "positive_cosine_mean": float(np.mean(cosine[diagonal, diagonal])),
        "negative_cosine_max_mean": float(np.mean(np.max(negative, axis=1))),
        "global_embeddings_sha256": _array_hash("global_embeddings", global_normalized),
        "local_embeddings_sha256": _array_hash("local_embeddings", local_normalized),
    }
    result["artifact_sha256"] = _sha256_json(result)
    return result


def select_information_aware_augmentation(
    candidates: Sequence[Mapping[str, Any]],
    *,
    baseline_id: str,
    minimum_improvement: float = 0.0,
) -> dict[str, Any]:
    """Select a candidate only from guarded train-fold metrics; never silently fallback."""
    if not candidates or not isinstance(baseline_id, str) or not baseline_id:
        raise InfoTSArtifactValidationError("candidates/baseline_id 无效")
    if not _finite_real(minimum_improvement) or minimum_improvement < 0.0:
        raise InfoTSArtifactValidationError("minimum_improvement 必须为非负有限值")
    normalized = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise InfoTSArtifactValidationError("candidate 必须为对象")
        candidate_id = candidate.get("augmentation_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise InfoTSArtifactValidationError("candidate augmentation_id 无效")
        required = ("info_nce_loss", "downstream_probe_score", "future_guard_passed", "state_guard_passed", "padding_guard_passed")
        if any(key not in candidate for key in required):
            raise InfoTSArtifactValidationError("candidate 缺少 selector 诊断字段")
        if not all(_finite_real(candidate[key]) for key in ("info_nce_loss", "downstream_probe_score")):
            raise InfoTSArtifactValidationError("selector 诊断必须为有限值")
        if not all(candidate[key] is True for key in required[2:]):
            continue
        score = float(candidate["downstream_probe_score"]) - float(candidate["info_nce_loss"])
        normalized.append({"augmentation_id": candidate_id, "score": score, "info_nce_loss": float(candidate["info_nce_loss"]), "downstream_probe_score": float(candidate["downstream_probe_score"])})
    if not normalized:
        raise InfoTSArtifactValidationError("没有通过 future/state/padding guard 的 selector candidate")
    baseline = next((candidate for candidate in normalized if candidate["augmentation_id"] == baseline_id), None)
    if baseline is None:
        raise InfoTSArtifactValidationError("baseline candidate 必须存在且通过 guard")
    winner = max(normalized, key=lambda candidate: (candidate["score"], candidate["augmentation_id"]))
    selected = winner if winner["score"] >= baseline["score"] + float(minimum_improvement) else baseline
    result = {
        "schema_version": 1,
        "baseline_id": baseline_id,
        "minimum_improvement": float(minimum_improvement),
        "candidates": sorted(normalized, key=lambda candidate: candidate["augmentation_id"]),
        "selected_id": selected["augmentation_id"],
        "selector_gate_passed": selected["augmentation_id"] == winner["augmentation_id"] or winner["score"] < baseline["score"] + float(minimum_improvement),
        "evidence_level": "REFERENCE_ONLY",
    }
    result["selector_sha256"] = _sha256_json(result)
    return result


def build_infots_pretraining_artifact(
    *,
    dataset_sha256: str,
    train_fold_end: Any,
    train_timestamps: Sequence[Any],
    augmentation_metadata: Mapping[str, Any],
    selector_artifact: Mapping[str, Any],
    supervised_budget: int,
    seed: int,
) -> dict[str, Any]:
    """Build a train-fold-only handoff artifact; no validation/test rows are accepted."""
    if not _digest_like(dataset_sha256):
        raise InfoTSArtifactValidationError("dataset_sha256 无效")
    if not _finite_real(train_fold_end):
        raise InfoTSArtifactValidationError("train_fold_end 必须为有限值")
    if not train_timestamps:
        raise InfoTSArtifactValidationError("train_timestamps 不得为空")
    if any(not _finite_real(timestamp) or float(timestamp) > float(train_fold_end) for timestamp in train_timestamps):
        raise InfoTSArtifactValidationError("pretraining artifact 含 train_fold_end 之后的数据")
    if not isinstance(augmentation_metadata, Mapping) or augmentation_metadata.get("future_guard_passed") is not True:
        raise InfoTSArtifactValidationError("augmentation metadata 未通过 future guard")
    if not isinstance(selector_artifact, Mapping) or selector_artifact.get("evidence_level") != "REFERENCE_ONLY":
        raise InfoTSArtifactValidationError("selector artifact evidence level 无效")
    if isinstance(supervised_budget, bool) or not isinstance(supervised_budget, Integral) or supervised_budget <= 0:
        raise InfoTSArtifactValidationError("supervised_budget 必须为正整数")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed <= 0:
        raise InfoTSArtifactValidationError("seed 必须为正整数")
    artifact = {
        "schema_version": 1,
        "role": "phase5_infots_pretraining_reference",
        "evidence_level": "REFERENCE_ONLY",
        "training_required": True,
        "phase_exit_eligible": False,
        "promotion_eligible": False,
        "dataset_sha256": dataset_sha256,
        "train_fold_end": float(train_fold_end),
        "train_timestamp_count": len(train_timestamps),
        "train_timestamps_sha256": _sha256_json(list(train_timestamps)),
        "augmentation_sha256": augmentation_metadata.get("augmentation_sha256"),
        "selector_sha256": selector_artifact.get("selector_sha256"),
        "supervised_budget": int(supervised_budget),
        "seed": int(seed),
        "output_semantics": [
            "expected_return", "expected_volatility", "direction_probability",
            "lower_quantile", "upper_quantile", "confidence",
        ],
        "limitations": ["CPU contract only", "no PyTorch training", "no OOS or cost gate"],
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    return artifact


def validate_infots_pretraining_artifact(artifact: Mapping[str, Any]) -> None:
    if not isinstance(artifact, Mapping) or artifact.get("schema_version") != 1:
        raise InfoTSArtifactValidationError("InfoTS artifact schema 无效")
    stored = artifact.get("artifact_sha256")
    if not _digest_like(stored):
        raise InfoTSArtifactValidationError("InfoTS artifact hash 无效")
    payload = dict(artifact)
    payload.pop("artifact_sha256", None)
    if _sha256_json(payload) != stored:
        raise InfoTSArtifactValidationError("InfoTS artifact hash 不匹配")
    if artifact.get("phase_exit_eligible") is not False or artifact.get("promotion_eligible") is not False:
        raise InfoTSArtifactValidationError("InfoTS reference artifact 不得晋级")
    if artifact.get("training_required") is not True:
        raise InfoTSArtifactValidationError("InfoTS artifact 必须声明需要训练")

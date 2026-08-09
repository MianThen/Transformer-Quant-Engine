"""Optional PyTorch InfoTS pretraining entrypoint with strict research guards."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np

from .infots import (
    InfoTSArtifactValidationError,
    InfoTSAugmentationSpecV1,
    _canonical_json,
    _digest_like,
    _sha256_json,
    apply_causal_augmentations,
)

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


PRODUCTION_OUTPUT_KEYS = (
    "expected_return",
    "expected_volatility",
    "direction_probability",
    "lower_quantile",
    "upper_quantile",
    "confidence",
)


class InfoTSTrainingUnavailable(RuntimeError):
    """Raised when a PyTorch/CUDA training operation is requested without PyTorch."""


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _array_sha256(name: str, value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(_canonical_json(list(contiguous.shape)).encode("ascii"))
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True)
class InfoTSTrainingSpecV1:
    schema_version: int = 1
    hypothesis_id: str = "INFOTS-CAUSAL-SIAMESE-V1"
    seed: int = 1
    epochs: int = 10
    batch_size: int = 64
    hidden_dim: int = 64
    embedding_dim: int = 32
    temperature: float = 0.1
    learning_rate: float = 1e-3
    train_fold_end: int = 0
    device: str = "cuda"
    deterministic_algorithms: bool = True
    production_eval: bool = False

    def validate(self) -> None:
        if self.schema_version != 1 or not self.hypothesis_id:
            raise InfoTSArtifactValidationError("InfoTS training spec schema/id 无效")
        for name in ("seed", "epochs", "batch_size", "hidden_dim", "embedding_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise InfoTSArtifactValidationError(f"{name} 必须为正整数")
        for name in ("temperature", "learning_rate"):
            if not _finite(getattr(self, name)):
                raise InfoTSArtifactValidationError(f"{name} 必须为有限值")
        if isinstance(self.train_fold_end, bool) or not isinstance(self.train_fold_end, Integral) or self.train_fold_end <= 0:
            raise InfoTSArtifactValidationError("train_fold_end 必须为正整数 timestamp")
        if self.temperature <= 0.0 or self.learning_rate <= 0.0:
            raise InfoTSArtifactValidationError("temperature/learning_rate 必须为正")
        if not isinstance(self.device, str) or self.device not in {"cuda", "cpu"}:
            raise InfoTSArtifactValidationError("device 只能为 cuda 或 cpu")
        if self.deterministic_algorithms is not True:
            raise InfoTSArtifactValidationError("必须开启 deterministic_algorithms")
        if self.production_eval is not False:
            raise InfoTSArtifactValidationError("InfoTS pretraining 禁止 production_eval")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "hypothesis_id": self.hypothesis_id,
            "seed": self.seed,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "temperature": self.temperature,
            "learning_rate": self.learning_rate,
            "train_fold_end": self.train_fold_end,
            "device": self.device,
            "deterministic_algorithms": self.deterministic_algorithms,
            "production_eval": self.production_eval,
        }

    @property
    def spec_sha256(self) -> str:
        return _sha256_json(self.to_dict())


def validate_six_output_contract(prediction: Mapping[str, Any]) -> None:
    if not isinstance(prediction, Mapping) or tuple(prediction.keys()) != PRODUCTION_OUTPUT_KEYS:
        raise InfoTSArtifactValidationError("模型必须保持固定六输出且顺序不变")
    for key in PRODUCTION_OUTPUT_KEYS:
        value = prediction[key]
        if torch is not None and isinstance(value, torch.Tensor):
            if value.ndim != 1 or not torch.isfinite(value).all():
                raise InfoTSArtifactValidationError(f"六输出 {key} 必须为有限一维 tensor")
        else:
            array = np.asarray(value)
            if array.ndim != 1 or not np.isfinite(array).all():
                raise InfoTSArtifactValidationError(f"六输出 {key} 必须为有限一维数组")


if torch is not None:

    class InfoTSCausalEncoder(nn.Module):
        def __init__(self, feature_count: int, hidden_dim: int, embedding_dim: int) -> None:
            super().__init__()
            if feature_count <= 0:
                raise InfoTSArtifactValidationError("feature_count 必须为正")
            self.input_projection = nn.Linear(feature_count, hidden_dim)
            self.recurrent = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embedding_dim),
            )

        def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
            if features.ndim != 3 or valid_mask.shape != features.shape[:2]:
                raise InfoTSArtifactValidationError("encoder 输入形状无效")
            mask = valid_mask.to(dtype=torch.bool)
            projected = self.input_projection(features)
            recurrent, _ = self.recurrent(projected)
            weighted = recurrent * mask.unsqueeze(-1)
            pooled = weighted.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(recurrent.dtype)
            return nn.functional.normalize(self.projection(pooled), dim=-1)


    class InfoTSSiamesePretrainer(nn.Module):
        def __init__(self, feature_count: int, hidden_dim: int, embedding_dim: int) -> None:
            super().__init__()
            self.encoder = InfoTSCausalEncoder(feature_count, hidden_dim, embedding_dim)

        def forward(self, global_view: torch.Tensor, local_view: torch.Tensor, global_mask: torch.Tensor, local_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.encoder(global_view, global_mask), self.encoder(local_view, local_mask)


    def _torch_info_nce(global_embedding: torch.Tensor, local_embedding: torch.Tensor, temperature: float) -> torch.Tensor:
        if global_embedding.shape != local_embedding.shape or global_embedding.ndim != 2:
            raise InfoTSArtifactValidationError("global/local embedding 形状无效")
        global_normalized = nn.functional.normalize(global_embedding, dim=-1)
        local_normalized = nn.functional.normalize(local_embedding, dim=-1)
        logits = global_normalized @ local_normalized.transpose(0, 1) / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.transpose(0, 1), labels))

else:

    class InfoTSCausalEncoder:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise InfoTSTrainingUnavailable("InfoTS encoder requires PyTorch")


    class InfoTSSiamesePretrainer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise InfoTSTrainingUnavailable("InfoTS pretrainer requires PyTorch")


def train_infots_pretraining(
    features: Any,
    valid_mask: Any,
    timestamps: Any,
    augmentation_spec: InfoTSAugmentationSpecV1,
    training_spec: InfoTSTrainingSpecV1,
    *,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run only train-fold InfoTS pretraining; real execution requires PyTorch and CUDA."""
    training_spec.validate()
    if torch is None:
        raise InfoTSTrainingUnavailable("Phase 5 training requires PyTorch; this machine has no torch")
    if training_spec.device == "cuda" and not torch.cuda.is_available():
        raise InfoTSTrainingUnavailable("Phase 5 training requires CUDA; CUDA is unavailable")
    source = np.asarray(features, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    timestamp_array = np.asarray(timestamps)
    if source.ndim != 3 or mask.shape != source.shape[:2]:
        raise InfoTSArtifactValidationError("features/valid_mask 形状无效")
    if timestamp_array.ndim == 1 and timestamp_array.shape[0] == source.shape[0]:
        decision_timestamps = timestamp_array.copy()
        causal_timestamps = np.broadcast_to(np.arange(source.shape[1], dtype=np.int64), mask.shape)
    else:
        if timestamp_array.ndim == 1 and timestamp_array.shape[0] == source.shape[1]:
            timestamp_array = np.broadcast_to(timestamp_array, mask.shape)
        if timestamp_array.shape != mask.shape:
            raise InfoTSArtifactValidationError("timestamps 形状无效")
        decision_timestamps = np.asarray([
            timestamp_array[row_index, np.flatnonzero(mask[row_index])[-1]]
            for row_index in range(source.shape[0])
        ])
        causal_timestamps = timestamp_array
    valid_timestamps = decision_timestamps
    try:
        if np.issubdtype(valid_timestamps.dtype, np.integer):
            numeric_decision_timestamps = valid_timestamps.astype(np.int64)
        else:
            numeric_decision_timestamps = valid_timestamps.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InfoTSArtifactValidationError("decision timestamps 必须为数值") from exc
    if numeric_decision_timestamps.size == 0 or not np.isfinite(numeric_decision_timestamps).all() or np.any(numeric_decision_timestamps > training_spec.train_fold_end):
        raise InfoTSArtifactValidationError("训练输入包含 train_fold_end 之后的数据")
    augmentation_spec.validate(feature_count=source.shape[2], time_steps=source.shape[1])
    augmented = apply_causal_augmentations(source, mask, causal_timestamps, augmentation_spec)
    torch.manual_seed(training_spec.seed)
    if training_spec.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    device = torch.device(training_spec.device)
    model = InfoTSSiamesePretrainer(source.shape[2], training_spec.hidden_dim, training_spec.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_spec.learning_rate)
    global_tensor = torch.as_tensor(augmented["global_view"], dtype=torch.float32, device=device)
    local_tensor = torch.as_tensor(augmented["local_view"], dtype=torch.float32, device=device)
    global_mask_tensor = torch.as_tensor(augmented["global_valid_mask"], dtype=torch.bool, device=device)
    local_mask_tensor = torch.as_tensor(augmented["local_valid_mask"], dtype=torch.bool, device=device)
    history: list[float] = []
    sample_count = int(source.shape[0])
    for _ in range(training_spec.epochs):
        permutation = torch.randperm(sample_count, device=device)
        epoch_losses: list[float] = []
        for start in range(0, sample_count, training_spec.batch_size):
            indices = permutation[start:start + training_spec.batch_size]
            optimizer.zero_grad(set_to_none=True)
            global_embedding, local_embedding = model(
                global_tensor[indices], local_tensor[indices], global_mask_tensor[indices], local_mask_tensor[indices],
            )
            loss = _torch_info_nce(global_embedding, local_embedding, training_spec.temperature)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        history.append(float(np.mean(epoch_losses)))
    with torch.no_grad():
        global_embedding, local_embedding = model(global_tensor, local_tensor, global_mask_tensor, local_mask_tensor)
    global_embedding_cpu = global_embedding.detach().cpu().numpy()
    local_embedding_cpu = local_embedding.detach().cpu().numpy()
    artifact = {
        "schema_version": 1,
        "role": "phase5_infots_pretraining_result",
        "evidence_level": "GPU_TRAINED_REFERENCE_PENDING_OOS",
        "phase_exit_eligible": False,
        "promotion_eligible": False,
        "hypothesis_id": training_spec.hypothesis_id,
        "training_spec_sha256": training_spec.spec_sha256,
        "augmentation_sha256": augmented["metadata"]["augmentation_sha256"],
        "train_fold_end": int(training_spec.train_fold_end),
        "sample_count": sample_count,
        "decision_timestamps_sha256": _array_sha256("decision_timestamps", np.asarray(decision_timestamps)),
        "epoch_count": training_spec.epochs,
        "loss_history": history,
        "global_embedding_sha256": _array_sha256("global_embedding", global_embedding_cpu),
        "local_embedding_sha256": _array_sha256("local_embedding", local_embedding_cpu),
        "embedding_dimension": int(global_embedding_cpu.shape[1]),
        "output_semantics": list(PRODUCTION_OUTPUT_KEYS),
        "checkpoint_sha256": None,
        "limitations": ["pretraining only", "no supervised six-output predictions", "no OOS/cost gate"],
    }
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "training_spec": training_spec.to_dict(), "artifact": artifact}, checkpoint)
        artifact["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    artifact["artifact_sha256"] = _sha256_json(artifact)
    return artifact


def validate_infots_training_artifact(artifact: Mapping[str, Any]) -> None:
    if not isinstance(artifact, Mapping) or artifact.get("schema_version") != 1:
        raise InfoTSArtifactValidationError("InfoTS training artifact schema 无效")
    stored_hash = artifact.get("artifact_sha256")
    if not _digest_like(stored_hash):
        raise InfoTSArtifactValidationError("InfoTS training artifact hash 无效")
    payload = dict(artifact)
    payload.pop("artifact_sha256", None)
    if _sha256_json(payload) != stored_hash:
        raise InfoTSArtifactValidationError("InfoTS training artifact hash 不匹配")
    if artifact.get("phase_exit_eligible") is not False or artifact.get("promotion_eligible") is not False:
        raise InfoTSArtifactValidationError("InfoTS training artifact 不得自动晋级")

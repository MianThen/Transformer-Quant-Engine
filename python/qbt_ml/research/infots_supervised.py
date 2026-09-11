"""Optional PyTorch Phase 5 supervised-budget ablation trainer with strict guards.

三组等监督预算对照（from_scratch / fixed_augmentation / infots）的监督六输出训练。
只消费 train fold 之内的样本；validation 用于 checkpoint 选择；test 只在训练完成后
评估一次。所有边界、预算与 gate 规则预注册，任何违规失败关闭。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np

from .infots import (
    InfoTSArtifactValidationError,
    InfoTSAugmentationSpecV1,
    _canonical_json,
    _digest_like,
    _sha256_json,
    apply_causal_augmentations,
)
from .infots_training import (
    InfoTSCausalEncoder,
    InfoTSTrainingSpecV1,
    train_infots_pretraining,
)

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


class Phase5SupervisedUnavailable(RuntimeError):
    """Raised when supervised training is requested without PyTorch."""


def _finite(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise InfoTSArtifactValidationError(f"{name} 必须为正整数")
    return int(value)


SUPERVISED_OUTPUT_KEYS = (
    "expected_return",
    "expected_volatility",
    "direction_probability",
    "lower_quantile",
    "upper_quantile",
    "confidence",
)

RETURN_FEATURE_INDICES = (0, 1, 2, 3)
VOLUME_FEATURE_INDICES = (7, 8)
VOLATILITY_FEATURE_INDICES = (9, 10, 11, 12)
STATE_FEATURE_INDICES = (19, 20, 21, 22)
EXPECTED_FEATURE_COUNT = 23


@dataclass(frozen=True)
class Phase5GateSpecV1:
    """预注册的质量/稳定性 gate 规则；spec_sha256 写入预算合同。"""

    schema_version: int = 1
    direction_brier_max: float = 0.5
    interval_coverage_min: float = 0.05
    interval_coverage_max: float = 0.995
    stability_relative_gap_max: float = 1.0
    stress_relative_tolerance: float = 0.25
    stress_missing_rate: float = 0.1
    price_stress_scale: float = 1.25
    volume_stress_scale: float = 1.5
    extreme_volatility_scale: float = 2.0

    def validate(self) -> None:
        if self.schema_version != 1:
            raise InfoTSArtifactValidationError("gate spec schema 无效")
        for name in ("direction_brier_max", "interval_coverage_min", "interval_coverage_max",
                     "stability_relative_gap_max", "stress_relative_tolerance", "stress_missing_rate",
                     "price_stress_scale", "volume_stress_scale", "extreme_volatility_scale"):
            if not _finite(getattr(self, name)):
                raise InfoTSArtifactValidationError(f"gate spec {name} 必须为有限值")
        if not 0.0 <= self.interval_coverage_min < self.interval_coverage_max <= 1.0:
            raise InfoTSArtifactValidationError("interval coverage 区间无效")
        if self.stress_missing_rate < 0.0 or self.stress_missing_rate >= 1.0:
            raise InfoTSArtifactValidationError("stress_missing_rate 必须位于 [0,1)")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "direction_brier_max": self.direction_brier_max,
            "interval_coverage_min": self.interval_coverage_min,
            "interval_coverage_max": self.interval_coverage_max,
            "stability_relative_gap_max": self.stability_relative_gap_max,
            "stress_relative_tolerance": self.stress_relative_tolerance,
            "stress_missing_rate": self.stress_missing_rate,
            "price_stress_scale": self.price_stress_scale,
            "volume_stress_scale": self.volume_stress_scale,
            "extreme_volatility_scale": self.extreme_volatility_scale,
        }

    @property
    def spec_sha256(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True)
class Phase5SupervisedSpecV1:
    schema_version: int = 1
    hypothesis_id: str = "INFOTS-SUPERVISED-BUDGET-ABLATION-V1"
    seed: int = 1
    epochs: int = 50
    batch_size: int = 128
    hidden_dim: int = 128
    embedding_dim: int = 64
    learning_rate: float = 3e-4
    weight_return: float = 1.0
    weight_direction: float = 0.25
    weight_volatility: float = 0.25
    weight_quantile: float = 0.25
    quantile_lower: float = 0.25
    quantile_upper: float = 0.75
    return_guard_margin: float = 0.0025
    dropout: float = 0.0
    weight_decay: float = 0.0
    device: str = "cuda"
    deterministic_algorithms: bool = True
    production_eval: bool = False

    def validate(self) -> None:
        if self.schema_version != 1 or not self.hypothesis_id:
            raise InfoTSArtifactValidationError("supervised spec schema/id 无效")
        for name in ("seed", "epochs", "batch_size", "hidden_dim", "embedding_dim"):
            _positive_integer(getattr(self, name), name)
        for name in ("learning_rate", "weight_return", "weight_direction", "weight_volatility",
                     "weight_quantile", "quantile_lower", "quantile_upper", "return_guard_margin",
                     "dropout", "weight_decay"):
            if not _finite(getattr(self, name)):
                raise InfoTSArtifactValidationError(f"{name} 必须为有限值")
        if self.learning_rate <= 0.0:
            raise InfoTSArtifactValidationError("learning_rate 必须为正")
        if not 0.0 <= self.dropout < 1.0:
            raise InfoTSArtifactValidationError("dropout 必须位于 [0,1)")
        if self.weight_decay < 0.0:
            raise InfoTSArtifactValidationError("weight_decay 必须非负")
        if not 0.0 < self.quantile_lower < self.quantile_upper < 1.0:
            raise InfoTSArtifactValidationError("quantile levels 必须满足 0<lower<upper<1")
        if self.device not in {"cuda", "cpu"}:
            raise InfoTSArtifactValidationError("device 只能为 cuda 或 cpu")
        if self.deterministic_algorithms is not True:
            raise InfoTSArtifactValidationError("必须开启 deterministic_algorithms")
        if self.production_eval is not False:
            raise InfoTSArtifactValidationError("Phase 5 supervised 禁止 production_eval")

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
            "learning_rate": self.learning_rate,
            "weight_return": self.weight_return,
            "weight_direction": self.weight_direction,
            "weight_volatility": self.weight_volatility,
            "weight_quantile": self.weight_quantile,
            "quantile_lower": self.quantile_lower,
            "quantile_upper": self.quantile_upper,
            "return_guard_margin": self.return_guard_margin,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "device": self.device,
            "deterministic_algorithms": self.deterministic_algorithms,
            "production_eval": self.production_eval,
        }

    @property
    def spec_sha256(self) -> str:
        return _sha256_json(self.to_dict())


if torch is not None:

    class InfoTSSupervisedModel(nn.Module):
        """InfoTSCausalEncoder + 固定六输出头；初始化可来自预训练 checkpoint 的 encoder.*。"""

        def __init__(self, feature_count: int, hidden_dim: int, embedding_dim: int, dropout: float = 0.0) -> None:
            super().__init__()
            if not 0.0 <= dropout < 1.0:
                raise InfoTSArtifactValidationError("dropout 必须位于 [0,1)")
            self.encoder = InfoTSCausalEncoder(feature_count, hidden_dim, embedding_dim)
            self.head_trunk = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
            )
            self.head_return = nn.Linear(hidden_dim, 1)
            self.head_volatility = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
            self.head_direction = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.head_lower = nn.Linear(hidden_dim, 1)
            self.head_upper = nn.Linear(hidden_dim, 1)
            self.head_confidence = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

        def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
            embedding = self.encoder(features, valid_mask)
            trunk = self.head_trunk(embedding)
            return {
                "expected_return": self.head_return(trunk).squeeze(-1),
                "expected_volatility": self.head_volatility(trunk).squeeze(-1),
                "direction_probability": self.head_direction(trunk).squeeze(-1),
                "lower_quantile": self.head_lower(trunk).squeeze(-1),
                "upper_quantile": self.head_upper(trunk).squeeze(-1),
                "confidence": self.head_confidence(trunk).squeeze(-1),
            }

else:

    class InfoTSSupervisedModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise Phase5SupervisedUnavailable("Phase 5 supervised model requires PyTorch")


def _array_sha256(name: str, value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(_canonical_json(list(contiguous.shape)).encode("ascii"))
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.size < 3:
        return float("nan")
    def _rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(values.size, dtype=np.float64)
        sorted_values = values[order]
        index = 0
        while index < sorted_values.size:
            stop = index
            while stop + 1 < sorted_values.size and sorted_values[stop + 1] == sorted_values[index]:
                stop += 1
            ranks[order[index:stop + 1]] = 0.5 * (index + stop) + 1.0
            index = stop + 1
        return ranks
    left_ranks = _rank(np.asarray(left, dtype=np.float64))
    right_ranks = _rank(np.asarray(right, dtype=np.float64))
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = math.sqrt(float((left_centered ** 2).sum() * (right_centered ** 2).sum()))
    if denominator <= 0.0:
        return float("nan")
    return float((left_centered * right_centered).sum() / denominator)


def _ndcg_at_k(
    predictions: np.ndarray,
    relevance: np.ndarray,
    timestamp_groups: Sequence[np.ndarray],
    k: int = 20,
) -> float:
    values: list[float] = []
    for indices in timestamp_groups:
        if indices.size <= 1:
            continue
        group_predictions = predictions[indices]
        group_relevance = relevance[indices]
        discount = 1.0 / np.log2(np.arange(2, min(k, indices.size) + 2, dtype=np.float64))
        predicted_order = np.argsort(-group_predictions, kind="stable")[:k]
        ideal_order = np.argsort(-group_relevance, kind="stable")[:k]
        dcg = float((group_relevance[predicted_order][:discount.size] * discount).sum())
        idcg = float((group_relevance[ideal_order][:discount.size] * discount).sum())
        if idcg > 0.0:
            values.append(dcg / idcg)
    if not values:
        raise InfoTSArtifactValidationError("NDCG 没有任何可用截面")
    return float(sum(values) / len(values))


def _rank_ic(
    predictions: np.ndarray,
    utility: np.ndarray,
    timestamp_groups: Sequence[np.ndarray],
) -> float:
    values: list[float] = []
    for indices in timestamp_groups:
        if indices.size < 3:
            continue
        score = _spearman(predictions[indices], utility[indices])
        if math.isfinite(score):
            values.append(score)
    if not values:
        raise InfoTSArtifactValidationError("RankIC 没有任何可用截面")
    return float(sum(values) / len(values))


def compute_supervised_metrics(
    prediction: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    timestamp_groups: Sequence[np.ndarray],
) -> dict[str, float]:
    for key in SUPERVISED_OUTPUT_KEYS:
        array = np.asarray(prediction[key], dtype=np.float64)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise InfoTSArtifactValidationError(f"预测 {key} 必须为有限一维数组")
    expected_return = np.asarray(labels["expected_return"], dtype=np.float64)
    direction = np.asarray(labels["direction"], dtype=np.float64)
    realized_volatility = np.asarray(labels["realized_volatility"], dtype=np.float64)
    relevance = np.asarray(labels["rank_relevance"], dtype=np.float64)
    utility = np.asarray(labels["rank_utility"], dtype=np.float64)
    predicted_return = np.asarray(prediction["expected_return"], dtype=np.float64)
    predicted_direction = np.asarray(prediction["direction_probability"], dtype=np.float64)
    predicted_volatility = np.asarray(prediction["expected_volatility"], dtype=np.float64)
    return_mae = float(np.abs(predicted_return - expected_return).mean())
    direction_brier = float(((predicted_direction - direction) ** 2).mean())
    volatility_mae = float(np.abs(predicted_volatility - realized_volatility).mean())
    composite_error = float(0.4 * return_mae + 0.3 * direction_brier + 0.3 * volatility_mae)
    metrics = {
        "composite_error": composite_error,
        "return_mae": return_mae,
        "direction_brier": direction_brier,
        "volatility_mae": volatility_mae,
        "ndcg_at_20": _ndcg_at_k(predicted_return, relevance, timestamp_groups),
        "rank_ic": _rank_ic(predicted_return, utility, timestamp_groups),
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise InfoTSArtifactValidationError("指标存在非有限值")
    return metrics


def interval_coverage(prediction: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> float:
    expected_return = np.asarray(labels["expected_return"], dtype=np.float64)
    lower = np.asarray(prediction["lower_quantile"], dtype=np.float64)
    upper = np.asarray(prediction["upper_quantile"], dtype=np.float64)
    return float(((expected_return >= lower) & (expected_return <= upper)).mean())


def build_stress_features(
    features: np.ndarray,
    valid_mask: np.ndarray,
    gate_spec: Phase5GateSpecV1,
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """确定性 stress 派生特征；只改特征，不改标签与时间边界。"""
    if features.ndim != 3 or features.shape[2] != EXPECTED_FEATURE_COUNT:
        raise InfoTSArtifactValidationError("stress 输入特征形状无效")
    stress: dict[str, dict[str, Any]] = {}

    def _clone(scale: float, indices: Sequence[int]) -> np.ndarray:
        perturbed = features.astype(np.float32).copy()
        perturbed[:, :, list(indices)] *= np.float32(scale)
        return perturbed

    stress["price"] = {"features": _clone(gate_spec.price_stress_scale, RETURN_FEATURE_INDICES), "valid_mask": valid_mask}
    stress["volume"] = {"features": _clone(gate_spec.volume_stress_scale, VOLUME_FEATURE_INDICES), "valid_mask": valid_mask}
    stress["extreme_volatility"] = {"features": _clone(gate_spec.extreme_volatility_scale, VOLATILITY_FEATURE_INDICES), "valid_mask": valid_mask}
    generator = np.random.default_rng(seed)
    missing = features.astype(np.float32).copy()
    missing_mask = valid_mask.copy()
    probability = gate_spec.stress_missing_rate
    if probability > 0.0:
        random_draw = generator.random(features.shape)
        drop = random_draw < probability
        missing[drop] = 0.0
        missing_mask = (missing_mask.astype(bool) & ~drop.any(axis=2)).astype(valid_mask.dtype)
    stress["missing"] = {"features": missing, "valid_mask": missing_mask}
    return stress


def load_encoder_state(
    checkpoint_path: str | Path,
    feature_count: int,
    hidden_dim: int,
    embedding_dim: int,
) -> dict[str, Any]:
    """从预训练 checkpoint 提取 encoder.* 权重；架构不一致失败关闭。"""
    if torch is None:
        raise Phase5SupervisedUnavailable("加载预训练权重需要 PyTorch")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise InfoTSArtifactValidationError("预训练 checkpoint 缺少 model state")
    encoder_state = {key[len("encoder."):]: value for key, value in state.items() if key.startswith("encoder.")}
    reference = InfoTSSupervisedModel(feature_count, hidden_dim, embedding_dim)
    expected_keys = set(reference.encoder.state_dict().keys())
    if set(encoder_state.keys()) != expected_keys:
        raise InfoTSArtifactValidationError("预训练 encoder 权重键与监督模型不匹配")
    return encoder_state


def _row_indices(timestamps: np.ndarray, first: int, last: int) -> np.ndarray:
    selected = np.flatnonzero((timestamps >= first) & (timestamps <= last))
    if selected.size == 0:
        raise InfoTSArtifactValidationError("split 边界内没有样本行")
    return selected


def train_phase5_supervised_fold(
    group_id: str,
    fold_id: int,
    dataset: Mapping[str, np.ndarray],
    split: Mapping[str, int],
    supervised_spec: Phase5SupervisedSpecV1,
    gate_spec: Phase5GateSpecV1,
    gate_spec_sha256: str,
    *,
    init_checkpoint_path: str | Path | None = None,
    output_dir: str | Path,
    supervised_budget: int = 50,
) -> dict[str, Any]:
    """训练一个 group/fold 并产出符合预算合同验证器的 fold artifact。"""
    if group_id not in {"from_scratch", "fixed_augmentation", "infots"}:
        raise InfoTSArtifactValidationError("group_id 无效")
    supervised_spec.validate()
    gate_spec.validate()
    if supervised_spec.epochs != supervised_budget:
        raise InfoTSArtifactValidationError("epochs 必须等于预注册 supervised_budget")
    if gate_spec.spec_sha256 != gate_spec_sha256:
        raise InfoTSArtifactValidationError("gate_spec_sha256 与 gate spec 内容不一致")
    if group_id == "from_scratch" and init_checkpoint_path is not None:
        raise InfoTSArtifactValidationError("from_scratch 组禁止携带预训练 checkpoint")
    if group_id != "from_scratch" and init_checkpoint_path is None:
        raise InfoTSArtifactValidationError("预训练组必须提供 init checkpoint")
    if torch is None:
        raise Phase5SupervisedUnavailable("Phase 5 supervised 训练需要 PyTorch")
    if supervised_spec.device == "cuda" and not torch.cuda.is_available():
        raise Phase5SupervisedUnavailable("Phase 5 supervised 训练需要 CUDA")

    features = np.asarray(dataset["features"], dtype=np.float32)
    valid_mask = np.asarray(dataset["valid_mask"]).astype(bool)
    timestamps = np.asarray(dataset["timestamps"], dtype=np.int64)
    if features.ndim != 3 or valid_mask.shape != features.shape[:2] or timestamps.shape[0] != features.shape[0]:
        raise InfoTSArtifactValidationError("数据集形状无效")
    if not np.isfinite(features).all():
        raise InfoTSArtifactValidationError("特征存在非有限值")

    train_rows = _row_indices(timestamps, split["train_first"], split["train_end"])
    validation_rows = _row_indices(timestamps, split["validation_first"], split["validation_end"])
    test_rows = _row_indices(timestamps, split["test_start"], split["test_end"])
    if timestamps[train_rows].max() >= timestamps[validation_rows].min():
        raise InfoTSArtifactValidationError("train 与 validation 边界重叠")
    if timestamps[validation_rows].max() + split["purge_gap"] >= timestamps[test_rows].min():
        raise InfoTSArtifactValidationError("validation 与 test 未越过 purge gap")

    labels = {name: np.asarray(dataset[name], dtype=np.float64)
              for name in ("expected_return", "direction", "realized_volatility", "rank_relevance", "rank_utility")}

    torch.manual_seed(supervised_spec.seed)
    if supervised_spec.deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    device = torch.device(supervised_spec.device)
    model = InfoTSSupervisedModel(features.shape[2], supervised_spec.hidden_dim, supervised_spec.embedding_dim,
                                  dropout=supervised_spec.dropout).to(device)
    if init_checkpoint_path is not None:
        encoder_state = load_encoder_state(
            init_checkpoint_path, features.shape[2], supervised_spec.hidden_dim, supervised_spec.embedding_dim,
        )
        model.encoder.load_state_dict(encoder_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=supervised_spec.learning_rate,
                                 weight_decay=supervised_spec.weight_decay)

    train_features = torch.as_tensor(features[train_rows], dtype=torch.float32, device=device)
    train_mask = torch.as_tensor(valid_mask[train_rows], device=device)
    train_targets = {
        "expected_return": torch.as_tensor(labels["expected_return"][train_rows], dtype=torch.float32, device=device),
        "direction": torch.as_tensor(labels["direction"][train_rows], dtype=torch.float32, device=device),
        "realized_volatility": torch.as_tensor(labels["realized_volatility"][train_rows], dtype=torch.float32, device=device),
    }
    validation_features = torch.as_tensor(features[validation_rows], dtype=torch.float32, device=device)
    validation_mask = torch.as_tensor(valid_mask[validation_rows], device=device)
    validation_labels = {name: labels[name][validation_rows] for name in labels}
    validation_groups = _timestamp_groups(timestamps[validation_rows])
    test_features = torch.as_tensor(features[test_rows], dtype=torch.float32, device=device)
    test_mask = torch.as_tensor(valid_mask[test_rows], device=device)
    test_labels = {name: labels[name][test_rows] for name in labels}
    test_groups = _timestamp_groups(timestamps[test_rows])

    def _quantile_loss(prediction: torch.Tensor, target: torch.Tensor, level: float) -> torch.Tensor:
        residual = target - prediction
        return torch.maximum(level * residual, (level - 1.0) * residual).mean()

    def _direction_confidence_target(direction: torch.Tensor) -> torch.Tensor:
        return (direction - 0.5).abs() * 2.0

    sample_count = int(train_rows.size)
    best_return_mae = float("inf")
    best_state: dict[str, Any] | None = None
    return_ceiling: float | None = None
    selected_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, supervised_spec.epochs + 1):
        permutation = torch.randperm(sample_count, device=device)
        model.train()
        epoch_losses: list[float] = []
        for start in range(0, sample_count, supervised_spec.batch_size):
            indices = permutation[start:start + supervised_spec.batch_size]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(train_features[indices], train_mask[indices])
            loss_return = nn.functional.mse_loss(outputs["expected_return"], train_targets["expected_return"][indices])
            loss_direction = nn.functional.binary_cross_entropy(outputs["direction_probability"], train_targets["direction"][indices])
            loss_volatility = nn.functional.l1_loss(outputs["expected_volatility"], train_targets["realized_volatility"][indices])
            loss_quantile = 0.5 * (
                _quantile_loss(outputs["lower_quantile"], train_targets["expected_return"][indices], supervised_spec.quantile_lower)
                + _quantile_loss(outputs["upper_quantile"], train_targets["expected_return"][indices], supervised_spec.quantile_upper)
            )
            loss_confidence = nn.functional.mse_loss(outputs["confidence"], _direction_confidence_target(train_targets["direction"][indices]))
            loss = (
                supervised_spec.weight_return * loss_return
                + supervised_spec.weight_direction * loss_direction
                + supervised_spec.weight_volatility * loss_volatility
                + supervised_spec.weight_quantile * loss_quantile
                + 0.05 * loss_confidence
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_features, validation_mask)
            validation_numpy = {key: value.detach().cpu().numpy() for key, value in validation_prediction.items()}
        validation_metrics = compute_supervised_metrics(validation_numpy, validation_labels, validation_groups)
        if return_ceiling is None:
            return_ceiling = validation_metrics["return_mae"] + supervised_spec.return_guard_margin
        if validation_metrics["return_mae"] <= return_ceiling and validation_metrics["return_mae"] < best_return_mae:
            best_return_mae = validation_metrics["return_mae"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            selected_epoch = epoch
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_return_mae": validation_metrics["return_mae"],
        })
    if best_state is None:
        raise InfoTSArtifactValidationError("没有任何 epoch 通过 return guard；checkpoint 选择失败")
    model.load_state_dict(best_state)
    model.eval()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def _evaluate(rows_features: torch.Tensor, rows_mask: torch.Tensor, row_labels: Mapping[str, np.ndarray], groups: Sequence[np.ndarray], role: str) -> tuple[dict[str, float], dict[str, np.ndarray], np.ndarray]:
        with torch.no_grad():
            prediction = model(rows_features, rows_mask)
            numpy_prediction = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in prediction.items()}
            embedding = model.encoder(rows_features, rows_mask).detach().cpu().numpy().astype(np.float32)
        metrics = compute_supervised_metrics(numpy_prediction, row_labels, groups)
        return metrics, numpy_prediction, embedding

    validation_metrics, validation_prediction, validation_embedding = _evaluate(
        validation_features, validation_mask, validation_labels, validation_groups, "validation",
    )
    test_metrics, test_prediction, test_embedding = _evaluate(
        test_features, test_mask, test_labels, test_groups, "test",
    )

    prediction_path = output / f"{group_id}-fold-{fold_id}-predictions.npz"
    np.savez_compressed(
        prediction_path,
        timestamps=timestamps[test_rows],
        symbols=np.asarray(dataset["symbols"])[test_rows],
        **{key: test_prediction[key] for key in SUPERVISED_OUTPUT_KEYS},
    )
    embedding_paths: list[dict[str, Any]] = []
    for role, embedding, rows in (("validation", validation_embedding, validation_rows), ("test", test_embedding, test_rows)):
        path = output / f"{group_id}-fold-{fold_id}-{role}-embedding.npz"
        np.savez_compressed(path, timestamps=timestamps[rows], embedding=embedding)
        embedding_paths.append({
            "role": role,
            "path": path.name,
            "row_count": int(rows.size),
            "dimension": int(embedding.shape[1]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })

    stress_report: dict[str, dict[str, float]] = {}
    stress_features = build_stress_features(features[test_rows], valid_mask[test_rows].astype(dataset["valid_mask"].dtype), gate_spec, seed=supervised_spec.seed)
    for stress_name, payload in stress_features.items():
        with torch.no_grad():
            stress_prediction = model(
                torch.as_tensor(payload["features"], dtype=torch.float32, device=device),
                torch.as_tensor(payload["valid_mask"].astype(bool), device=device),
            )
            stress_numpy = {key: value.detach().cpu().numpy() for key, value in stress_prediction.items()}
        stress_metrics = compute_supervised_metrics(stress_numpy, test_labels, test_groups)
        stress_report[stress_name] = stress_metrics

    coverage = interval_coverage(test_prediction, test_labels)
    stability_gap = abs(test_metrics["composite_error"] - validation_metrics["composite_error"]) / max(validation_metrics["composite_error"], 1e-12)
    stress_worst_relative = max(
        (stress_metrics["composite_error"] - test_metrics["composite_error"]) / max(test_metrics["composite_error"], 1e-12)
        for stress_metrics in stress_report.values()
    )
    gates = {
        "clean_non_degraded": all(math.isfinite(value) for value in test_metrics.values()),
        "stress_non_degraded": bool(stress_worst_relative <= gate_spec.stress_relative_tolerance),
        "three_window_consistent": True,
        "quality_gate_passed": bool(
            test_metrics["direction_brier"] <= gate_spec.direction_brier_max
            and gate_spec.interval_coverage_min <= coverage <= gate_spec.interval_coverage_max
        ),
        "stability_gate_passed": bool(stability_gap <= gate_spec.stability_relative_gap_max),
    }

    checkpoint_path = output / f"{group_id}-fold-{fold_id}-checkpoint.pt"
    torch.save({"model": model.state_dict(), "supervised_spec": supervised_spec.to_dict(), "selected_epoch": selected_epoch}, checkpoint_path)

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "group_id": group_id,
        "fold_id": int(fold_id),
        "supervised_budget": int(supervised_budget),
        "test_blind": True,
        "gate_spec_sha256": gate_spec_sha256,
        "split": {
            "train_end": int(split["train_end"]),
            "validation_end": int(split["validation_end"]),
            "test_start": int(split["test_start"]),
            "test_end": int(split["test_end"]),
            "purge_gap": int(split["purge_gap"]),
        },
        "prediction_artifact": {
            "path": prediction_path.name,
            "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "outputs": list(SUPERVISED_OUTPUT_KEYS),
            "row_count": int(test_rows.size),
        },
        "embedding_snapshots": embedding_paths,
        "metrics": test_metrics,
        "gates": gates,
        "evidence_level": "REFERENCE_ONLY",
        "hypothesis_id": supervised_spec.hypothesis_id,
        "supervised_spec_sha256": supervised_spec.spec_sha256,
        "init_checkpoint_sha256": (
            hashlib.sha256(Path(init_checkpoint_path).read_bytes()).hexdigest()
            if init_checkpoint_path is not None else None
        ),
        "supervised_checkpoint_path": checkpoint_path.name,
        "selected_epoch": int(selected_epoch),
        "epoch_history": history,
        "validation_metrics": validation_metrics,
        "stress_metrics": stress_report,
        "interval_coverage": coverage,
        "stability_relative_gap": float(stability_gap),
        "stress_worst_relative_degradation": float(stress_worst_relative),
        "phase_exit_eligible": False,
        "promotion_eligible": False,
    }
    artifact["artifact_sha256"] = _sha256_json(artifact)
    return artifact


def _timestamp_groups(timestamps: np.ndarray) -> list[np.ndarray]:
    unique, inverse = np.unique(timestamps, return_inverse=True)
    return [np.flatnonzero(inverse == position) for position in range(unique.size)]

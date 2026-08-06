"""Phase 2B robust-training candidates and auditable stress fixtures.

The candidates in this module are research-only.  They are intentionally
opt-in and never change the evaluation or export graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import numpy as np


PROTECTED_FEATURE_NAMES = frozenset({
    "is_suspended", "is_listed", "is_st", "is_tradable",
    "cross_section_return_rank", "price_position_20", "breakout_20",
})
VALID_MODES = frozenset({
    "none", "direction_apl", "latent_fgm", "feature_pgd", "structured_missing",
})
VALID_MISSING_MODES = frozenset({"none", "continuous_center_impute"})


@dataclass(frozen=True)
class RobustTrainingSpec:
    mode: str = "none"
    hypothesis_id: str | None = None
    epsilon: float = 0.01
    beta: float = 0.5
    pgd_steps: int = 3
    pgd_step_size: float | None = None
    apl_alpha: float = 1.0
    apl_beta: float = 1.0
    apl_label_clip: float = 1e-4
    feature_names: tuple[str, ...] = ()
    missing_mode: str = "none"
    missing_rate: float = 0.0

    def validate(self) -> "RobustTrainingSpec":
        if self.mode not in VALID_MODES:
            raise ValueError(
                "robust_training.mode 必须为 none/direction_apl/latent_fgm/feature_pgd/structured_missing"
            )
        if self.mode != "none" and not self.hypothesis_id:
            raise ValueError("Phase 2B challenger 必须提供 hypothesis_id")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("robust_training.epsilon 必须为有限正数")
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("robust_training.beta 不能为负数")
        if self.pgd_steps <= 0:
            raise ValueError("robust_training.pgd_steps 必须为正数")
        if self.pgd_step_size is not None and (
            not np.isfinite(self.pgd_step_size) or self.pgd_step_size <= 0
        ):
            raise ValueError("robust_training.pgd_step_size 必须为有限正数")
        if self.apl_alpha < 0 or self.apl_beta < 0:
            raise ValueError("APL 权重不能为负数")
        if self.apl_alpha + self.apl_beta <= 0:
            raise ValueError("APL 至少需要一个正权重")
        if not 0 < self.apl_label_clip < 0.5:
            raise ValueError("APL label_clip 必须位于 (0, 0.5)")
        if self.missing_mode not in VALID_MISSING_MODES:
            raise ValueError("missing_mode 必须为 none/continuous_center_impute")
        if not 0 <= self.missing_rate <= 1 or not np.isfinite(self.missing_rate):
            raise ValueError("missing_rate 必须位于 [0, 1]")
        if self.mode == "structured_missing" and self.missing_mode == "none":
            raise ValueError("structured_missing 必须指定 missing_mode")
        if self.mode == "structured_missing" and self.missing_rate <= 0:
            raise ValueError("structured_missing 的 missing_rate 必须为正数")
        if self.missing_mode != "none" and self.missing_rate <= 0:
            raise ValueError("启用 missing_mode 时 missing_rate 必须为正数")
        if self.mode == "feature_pgd":
            if not self.feature_names:
                raise ValueError("feature_pgd 必须提供训练特征名")
            if self.missing_mode != "none":
                raise ValueError("feature_pgd 默认必须是纯 PGD；缺失增强请单独使用 structured_missing")
        return self

    @property
    def candidate_id(self) -> str:
        return self.mode if self.hypothesis_id is None else self.hypothesis_id

    @property
    def adversarial_enabled(self) -> bool:
        return self.mode in {"latent_fgm", "feature_pgd"}


def validate_robust_training_config(
    config: Mapping[str, object] | None,
    *,
    feature_names: Sequence[str] = (),
) -> RobustTrainingSpec:
    """Parse the single-candidate Phase 2B configuration.

    A missing section is equivalent to the frozen baseline.  Unknown keys are
    rejected so a candidate cannot silently change the training contract.
    """
    value = {} if config is None else dict(config)
    allowed = {
        "mode", "hypothesis_id", "epsilon", "beta", "pgd_steps",
        "pgd_step_size", "apl_alpha", "apl_beta", "apl_label_clip",
        "production_eval", "feature_names", "missing_mode", "missing_rate",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            "robust_training 包含不支持的字段: " + ", ".join(unknown)
        )
    if value.get("production_eval") is True:
        raise ValueError("Phase 2B 对抗分支不得进入 eval/ONNX 图")
    configured_names = tuple(value.get("feature_names", feature_names))
    spec = RobustTrainingSpec(
        mode=str(value.get("mode", "none")),
        hypothesis_id=(
            str(value["hypothesis_id"]) if value.get("hypothesis_id") else None
        ),
        epsilon=float(value.get("epsilon", 0.01)),
        beta=float(value.get("beta", 0.5)),
        pgd_steps=int(value.get("pgd_steps", 3)),
        pgd_step_size=(
            float(value["pgd_step_size"])
            if value.get("pgd_step_size") is not None else None
        ),
        apl_alpha=float(value.get("apl_alpha", 1.0)),
        apl_beta=float(value.get("apl_beta", 1.0)),
        apl_label_clip=float(value.get("apl_label_clip", 1e-4)),
        feature_names=configured_names,
        missing_mode=str(value.get("missing_mode", "none")),
        missing_rate=float(value.get("missing_rate", 0.0)),
    )
    return spec.validate()


def _reduce_loss(value, reduction: str):
    import torch

    if reduction == "none":
        return value
    if reduction == "sum":
        return value.sum()
    if reduction == "mean":
        return value.mean()
    raise ValueError("reduction 必须为 none/sum/mean")


def apl_direction_loss(
    probability,
    target,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    label_clip: float = 1e-4,
    reduction: str = "mean",
):
    """NCE+RCE normalized APL for the direction head only."""
    import torch

    if alpha < 0 or beta < 0 or alpha + beta <= 0:
        raise ValueError("APL alpha/beta 必须非负且至少一个为正")
    if not 0 < label_clip < 0.5:
        raise ValueError("APL label_clip 必须位于 (0, 0.5)")
    probability = probability.clamp(label_clip, 1.0 - label_clip)
    target = target.to(dtype=probability.dtype)
    target_clipped = target.clamp(label_clip, 1.0 - label_clip)
    log_two = math.log(2.0)
    nce = -(
        target * torch.log(probability)
        + (1.0 - target) * torch.log1p(-probability)
    ) / log_two
    rce = -(
        probability * torch.log(target_clipped)
        + (1.0 - probability) * torch.log1p(-target_clipped)
    ) / log_two
    return _reduce_loss((alpha * nce + beta * rce) / (alpha + beta), reduction)


def direction_noise_audit(
    labels: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray | None = None,
    *,
    flip_rates: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
    seed: int = 20260804,
) -> dict:
    """Build a deterministic artificial-label-flip audit without mutating labels."""
    labels_array = np.asarray(labels, dtype=np.float64)
    if labels_array.ndim != 1 or labels_array.size == 0:
        raise ValueError("direction labels 必须是非空一维数组")
    if (
        not np.isfinite(labels_array).all()
        or (labels_array < 0.0).any()
        or (labels_array > 1.0).any()
    ):
        raise ValueError("direction labels 必须是 [0, 1] 内的有限值")
    probabilities_array = None
    if probabilities is not None:
        probabilities_array = np.asarray(probabilities, dtype=np.float64)
        if probabilities_array.shape != labels_array.shape:
            raise ValueError("probabilities 与 labels 形状必须一致")
        if not np.isfinite(probabilities_array).all():
            raise ValueError("probabilities 必须是有限值")
        probabilities_array = np.clip(probabilities_array, 1e-7, 1.0 - 1e-7)
    records = []
    for flip_rate in flip_rates:
        rate = float(flip_rate)
        if not 0 <= rate <= 1:
            raise ValueError("flip_rates 必须位于 [0, 1]")
        generator = np.random.default_rng(seed + len(records))
        flip_mask = generator.random(labels_array.size) < rate
        noisy = np.where(flip_mask, 1.0 - labels_array, labels_array)
        record = {
            "flip_rate": rate,
            "realized_flip_rate": float(flip_mask.mean()),
            "flipped_count": int(flip_mask.sum()),
            "sample_count": int(labels_array.size),
            "label_mean": float(noisy.mean()),
        }
        if probabilities_array is not None:
            record["brier"] = float(np.mean((probabilities_array - noisy) ** 2))
            record["bce"] = float(np.mean(
                -(noisy * np.log(probabilities_array)
                  + (1.0 - noisy) * np.log1p(-probabilities_array))
            ))
        records.append(record)
    return {
        "schema_version": 1,
        "kind": "direction_noise_audit",
        "seed": int(seed),
        "records": records,
    }


def _continuous_feature_mask(
    feature_count: int,
    feature_names: Sequence[str] | None,
):
    import torch

    if feature_names is None:
        return torch.ones(feature_count, dtype=torch.bool)
    if len(feature_names) != feature_count:
        raise ValueError("feature_names 长度必须等于 feature_count")
    return torch.tensor(
        [name not in PROTECTED_FEATURE_NAMES for name in feature_names],
        dtype=torch.bool,
    )


def feature_pgd_perturbation(
    features,
    valid_mask,
    loss_fn: Callable,
    *,
    epsilon: float,
    steps: int = 3,
    step_size: float | None = None,
    feature_names: Sequence[str] | None = None,
):
    """Mask-aware L-infinity PGD over continuous features only."""
    import torch

    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("PGD epsilon 必须为有限正数")
    if steps <= 0:
        raise ValueError("PGD steps 必须为正数")
    if step_size is None:
        step_size = epsilon / steps
    if step_size <= 0 or not np.isfinite(step_size):
        raise ValueError("PGD step_size 必须为有限正数")
    original = features.detach()
    feature_mask = _continuous_feature_mask(
        features.shape[-1], feature_names,
    ).to(device=features.device)
    update_mask = valid_mask.to(dtype=torch.bool).unsqueeze(-1) & feature_mask.view(1, 1, -1)
    adversarial = original.clone()
    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        objective = loss_fn(adversarial)
        gradient, = torch.autograd.grad(objective, adversarial, only_inputs=True)
        adversarial = adversarial.detach() + step_size * gradient.sign()
        adversarial = torch.maximum(
            torch.minimum(adversarial, original + epsilon), original - epsilon,
        )
        adversarial = torch.where(update_mask, adversarial, original)
    return adversarial.detach()


def structured_missing_augmentation(
    features,
    valid_mask,
    *,
    feature_names: Sequence[str],
    center,
    rate: float,
    generator=None,
):
    """Replace continuous features at valid tokens with train-fold centers."""
    import torch

    if not 0 < rate <= 1 or not np.isfinite(rate):
        raise ValueError("missing augmentation rate 必须位于 (0, 1]")
    if features.ndim != 3 or valid_mask.shape != features.shape[:2]:
        raise ValueError("features/valid_mask 形状不一致")
    if len(feature_names) != features.shape[-1]:
        raise ValueError("feature_names 长度必须等于 feature_count")
    center = torch.as_tensor(center, dtype=features.dtype, device=features.device)
    if center.ndim != 1 or center.numel() != features.shape[-1]:
        raise ValueError("center 长度必须等于 feature_count")
    continuous = torch.tensor(
        [name not in PROTECTED_FEATURE_NAMES for name in feature_names],
        dtype=torch.bool, device=features.device,
    )
    if generator is None:
        selector = torch.rand(
            valid_mask.shape, device=features.device,
        ) < rate
    else:
        selector = torch.rand(
            valid_mask.shape, generator=generator, device=features.device,
        ) < rate
    selector = selector & valid_mask.to(dtype=torch.bool)
    update = selector.unsqueeze(-1) & continuous.view(1, 1, -1)
    augmented = torch.where(update, center.view(1, 1, -1), features)
    return augmented, update


def latent_fgm_loss(
    model,
    features,
    valid_mask,
    target: Mapping[str, object],
    clean_loss_fn: Callable,
    *,
    epsilon: float,
    beta: float,
):
    """Return clean + latent-FGM loss while keeping the model graph intact."""
    import torch

    if epsilon <= 0 or beta < 0:
        raise ValueError("latent FGM epsilon 必须为正，beta 不能为负")
    clean_prediction = model(features, valid_mask, return_embedding=True)
    clean_loss = clean_loss_fn(clean_prediction, target)
    embedding = clean_prediction.get("embedding")
    if embedding is None:
        raise ValueError("模型必须返回 embedding 才能执行 latent FGM")
    gradient, = torch.autograd.grad(
        clean_loss, embedding, retain_graph=True, create_graph=False,
    )
    norm = gradient.flatten(start_dim=1).norm(p=2, dim=1, keepdim=True)
    delta = epsilon * gradient / (norm.view(-1, 1) + 1e-12)
    adversarial_prediction = model.predict_from_embedding(embedding + delta.detach())
    adversarial_loss = clean_loss_fn(adversarial_prediction, target)
    total = clean_loss + beta * adversarial_loss
    return total, {
        "clean_loss": float(clean_loss.detach()),
        "adversarial_loss": float(adversarial_loss.detach()),
        "latent_perturbation_norm": float(delta.flatten(start_dim=1).norm(p=2, dim=1).mean().detach()),
    }


def build_stress_sets(
    features: np.ndarray,
    valid_mask: np.ndarray,
    *,
    feature_names: Sequence[str],
    seed: int = 20260804,
    missing_mode: str = "raw_zero",
    missing_center: Sequence[float] | None = None,
) -> dict[str, dict[str, object]]:
    """Create deterministic price/volume/missing/extreme-volatility fixtures."""
    values = np.asarray(features, dtype=np.float32)
    masks = np.asarray(valid_mask, dtype=np.uint8)
    if values.ndim != 3 or masks.shape != values.shape[:2]:
        raise ValueError("features 必须为 [N,T,F] 且 valid_mask 必须为 [N,T]")
    names = tuple(feature_names)
    if len(names) != values.shape[-1]:
        raise ValueError("feature_names 长度必须等于 features 的 F")
    valid = masks.astype(bool)[..., None]
    continuous = np.asarray(
        [name not in PROTECTED_FEATURE_NAMES for name in names], dtype=bool,
    ).reshape(1, 1, -1)
    index = {name: position for position, name in enumerate(names)}
    generator = np.random.default_rng(seed)

    def fresh() -> np.ndarray:
        return values.copy()

    stress: dict[str, dict[str, object]] = {}
    price_names = [name for name in names if name.startswith(("log_return", "intraday", "close_open", "overnight", "ma_deviation", "price_position", "breakout", "cross_section"))]
    price_indices = [index[name] for name in price_names]
    price = fresh()
    if price_indices:
        price[..., price_indices] += generator.normal(0.0, 0.02, size=price[..., price_indices].shape).astype(np.float32) * valid
    stress["price"] = {"features": price, "valid_mask": masks.copy(), "metadata": {"kind": "price", "scale": 0.02}}

    volume = fresh()
    for name in ("log_volume", "volume_zscore_20"):
        if name in index:
            volume[..., index[name]] += generator.normal(0.0, 0.25, size=volume[..., index[name]].shape).astype(np.float32) * masks
    stress["volume"] = {"features": volume, "valid_mask": masks.copy(), "metadata": {"kind": "volume", "scale": 0.25}}

    if missing_mode not in {"raw_zero", "continuous_center_impute"}:
        raise ValueError("missing_mode 必须为 raw_zero/continuous_center_impute")
    if missing_mode == "continuous_center_impute":
        if missing_center is None:
            raise ValueError("continuous_center_impute 需要 missing_center")
        center = np.asarray(missing_center, dtype=np.float32)
        if center.shape != (values.shape[-1],) or not np.isfinite(center).all():
            raise ValueError("missing_center 必须是有限的 [F] 向量")
    else:
        center = np.zeros(values.shape[-1], dtype=np.float32)
    missing = fresh()
    missing_selector = generator.random(values.shape[:2]) < 0.10
    missing[..., :] = np.where(
        missing_selector[..., None] & valid & continuous,
        center.reshape(1, 1, -1), missing,
    )
    stress["missing"] = {
        "features": missing,
        "valid_mask": masks.copy(),
        "metadata": {
            "kind": "missing",
            "mode": missing_mode,
            "rate": 0.10,
        },
    }

    volatility = fresh()
    vol_names = [name for name in names if "volatility" in name or name.startswith("log_return") or name in {"intraday_range", "close_open_return", "overnight_gap"}]
    vol_indices = [index[name] for name in vol_names]
    if vol_indices:
        volatility[..., vol_indices] *= np.where(valid, 2.0, 1.0)
    stress["extreme_volatility"] = {"features": volatility, "valid_mask": masks.copy(), "metadata": {"kind": "extreme_volatility", "multiplier": 2.0}}
    return stress


def stress_degradation(
    clean_metric: float,
    stress_metrics: Mapping[str, float],
) -> dict[str, float]:
    """Convert stress metrics to comparable additive degradation values."""
    if not np.isfinite(clean_metric):
        raise ValueError("clean_metric 必须为有限值")
    return {name: float(value - clean_metric) for name, value in stress_metrics.items()}


__all__ = [
    "PROTECTED_FEATURE_NAMES", "RobustTrainingSpec", "apl_direction_loss",
    "build_stress_sets", "direction_noise_audit", "feature_pgd_perturbation",
    "structured_missing_augmentation",
    "latent_fgm_loss", "stress_degradation", "validate_robust_training_config",
]

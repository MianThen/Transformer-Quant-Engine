from __future__ import annotations

import random

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

from .ranking import lambda_loss_at_k, legacy_rank_loss, listmle_loss


KENDALL_TASKS = ("return", "direction", "volatility", "quantile")


if nn is not None:
    class KendallTaskWeights(nn.Module):
        """Global homoscedastic task weights; rank remains explicitly weighted."""

        def __init__(
            self,
            *,
            initial_log_variance: float = 0.0,
            regularizer: float = 0.5,
            minimum: float = -6.0,
            maximum: float = 6.0,
        ) -> None:
            super().__init__()
            values = (initial_log_variance, regularizer, minimum, maximum)
            if not np.isfinite(values).all() or regularizer <= 0 or minimum >= maximum:
                raise ValueError("Kendall 参数必须有限，regularizer 为正且 clamp 有序")
            self.log_variances = nn.ParameterDict({
                task: nn.Parameter(torch.tensor(float(initial_log_variance)))
                for task in KENDALL_TASKS
            })
            self.regularizer = float(regularizer)
            self.minimum = float(minimum)
            self.maximum = float(maximum)

        def forward(self, components):
            weighted = {}
            total = None
            for task in KENDALL_TASKS:
                log_variance = self.log_variances[task].clamp(
                    self.minimum, self.maximum
                )
                value = torch.exp(-log_variance) * components[task]
                weighted[task] = value
                term = value + self.regularizer * log_variance
                total = term if total is None else total + term
            return total, weighted

        def clamp_(self) -> None:
            with torch.no_grad():
                for value in self.log_variances.values():
                    value.clamp_(self.minimum, self.maximum)
else:
    class KendallTaskWeights:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("Kendall 动态权重需要安装项目 ml 可选依赖")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ranking_score(prediction, mode: str, risk_floor: float):
    if mode == "raw_return":
        return prediction["expected_return"]
    if mode == "risk_adjusted_return":
        return prediction["expected_return"] / prediction["expected_volatility"].clamp_min(
            risk_floor
        )
    raise ValueError(f"不支持的 ranking score mode: {mode}")


def _timestamp_rank_loss(
    prediction,
    target,
    *,
    variant: str,
    cutoff: int,
    temperature: float,
    score_mode: str,
    risk_floor: float,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("训练功能未安装；请安装 ml 可选依赖") from exc
    if variant == "none":
        return prediction["expected_return"].sum() * 0.0
    score = _ranking_score(prediction, score_mode, risk_floor)
    timestamp = target["timestamp"]
    mask = target.get("rank_mask")
    losses = []
    for value in torch.unique(timestamp, sorted=True):
        group = timestamp == value
        group_mask = group if mask is None else group & mask.bool()
        if variant == "legacy":
            losses.append(legacy_rank_loss(score, target["rank_utility"], group_mask))
        elif variant == "listmle":
            losses.append(listmle_loss(
                score,
                target["rank_utility"],
                mask=group_mask,
                tie_breaker=target["symbol_tie_breaker"],
            ))
        elif variant == "lambda":
            losses.append(lambda_loss_at_k(
                score,
                target["rank_relevance"],
                cutoff=cutoff,
                temperature=temperature,
                mask=group_mask,
            ))
        else:
            raise ValueError(f"不支持的 ranking loss variant: {variant}")
    if not losses:
        return score.sum() * 0.0
    return torch.stack(losses).mean()


def multitask_loss_components(
    prediction,
    target,
    *,
    ranking_variant="none",
    ranking_cutoff=20,
    rank_temperature=1.0,
    ranking_score_mode="raw_return",
    risk_floor=1e-4,
):
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("训练功能未安装；请安装 ml 可选依赖") from exc
    expected_return = target["expected_return"]
    lower_error = expected_return - prediction["lower_quantile"]
    upper_error = expected_return - prediction["upper_quantile"]
    return {
        "return": functional.huber_loss(
            prediction["expected_return"], target["expected_return"]
        ),
        "direction": functional.binary_cross_entropy(
            prediction["direction_probability"], target["direction"]
        ),
        "volatility": functional.huber_loss(
            prediction["expected_volatility"], target["realized_volatility"]
        ),
        "quantile": (
            torch.maximum(0.10 * lower_error, -0.90 * lower_error).mean()
            + torch.maximum(0.90 * upper_error, -0.10 * upper_error).mean()
        ),
        "rank": _timestamp_rank_loss(
            prediction,
            target,
            variant=ranking_variant,
            cutoff=ranking_cutoff,
            temperature=rank_temperature,
            score_mode=ranking_score_mode,
            risk_floor=risk_floor,
        ),
    }


def multitask_loss(
    prediction,
    target,
    *,
    return_weight=1.0,
    direction_weight=0.25,
    volatility_weight=0.25,
    quantile_weight=0.25,
    rank_weight=0.0,
    ranking_variant="none",
    ranking_cutoff=20,
    rank_temperature=1.0,
    ranking_score_mode="raw_return",
    risk_floor=1e-4,
):
    components = multitask_loss_components(
        prediction,
        target,
        ranking_variant=ranking_variant,
        ranking_cutoff=ranking_cutoff,
        rank_temperature=rank_temperature,
        ranking_score_mode=ranking_score_mode,
        risk_floor=risk_floor,
    )
    return (
        return_weight * components["return"]
        + direction_weight * components["direction"]
        + volatility_weight * components["volatility"]
        + quantile_weight * components["quantile"]
        + rank_weight * components["rank"]
    )


def train_epoch(model, batches, optimizer, device="cpu") -> float:
    model.train()
    total, count = 0.0, 0
    for batch in batches:
        features = batch["features"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        static_features = batch.get("static_features")
        if static_features is not None:
            static_features = static_features.to(device)
        target = {name: value.to(device) for name, value in batch["target"].items()}
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features, valid_mask, static_features)
        loss = multitask_loss(prediction, target)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * features.shape[0]
        count += features.shape[0]
    if count == 0:
        raise ValueError("训练 batch 不能为空")
    return total / count

from __future__ import annotations

import random

import numpy as np


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


def pinball_loss(prediction, target, quantile: float):
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile 必须位于 (0,1)")
    error = target - prediction
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("训练功能未安装；请安装 ml 可选依赖") from exc
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def multitask_loss_components(
    prediction,
    target,
    *,
    return_weight=1.0,
    direction_weight=0.25,
    volatility_weight=0.25,
    quantile_weight=0.25,
    rank_weight=0.0,
):
    try:
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("训练功能未安装；请安装 ml 可选依赖") from exc
    raw = {
        "return": functional.huber_loss(
            prediction["expected_return"], target["expected_return"]
        ),
        "direction": functional.binary_cross_entropy_with_logits(
            prediction["direction_logits"], target["direction"]
        ),
        "volatility": functional.huber_loss(
            prediction["expected_volatility"], target["realized_volatility"]
        ),
        "quantile": (
            pinball_loss(prediction["lower_quantile"], target["expected_return"], 0.10)
            + pinball_loss(prediction["upper_quantile"], target["expected_return"], 0.90)
        ),
    }
    if rank_weight:
        if "timestamps" not in target:
            raise ValueError("启用 rank loss 时 target 必须包含 timestamps")
        raw["rank"] = cross_section_rank_loss(
            prediction["expected_return"], target["expected_return"], target["timestamps"]
        )
    weighted = {
        "return": return_weight * raw["return"],
        "direction": direction_weight * raw["direction"],
        "volatility": volatility_weight * raw["volatility"],
        "quantile": quantile_weight * raw["quantile"],
    }
    if rank_weight:
        weighted["rank"] = rank_weight * raw["rank"]
    weighted["total"] = sum(weighted.values())
    return weighted


def multitask_loss(prediction, target, **weights):
    return multitask_loss_components(prediction, target, **weights)["total"]


def cross_section_rank_loss(prediction, target, timestamps):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("训练功能未安装；请安装 ml 可选依赖") from exc
    losses = []
    for timestamp in torch.unique(timestamps):
        selected = timestamps == timestamp
        section_prediction = prediction[selected]
        section_target = target[selected]
        if section_prediction.numel() < 2:
            continue
        target_rank = torch.argsort(torch.argsort(section_target)).to(section_prediction.dtype)
        prediction_centered = section_prediction - section_prediction.mean()
        rank_centered = target_rank - target_rank.mean()
        prediction_norm = torch.sqrt(prediction_centered.square().sum() + 1e-12)
        rank_norm = torch.sqrt(rank_centered.square().sum() + 1e-12)
        denominator = prediction_norm * rank_norm
        losses.append(1.0 - (prediction_centered * rank_centered).sum() / denominator)
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


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

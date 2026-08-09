"""PyTorch bridge for real shared-backbone gradient observations."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from .gradient_conflict import (
    GradientConflictArtifactValidationError,
    compute_gradient_conflict_metrics,
    pcgrad_project,
    shared_parameter_set_sha256,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]


class TorchGradientAdapterError(RuntimeError):
    """Raised when the PyTorch gradient adapter cannot collect valid gradients."""


def _require_torch() -> Any:
    if torch is None:
        raise TorchGradientAdapterError(
            "未安装 PyTorch；请复用 "
            "/Users/Zhuanz/PycharmProjects/PythonProject/.venv/bin/python，"
            "或按 docs/phase1e_pytorch.md 配置 .[ml]"
        )
    return torch


def _parameter_items(
    shared_parameters: Mapping[str, Any] | Sequence[Any],
) -> tuple[tuple[str, Any], ...]:
    torch_module = _require_torch()
    if isinstance(shared_parameters, Mapping):
        items = tuple(shared_parameters.items())
    else:
        items = tuple(
            (f"shared.{index}", parameter)
            for index, parameter in enumerate(shared_parameters)
        )
    if not items or len({name for name, _ in items}) != len(items):
        raise TorchGradientAdapterError("shared_parameters 名称必须非空且唯一")
    for name, parameter in items:
        if not isinstance(name, str) or not name:
            raise TorchGradientAdapterError("shared parameter 名称无效")
        if not isinstance(parameter, torch_module.nn.Parameter):
            raise TorchGradientAdapterError(f"shared parameter {name} 不是 torch.nn.Parameter")
        if not parameter.requires_grad:
            raise TorchGradientAdapterError(f"shared parameter {name} 必须 requires_grad")
    return items


def _loss_items(task_losses: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    torch_module = _require_torch()
    if not isinstance(task_losses, Mapping) or not task_losses:
        raise TorchGradientAdapterError("task_losses 必须是非空对象")
    items = tuple(task_losses.items())
    if len({name for name, _ in items}) != len(items):
        raise TorchGradientAdapterError("task_losses 名称不得重复")
    for name, loss in items:
        if not isinstance(name, str) or not name:
            raise TorchGradientAdapterError("task loss 名称无效")
        if not isinstance(loss, torch_module.Tensor):
            raise TorchGradientAdapterError(f"task loss {name} 必须是 torch.Tensor")
        if loss.ndim != 0:
            raise TorchGradientAdapterError(f"task loss {name} 必须是标量张量")
        if not loss.requires_grad:
            raise TorchGradientAdapterError(f"task loss {name} 必须保留 autograd 图")
        if not bool(torch_module.isfinite(loss.detach()).item()):
            raise TorchGradientAdapterError(f"task loss {name} 必须是有限值")
    return items


def _flatten_gradient(gradient: Any, parameters: Sequence[Any]) -> list[float]:
    torch_module = _require_torch()
    chunks: list[Any] = []
    for parameter, value in zip(parameters, gradient):
        if value is None:
            chunks.append(torch_module.zeros_like(parameter, memory_format=torch_module.preserve_format))
        else:
            chunks.append(value.detach())
    return torch_module.cat(
        [chunk.reshape(-1).to(device="cpu", dtype=torch_module.float64) for chunk in chunks]
    ).tolist()


def collect_shared_task_gradients(
    task_losses: Mapping[str, Any],
    shared_parameters: Mapping[str, Any] | Sequence[Any],
    *,
    amp_scaler: Any | None = None,
    losses_are_scaled: bool = False,
) -> dict[str, Any]:
    """Collect unscaled per-task shared gradients without touching ``parameter.grad``."""
    torch_module = _require_torch()
    loss_items = _loss_items(task_losses)
    parameter_items = _parameter_items(shared_parameters)
    parameters = tuple(parameter for _, parameter in parameter_items)
    parameter_names = tuple(name for name, _ in parameter_items)
    if losses_are_scaled and amp_scaler is None:
        raise TorchGradientAdapterError("losses_are_scaled=True 必须提供 amp_scaler")
    scale = 1.0
    if amp_scaler is not None:
        if not hasattr(amp_scaler, "get_scale"):
            raise TorchGradientAdapterError("amp_scaler 缺少 get_scale")
        scale = float(amp_scaler.get_scale())
        if not scale > 0.0:
            raise TorchGradientAdapterError("AMP loss scale 必须为正")
    gradients: dict[str, list[float]] = {}
    raw_losses: dict[str, float] = {}
    for index, (name, loss) in enumerate(loss_items):
        raw_losses[name] = float(loss.detach().cpu().item())
        scaled_loss = loss * scale if losses_are_scaled else loss
        values = torch_module.autograd.grad(
            scaled_loss,
            parameters,
            retain_graph=index + 1 < len(loss_items),
            create_graph=False,
            allow_unused=True,
        )
        if losses_are_scaled:
            values = tuple(value / scale if value is not None else None for value in values)
        gradients[name] = _flatten_gradient(values, parameters)
    return {
        "task_gradients": gradients,
        "task_losses": raw_losses,
        "shared_parameter_names": list(parameter_names),
        "shared_parameter_set_sha256": shared_parameter_set_sha256(parameter_names),
        "amp_enabled": amp_scaler is not None,
        "amp_unscale_before_measurement": True,
        "loss_scaling_mode": "dynamic" if amp_scaler is not None else "none",
        "loss_scale": scale if amp_scaler is not None else None,
        "parameter_grad_untouched": all(parameter.grad is None for parameter in parameters),
    }


def build_torch_gradient_conflict_observation(
    task_losses: Mapping[str, Any],
    shared_parameters: Mapping[str, Any] | Sequence[Any],
    *,
    fold: str,
    epoch: int,
    market_regime: str,
    initial_losses: Mapping[str, Any] | None = None,
    task_weights: Mapping[str, Any] | None = None,
    amp_scaler: Any | None = None,
    losses_are_scaled: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    collected = collect_shared_task_gradients(
        task_losses,
        shared_parameters,
        amp_scaler=amp_scaler,
        losses_are_scaled=losses_are_scaled,
    )
    metrics = compute_gradient_conflict_metrics(
        collected["task_gradients"],
        collected["task_losses"],
        initial_losses=initial_losses,
        task_weights=task_weights,
        task_names=tuple(collected["task_gradients"]),
    )
    observation = {
        "fold": fold,
        "epoch": epoch,
        "market_regime": market_regime,
        "overhead_ms": (time.perf_counter() - started) * 1000.0,
        "task_gradients": collected["task_gradients"],
        "task_losses": collected["task_losses"],
        "initial_losses": dict(initial_losses) if initial_losses is not None else None,
        "task_weights": dict(task_weights) if task_weights is not None else None,
        "metrics": metrics,
        "torch": {
            "shared_parameter_names": collected["shared_parameter_names"],
            "shared_parameter_set_sha256": collected["shared_parameter_set_sha256"],
            "amp_enabled": collected["amp_enabled"],
            "amp_unscale_before_measurement": collected["amp_unscale_before_measurement"],
            "loss_scaling_mode": collected["loss_scaling_mode"],
            "loss_scale": collected["loss_scale"],
            "parameter_grad_untouched": collected["parameter_grad_untouched"],
        },
    }
    if not collected["parameter_grad_untouched"]:
        raise TorchGradientAdapterError("梯度诊断不应写入 parameter.grad")
    return observation


def apply_pcgrad_shared_gradients(
    task_losses: Mapping[str, Any],
    shared_parameters: Mapping[str, Any] | Sequence[Any],
    *,
    seed: int,
    task_weights: Mapping[str, Any] | None = None,
    retain_graph: bool = False,
) -> dict[str, Any]:
    """Apply an explicit PCGrad shared-backbone update; heads remain untouched."""
    torch_module = _require_torch()
    loss_items = _loss_items(task_losses)
    parameter_items = _parameter_items(shared_parameters)
    parameters = tuple(parameter for _, parameter in parameter_items)
    names = tuple(name for name, _ in loss_items)
    raw_gradients: dict[str, list[float]] = {}
    tensor_gradients: dict[str, tuple[Any, ...]] = {}
    for index, (name, loss) in enumerate(loss_items):
        values = torch_module.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph or index + 1 < len(loss_items),
            create_graph=False,
            allow_unused=True,
        )
        tensor_gradients[name] = values
        raw_gradients[name] = _flatten_gradient(values, parameters)
    projected = pcgrad_project(
        raw_gradients,
        task_order=names,
        seed=seed,
        task_weights=task_weights,
    )
    offsets: list[tuple[int, int]] = []
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        offsets.append((offset, offset + size))
        offset += size
    aggregate = projected["aggregate_gradient"]
    for parameter, (start, end) in zip(parameters, offsets):
        parameter.grad = torch_module.tensor(
            aggregate[start:end], dtype=parameter.dtype, device=parameter.device
        ).reshape_as(parameter).clone()
    projected["shared_parameter_names"] = list(name for name, _ in parameter_items)
    projected["task_names"] = list(names)
    projected["raw_gradients"] = raw_gradients
    projected["head_gradients_untouched"] = True
    return projected


__all__ = [
    "TorchGradientAdapterError",
    "apply_pcgrad_shared_gradients",
    "build_torch_gradient_conflict_observation",
    "collect_shared_task_gradients",
]

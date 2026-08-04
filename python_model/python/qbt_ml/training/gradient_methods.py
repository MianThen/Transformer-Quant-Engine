from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


GRADIENT_TASKS = ("return", "direction", "volatility", "quantile", "rank")
GRADNORM_TASKS = ("return", "direction", "volatility", "quantile")


def _canonical_json(value: Any) -> bytes:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("Phase 1E 梯度方法需要安装项目 ml 可选依赖")


@dataclass(frozen=True)
class GradientDiagnosticSpecV1:
    schema_version: int = 1
    cadence_steps: int = 1
    seed: int = 20260801
    accumulation_steps: int = 1
    amp_enabled: bool = False
    loss_scale: float = 1.0
    report_version: str = "gradient-conflict-v1"

    def __post_init__(self) -> None:
        if (self.schema_version != 1 or self.cadence_steps < 1 or self.seed == 0 or
                self.accumulation_steps < 1 or not math.isfinite(self.loss_scale) or
                self.loss_scale <= 0.0 or not self.report_version):
            raise ValueError("GradientDiagnosticSpec V1 无效")

    @property
    def sha256(self) -> str:
        return _sha256(self)


@dataclass(frozen=True)
class PCGradSpecV1:
    schema_version: int = 1
    seed: int = 20260801
    zero_norm_epsilon: float = 1e-12
    accumulation_steps: int = 1
    gradients_unscaled: bool = True
    hypothesis_id: str = ""

    def __post_init__(self) -> None:
        if (self.schema_version != 1 or self.seed == 0 or
                not math.isfinite(self.zero_norm_epsilon) or
                self.zero_norm_epsilon <= 0.0 or self.accumulation_steps < 1 or
                not self.gradients_unscaled or not self.hypothesis_id):
            raise ValueError("PCGradSpec V1 必须冻结 seed、unscale 和 hypothesis_id")

    @property
    def sha256(self) -> str:
        return _sha256(self)


@dataclass(frozen=True)
class GradNormSpecV1:
    schema_version: int = 1
    alpha: float = 1.5
    rank_weight: float = 0.1
    update_every_steps: int = 1
    epsilon: float = 1e-8
    hypothesis_id: str = ""
    adaptive_tasks: tuple[str, ...] = GRADNORM_TASKS

    def __post_init__(self) -> None:
        if (self.schema_version != 1 or not math.isfinite(self.alpha) or self.alpha < 0.0 or
                self.rank_weight != 0.1 or self.update_every_steps < 1 or
                not math.isfinite(self.epsilon) or self.epsilon <= 0.0 or
                self.adaptive_tasks != GRADNORM_TASKS or not self.hypothesis_id):
            raise ValueError("GradNormSpec V1 仅允许四任务自适应且 rank 固定为 0.1")

    @property
    def sha256(self) -> str:
        return _sha256(self)


def shared_named_parameters(model) -> tuple[tuple[str, Any], ...]:
    _require_torch()
    values = []
    for name, parameter in model.named_parameters():
        root = name.split(".", 1)[0]
        if root.endswith("_head"):
            continue
        if parameter.requires_grad:
            values.append((name, parameter))
    if not values:
        raise ValueError("模型没有可诊断的共享参数")
    return tuple(values)


def shared_parameter_set_sha256(named_parameters: Sequence[tuple[str, Any]]) -> str:
    metadata = []
    for name, parameter in named_parameters:
        metadata.append({
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
        })
    return _sha256(metadata)


def _flatten_gradients(loss, parameters: Sequence[Any], *, create_graph: bool = False):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, create_graph=create_graph,
        allow_unused=True)
    flattened = []
    for parameter, gradient in zip(parameters, gradients):
        flattened.append((torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1))
    return torch.cat(flattened)


def _coordinate_summary(gradient) -> tuple[float, float]:
    absolute = gradient.detach().abs()
    if absolute.numel() == 0:
        return 0.0, 0.0
    return float(absolute.max()), float(absolute.median())


@dataclass
class GradientConflictRecorder:
    spec: GradientDiagnosticSpecV1
    initial_losses: dict[str, float] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    overhead_seconds: float = 0.0

    def sample(self, components: Mapping[str, Any], named_parameters,
               task_weights: Mapping[str, float], *, step: int, epoch: int,
               fold: int, regime: str = "ALL") -> dict[str, Any] | None:
        _require_torch()
        if step % self.spec.cadence_steps:
            return None
        if tuple(components) != GRADIENT_TASKS:
            raise ValueError("diagnostics 需要冻结的五任务顺序")
        if set(task_weights) != set(GRADIENT_TASKS):
            raise ValueError("diagnostics task weights 不完整")
        started = time.perf_counter()
        parameters = tuple(parameter for _, parameter in named_parameters)
        gradients = {}
        raw_losses = {}
        normalized_losses = {}
        norms = {}
        coordinate_max = {}
        coordinate_median = {}
        for task in GRADIENT_TASKS:
            raw = float(components[task].detach())
            if task not in self.initial_losses:
                self.initial_losses[task] = max(abs(raw), 1e-12)
            raw_losses[task] = raw
            normalized_losses[task] = raw / self.initial_losses[task]
            gradient = _flatten_gradients(
                task_weights[task] * components[task], parameters).detach()
            gradients[task] = gradient
            norms[task] = float(torch.linalg.vector_norm(gradient))
            coordinate_max[task], coordinate_median[task] = _coordinate_summary(gradient)
        dot_products = {}
        cosine = {}
        conflicts = 0
        pair_count = 0
        for left_index, left in enumerate(GRADIENT_TASKS):
            for right in GRADIENT_TASKS[left_index + 1:]:
                key = f"{left}|{right}"
                dot = float(torch.dot(gradients[left], gradients[right]))
                denominator = norms[left] * norms[right]
                similarity = dot / denominator if denominator > 0.0 else 0.0
                dot_products[key] = dot
                cosine[key] = similarity
                conflicts += int(similarity < 0.0)
                pair_count += 1
        aggregate = torch.stack(tuple(gradients.values())).sum(dim=0)
        aggregate_squared = float(torch.dot(aggregate, aggregate))
        norm_sum = sum(norms.values())
        projection = {
            task: (float(torch.dot(gradient, aggregate)) / aggregate_squared
                   if aggregate_squared > 0.0 else 0.0)
            for task, gradient in gradients.items()
        }
        dominance = {
            task: norms[task] / norm_sum if norm_sum > 0.0 else 0.0
            for task in GRADIENT_TASKS
        }
        relative = np.asarray(
            [normalized_losses[task] for task in GRADIENT_TASKS], dtype=np.float64)
        relative_mean = float(relative.mean())
        training_rate = {
            task: normalized_losses[task] / relative_mean if relative_mean > 0.0 else 0.0
            for task in GRADIENT_TASKS
        }
        positive_norms = [value for value in norms.values() if value > 0.0]
        norm_ratio = (max(positive_norms) / min(positive_norms)
                      if positive_norms else 0.0)
        sample = {
            "step": step,
            "epoch": epoch,
            "fold": fold,
            "regime": regime,
            "raw_losses": raw_losses,
            "normalized_losses": normalized_losses,
            "gradient_norms": norms,
            "coordinate_max_abs": coordinate_max,
            "coordinate_median_abs": coordinate_median,
            "dot_products": dot_products,
            "cosine_similarity": cosine,
            "negative_cosine_conflict_rate": conflicts / pair_count,
            "maximum_minimum_norm_ratio": norm_ratio,
            "aggregate_projection": projection,
            "dominance_ratio": dominance,
            "relative_training_rate": training_rate,
        }
        self.overhead_seconds += time.perf_counter() - started
        self.samples.append(sample)
        return sample

    def _aggregates(self) -> list[dict[str, Any]]:
        groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
        for sample in self.samples:
            key = (sample["epoch"], sample["fold"], sample["regime"])
            groups.setdefault(key, []).append(sample)
        aggregates = []
        for (epoch, fold, regime), samples in sorted(groups.items()):
            pair_keys = tuple(samples[0]["cosine_similarity"])
            conflict_matrix = {
                pair: {
                    "mean_cosine": float(np.mean([
                        sample["cosine_similarity"][pair] for sample in samples
                    ])),
                    "negative_rate": float(np.mean([
                        sample["cosine_similarity"][pair] < 0.0 for sample in samples
                    ])),
                }
                for pair in pair_keys
            }
            aggregates.append({
                "epoch": epoch,
                "fold": fold,
                "regime": regime,
                "sample_count": len(samples),
                "negative_cosine_conflict_rate": float(np.mean([
                    sample["negative_cosine_conflict_rate"] for sample in samples
                ])),
                "maximum_minimum_norm_ratio": float(np.mean([
                    sample["maximum_minimum_norm_ratio"] for sample in samples
                ])),
                "gradient_norms": {
                    task: float(np.mean([
                        sample["gradient_norms"][task] for sample in samples
                    ]))
                    for task in GRADIENT_TASKS
                },
                "dominance_ratio": {
                    task: float(np.mean([
                        sample["dominance_ratio"][task] for sample in samples
                    ]))
                    for task in GRADIENT_TASKS
                },
                "relative_training_rate": {
                    task: float(np.mean([
                        sample["relative_training_rate"][task] for sample in samples
                    ]))
                    for task in GRADIENT_TASKS
                },
                "conflict_matrix": conflict_matrix,
            })
        return aggregates

    def artifact(self, named_parameters, *, model_contract_sha256: str,
                 dataset_sha256: str) -> dict[str, Any]:
        if len(model_contract_sha256) != 64 or len(dataset_sha256) != 64:
            raise ValueError("artifact source hashes 必须是 SHA-256")
        payload = {
            "schema_version": 1,
            "report_version": self.spec.report_version,
            "diagnostic_spec_sha256": self.spec.sha256,
            "model_contract_sha256": model_contract_sha256,
            "dataset_sha256": dataset_sha256,
            "shared_parameter_set_sha256": shared_parameter_set_sha256(named_parameters),
            "amp_enabled": self.spec.amp_enabled,
            "loss_scale": self.spec.loss_scale,
            "accumulation_steps": self.spec.accumulation_steps,
            "sampling_seed": self.spec.seed,
            "overhead_seconds": self.overhead_seconds,
            "samples": self.samples,
            "aggregates": self._aggregates(),
        }
        return {**payload, "report_sha256": _sha256(payload)}


def _pcgrad_project_with_metrics(task_gradients: Mapping[str, Any], spec: PCGradSpecV1,
                                 *, step: int = 0):
    _require_torch()
    if tuple(task_gradients) != GRADIENT_TASKS:
        raise ValueError("PCGrad 需要冻结的五任务顺序")
    shapes = {tuple(value.shape) for value in task_gradients.values()}
    if len(shapes) != 1:
        raise ValueError("PCGrad task gradient shape 必须一致")
    randomizer = random.Random(spec.seed + step)
    projected = {task: gradient.clone() for task, gradient in task_gradients.items()}
    tasks = list(GRADIENT_TASKS)
    projection_count = 0
    zero_norm_skip_count = 0
    for task in tasks:
        others = [other for other in tasks if other != task]
        randomizer.shuffle(others)
        for other in others:
            denominator = torch.dot(task_gradients[other], task_gradients[other])
            if float(denominator.detach()) <= spec.zero_norm_epsilon:
                zero_norm_skip_count += 1
                continue
            dot = torch.dot(projected[task], task_gradients[other])
            if float(dot.detach()) < 0.0:
                projected[task] = projected[task] - dot / denominator * task_gradients[other]
                projection_count += 1
    aggregate = torch.stack(tuple(projected.values())).sum(dim=0)
    return projected, aggregate, projection_count, zero_norm_skip_count


def pcgrad_project(task_gradients: Mapping[str, Any], spec: PCGradSpecV1,
                   *, step: int = 0) -> tuple[dict[str, Any], Any]:
    projected, aggregate, _, _ = _pcgrad_project_with_metrics(
        task_gradients, spec, step=step
    )
    return projected, aggregate


def _assign_flat_gradient(parameters: Sequence[Any], gradient, previous) -> None:
    offset = 0
    for parameter, old_gradient in zip(parameters, previous):
        count = parameter.numel()
        value = gradient[offset:offset + count].view_as(parameter)
        parameter.grad = value.clone() if old_gradient is None else old_gradient + value
        offset += count
    if offset != gradient.numel():
        raise ValueError("flat gradient 与共享参数维度不一致")


def backward_pcgrad(weighted_task_losses: Mapping[str, Any], total_loss,
                    shared_parameters: Sequence[Any], spec: PCGradSpecV1,
                    *, step: int = 0) -> dict[str, Any]:
    _require_torch()
    if tuple(weighted_task_losses) != GRADIENT_TASKS:
        raise ValueError("PCGrad 需要冻结的五任务顺序")
    parameters = tuple(shared_parameters)
    if not parameters:
        raise ValueError("PCGrad shared parameters 不能为空")
    gradients = {
        task: _flatten_gradients(weighted_task_losses[task], parameters).detach()
        for task in GRADIENT_TASKS
    }
    previous = [None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in parameters]
    total_loss.backward()
    projected, aggregate, projection_count, zero_norm_skip_count = \
        _pcgrad_project_with_metrics(gradients, spec, step=step)
    _assign_flat_gradient(parameters, aggregate, previous)
    pair_opportunities = len(GRADIENT_TASKS) * (len(GRADIENT_TASKS) - 1)
    return {
        "spec_sha256": spec.sha256,
        "step": step,
        "raw_shared_norms": {
            task: float(torch.linalg.vector_norm(value))
            for task, value in gradients.items()
        },
        "projected_shared_norms": {
            task: float(torch.linalg.vector_norm(value))
            for task, value in projected.items()
        },
        "aggregate_shared_norm": float(torch.linalg.vector_norm(aggregate)),
        "projection_count": projection_count,
        "projection_frequency": projection_count / pair_opportunities,
        "zero_norm_skip_count": zero_norm_skip_count,
    }


if nn is not None:
    class GradNormController(nn.Module):
        def __init__(self, spec: GradNormSpecV1) -> None:
            super().__init__()
            self.spec = spec
            inverse_softplus_one = math.log(math.e - 1.0)
            self.raw_weights = nn.Parameter(torch.full(
                (len(spec.adaptive_tasks),), inverse_softplus_one,
                dtype=torch.float32))
            self.register_buffer(
                "initial_losses", torch.full((len(spec.adaptive_tasks),), float("nan")))

        def normalized_weights(self):
            positive = torch.nn.functional.softplus(self.raw_weights) + self.spec.epsilon
            return positive * (len(self.spec.adaptive_tasks) / positive.sum())

        def weighted_task_losses(self, components: Mapping[str, Any]) -> dict[str, Any]:
            if tuple(components) != GRADIENT_TASKS:
                raise ValueError("GradNorm 需要冻结的五任务顺序")
            weights = self.normalized_weights()
            return {
                task: weights[index] * components[task]
                for index, task in enumerate(self.spec.adaptive_tasks)
            }

        def model_loss(self, components: Mapping[str, Any]):
            weighted = self.weighted_task_losses(components)
            total = torch.stack(tuple(weighted.values())).sum()
            return total + self.spec.rank_weight * components["rank"]

        def gradnorm_objective(self, components: Mapping[str, Any],
                               shared_parameters: Sequence[Any]):
            if tuple(components) != GRADIENT_TASKS:
                raise ValueError("GradNorm 需要冻结的五任务顺序")
            parameters = tuple(shared_parameters)
            if not parameters:
                raise ValueError("GradNorm shared parameters 不能为空")
            losses = torch.stack(tuple(components[task] for task in self.spec.adaptive_tasks))
            with torch.no_grad():
                missing = torch.isnan(self.initial_losses)
                self.initial_losses[missing] = losses.detach()[missing].clamp_min(
                    self.spec.epsilon)
            weights = self.normalized_weights()
            unweighted_norms = torch.stack(tuple(
                torch.linalg.vector_norm(_flatten_gradients(
                    components[task], parameters).detach())
                for task in self.spec.adaptive_tasks
            ))
            norms = weights * unweighted_norms
            relative = losses.detach().clamp_min(0.0) / self.initial_losses.clamp_min(
                self.spec.epsilon
            )
            rates = relative / relative.mean().clamp_min(self.spec.epsilon)
            targets = (norms.detach().mean() * rates.pow(self.spec.alpha)).detach()
            objective = torch.abs(norms - targets).sum()
            diagnostics = {
                "weights": {
                    task: float(value.detach())
                    for task, value in zip(self.spec.adaptive_tasks,
                                           self.normalized_weights())
                },
                "gradient_norms": {
                    task: float(value.detach())
                    for task, value in zip(self.spec.adaptive_tasks, norms)
                },
                "relative_training_rates": {
                    task: float(value)
                    for task, value in zip(self.spec.adaptive_tasks, rates)
                },
                "target_gradient_norms": {
                    task: float(value)
                    for task, value in zip(self.spec.adaptive_tasks, targets)
                },
                "rank_weight": self.spec.rank_weight,
                "spec_sha256": self.spec.sha256,
            }
            return objective, diagnostics
else:
    class GradNormController:
        def __init__(self, _spec: GradNormSpecV1) -> None:
            _require_torch()

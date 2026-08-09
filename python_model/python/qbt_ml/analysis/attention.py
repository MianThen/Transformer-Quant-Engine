from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, Iterator, Mapping, Sequence

try:
    import torch
except ImportError:
    torch = None


PRODUCTION_OUTPUT_KEYS = (
    "expected_return",
    "expected_volatility",
    "direction_probability",
    "lower_quantile",
    "upper_quantile",
    "confidence",
)


class AttentionAnalysisError(ValueError):
    pass


def _require_torch():
    if torch is None:
        raise RuntimeError("Attention Analysis 需要安装 PyTorch")
    return torch


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _update_tensor_hash(hasher, name: str, value) -> None:
    tensor = value.detach().cpu().contiguous()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(tensor.dtype).encode("ascii"))
    hasher.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    hasher.update(tensor.view(torch.uint8).numpy().tobytes())


def _model_sha256(model) -> str:
    hasher = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        _update_tensor_hash(hasher, name, value)
    return hasher.hexdigest()


def _input_sha256(features, valid_mask, static_features, timestamp_rows) -> str:
    hasher = hashlib.sha256()
    mask = valid_mask.to(dtype=torch.bool, device=features.device)
    masked_features = torch.where(
        mask.unsqueeze(-1), features, torch.zeros_like(features),
    )
    _update_tensor_hash(hasher, "features_at_valid_times", masked_features)
    _update_tensor_hash(hasher, "valid_mask", mask)
    if static_features is not None:
        _update_tensor_hash(hasher, "static_features", static_features)
    valid_timestamps = [
        [row[index] for index, keep in enumerate(mask_row) if keep]
        for row, mask_row in zip(timestamp_rows, mask.detach().cpu().tolist())
    ]
    hasher.update(json.dumps(
        valid_timestamps, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8"))
    return hasher.hexdigest()


def _as_timestamp_rows(timestamps, batch_size: int, time_steps: int):
    if timestamps is None:
        return [list(range(time_steps)) for _ in range(batch_size)]
    if hasattr(timestamps, "detach"):
        timestamps = timestamps.detach().cpu().tolist()
    else:
        timestamps = list(timestamps)
    nested = (
        len(timestamps) == batch_size
        and all(
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            for value in timestamps
        )
    )
    if nested:
        rows = [list(row) for row in timestamps]
    elif len(timestamps) == time_steps:
        rows = [list(timestamps) for _ in range(batch_size)]
    else:
        raise AttentionAnalysisError(
            "timestamps 必须为 [T] 或 [N,T]，并与 features 对齐",
        )
    if any(len(row) != time_steps for row in rows):
        raise AttentionAnalysisError("timestamps 的时间维度与 features 不一致")
    serialized_rows = []
    for row in rows:
        serialized = []
        for value in row:
            if isinstance(value, Real) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    raise AttentionAnalysisError("timestamp 必须有限")
                serialized.append(
                    int(value) if isinstance(value, Integral) else float(value),
                )
            elif isinstance(value, str):
                serialized.append(value)
            elif hasattr(value, "isoformat"):
                serialized.append(value.isoformat())
            else:
                raise AttentionAnalysisError("timestamp 必须是数值、字符串或日期时间")
        serialized_rows.append(serialized)
    return serialized_rows


def _validate_timestamp_order(timestamp_rows, valid_mask) -> None:
    for sample_index, (row, mask_row) in enumerate(zip(
        timestamp_rows, valid_mask.detach().cpu().to(dtype=torch.bool).tolist(),
    )):
        valid_values = [value for value, keep in zip(row, mask_row) if keep]
        if not valid_values:
            raise AttentionAnalysisError(f"样本 {sample_index} 没有有效时间点")
        numeric = all(isinstance(value, Real) for value in valid_values)
        textual = all(isinstance(value, str) for value in valid_values)
        if not numeric and not textual:
            raise AttentionAnalysisError("同一样本的有效 timestamp 类型必须一致")
        if any(left >= right for left, right in zip(valid_values, valid_values[1:])):
            raise AttentionAnalysisError(
                f"样本 {sample_index} 的有效 timestamp 必须严格递增",
            )


def _validate_inputs(model, features, valid_mask, static_features, top_k: int) -> None:
    if features.ndim != 3:
        raise AttentionAnalysisError("features 必须为 [N,T,F]")
    if valid_mask.shape != features.shape[:2]:
        raise AttentionAnalysisError("valid_mask 必须为 [N,T] 并与 features 对齐")
    if features.device != valid_mask.device:
        raise AttentionAnalysisError("features 与 valid_mask 必须位于同一设备")
    if static_features is not None and static_features.shape[0] != features.shape[0]:
        raise AttentionAnalysisError("static_features 的 batch 维度不一致")
    if static_features is not None and static_features.device != features.device:
        raise AttentionAnalysisError("static_features 与 features 必须位于同一设备")
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
        raise AttentionAnalysisError("top_k 必须为正整数")
    valid_counts = valid_mask.to(dtype=torch.bool).sum(dim=1)
    insufficient = torch.nonzero(valid_counts < 2 * top_k, as_tuple=False).flatten()
    if insufficient.numel():
        indices = insufficient.detach().cpu().tolist()
        raise AttentionAnalysisError(
            f"样本 {indices} 的有效时间点不足以构造互斥 top/random 对照",
        )
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None or not len(layers):
        raise AttentionAnalysisError("模型必须暴露 encoder.layers")
    for layer_index, layer in enumerate(layers):
        if not hasattr(layer, "self_attn"):
            raise AttentionAnalysisError(
                f"encoder layer {layer_index} 未暴露 self_attn",
            )


@contextmanager
def _temporary_eval(model) -> Iterator[None]:
    modules = list(model.modules())
    training_states = [module.training for module in modules]
    model.eval()
    try:
        yield
    finally:
        for module, state in zip(modules, training_states):
            module.training = state


@contextmanager
def _capture_attention(model) -> Iterator[list[list[Any]]]:
    captured = [[] for _ in model.encoder.layers]
    restorations = []
    handles = []
    try:
        for layer_index, layer in enumerate(model.encoder.layers):
            attention = layer.self_attn
            had_instance_forward = "forward" in attention.__dict__
            instance_forward = attention.__dict__.get("forward")
            original_forward = attention.forward

            def capture_hook(_module, _inputs, output, *, index=layer_index):
                if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                    raise AttentionAnalysisError(
                        f"encoder layer {index} 未返回注意力权重",
                    )
                captured[index].append(output[1].detach().clone())

            def analysis_forward(*args, _original=original_forward, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return _original(*args, **kwargs)

            attention.forward = analysis_forward
            restorations.append((attention, had_instance_forward, instance_forward))
            handles.append(attention.register_forward_hook(capture_hook))
        yield captured
    finally:
        for handle in reversed(handles):
            handle.remove()
        for attention, had_instance_forward, instance_forward in reversed(restorations):
            if had_instance_forward:
                attention.forward = instance_forward
            elif "forward" in attention.__dict__:
                del attention.forward


def _model_forward(model, features, valid_mask, static_features):
    if static_features is None:
        return model(features, valid_mask)
    return model(features, valid_mask, static_features=static_features)


def _validate_prediction(prediction: Mapping[str, Any]) -> None:
    if not isinstance(prediction, Mapping):
        raise AttentionAnalysisError("模型 forward 必须返回六输出映射")
    keys = tuple(prediction.keys())
    if set(keys) != set(PRODUCTION_OUTPUT_KEYS):
        raise AttentionAnalysisError(
            "分析 forward 要求模型保持且仅返回生产六输出",
        )
    for key in PRODUCTION_OUTPUT_KEYS:
        value = prediction[key]
        if not hasattr(value, "shape") or value.ndim != 1:
            raise AttentionAnalysisError(f"输出 {key} 必须为 [N]")
        if not torch.isfinite(value).all():
            raise AttentionAnalysisError(f"输出 {key} 含非有限值")


def _prediction_parity(reference, captured, *, atol: float = 1e-6, rtol: float = 1e-5):
    differences = {}
    for key in PRODUCTION_OUTPUT_KEYS:
        left = reference[key]
        right = captured[key]
        if left.shape != right.shape:
            raise AttentionAnalysisError(f"捕获路径改变了输出 {key} 的形状")
        difference = (left - right).abs()
        differences[key] = float(difference.max().item()) if difference.numel() else 0.0
        if not torch.allclose(left, right, atol=atol, rtol=rtol):
            raise AttentionAnalysisError(f"捕获路径改变了生产输出 {key}")
    return {
        "atol": atol,
        "rtol": rtol,
        "max_abs_difference_by_output": differences,
        "max_abs_difference": max(differences.values()),
        "passed": True,
    }


def _prepare_attention(captured, valid_mask):
    layer_weights = []
    for layer_index, calls in enumerate(captured):
        if len(calls) != 1:
            raise AttentionAnalysisError(
                f"encoder layer {layer_index} 捕获次数应为 1，实际为 {len(calls)}",
            )
        weights = calls[0]
        if weights.ndim != 4:
            raise AttentionAnalysisError(
                "注意力权重必须为 [N,H,T,T]；当前模型未提供逐 head 权重",
            )
        layer_weights.append(weights)
    stacked = torch.stack(layer_weights, dim=0).to(dtype=torch.float64)
    layer_count, batch_size, _, query_steps, key_steps = stacked.shape
    if query_steps != key_steps or query_steps != valid_mask.shape[1]:
        raise AttentionAnalysisError("注意力时间维度与输入不一致")
    finite = torch.where(torch.isfinite(stacked), stacked, torch.zeros_like(stacked))
    if float(finite.min().item()) < -1e-7:
        raise AttentionAnalysisError("注意力权重包含负值")
    finite = finite.clamp_min(0.0)
    future = torch.triu(torch.ones(
        query_steps, key_steps, dtype=torch.bool, device=stacked.device,
    ), diagonal=1)
    future_values = finite.masked_select(future.view(1, 1, 1, query_steps, key_steps))
    max_future = float(future_values.abs().max().item()) if future_values.numel() else 0.0
    if max_future > 1e-7:
        raise AttentionAnalysisError("捕获到未来时间点注意力，拒绝生成分析产物")
    valid = valid_mask.to(dtype=torch.bool, device=stacked.device)
    invalid_key = (~valid).view(1, batch_size, 1, 1, key_steps)
    invalid_key_values = finite.masked_select(invalid_key)
    max_invalid_key = (
        float(invalid_key_values.abs().max().item())
        if invalid_key_values.numel() else 0.0
    )
    if max_invalid_key > 1e-7:
        raise AttentionAnalysisError("捕获到 padding key 注意力，拒绝生成分析产物")
    causal = ~future
    allowed = (
        valid.view(1, batch_size, 1, query_steps, 1)
        & valid.view(1, batch_size, 1, 1, key_steps)
        & causal.view(1, 1, 1, query_steps, key_steps)
    )
    normalized = torch.where(allowed, finite, torch.zeros_like(finite))
    row_sum = normalized.sum(dim=-1, keepdim=True)
    normalized = torch.where(
        row_sum > 0.0,
        normalized / row_sum.clamp_min(torch.finfo(normalized.dtype).tiny),
        torch.zeros_like(normalized),
    )
    return normalized, max_future, max_invalid_key


def _last_indices(valid_mask):
    positions = torch.arange(valid_mask.shape[1], device=valid_mask.device).expand_as(valid_mask)
    return positions.masked_fill(~valid_mask.to(dtype=torch.bool), -1).max(dim=1).values


def _attention_metrics(weights, valid_mask):
    layer_count, batch_size, head_count, time_steps, _ = weights.shape
    indices = _last_indices(valid_mask)
    last_attention = torch.stack([
        weights[:, sample_index, :, int(indices[sample_index].item()), :]
        for sample_index in range(batch_size)
    ], dim=0)
    positive = last_attention.clamp_min(torch.finfo(last_attention.dtype).tiny)
    entropy = -(last_attention * positive.log()).sum(dim=-1)
    valid_count = valid_mask.to(dtype=torch.bool).sum(dim=1).to(dtype=weights.dtype)
    entropy_scale = valid_count.log().view(batch_size, 1, 1)
    normalized_entropy = torch.where(
        entropy_scale > 0.0, entropy / entropy_scale, torch.zeros_like(entropy),
    )
    positions = torch.arange(time_steps, device=weights.device, dtype=weights.dtype)
    distance = indices.to(dtype=weights.dtype).view(batch_size, 1, 1, 1) - positions
    span = (last_attention * distance.clamp_min(0.0)).sum(dim=-1)
    maximum_span = torch.stack([
        indices[sample_index] - torch.nonzero(
            valid_mask[sample_index].to(dtype=torch.bool), as_tuple=False,
        ).flatten().min()
        for sample_index in range(batch_size)
    ]).to(dtype=weights.dtype).view(batch_size, 1, 1)
    normalized_span = torch.where(
        maximum_span > 0.0, span / maximum_span, torch.zeros_like(span),
    )
    identity = torch.eye(time_steps, dtype=weights.dtype, device=weights.device)
    rollout = identity.unsqueeze(0).expand(batch_size, -1, -1).clone()
    for layer_index in range(layer_count):
        mean_attention = weights[layer_index].mean(dim=1)
        residual_attention = mean_attention + identity.unsqueeze(0)
        residual_attention = residual_attention / residual_attention.sum(
            dim=-1, keepdim=True,
        ).clamp_min(torch.finfo(weights.dtype).tiny)
        rollout = residual_attention @ rollout
    last_rollout = torch.stack([
        rollout[sample_index, int(indices[sample_index].item()), :]
        for sample_index in range(batch_size)
    ])
    return {
        "last_indices": indices,
        "last_attention": last_attention,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "span": span,
        "normalized_span": normalized_span,
        "rollout": rollout,
        "last_rollout": last_rollout,
        "layer_count": layer_count,
        "head_count": head_count,
    }


def _select_occlusion_indices(last_rollout, valid_mask, top_k: int, random_seed: int):
    top_indices = []
    control_indices = []
    top_scores = []
    for sample_index, mask_row in enumerate(
        valid_mask.detach().cpu().to(dtype=torch.bool).tolist(),
    ):
        valid_indices = [index for index, keep in enumerate(mask_row) if keep]
        score_row = last_rollout[sample_index].detach().cpu().tolist()
        selected = sorted(
            valid_indices, key=lambda index: (-score_row[index], index),
        )[:top_k]
        candidates = [index for index in valid_indices if index not in selected]
        controls = sorted(
            candidates,
            key=lambda index: hashlib.sha256(
                f"{random_seed}:{sample_index}:{index}".encode("ascii"),
            ).digest(),
        )[:top_k]
        top_indices.append(selected)
        control_indices.append(controls)
        top_scores.append([float(score_row[index]) for index in selected])
    return top_indices, control_indices, top_scores


def _neutral_feature_baseline(model, features):
    baseline = torch.zeros(features.shape[-1], dtype=features.dtype, device=features.device)
    policy = "raw_zero"
    mean = getattr(model, "input_mean", None)
    if mean is not None and mean.numel() == features.shape[-1]:
        baseline = mean.detach().to(dtype=features.dtype, device=features.device).reshape(-1)
        protected = getattr(model, "input_protected", None)
        if protected is not None and protected.numel() == features.shape[-1]:
            protected = protected.detach().to(dtype=torch.bool, device=features.device).reshape(-1)
            baseline = torch.where(protected, torch.zeros_like(baseline), baseline)
            policy = "model_input_mean_unprotected_zero_protected"
        else:
            policy = "model_input_mean"
    return baseline, policy


def _occlude(features, selections, baseline):
    result = features.detach().clone()
    for sample_index, indices in enumerate(selections):
        result[sample_index, indices, :] = baseline
    return result


def _float_list(value):
    return value.detach().cpu().to(dtype=torch.float64).tolist()


def analyze_temporal_attention(
    model,
    features,
    valid_mask,
    *,
    static_features=None,
    timestamps=None,
    target_output: str = "expected_return",
    top_k: int = 1,
    random_seed: int = 1729,
) -> dict[str, Any]:
    """Build a deterministic, analysis-only attention and occlusion artifact."""
    _require_torch()
    if target_output not in PRODUCTION_OUTPUT_KEYS:
        raise AttentionAnalysisError(f"未知的 target_output: {target_output}")
    _validate_inputs(model, features, valid_mask, static_features, top_k)
    timestamp_rows = _as_timestamp_rows(
        timestamps, features.shape[0], features.shape[1],
    )
    _validate_timestamp_order(timestamp_rows, valid_mask)
    model_hash = _model_sha256(model)
    input_hash = _input_sha256(
        features, valid_mask, static_features, timestamp_rows,
    )
    with _temporary_eval(model), torch.inference_mode():
        reference_prediction = _model_forward(
            model, features, valid_mask, static_features,
        )
        _validate_prediction(reference_prediction)
        with _capture_attention(model) as captured:
            captured_prediction = _model_forward(
                model, features, valid_mask, static_features,
            )
        _validate_prediction(captured_prediction)
        parity = _prediction_parity(reference_prediction, captured_prediction)
        attention, max_future, max_invalid_key = _prepare_attention(
            captured, valid_mask,
        )
        metrics = _attention_metrics(attention, valid_mask)
        occlusion_attention = metrics["last_attention"][:, -1, :, :].mean(dim=1)
        top_indices, control_indices, top_scores = _select_occlusion_indices(
            occlusion_attention, valid_mask, top_k, random_seed,
        )
        baseline, baseline_policy = _neutral_feature_baseline(model, features)
        top_prediction = _model_forward(
            model, _occlude(features, top_indices, baseline), valid_mask,
            static_features,
        )
        control_prediction = _model_forward(
            model, _occlude(features, control_indices, baseline), valid_mask,
            static_features,
        )
        _validate_prediction(top_prediction)
        _validate_prediction(control_prediction)
        reference_target = reference_prediction[target_output]
        top_effect = (reference_target - top_prediction[target_output]).abs()
        control_effect = (reference_target - control_prediction[target_output]).abs()
    mask_rows = valid_mask.detach().cpu().to(dtype=torch.bool).tolist()
    last_indices = metrics["last_indices"].detach().cpu().tolist()
    cutoff_timestamps = [
        timestamp_rows[sample_index][last_index]
        for sample_index, last_index in enumerate(last_indices)
    ]
    top_timestamps = [
        [timestamp_rows[sample_index][index] for index in indices]
        for sample_index, indices in enumerate(top_indices)
    ]
    control_timestamps = [
        [timestamp_rows[sample_index][index] for index in indices]
        for sample_index, indices in enumerate(control_indices)
    ]
    top_effect_list = _float_list(top_effect)
    control_effect_list = _float_list(control_effect)
    gap_list = [top - control for top, control in zip(
        top_effect_list, control_effect_list,
    )]
    top_mean = sum(top_effect_list) / len(top_effect_list)
    control_mean = sum(control_effect_list) / len(control_effect_list)
    artifact = {
        "schema_version": "AttentionAnalysisArtifactV1",
        "analysis_only": True,
        "production_forward_modified": False,
        "attention_is_causal_explanation": False,
        "model_sha256": model_hash,
        "input_sha256": input_hash,
        "target_output": target_output,
        "shape": {
            "batch": int(features.shape[0]),
            "time": int(features.shape[1]),
            "features": int(features.shape[2]),
            "layers": int(metrics["layer_count"]),
            "heads": int(metrics["head_count"]),
        },
        "tensor_layout": {
            "per_layer_head_attention": "[layer,batch,head,query_time,key_time]",
            "last_token_attention": "[batch,layer,head,key_time]",
            "rollout": "[batch,query_time,key_time]",
            "last_token_rollout": "[batch,key_time]",
            "head_metrics": "[batch,layer,head]",
        },
        "valid_mask": mask_rows,
        "timestamps": timestamp_rows,
        "last_valid_index": last_indices,
        "cutoff_timestamp": cutoff_timestamps,
        "per_layer_head_attention": _float_list(attention),
        "last_token_attention": _float_list(metrics["last_attention"]),
        "rollout": _float_list(metrics["rollout"]),
        "last_token_rollout": _float_list(metrics["last_rollout"]),
        "head_entropy_nats": _float_list(metrics["entropy"]),
        "head_normalized_entropy": _float_list(metrics["normalized_entropy"]),
        "head_mean_span": _float_list(metrics["span"]),
        "head_normalized_span": _float_list(metrics["normalized_span"]),
        "future_guard": {
            "causal_attention_verified": True,
            "max_future_attention": max_future,
            "padding_key_mask_verified": True,
            "max_padding_key_attention": max_invalid_key,
            "timestamps_strictly_increasing_on_valid_rows": True,
            "invalid_feature_values_excluded_from_input_hash": True,
        },
        "capture_parity": parity,
        "occlusion": {
            "top_k": top_k,
            "random_seed": random_seed,
            "selection_source": "last_layer_last_token_head_mean_attention",
            "random_control_disjoint_from_top": True,
            "baseline_policy": baseline_policy,
            "top_indices": top_indices,
            "random_control_indices": control_indices,
            "top_timestamps": top_timestamps,
            "random_control_timestamps": control_timestamps,
            "top_attention_scores": top_scores,
            "target_before_occlusion": _float_list(reference_target),
            "top_absolute_effect": top_effect_list,
            "random_control_absolute_effect": control_effect_list,
            "faithfulness_gap": gap_list,
            "mean_top_absolute_effect": top_mean,
            "mean_random_control_absolute_effect": control_mean,
            "mean_faithfulness_gap": top_mean - control_mean,
            "top_beats_random_control": top_mean > control_mean,
        },
    }
    artifact["analysis_sha256"] = _canonical_sha256(artifact)
    return artifact


def _validate_attention_artifact(
    artifact: Mapping[str, Any], model, features, valid_mask, static_features,
) -> None:
    if artifact.get("schema_version") != "AttentionAnalysisArtifactV1":
        raise AttentionAnalysisError("归因交叉验证只接受 AttentionAnalysisArtifactV1")
    stored_hash = artifact.get("analysis_sha256")
    if not isinstance(stored_hash, str) or len(stored_hash) != 64:
        raise AttentionAnalysisError("attention artifact 缺少有效 analysis_sha256")
    payload = dict(artifact)
    del payload["analysis_sha256"]
    if _canonical_sha256(payload) != stored_hash:
        raise AttentionAnalysisError("attention artifact hash 校验失败")
    if artifact.get("production_forward_modified") is not False:
        raise AttentionAnalysisError("attention artifact 未声明保持生产 forward")
    if artifact.get("model_sha256") != _model_sha256(model):
        raise AttentionAnalysisError("attention artifact 与当前模型状态不一致")
    timestamp_rows = _as_timestamp_rows(
        artifact.get("timestamps"), features.shape[0], features.shape[1],
    )
    _validate_timestamp_order(timestamp_rows, valid_mask)
    if artifact.get("input_sha256") != _input_sha256(
        features, valid_mask, static_features, timestamp_rows,
    ):
        raise AttentionAnalysisError("attention artifact 与当前输入不一致")
    mask_rows = valid_mask.detach().cpu().to(dtype=torch.bool).tolist()
    if artifact.get("valid_mask") != mask_rows:
        raise AttentionAnalysisError("attention artifact 的 valid_mask 不一致")
    shape = artifact.get("shape", {})
    expected_shape = {
        "batch": int(features.shape[0]),
        "time": int(features.shape[1]),
        "features": int(features.shape[2]),
    }
    if any(shape.get(name) != value for name, value in expected_shape.items()):
        raise AttentionAnalysisError("attention artifact 的输入形状不一致")


def _normalize_feature_groups(
    feature_groups: Mapping[str, Sequence[int]], feature_count: int,
):
    if not isinstance(feature_groups, Mapping) or len(feature_groups) < 2:
        raise AttentionAnalysisError("feature_groups 至少需要两个特征组")
    if any(not isinstance(name, str) or not name for name in feature_groups):
        raise AttentionAnalysisError("特征组名称必须为非空字符串")
    normalized = {}
    occupied = set()
    for name in sorted(feature_groups):
        try:
            raw_indices = list(feature_groups[name])
        except TypeError as error:
            raise AttentionAnalysisError(
                f"特征组 {name} 必须提供索引序列",
            ) from error
        if not raw_indices:
            raise AttentionAnalysisError(f"特征组 {name} 不能为空")
        indices = []
        for index in raw_indices:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise AttentionAnalysisError(f"特征组 {name} 的索引必须为整数")
            index = int(index)
            if index < 0 or index >= feature_count:
                raise AttentionAnalysisError(f"特征组 {name} 含越界索引 {index}")
            indices.append(index)
        if len(set(indices)) != len(indices):
            raise AttentionAnalysisError(f"特征组 {name} 含重复索引")
        overlap = occupied.intersection(indices)
        if overlap:
            raise AttentionAnalysisError(
                f"特征组必须互斥，重复索引为 {sorted(overlap)}",
            )
        occupied.update(indices)
        normalized[name] = tuple(sorted(indices))
    return normalized, sorted(occupied), sorted(set(range(feature_count)) - occupied)


def _integrated_gradients(
    model,
    features,
    valid_mask,
    static_features,
    target_output: str,
    integration_steps: int,
):
    if (
        isinstance(integration_steps, bool)
        or not isinstance(integration_steps, Integral)
        or integration_steps <= 0
    ):
        raise AttentionAnalysisError("integration_steps 必须为正整数")
    if not features.is_floating_point():
        raise AttentionAnalysisError("Integrated Gradients 要求浮点 features")
    if torch.is_inference_mode_enabled():
        raise AttentionAnalysisError(
            "Integrated Gradients 不能在 torch.inference_mode() 内运行",
        )
    baseline_vector, baseline_policy = _neutral_feature_baseline(model, features)
    baseline = baseline_vector.view(1, 1, -1).expand_as(features).detach()
    mask = valid_mask.to(dtype=torch.bool, device=features.device).unsqueeze(-1)
    endpoint = torch.where(mask, features.detach(), baseline)
    delta = endpoint - baseline
    gradient_sum = torch.zeros_like(features)
    with _temporary_eval(model), torch.enable_grad():
        for step_index in range(integration_steps + 1):
            alpha = step_index / integration_steps
            point = (baseline + alpha * delta).detach().requires_grad_(True)
            prediction = _model_forward(model, point, valid_mask, static_features)
            _validate_prediction(prediction)
            gradient = torch.autograd.grad(
                prediction[target_output].sum(), point,
                retain_graph=False, create_graph=False, allow_unused=True,
            )[0]
            if gradient is None:
                gradient = torch.zeros_like(point)
            coefficient = 0.5 if step_index in (0, integration_steps) else 1.0
            gradient_sum = gradient_sum + coefficient * gradient.detach()
        attribution = delta * (gradient_sum / integration_steps)
        with torch.no_grad():
            baseline_prediction = _model_forward(
                model, baseline, valid_mask, static_features,
            )
            endpoint_prediction = _model_forward(
                model, endpoint, valid_mask, static_features,
            )
        _validate_prediction(baseline_prediction)
        _validate_prediction(endpoint_prediction)
    attribution = torch.where(mask, attribution, torch.zeros_like(attribution))
    target_delta = (
        endpoint_prediction[target_output] - baseline_prediction[target_output]
    )
    attribution_sum = attribution.sum(dim=(1, 2))
    completeness_error = (target_delta - attribution_sum).abs()
    completeness_threshold = 1e-4 + 0.02 * target_delta.abs()
    return {
        "attribution": attribution,
        "baseline": baseline_vector,
        "baseline_policy": baseline_policy,
        "baseline_prediction": baseline_prediction,
        "endpoint_prediction": endpoint_prediction,
        "target_delta": target_delta,
        "attribution_sum": attribution_sum,
        "completeness_error": completeness_error,
        "completeness_threshold": completeness_threshold,
        "completeness_passed": completeness_error <= completeness_threshold,
    }


def _feature_group_scores(attribution, feature_groups):
    signed = []
    absolute = []
    for indices in feature_groups.values():
        group = attribution[:, :, list(indices)]
        signed.append(group.sum(dim=(1, 2)))
        absolute.append(group.abs().sum(dim=(1, 2)))
    return torch.stack(signed, dim=1), torch.stack(absolute, dim=1)


def _feature_group_occlusion(
    model,
    features,
    valid_mask,
    static_features,
    target_output: str,
    feature_groups,
    baseline_vector,
):
    with _temporary_eval(model), torch.inference_mode():
        reference = _model_forward(model, features, valid_mask, static_features)
        _validate_prediction(reference)
        after = []
        for indices in feature_groups.values():
            occluded = features.detach().clone()
            selected = list(indices)
            occluded[:, :, selected] = baseline_vector[selected]
            prediction = _model_forward(
                model, occluded, valid_mask, static_features,
            )
            _validate_prediction(prediction)
            after.append(prediction[target_output])
    after_tensor = torch.stack(after, dim=1)
    signed_effect = reference[target_output].unsqueeze(1) - after_tensor
    return reference, after_tensor, signed_effect, signed_effect.abs()


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]):
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    ) / denominator


def _safe_cosine(left: Sequence[float], right: Sequence[float]):
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _profile_cross_validation(
    left: Sequence[float],
    right: Sequence[float],
    candidate_indices: Sequence[int],
    top_k: int,
):
    left_values = [float(left[index]) for index in candidate_indices]
    right_values = [float(right[index]) for index in candidate_indices]
    left_top = sorted(
        candidate_indices, key=lambda index: (-float(left[index]), index),
    )[:top_k]
    right_top = sorted(
        candidate_indices, key=lambda index: (-float(right[index]), index),
    )[:top_k]
    left_set = set(left_top)
    right_set = set(right_top)
    return {
        "cosine": _safe_cosine(left_values, right_values),
        "spearman": _pearson(
            _average_ranks(left_values), _average_ranks(right_values),
        ),
        "top_k_jaccard": len(left_set & right_set) / len(left_set | right_set),
        "left_top_indices": left_top,
        "right_top_indices": right_top,
    }


def _defined_mean(records, key: str):
    values = [record[key] for record in records if record[key] is not None]
    return sum(values) / len(values) if values else None


def cross_validate_attention_attribution(
    model,
    features,
    valid_mask,
    attention_artifact: Mapping[str, Any],
    *,
    feature_groups: Mapping[str, Sequence[int]],
    static_features=None,
    integration_steps: int = 32,
    top_k: int = 1,
) -> dict[str, Any]:
    """Cross-check attention with IG and feature-group occlusion on frozen inputs."""
    _require_torch()
    if top_k <= 0:
        raise AttentionAnalysisError("top_k 必须为正整数")
    _validate_inputs(model, features, valid_mask, static_features, top_k=1)
    _validate_attention_artifact(
        attention_artifact, model, features, valid_mask, static_features,
    )
    target_output = attention_artifact.get("target_output")
    if target_output not in PRODUCTION_OUTPUT_KEYS:
        raise AttentionAnalysisError("attention artifact 的 target_output 无效")
    groups, covered, ungrouped = _normalize_feature_groups(
        feature_groups, features.shape[2],
    )
    if top_k > len(groups):
        raise AttentionAnalysisError("top_k 不能大于特征组数量")
    valid_counts = valid_mask.to(dtype=torch.bool).sum(dim=1)
    if bool((valid_counts < top_k).any()):
        raise AttentionAnalysisError("有效时间点少于 cross-validation top_k")
    model_hash_before = _model_sha256(model)
    integrated = _integrated_gradients(
        model,
        features,
        valid_mask,
        static_features,
        target_output,
        integration_steps,
    )
    group_ig_signed, group_ig_absolute = _feature_group_scores(
        integrated["attribution"], groups,
    )
    reference, group_after, group_effect_signed, group_effect_absolute = (
        _feature_group_occlusion(
            model,
            features,
            valid_mask,
            static_features,
            target_output,
            groups,
            integrated["baseline"],
        )
    )
    masked_endpoint_parity = _prediction_parity(
        integrated["endpoint_prediction"], reference,
    )
    with _temporary_eval(model), torch.inference_mode():
        after_analysis = _model_forward(
            model, features, valid_mask, static_features,
        )
        _validate_prediction(after_analysis)
    production_parity = _prediction_parity(reference, after_analysis)
    model_hash_after = _model_sha256(model)
    if model_hash_before != model_hash_after:
        raise AttentionAnalysisError("归因分析改变了模型 state_dict")
    attribution = integrated["attribution"]
    temporal_ig_absolute = attribution.abs().sum(dim=2)
    attention_profiles = []
    for sample in attention_artifact["last_token_attention"]:
        final_layer = sample[-1]
        if not final_layer:
            raise AttentionAnalysisError("attention artifact 缺少末层 head")
        attention_profiles.append([
            sum(float(head[index]) for head in final_layer) / len(final_layer)
            for index in range(features.shape[1])
        ])
    temporal_ig_rows = _float_list(temporal_ig_absolute)
    group_ig_rows = _float_list(group_ig_absolute)
    group_effect_rows = _float_list(group_effect_absolute)
    temporal_cross = []
    group_cross = []
    mask_rows = valid_mask.detach().cpu().to(dtype=torch.bool).tolist()
    for sample_index, mask_row in enumerate(mask_rows):
        valid_indices = [index for index, keep in enumerate(mask_row) if keep]
        temporal_record = _profile_cross_validation(
            attention_profiles[sample_index],
            temporal_ig_rows[sample_index],
            valid_indices,
            top_k,
        )
        temporal_record["sample_index"] = sample_index
        temporal_cross.append(temporal_record)
        group_record = _profile_cross_validation(
            group_ig_rows[sample_index],
            group_effect_rows[sample_index],
            list(range(len(groups))),
            top_k,
        )
        group_record["sample_index"] = sample_index
        group_record["left_top_groups"] = [
            list(groups)[index] for index in group_record["left_top_indices"]
        ]
        group_record["right_top_groups"] = [
            list(groups)[index] for index in group_record["right_top_indices"]
        ]
        group_cross.append(group_record)
    invalid = (~valid_mask.to(dtype=torch.bool)).unsqueeze(-1).expand_as(attribution)
    invalid_values = attribution.masked_select(invalid)
    max_invalid_attribution = (
        float(invalid_values.abs().max().item()) if invalid_values.numel() else 0.0
    )
    report = {
        "schema_version": "AttentionAttributionCrossValidationV1",
        "analysis_only": True,
        "production_forward_modified": False,
        "causal_attribution_claim": False,
        "attention_analysis_sha256": attention_artifact["analysis_sha256"],
        "model_sha256": model_hash_after,
        "input_sha256": attention_artifact["input_sha256"],
        "target_output": target_output,
        "integration_method": "straight_line_trapezoidal_integrated_gradients",
        "integration_steps": integration_steps,
        "baseline_policy": integrated["baseline_policy"],
        "feature_groups": {
            name: list(indices) for name, indices in groups.items()
        },
        "feature_group_order": list(groups),
        "covered_feature_indices": covered,
        "ungrouped_feature_indices": ungrouped,
        "integrated_gradients": {
            "tensor_layout": "[batch,time,feature]",
            "attribution": _float_list(attribution),
            "temporal_absolute_score": temporal_ig_rows,
            "feature_group_signed_score": _float_list(group_ig_signed),
            "feature_group_absolute_score": group_ig_rows,
            "target_at_baseline": _float_list(
                integrated["baseline_prediction"][target_output],
            ),
            "target_at_input": _float_list(
                integrated["endpoint_prediction"][target_output],
            ),
            "target_delta": _float_list(integrated["target_delta"]),
            "attribution_sum": _float_list(integrated["attribution_sum"]),
            "completeness_absolute_error": _float_list(
                integrated["completeness_error"],
            ),
            "completeness_threshold": _float_list(
                integrated["completeness_threshold"],
            ),
            "completeness_passed": integrated[
                "completeness_passed"
            ].detach().cpu().tolist(),
        },
        "feature_group_occlusion": {
            "tensor_layout": "[batch,feature_group]",
            "target_before_occlusion": _float_list(reference[target_output]),
            "target_after_occlusion": _float_list(group_after),
            "signed_effect": _float_list(group_effect_signed),
            "absolute_effect": group_effect_rows,
        },
        "cross_validation": {
            "top_k": top_k,
            "attention_vs_integrated_gradients_temporal": temporal_cross,
            "integrated_gradients_vs_feature_group_occlusion": group_cross,
            "summary": {
                "attention_ig_cosine_mean": _defined_mean(temporal_cross, "cosine"),
                "attention_ig_spearman_mean": _defined_mean(temporal_cross, "spearman"),
                "attention_ig_top_k_jaccard_mean": _defined_mean(
                    temporal_cross, "top_k_jaccard",
                ),
                "group_ig_occlusion_cosine_mean": _defined_mean(
                    group_cross, "cosine",
                ),
                "group_ig_occlusion_spearman_mean": _defined_mean(
                    group_cross, "spearman",
                ),
                "group_ig_occlusion_top_k_jaccard_mean": _defined_mean(
                    group_cross, "top_k_jaccard",
                ),
            },
            "promotion_threshold_pre_registered": False,
        },
        "future_guard": {
            "invalid_time_attribution_zero": max_invalid_attribution == 0.0,
            "max_invalid_time_attribution": max_invalid_attribution,
            "masked_endpoint_parity": masked_endpoint_parity,
            "attention_artifact_hash_verified": True,
            "attention_input_hash_verified": True,
        },
        "production_parity": production_parity,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise AttentionAnalysisError("rollout profile 的范数必须为正")
    return numerator / (left_norm * right_norm)


def attention_stability_report(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str] | None = None,
    top_k: int = 1,
) -> dict[str, Any]:
    """Compare seed, window, or regime attention profiles without refitting."""
    if len(artifacts) < 2:
        raise AttentionAnalysisError("稳定性报告至少需要两个 attention artifact")
    if top_k <= 0:
        raise AttentionAnalysisError("top_k 必须为正整数")
    if labels is None:
        labels = [f"artifact-{index}" for index in range(len(artifacts))]
    else:
        labels = list(labels)
    if len(labels) != len(artifacts) or len(set(labels)) != len(labels):
        raise AttentionAnalysisError("labels 必须与 artifacts 一一对应且唯一")
    profiles = []
    masks = []
    for artifact in artifacts:
        if artifact.get("schema_version") != "AttentionAnalysisArtifactV1":
            raise AttentionAnalysisError("稳定性报告只接受 AttentionAnalysisArtifactV1")
        profiles.append(artifact["last_token_rollout"])
        masks.append(artifact["valid_mask"])
    reference_shape = (
        len(profiles[0]), len(profiles[0][0]) if profiles[0] else 0,
    )
    for profile, mask in zip(profiles, masks):
        shape = (len(profile), len(profile[0]) if profile else 0)
        if shape != reference_shape or any(len(row) != shape[1] for row in profile):
            raise AttentionAnalysisError("用于稳定性比较的 rollout 形状必须一致")
        if len(mask) != shape[0] or any(len(row) != shape[1] for row in mask):
            raise AttentionAnalysisError("用于稳定性比较的 valid_mask 形状不一致")
    pairwise = []
    for left_index in range(len(artifacts)):
        for right_index in range(left_index + 1, len(artifacts)):
            if masks[left_index] != masks[right_index]:
                raise AttentionAnalysisError("稳定性比较要求相同 valid_mask")
            cosines = []
            overlaps = []
            for sample_index, mask_row in enumerate(masks[left_index]):
                valid_indices = [index for index, keep in enumerate(mask_row) if keep]
                if len(valid_indices) < top_k:
                    raise AttentionAnalysisError("有效时间点少于 stability top_k")
                left_profile = profiles[left_index][sample_index]
                right_profile = profiles[right_index][sample_index]
                cosines.append(_cosine(left_profile, right_profile))
                left_top = set(sorted(
                    valid_indices, key=lambda index: (-left_profile[index], index),
                )[:top_k])
                right_top = set(sorted(
                    valid_indices, key=lambda index: (-right_profile[index], index),
                )[:top_k])
                overlaps.append(len(left_top & right_top) / len(left_top | right_top))
            pairwise.append({
                "left": labels[left_index],
                "right": labels[right_index],
                "cosine_mean": sum(cosines) / len(cosines),
                "cosine_min": min(cosines),
                "top_k_jaccard_mean": sum(overlaps) / len(overlaps),
            })
    report = {
        "schema_version": "AttentionStabilityReportV1",
        "labels": labels,
        "top_k": top_k,
        "artifact_sha256": [artifact["analysis_sha256"] for artifact in artifacts],
        "pairwise": pairwise,
        "interpretation_scope": "seed/window/regime profile stability; not causal attribution",
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report

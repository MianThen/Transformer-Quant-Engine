from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from python.qbt_ml.analysis.attention import (
    AttentionAnalysisError,
    PRODUCTION_OUTPUT_KEYS,
    analyze_temporal_attention,
    attention_stability_report,
    cross_validate_attention_attribution,
)
try:
    from work.phase2b_feature_pgd_v2_r3_source.python.qbt_ml.models.temporal_transformer import (
        TemporalTransformerConfig,
        TemporalTransformerV1,
    )
except ModuleNotFoundError:
    from python.qbt_ml.models.temporal_transformer import (
        TemporalTransformerConfig,
        TemporalTransformerV1,
    )


def _model(*, lookback=5, layers=2, dropout=0.0):
    torch.manual_seed(17)
    return TemporalTransformerV1(TemporalTransformerConfig(
        feature_count=3,
        lookback=lookback,
        d_model=8,
        nhead=2,
        num_layers=layers,
        dim_feedforward=16,
        dropout=dropout,
    ))


class _ToyAttentionLayer(nn.Module):
    def __init__(self, embed_dim=1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, dropout=0.0, bias=False,
            batch_first=True,
        )
        with torch.no_grad():
            if embed_dim == 1:
                self.self_attn.in_proj_weight.fill_(1.0)
                self.self_attn.out_proj.weight.fill_(1.0)
            else:
                self.self_attn.in_proj_weight.zero_()
                self.self_attn.out_proj.weight.zero_()

    def forward(self, values, *, mask, key_padding_mask):
        return self.self_attn(
            values, values, values,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]


class _ToyEncoder(nn.Module):
    def __init__(self, embed_dim=1):
        super().__init__()
        self.layers = nn.ModuleList([_ToyAttentionLayer(embed_dim)])

    def forward(self, values, *, mask, key_padding_mask):
        for layer in self.layers:
            values = layer(values, mask=mask, key_padding_mask=key_padding_mask)
        return values


class _ToyTemporalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _ToyEncoder()

    def forward(self, features, valid_mask):
        mask = valid_mask.to(dtype=torch.bool)
        causal = torch.triu(torch.ones(
            features.shape[1], features.shape[1], dtype=torch.bool,
            device=features.device,
        ), diagonal=1)
        encoded = self.encoder(
            features, mask=causal, key_padding_mask=~mask,
        )
        positions = torch.arange(mask.shape[1], device=features.device).expand_as(mask)
        last_index = positions.masked_fill(~mask, -1).max(dim=1).values
        value = encoded[
            torch.arange(encoded.shape[0], device=encoded.device), last_index, 0,
        ]
        return {
            "expected_return": value,
            "expected_volatility": value.abs() + 0.1,
            "direction_probability": torch.sigmoid(value),
            "lower_quantile": value - 1.0,
            "upper_quantile": value + 1.0,
            "confidence": torch.sigmoid(value.abs()),
        }


class _AttributionProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _ToyEncoder(embed_dim=3)

    def forward(self, features, valid_mask):
        mask = valid_mask.to(dtype=torch.bool)
        causal = torch.triu(torch.ones(
            features.shape[1], features.shape[1], dtype=torch.bool,
            device=features.device,
        ), diagonal=1)
        self.encoder(features, mask=causal, key_padding_mask=~mask)
        valid_features = torch.where(
            mask.unsqueeze(-1), features, torch.zeros_like(features),
        )
        value = (
            10.0 * valid_features[:, :, 0]
            + 2.0 * valid_features[:, :, 1]
            + 0.5 * valid_features[:, :, 2]
        ).sum(dim=1)
        return {
            "expected_return": value,
            "expected_volatility": value.abs() + 0.1,
            "direction_probability": torch.sigmoid(value),
            "lower_quantile": value - 1.0,
            "upper_quantile": value + 1.0,
            "confidence": torch.sigmoid(value.abs()),
        }


def test_analysis_off_preserves_forward_six_outputs_and_module_state():
    model = _model(dropout=0.2).eval()
    features = torch.randn(2, 5, 3)
    valid_mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
    ], dtype=torch.uint8)
    forward_attributes = ["forward" in layer.self_attn.__dict__ for layer in model.encoder.layers]
    with torch.inference_mode():
        before = {key: value.clone() for key, value in model(features, valid_mask).items()}
    artifact = analyze_temporal_attention(
        model,
        features,
        valid_mask,
        timestamps=[20260101, 20260102, 20260103, 20260104, 20260105],
    )
    with torch.inference_mode():
        after = model(features, valid_mask)
    assert tuple(before) == PRODUCTION_OUTPUT_KEYS
    assert tuple(after) == PRODUCTION_OUTPUT_KEYS
    for key in PRODUCTION_OUTPUT_KEYS:
        torch.testing.assert_close(before[key], after[key], rtol=0.0, atol=0.0)
    assert not model.training
    assert forward_attributes == [
        "forward" in layer.self_attn.__dict__ for layer in model.encoder.layers
    ]
    assert artifact["production_forward_modified"] is False
    assert artifact["capture_parity"]["passed"] is True
    assert artifact["shape"] == {
        "batch": 2, "time": 5, "features": 3, "layers": 2, "heads": 2,
    }
    assert artifact["future_guard"]["causal_attention_verified"] is True
    assert artifact["future_guard"]["max_future_attention"] == 0.0
    assert len(artifact["per_layer_head_attention"]) == 2
    assert len(artifact["last_token_attention"][0]) == 2
    assert len(artifact["last_token_attention"][0][0]) == 2


def test_attention_is_deterministic_and_future_padding_is_ignored():
    model = _model(lookback=5, layers=1).eval()
    features = torch.randn(1, 5, 3)
    valid_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.uint8)
    kwargs = {
        "timestamps": [1, 2, 3, 4, 999],
        "top_k": 1,
        "random_seed": 31,
    }
    first = analyze_temporal_attention(model, features, valid_mask, **kwargs)
    second = analyze_temporal_attention(model, features, valid_mask, **kwargs)
    mutated = features.clone()
    mutated[:, 4, :] = torch.tensor([1e8, -1e8, 7e7])
    future_mutated = analyze_temporal_attention(
        model, mutated, valid_mask, **kwargs,
    )
    assert first == second
    assert first["analysis_sha256"] == future_mutated["analysis_sha256"]
    assert first["input_sha256"] == future_mutated["input_sha256"]
    assert first["last_token_rollout"] == future_mutated["last_token_rollout"]
    assert first["occlusion"] == future_mutated["occlusion"]


def test_top_attention_occlusion_beats_disjoint_random_control():
    model = _ToyTemporalModel().eval()
    features = torch.tensor([[[10.0], [0.0], [0.0], [0.1]]])
    valid_mask = torch.ones((1, 4), dtype=torch.uint8)
    artifact = analyze_temporal_attention(
        model,
        features,
        valid_mask,
        timestamps=["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        top_k=1,
        random_seed=7,
    )
    occlusion = artifact["occlusion"]
    assert occlusion["top_indices"] == [[0]]
    assert set(occlusion["top_indices"][0]).isdisjoint(
        occlusion["random_control_indices"][0],
    )
    assert occlusion["mean_top_absolute_effect"] > occlusion[
        "mean_random_control_absolute_effect"
    ]
    assert occlusion["mean_faithfulness_gap"] > 0.0
    assert occlusion["top_beats_random_control"] is True


def test_stability_report_is_hashed_and_identical_profiles_score_one():
    model = _model(lookback=4, layers=1).eval()
    features = torch.randn(1, 4, 3)
    valid_mask = torch.ones((1, 4), dtype=torch.uint8)
    artifact = analyze_temporal_attention(model, features, valid_mask)
    copied = copy.deepcopy(artifact)
    report = attention_stability_report(
        [artifact, copied], labels=["seed-17", "seed-17-repeat"], top_k=2,
    )
    assert report == attention_stability_report(
        [artifact, copied], labels=["seed-17", "seed-17-repeat"], top_k=2,
    )
    assert len(report["report_sha256"]) == 64
    assert report["pairwise"] == [{
        "left": "seed-17",
        "right": "seed-17-repeat",
        "cosine_mean": pytest.approx(1.0),
        "cosine_min": pytest.approx(1.0),
        "top_k_jaccard_mean": 1.0,
    }]


def test_timestamp_and_control_future_guards_fail_closed():
    model = _model(lookback=4, layers=1).eval()
    features = torch.randn(1, 4, 3)
    valid_mask = torch.ones((1, 4), dtype=torch.uint8)
    with pytest.raises(AttentionAnalysisError, match="严格递增"):
        analyze_temporal_attention(
            model, features, valid_mask, timestamps=[1, 3, 2, 4],
        )
    with pytest.raises(AttentionAnalysisError, match="互斥"):
        analyze_temporal_attention(
            model, features, valid_mask, top_k=3,
        )


def test_integrated_gradients_and_feature_group_occlusion_cross_validate():
    model = _AttributionProbeModel().eval()
    features = torch.tensor([[
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]])
    valid_mask = torch.ones((1, 4), dtype=torch.uint8)
    attention = analyze_temporal_attention(
        model, features, valid_mask, timestamps=[1, 2, 3, 4],
    )
    first = cross_validate_attention_attribution(
        model,
        features,
        valid_mask,
        attention,
        feature_groups={"secondary": [1, 2], "dominant": [0]},
        integration_steps=8,
        top_k=1,
    )
    second = cross_validate_attention_attribution(
        model,
        features,
        valid_mask,
        attention,
        feature_groups={"dominant": [0], "secondary": [1, 2]},
        integration_steps=8,
        top_k=1,
    )
    assert first == second
    assert first["schema_version"] == "AttentionAttributionCrossValidationV1"
    assert len(first["report_sha256"]) == 64
    assert first["feature_group_order"] == ["dominant", "secondary"]
    integrated = first["integrated_gradients"]
    assert integrated["completeness_passed"] == [True]
    assert integrated["completeness_absolute_error"][0] == pytest.approx(0.0)
    group_cross = first["cross_validation"][
        "integrated_gradients_vs_feature_group_occlusion"
    ][0]
    assert group_cross["left_top_groups"] == ["dominant"]
    assert group_cross["right_top_groups"] == ["dominant"]
    assert group_cross["cosine"] == pytest.approx(1.0)
    assert group_cross["spearman"] == pytest.approx(1.0)
    assert group_cross["top_k_jaccard"] == 1.0
    assert first["production_parity"]["passed"] is True


def test_attribution_analysis_preserves_model_state_gradients_and_forward():
    model = _model(lookback=4, layers=1, dropout=0.2).eval()
    features = torch.randn(1, 4, 3, requires_grad=True)
    valid_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.uint8)
    parameter = next(model.parameters())
    parameter.grad = torch.full_like(parameter, 7.0)
    saved_gradient = parameter.grad.clone()
    saved_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    with torch.inference_mode():
        before = {key: value.clone() for key, value in model(features, valid_mask).items()}
    attention = analyze_temporal_attention(
        model, features, valid_mask, timestamps=[1, 2, 3, 999],
    )
    report = cross_validate_attention_attribution(
        model,
        features,
        valid_mask,
        attention,
        feature_groups={"feature-0": [0], "feature-1-2": [1, 2]},
        integration_steps=4,
    )
    with torch.inference_mode():
        after = model(features, valid_mask)
    for key in PRODUCTION_OUTPUT_KEYS:
        torch.testing.assert_close(before[key], after[key], rtol=0.0, atol=0.0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(saved_state[name], value, rtol=0.0, atol=0.0)
    torch.testing.assert_close(parameter.grad, saved_gradient, rtol=0.0, atol=0.0)
    assert features.grad is None
    assert not model.training
    assert report["future_guard"]["invalid_time_attribution_zero"] is True
    assert report["future_guard"]["max_invalid_time_attribution"] == 0.0
    assert report["future_guard"]["masked_endpoint_parity"]["passed"] is True


def test_attribution_cross_validation_rejects_tampering_and_group_overlap():
    model = _AttributionProbeModel().eval()
    features = torch.ones((1, 4, 3))
    valid_mask = torch.ones((1, 4), dtype=torch.uint8)
    attention = analyze_temporal_attention(model, features, valid_mask)
    tampered = copy.deepcopy(attention)
    tampered["last_token_attention"][0][0][0][0] += 0.01
    with pytest.raises(AttentionAnalysisError, match="hash 校验失败"):
        cross_validate_attention_attribution(
            model,
            features,
            valid_mask,
            tampered,
            feature_groups={"first": [0], "rest": [1, 2]},
            integration_steps=2,
        )
    with pytest.raises(AttentionAnalysisError, match="互斥"):
        cross_validate_attention_attribution(
            model,
            features,
            valid_mask,
            attention,
            feature_groups={"first": [0, 1], "overlap": [1, 2]},
            integration_steps=2,
        )

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # 现有回测环境不需要安装训练依赖。
    torch = None
    nn = None


@dataclass(frozen=True)
class TemporalTransformerConfig:
    feature_count: int
    lookback: int = 64
    static_feature_count: int = 0
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
    normalization_clip: float = 8.0


if nn is not None:
    class TemporalTransformerV1(nn.Module):
        def __init__(self, config: TemporalTransformerConfig) -> None:
            super().__init__()
            if config.feature_count <= 0 or config.lookback <= 0:
                raise ValueError("feature_count/lookback 必须为正数")
            if config.d_model % config.nhead:
                raise ValueError("d_model 必须能被 nhead 整除")
            if config.normalization_clip <= 0:
                raise ValueError("normalization_clip 必须为正数")
            self.config = config
            self.register_buffer("normalization_mean", torch.zeros(config.feature_count))
            self.register_buffer("normalization_scale", torch.ones(config.feature_count))
            self.register_buffer("direction_calibration_intercept", torch.zeros(1))
            self.register_buffer("direction_calibration_slope", torch.ones(1))
            self.register_buffer(
                "causal_mask",
                torch.triu(torch.ones(
                    config.lookback, config.lookback, dtype=torch.bool
                ), diagonal=1),
                persistent=False,
            )
            self.register_buffer(
                "position_identity",
                torch.eye(config.lookback, dtype=torch.bool),
                persistent=False,
            )
            self.feature_projection = nn.Linear(config.feature_count, config.d_model)
            self.position = nn.Parameter(torch.zeros(1, config.lookback, config.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model, nhead=config.nhead,
                dim_feedforward=config.dim_feedforward, dropout=config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, config.num_layers)
            input_size = config.d_model + config.static_feature_count
            self.shared = nn.Sequential(nn.LayerNorm(input_size), nn.Linear(input_size, config.d_model), nn.GELU())
            self.return_head = nn.Linear(config.d_model, 1)
            self.volatility_head = nn.Linear(config.d_model, 1)
            self.direction_head = nn.Linear(config.d_model, 1)
            self.quantile_head = nn.Linear(config.d_model, 2)

        def set_normalization(self, mean, scale) -> None:
            mean = torch.as_tensor(mean, dtype=self.normalization_mean.dtype)
            scale = torch.as_tensor(scale, dtype=self.normalization_scale.dtype)
            if mean.shape != self.normalization_mean.shape or scale.shape != self.normalization_scale.shape:
                raise ValueError("normalizer shape 与 feature_count 不一致")
            if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
                raise ValueError("normalizer 必须为有限数值")
            if (scale <= 0).any():
                raise ValueError("normalizer scale 必须为正数")
            self.normalization_mean.copy_(mean)
            self.normalization_scale.copy_(scale)

        def set_direction_calibration(self, intercept, slope) -> None:
            values = torch.as_tensor(
                [intercept, slope], dtype=self.direction_calibration_intercept.dtype
            )
            if not torch.isfinite(values).all():
                raise ValueError("方向概率校准参数必须为有限数值")
            self.direction_calibration_intercept.copy_(values[0:1])
            self.direction_calibration_slope.copy_(values[1:2])

        def forward(self, features, valid_mask, static_features=None):
            mask = valid_mask.to(dtype=torch.bool)
            safe_mask = mask.clone()
            empty_rows = ~safe_mask.any(dim=1)
            safe_mask[empty_rows, -1] = True
            normalized = (features - self.normalization_mean) / self.normalization_scale
            normalized = torch.clamp(
                normalized, -self.config.normalization_clip, self.config.normalization_clip
            )
            masked_features = normalized * mask.unsqueeze(-1).to(dtype=features.dtype)
            encoded = self.feature_projection(masked_features) + self.position[:, :features.shape[1]]
            length = features.shape[1]
            causal = self.causal_mask[:length, :length].unsqueeze(0)
            invalid_keys = (~mask).unsqueeze(1).expand(-1, length, -1)
            invalid_query_self = (
                (~mask).unsqueeze(2)
                & self.position_identity[:length, :length].unsqueeze(0)
            )
            attention_mask = (causal | invalid_keys) & ~invalid_query_self
            attention_mask = attention_mask.repeat_interleave(
                self.config.nhead, dim=0
            )
            encoded = self.encoder(encoded, mask=attention_mask)
            positions = torch.arange(mask.shape[1], device=features.device).expand_as(mask)
            last_index = positions.masked_fill(~safe_mask, -1).max(dim=1).values.clamp(min=0)
            pooled = encoded[torch.arange(encoded.shape[0], device=encoded.device), last_index]
            if self.config.static_feature_count:
                if static_features is None:
                    raise ValueError("模型要求 static_features")
                pooled = torch.cat([pooled, static_features], dim=1)
            hidden = self.shared(pooled)
            expected_return = self.return_head(hidden).squeeze(1)
            raw_quantile_delta = self.quantile_head(hidden)
            lower_quantile = expected_return - torch.nn.functional.softplus(
                raw_quantile_delta[:, 0]
            )
            upper_quantile = expected_return + torch.nn.functional.softplus(
                raw_quantile_delta[:, 1]
            )
            direction_logits = self.direction_head(hidden).squeeze(1)
            direction_probability = torch.sigmoid(
                self.direction_calibration_intercept
                + self.direction_calibration_slope * direction_logits
            )
            return {
                "expected_return": expected_return,
                "expected_volatility": torch.nn.functional.softplus(
                    self.volatility_head(hidden).squeeze(1)
                ),
                "direction_probability": direction_probability,
                "lower_quantile": lower_quantile,
                "upper_quantile": upper_quantile,
                "confidence": torch.maximum(direction_probability, 1.0 - direction_probability),
                "direction_logits": direction_logits,
            }
else:
    class TemporalTransformerV1:
        def __init__(self, _config: TemporalTransformerConfig) -> None:
            raise RuntimeError("训练功能未安装；请在训练机安装项目的 ml 可选依赖")

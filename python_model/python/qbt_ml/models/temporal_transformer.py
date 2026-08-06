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
    input_mean: tuple[float, ...] = ()
    input_scale: tuple[float, ...] = ()
    input_protected: tuple[bool, ...] = ()


if nn is not None:
    class TemporalTransformerV1(nn.Module):
        def __init__(self, config: TemporalTransformerConfig) -> None:
            super().__init__()
            if config.feature_count <= 0 or config.lookback <= 0:
                raise ValueError("feature_count/lookback 必须为正数")
            if config.d_model % config.nhead:
                raise ValueError("d_model 必须能被 nhead 整除")
            standardizer_fields = (
                config.input_mean, config.input_scale, config.input_protected,
            )
            if any(standardizer_fields) and not all(standardizer_fields):
                raise ValueError("input standardizer 字段必须同时提供")
            if config.input_mean and len(config.input_mean) != config.feature_count:
                raise ValueError("input_mean 长度必须等于 feature_count")
            if config.input_scale and len(config.input_scale) != config.feature_count:
                raise ValueError("input_scale 长度必须等于 feature_count")
            if config.input_protected and len(config.input_protected) != config.feature_count:
                raise ValueError("input_protected 长度必须等于 feature_count")
            if config.input_scale and any(value <= 0 for value in config.input_scale):
                raise ValueError("input_scale 必须为正数")
            self.config = config
            if config.input_mean:
                mean = torch.tensor(config.input_mean, dtype=torch.float32)
                scale = torch.tensor(config.input_scale, dtype=torch.float32)
                protected = torch.tensor(config.input_protected, dtype=torch.bool)
            else:
                mean = torch.zeros(config.feature_count, dtype=torch.float32)
                scale = torch.ones(config.feature_count, dtype=torch.float32)
                protected = torch.zeros(config.feature_count, dtype=torch.bool)
            self.register_buffer("input_mean", mean.view(1, 1, -1))
            self.register_buffer("input_scale", scale.view(1, 1, -1))
            self.register_buffer("input_protected", protected.view(1, 1, -1))
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
            self.confidence_head = nn.Linear(config.d_model, 1)

        def _heads_from_embedding(self, hidden, *, return_embedding=False):
            quantiles = torch.sort(self.quantile_head(hidden), dim=1).values
            outputs = {
                "expected_return": self.return_head(hidden).squeeze(1),
                "expected_volatility": torch.nn.functional.softplus(
                    self.volatility_head(hidden).squeeze(1)
                ),
                "direction_probability": torch.sigmoid(self.direction_head(hidden).squeeze(1)),
                "lower_quantile": quantiles[:, 0],
                "upper_quantile": quantiles[:, 1],
                "confidence": torch.sigmoid(self.confidence_head(hidden).squeeze(1)),
            }
            if return_embedding:
                outputs["embedding"] = hidden
            return outputs

        def predict_from_embedding(self, embedding, *, return_embedding=False):
            """Apply prediction heads to a shared latent for research-time FGM."""
            if embedding.ndim != 2 or embedding.shape[1] != self.config.d_model:
                raise ValueError("embedding 形状必须为 [batch, d_model]")
            return self._heads_from_embedding(
                embedding, return_embedding=return_embedding,
            )

        @property
        def has_input_standardizer(self):
            return bool(self.config.input_mean)

        def normalize_features(self, features, valid_mask=None):
            if features.ndim != 3 or features.shape[-1] != self.config.feature_count:
                raise ValueError("features 必须为 [N,T,F] 且 F 与模型一致")
            normalized = (features - self.input_mean) / self.input_scale
            normalized = torch.where(
                self.input_protected, features, normalized,
            )
            if valid_mask is not None:
                if valid_mask.shape != features.shape[:2]:
                    raise ValueError("valid_mask 形状与 features 不一致")
                normalized = torch.where(
                    valid_mask.to(dtype=torch.bool).unsqueeze(-1),
                    normalized, torch.zeros_like(normalized),
                )
            return normalized

        def forward_standardized(self, features, valid_mask, static_features=None,
                                 return_embedding=False):
            mask = valid_mask.to(dtype=torch.bool)
            safe_mask = mask.clone()
            empty_rows = ~safe_mask.any(dim=1)
            safe_mask[empty_rows, -1] = True
            encoded = self.feature_projection(features) + self.position[:, :features.shape[1]]
            causal = torch.triu(torch.ones(
                features.shape[1], features.shape[1], dtype=torch.bool,
                device=features.device,
            ), diagonal=1)
            encoded = self.encoder(encoded, mask=causal, src_key_padding_mask=~safe_mask)
            positions = torch.arange(mask.shape[1], device=features.device).expand_as(mask)
            last_index = positions.masked_fill(~safe_mask, -1).max(dim=1).values.clamp(min=0)
            pooled = encoded[torch.arange(encoded.shape[0], device=encoded.device), last_index]
            if self.config.static_feature_count:
                if static_features is None:
                    raise ValueError("模型要求 static_features")
                pooled = torch.cat([pooled, static_features], dim=1)
            hidden = self.shared(pooled)
            return self._heads_from_embedding(
                hidden, return_embedding=return_embedding,
            )

        def forward(self, features, valid_mask, static_features=None,
                    return_embedding=False):
            return self.forward_standardized(
                self.normalize_features(features, valid_mask), valid_mask,
                static_features, return_embedding=return_embedding,
            )
else:
    class TemporalTransformerV1:
        def __init__(self, _config: TemporalTransformerConfig) -> None:
            raise RuntimeError("训练功能未安装；请在训练机安装项目的 ml 可选依赖")


def load_temporal_transformer_state_dict(model, state_dict):
    """Load old checkpoints while requiring every non-scaler key to match."""
    if torch is None:
        raise RuntimeError("训练功能未安装；请在训练机安装项目的 ml 可选依赖")
    state = dict(state_dict)
    for name in ("input_mean", "input_scale", "input_protected"):
        if name not in state:
            state[name] = getattr(model, name).detach().clone()
    model.load_state_dict(state, strict=True)
    return model


__all__ = [
    "TemporalTransformerConfig", "TemporalTransformerV1",
    "load_temporal_transformer_state_dict",
]

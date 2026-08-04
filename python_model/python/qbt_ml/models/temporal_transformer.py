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


if nn is not None:
    class TemporalTransformerV1(nn.Module):
        def __init__(self, config: TemporalTransformerConfig) -> None:
            super().__init__()
            if config.feature_count <= 0 or config.lookback <= 0:
                raise ValueError("feature_count/lookback 必须为正数")
            if config.d_model % config.nhead:
                raise ValueError("d_model 必须能被 nhead 整除")
            self.config = config
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

        def forward(self, features, valid_mask, static_features=None,
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
else:
    class TemporalTransformerV1:
        def __init__(self, _config: TemporalTransformerConfig) -> None:
            raise RuntimeError("训练功能未安装；请在训练机安装项目的 ml 可选依赖")

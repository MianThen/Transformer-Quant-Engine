from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # 普通回测环境不加载深度学习依赖。
    torch = None
    nn = None


@dataclass(frozen=True)
class DeepBaselineConfig:
    feature_count: int
    lookback: int
    hidden_size: int = 64
    num_layers: int = 2
    kernel_size: int = 3
    dropout: float = 0.1
    normalization_clip: float = 8.0


if nn is not None:
    class _MultiTaskBaseline(nn.Module):
        def __init__(self, config: DeepBaselineConfig) -> None:
            super().__init__()
            if config.feature_count <= 0 or config.lookback <= 0:
                raise ValueError("feature_count/lookback 必须为正数")
            if config.hidden_size <= 0 or config.num_layers <= 0:
                raise ValueError("hidden_size/num_layers 必须为正数")
            if not 0 <= config.dropout < 1 or config.normalization_clip <= 0:
                raise ValueError("dropout/normalization_clip 无效")
            self.config = config
            self.register_buffer("normalization_mean", torch.zeros(config.feature_count))
            self.register_buffer("normalization_scale", torch.ones(config.feature_count))
            self.register_buffer("direction_calibration_intercept", torch.zeros(1))
            self.register_buffer("direction_calibration_slope", torch.ones(1))
            self.shared = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
            )
            self.return_head = nn.Linear(config.hidden_size, 1)
            self.volatility_head = nn.Linear(config.hidden_size, 1)
            self.direction_head = nn.Linear(config.hidden_size, 1)
            self.quantile_head = nn.Linear(config.hidden_size, 2)

        def set_normalization(self, mean, scale) -> None:
            mean = torch.as_tensor(mean, dtype=self.normalization_mean.dtype)
            scale = torch.as_tensor(scale, dtype=self.normalization_scale.dtype)
            if mean.shape != self.normalization_mean.shape or scale.shape != self.normalization_scale.shape:
                raise ValueError("normalizer shape 与 feature_count 不一致")
            if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("normalizer 必须为有限数值且 scale 为正")
            self.normalization_mean.copy_(mean)
            self.normalization_scale.copy_(scale)

        def set_direction_calibration(self, intercept, slope) -> None:
            values = torch.as_tensor(
                [intercept, slope], dtype=self.direction_calibration_intercept.dtype
            )
            if not torch.isfinite(values).all():
                raise ValueError("方向概率校准参数必须为有限数值")
            self.direction_calibration_intercept.copy_(values[:1])
            self.direction_calibration_slope.copy_(values[1:])

        def _inputs(self, features, valid_mask):
            mask = valid_mask.to(dtype=torch.bool)
            normalized = (features - self.normalization_mean) / self.normalization_scale
            normalized = torch.clamp(
                normalized, -self.config.normalization_clip,
                self.config.normalization_clip,
            )
            return normalized * mask.unsqueeze(-1).to(features.dtype), mask

        def _predictions(self, encoded):
            hidden = self.shared(encoded)
            expected_return = self.return_head(hidden).squeeze(1)
            deltas = self.quantile_head(hidden)
            direction_logits = self.direction_head(hidden).squeeze(1)
            probability = torch.sigmoid(
                self.direction_calibration_intercept
                + self.direction_calibration_slope * direction_logits
            )
            return {
                "expected_return": expected_return,
                "expected_volatility": torch.nn.functional.softplus(
                    self.volatility_head(hidden).squeeze(1)
                ),
                "direction_probability": probability,
                "lower_quantile": expected_return - torch.nn.functional.softplus(deltas[:, 0]),
                "upper_quantile": expected_return + torch.nn.functional.softplus(deltas[:, 1]),
                "confidence": torch.maximum(probability, 1.0 - probability),
                "direction_logits": direction_logits,
            }


    class MLPSequenceBaseline(_MultiTaskBaseline):
        def __init__(self, config: DeepBaselineConfig) -> None:
            super().__init__(config)
            layers = [
                nn.Flatten(),
                nn.Linear(config.lookback * config.feature_count, config.hidden_size),
                nn.GELU(), nn.Dropout(config.dropout),
            ]
            for _ in range(config.num_layers - 1):
                layers.extend([
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.GELU(), nn.Dropout(config.dropout),
                ])
            self.encoder = nn.Sequential(*layers)

        def forward(self, features, valid_mask, static_features=None):
            del static_features
            values, _ = self._inputs(features, valid_mask)
            return self._predictions(self.encoder(values))


    class CausalTCNBaseline(_MultiTaskBaseline):
        def __init__(self, config: DeepBaselineConfig) -> None:
            super().__init__(config)
            if config.kernel_size <= 1:
                raise ValueError("TCN kernel_size 必须大于 1")
            layers = []
            channels = config.feature_count
            for index in range(config.num_layers):
                dilation = 2 ** index
                padding = (config.kernel_size - 1) * dilation
                layers.extend([
                    nn.Conv1d(channels, config.hidden_size, config.kernel_size,
                              dilation=dilation, padding=padding),
                    nn.GELU(), nn.Dropout(config.dropout),
                ])
                channels = config.hidden_size
            self.layers = nn.ModuleList(layers)

        def forward(self, features, valid_mask, static_features=None):
            del static_features
            values, mask = self._inputs(features, valid_mask)
            encoded = values.transpose(1, 2)
            layer_index = 0
            for _ in range(self.config.num_layers):
                convolution = self.layers[layer_index]
                encoded = convolution(encoded)[..., :values.shape[1]]
                encoded = self.layers[layer_index + 1](encoded)
                encoded = self.layers[layer_index + 2](encoded)
                layer_index += 3
            positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
            last = positions.masked_fill(~mask, -1).max(dim=1).values.clamp(min=0)
            pooled = encoded.transpose(1, 2)[torch.arange(encoded.shape[0], device=mask.device), last]
            return self._predictions(pooled)


    class GRUSequenceBaseline(_MultiTaskBaseline):
        def __init__(self, config: DeepBaselineConfig) -> None:
            super().__init__(config)
            self.encoder = nn.GRU(
                config.feature_count, config.hidden_size,
                num_layers=config.num_layers, batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )

        def forward(self, features, valid_mask, static_features=None):
            del static_features
            values, mask = self._inputs(features, valid_mask)
            encoded, _ = self.encoder(values)
            positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
            last = positions.masked_fill(~mask, -1).max(dim=1).values.clamp(min=0)
            pooled = encoded[torch.arange(encoded.shape[0], device=mask.device), last]
            return self._predictions(pooled)


    BASELINE_CLASSES = {
        "mlp": MLPSequenceBaseline,
        "tcn": CausalTCNBaseline,
        "gru": GRUSequenceBaseline,
    }
else:
    BASELINE_CLASSES = {}


def build_deep_baseline(name: str, feature_count: int, lookback: int, values: dict):
    if nn is None:
        raise RuntimeError("深度基线需要安装项目的 ml 可选依赖")
    if name not in BASELINE_CLASSES:
        raise ValueError(f"未知深度基线: {name}")
    allowed = {"hidden_size", "num_layers", "kernel_size", "dropout", "normalization_clip"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{name} 包含不支持的模型参数: " + ", ".join(unknown))
    config = DeepBaselineConfig(feature_count=feature_count, lookback=lookback, **values)
    return BASELINE_CLASSES[name](config)

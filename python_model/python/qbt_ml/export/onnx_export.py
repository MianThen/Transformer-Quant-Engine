from __future__ import annotations

from pathlib import Path

from .manifest import OUTPUT_NAMES


def export_temporal_transformer(model, output: str | Path, *, batch_size: int = 2,
                                opset: int = 17) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("ONNX 导出功能未安装；请安装 ml 可选依赖") from exc
    config = model.config
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    class ExportWrapper(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            self.source = source

        def forward(self, features, valid_mask, static_features=None):
            value = self.source(features, valid_mask, static_features)
            return tuple(value[name] for name in OUTPUT_NAMES)

    wrapper = ExportWrapper(model).eval()
    features = torch.zeros(batch_size, config.lookback, config.feature_count)
    valid_mask = torch.ones(batch_size, config.lookback, dtype=torch.uint8)
    inputs = (features, valid_mask)
    input_names = ["features", "valid_mask"]
    dynamic_axes = {name: {0: "batch"} for name in (*input_names, *OUTPUT_NAMES)}
    if config.static_feature_count:
        static = torch.zeros(batch_size, config.static_feature_count)
        inputs += (static,)
        input_names.append("static_features")
        dynamic_axes["static_features"] = {0: "batch"}
    torch.onnx.export(
        wrapper, inputs, output, input_names=input_names,
        output_names=list(OUTPUT_NAMES), dynamic_axes=dynamic_axes,
        opset_version=opset, do_constant_folding=True,
    )
    return output

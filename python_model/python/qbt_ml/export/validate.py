from __future__ import annotations

import numpy as np

from .manifest import OUTPUT_NAMES


def validate_onnx_parity(model, model_path, features, valid_mask,
                         static_features=None, *, atol=1e-5, rtol=1e-4):
    try:
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise RuntimeError("ONNX 一致性验证需要 ml 可选依赖") from exc
    model.eval()
    with torch.no_grad():
        expected = model(
            torch.as_tensor(features), torch.as_tensor(valid_mask),
            None if static_features is None else torch.as_tensor(static_features),
        )
    inputs = {
        "features": np.asarray(features, dtype=np.float32),
        "valid_mask": np.asarray(valid_mask, dtype=np.uint8),
    }
    if static_features is not None:
        inputs["static_features"] = np.asarray(static_features, dtype=np.float32)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = session.run(list(OUTPUT_NAMES), inputs)
    errors = {}
    for name, value in zip(OUTPUT_NAMES, actual):
        reference = expected[name].detach().cpu().numpy()
        if not np.allclose(reference, value, atol=atol, rtol=rtol):
            errors[name] = float(np.max(np.abs(reference - value)))
    if errors:
        raise ValueError(f"PyTorch/ONNX 输出不一致: {errors}")
    return {name: value for name, value in zip(OUTPUT_NAMES, actual)}

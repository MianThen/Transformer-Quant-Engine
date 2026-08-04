# Optional ML runtime

The ML runtime is excluded from normal builds and from the Python wheel:

```bash
cmake -S . -B build/default -DQBT_BUILD_PYTHON=OFF
```

Use the deterministic backend to test feature windows and strategy integration without a model:

```bash
cmake -S . -B build/ml-mock \
  -DQBT_BUILD_PYTHON=OFF \
  -DQBT_ENABLE_ML=ON \
  -DQBT_BUILD_ML_TESTS=ON \
  -DQBT_ML_BACKEND=mock
```

On a deployment host with ONNX Runtime installed, enable the CPU backend explicitly:

```bash
cmake -S . -B build/ml-onnx \
  -DQBT_BUILD_PYTHON=OFF \
  -DQBT_ENABLE_ML=ON \
  -DQBT_ML_BACKEND=onnxruntime \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime
```

To build the Python wheel with this optional backend, keep normal builds unchanged and set the
activation variables only on the deployment host:

```bash
QBT_ENABLE_ML=ON \
QBT_ML_BACKEND=onnxruntime \
ONNXRUNTIME_ROOT=/opt/onnxruntime \
python -m pip install .
```

The resulting module reports `cpp_engine.__ml_enabled__ == True` and
`cpp_engine.__ml_backend__ == "onnxruntime"`. A normal wheel reports `disabled`.

After an artifact passes validation, attach it before processing market data:

```python
engine.set_model_strategy(
    artifact_path="models/bar-temporal-transformer-v1",
    policy_config={
        "max_positions": 20,
        "max_position_weight": 0.05,
        "minimum_confidence": 0.55,
    },
    risk_config={"max_order_quantity": 100_000},
    runtime_config={"intra_op_threads": 1, "max_batch_size": 4096},
)
```

The backend requires the fixed input names `features`, `valid_mask`, and optionally
`static_features`. It requires the six versioned prediction outputs declared by the model
manifest. Schema mismatch, output overflow, timeout, backend errors, and non-finite values are
reported as statuses and must not be converted into orders.

The first backend intentionally supports CPU only. CUDA must be enabled as a separate, tested
deployment path after CPU golden parity is established.

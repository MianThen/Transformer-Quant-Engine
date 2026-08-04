# Release benchmark and replay baseline

All performance numbers must come from a Release preset or an installed Release
wheel. Reports record compiler, build type, LTO state, artifact SHA-256, dataset,
hardware, all five raw runs, median and p95. The validated MinGW GNU 13.1.0 build
does not support the requested LTO configuration, so its reports correctly record
`lto=false`; do not describe that artifact as LTO-enabled.

```powershell
cmake --preset benchmark-release
cmake --build --preset benchmark-release --parallel
ctest --preset benchmark-release
python tools/run_cpp_benchmarks.py `
  build/benchmark-release/cpp_engine/qbt_benchmark.exe `
  --runs 5 --baseline benchmarks/reports/m0-current.json `
  --output benchmarks/reports/vnext-current.json
build/benchmark-release/cpp_engine/qbt_golden_replay.exe `
  --verify benchmarks/golden/qbt-m0-golden-v1.json
```

Use `--quick` only for smoke checks. Full reports contain compiler, Release/LTO,
artifact SHA-256, dataset identity, hardware, raw runs, median, p95, behavior
checksums, and separate `market_data`, `orders`, and `positions` attribution.
Pass `--baseline old.json` to attach old/new median deltas to every scenario.

Generate a new golden file only after reviewing intentional behavior changes:

```powershell
build/benchmark-release/cpp_engine/qbt_golden_replay.exe `
  --output benchmarks/golden/qbt-m0-golden-v1.json
python tools/compare_golden.py old-golden.json benchmarks/golden/qbt-m0-golden-v1.json
```

The live feed benchmark uses the dedicated Release preset and keeps the original
allocating Decoder as a benchmark-only comparison target:

```powershell
cmake --preset live-benchmark-release
cmake --build --preset live-benchmark-release --parallel
ctest --preset live-benchmark-release
python tools/run_cpp_benchmarks.py `
  build/live-benchmark-release/trading_engine/te_benchmark_legacy.exe `
  --runs 5 --output benchmarks/reports/te-decoder-legacy.json
python tools/run_cpp_benchmarks.py `
  build/live-benchmark-release/trading_engine/te_benchmark.exe `
  --runs 5 --baseline benchmarks/reports/te-decoder-legacy.json `
  --output benchmarks/reports/te-decoder-current.json
```

The current Decoder report must show zero timed allocations. Queue overflow enters
`DEGRADED` by default; `DROP_AND_RESYNC` requires an explicit
`mark_resynchronized()` call before market data becomes trusted again.

## Release wheel boundary benchmark

Build and install the wheel before measuring Python overhead. The runner rejects
non-Release extensions by checking `cpp_engine.__build_type__`.

```powershell
python -m build --wheel
$wheel = Get-ChildItem dist\*.whl | Sort-Object LastWriteTime | Select-Object -Last 1
python -m pip install --force-reinstall $wheel.FullName
python tools/run_python_boundary_benchmarks.py `
  --runs 5 --symbols 10000 `
  --output benchmarks/reports/python-boundary-current.json
```

## Current archived reports

- `benchmarks/reports/vnext-current.json`: C++ market data, orders, positions and behavior.
- `benchmarks/reports/te-decoder-current.json`: fixed-buffer Decoder versus allocating legacy Decoder.
- `benchmarks/reports/python-boundary-current.json`: pure C++, cross-section Python and per-row Python boundary cost.
- `te_pipeline_benchmark`: feed bytes → decoder → strategy → order gateway → ACK 的端到端场景，包含突发、半包、背压、乱序 ACK、WAL 恢复和重连。
- `tools/check_pipeline_budget.py`: 稳定场景回退超过 5% 输出告警，超过 10% 返回失败并阻止发布。
- `benchmarks/golden/qbt-m0-golden-v1.json`: deterministic inputs, orders, fills, cash, positions, equity and final metrics.

Feed capture/replay, WAL recovery/reconciliation, epoll/IOCP, runtime modes and
operational metrics are covered by the Release CTest suite and documented in
`benchmarks/VNEXT_COMPLETION.md`.

## ML runtime benchmark

`qbt_ml_benchmark` must be built with Release, LTO and the real ONNX Runtime CPU
backend. The wrapper rejects non-Release/non-LTO output and records model lineage,
binary SHA-256, hardware, compiler, ORT version and git revision.

```bash
ONNXRUNTIME_ROOT=/path/to/onnxruntime cmake -S . -B build/ml-benchmark-release \
  -DCMAKE_BUILD_TYPE=Release -DQBT_BUILD_PYTHON=OFF -DQBT_ENABLE_ML=ON \
  -DQBT_ML_BACKEND=onnxruntime -DQBT_BUILD_BENCHMARKS=ON -DQBT_ENABLE_LTO=ON
cmake --build build/ml-benchmark-release --target qbt_ml_benchmark --parallel
python3 tools/run_ml_runtime_benchmarks.py \
  --executable build/ml-benchmark-release/strategy_runtime/qbt_ml_benchmark \
  --artifact /path/to/manifest-v2-artifact \
  --output benchmarks/reports/ml-runtime-current.json \
  --iterations 100 --chunk-size 512 \
  --benchmark-scope production_candidate
```

The Phase D M4 Pro baseline is stored in
`benchmarks/reports/ml-runtime-m4-pro-phase-d.json`. It uses the frozen small V2
test artifact and is an engineering baseline, not a production-model quality claim.
The default scope remains `engineering_test_artifact`; Phase E only accepts a report
that explicitly uses `--benchmark-scope production_candidate` and matches the
candidate model and training-dataset lineage.

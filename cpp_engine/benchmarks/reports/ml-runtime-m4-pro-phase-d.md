# Phase D ML Runtime Baseline

- Run date: 2026-07-29
- Code revision: `3e9293c6b8a39e12be077c5b8a4a4f91f5616277`
- Hardware: Apple M4 Pro
- Build: Release, LTO enabled, AppleClang 21.0.0.21000101
- Runtime: ONNX Runtime 1.17.3, CPUExecutionProvider, intra/inter-op threads 1/1
- Chunk size: 512
Measured iterations: 100 per batch size

| Batch | End-to-end p50 | End-to-end p99 | ORT p50 | ORT p99 | Peak RSS |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.057 ms | 0.145 ms | 0.056 ms | 0.142 ms | 22.44 MiB |
| 64 | 0.253 ms | 0.410 ms | 0.218 ms | 0.375 ms | 23.86 MiB |
| 512 | 1.724 ms | 1.922 ms | 1.363 ms | 1.524 ms | 32.44 MiB |
| 4096 | 14.411 ms | 14.841 ms | 10.958 ms | 11.285 ms | 84.23 MiB |

At batch 4096, feature/policy/risk p99 are 3.654/0.101/0.075 ms. Deterministic
chunking uses chunks of at most 512 symbols. The full machine-readable report includes
artifact hashes, training dataset fingerprint and benchmark executable SHA-256.

This is a software performance baseline produced with the small frozen V2 test artifact.
It does not establish production-model returns, production all-A data throughput, or the
performance of a future minute-frequency model.

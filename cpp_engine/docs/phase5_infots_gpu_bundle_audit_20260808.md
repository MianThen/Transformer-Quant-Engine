# Phase 5 InfoTS GPU 输入包审计

更新时间：2026-08-08

## 归档

- 文件：`work/phase5_infots_gpu_training_bundle_20260808.tar.gz`
- sidecar：`work/phase5_infots_gpu_training_bundle_20260808.tar.gz.sha256`
- SHA-256：`166abd15e9400c13339f20d1b6f28ca4df39b0e62baf573ffe56fa2591f0eeff`
- 根目录：`qbt_phase5_infots_gpu_v1/`
- manifest 条目：19
- 数据 SHA-256：`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`

## 审计结果

```text
archive SHA-256 == sidecar             PASS
manifest file hashes (13/13)           PASS
dataset SHA-256                        PASS
absolute path / .. traversal           PASS
Python cache / .pyc                    PASS
source/config py_compile               PASS
no-Torch CUDA preflight                FAIL-CLOSED (exit 1)
```

该归档是外部 CUDA 训练输入，不是训练结果；不包含 checkpoint、prediction、embedding 或 OOS 报告。
训练主机必须先执行 `phase5_infots_gpu_preflight.py`，再运行 InfoTS pretraining runner。训练回传后仍需
独立验证三组监督预算、purged OOS、六输出/embedding provenance、质量/稳定性 gate 和 promotion 条件。

# Phase 5 InfoTS GPU 输入包审计（2026-08-09）

- 归档：`work/phase5_infots_gpu_training_bundle_20260809.tar.gz`
- sidecar：`work/phase5_infots_gpu_training_bundle_20260809.tar.gz.sha256`
- SHA-256：`ee9e28b0806cd5c969556228bc930133a6a31d65b9ebd049eb7dcd14a6434c2e`
- 根目录：`qbt_phase5_infots_gpu_v1/`
- manifest 条目：24
- 数据 SHA-256：`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`

审计结果：archive/sidecar、manifest 全文件 hash、数据 hash、绝对路径与 `..` 穿越、Python cache、源码/配置语法均通过；本机无 Torch/CUDA 时 preflight 按合同非零退出。新包增加 `infots_replay.py`、C++ report hash 复算、C++ replay runner、九组 joint acceptance 和对应测试/文档。

该归档仍是外部 CUDA 训练输入，不是 checkpoint 或实验结果。真实训练完成后必须回传九组 fold artifact、六输出、validation/test embedding、purged OOS 报告和 gate，再运行联合验收。

# Phase 5 InfoTS GPU runner

该 runner 只允许 train-fold-only 的 InfoTS 对比预训练，不执行监督六输出评估，不自动替换 frozen
champion，也不输出正式 OOS 或生产 gate。

## 输入

- 数据集必须包含 `features`、`valid_mask`、`timestamps`；数据文件 SHA-256 必须与配置一致。
- `timestamps` 可以是每个样本的 decision timestamp `[N]`，也可以是完整时间矩阵 `[N,T]`。
- 配置固定 `device=cuda`、`deterministic_algorithms=true`、`production_eval=false`；没有 CUDA 或 Torch
  时 runner 非零退出，禁止静默回退 CPU。

## 运行

先执行 CUDA/data preflight；任何失败都必须停止：

```text
python tools/phase5_infots_gpu_preflight.py \
  --dataset data/research/phase1e_pit_120_dataset.npz \
  --output runs/phase5-infots/runtime_preflight.json
```

通过后再运行训练：

```text
python tools/run_phase5_infots_pretraining.py \
  --dataset data/research/phase1e_pit_120_dataset.npz \
  --config configs/ml/phase5_infots_pretraining_cuda.json \
  --output runs/phase5-infots/reference_artifact.json \
  --checkpoint runs/phase5-infots/infots_checkpoint.pt
```

输出 artifact 固定保存 training/augmentation/spec hash、loss history、global/local embedding hash、
checkpoint hash、六输出语义和 `phase_exit_eligible=false`/`promotion_eligible=false`。训练产物回传后仍需
使用 `python/qbt_ml/evaluation/infots_ablation.py` 独立完成 from-scratch、固定增强、InfoTS 三组等监督
预算比较、purged OOS、embedding/prediction provenance 和质量/稳定性 gate。

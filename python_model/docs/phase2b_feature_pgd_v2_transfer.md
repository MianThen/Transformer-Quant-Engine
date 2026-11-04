# Phase 2B Feature-PGD V2 迁移训练包

## 包内容

- `python/qbt_ml/`：训练、模型、数据、标签、Phase 2B OOS 与 bootstrap 代码。
- `configs/ml/phase2b_feature_pgd_v2.local.json`：V2 预注册配置。
- `configs/ml/phase2b_feature_pgd_v2_validation.local.json`：validation-only 消融配置，不访问旧 test 窗口。
- `data/research/phase1e_pit_120_dataset.npz`：训练数据集。
- `data/research/phase1e_pit_120_audit.json`：数据审计与 cutoff。
- `requirements-ml.txt`：训练依赖参考版本。

## 环境

- Python `>=3.9`。
- PyTorch `2.8.0`（GPU 机器请先按 CUDA/ROCm 平台安装对应 wheel）。
- NumPy `2.0.2`、Pandas `2.3.3`、PyYAML `6.0.3`、PyTest `8.4.2`。

## 运行前检查

```bash
python -m pytest -q -p no:cacheprovider python/tests/test_phase2b_robust_training.py
```

## 启动 V2 三折训练

```bash
python -m python.qbt_ml.cli phase2b-oos \
  --config configs/ml/phase2b_feature_pgd_v2.local.json \
  --dataset data/research/phase1e_pit_120_dataset.npz \
  --output runs/phase2b-feature-pgd-v2
```

输出包含每折 `checkpoint.pt`、validation/test predictions、embedding snapshots、metrics、预注册合同和 Phase 2B report。训练前不要修改 seed、split、scaler、missing mode、epsilon、beta 或 bootstrap 配置。

## 先做 validation-only 消融

```bash
python -m python.qbt_ml.cli phase2b-oos \
  --config configs/ml/phase2b_feature_pgd_v2_validation.local.json \
  --dataset data/research/phase1e_pit_120_dataset.npz \
  --output runs/phase2b-feature-pgd-v2-validation
```

该配置的 `evaluation_split=validation` 只用于选择候选；冻结 hypothesis 和 margin 后，再使用正式配置访问未触碰 test/OOS。

## 校验

```bash
shasum -a 256 data/research/phase1e_pit_120_dataset.npz
```

归档根目录的 `MANIFEST.sha256` 是迁移包文件清单；数据 hash 必须与清单一致。当前 V1 test 窗口只能作诊断，不能把本包 V2 结果直接当作晋级证据。

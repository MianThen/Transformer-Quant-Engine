# Phase 2B Feature-PGD V2 施工状态

更新时间：2026-08-05

`Feature-PGD V1` 保持 `TRAINING_COMPLETE_GATE_FAILED`。V2 已完成 train-fold-only scaler、标准化 z-space PGD、center-impute missing 语义、旧 checkpoint identity 兼容、relative stress diagnostics 和 paired block-bootstrap CI；尚未运行正式 50-epoch OOS。

- 配置：`configs/ml/phase2b_feature_pgd_v2.local.json`
- validation-only 配置：`configs/ml/phase2b_feature_pgd_v2_validation.local.json`
- 新 scaler：`python/qbt_ml/features/scaling.py`
- 训练/模型/评估改动：`python/qbt_ml/cli.py`、`python/qbt_ml/models/temporal_transformer.py`、`python/qbt_ml/training/robust_training.py`、`python/qbt_ml/training/phase2b.py`
- 定向测试：`12 passed`
- 全量测试：`182 passed, 8 skipped`

下一步先在 validation-only 口径完成三组消融，再冻结新 hypothesis 与未触碰 OOS gate；现有 V1 test 窗口不能直接用于 V2 晋级。

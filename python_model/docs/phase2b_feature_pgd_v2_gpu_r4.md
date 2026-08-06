# Phase 2B Feature-PGD V2 GPU 迁移包 r4

本包修复 r2 的三个问题：CUDA 不再写死为 CPU；validation-only 不再执行或写出 test；
预注册合同绑定训练配置、实现 hash 与每个 checkpoint 的 SHA-256。

r4 额外修复：Feature-PGD 为纯 PGD，structured missing 改为独立候选；PGD 不扰动受保护和派生
语义特征；训练输出 checkpoint_best.pt、checkpoint_last.pt 及头级攻击诊断；门禁使用预注册
non-inferiority/relative-stress margin 和 paired bootstrap。

## 环境与数据

- Python `>=3.9`；先按训练机 CUDA 版本安装 PyTorch，再安装 `requirements-ml.txt`。
- 数据：`data/research/phase1e_pit_120_dataset.npz`。
- 数据 SHA-256：`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`。
- CUDA 不可用时训练会失败，不会静默退回 CPU。

## 顺序

先运行真正 test-blind 的 validation：

```bash
bash run_validation_cuda.sh
```

validation 目录的每个 fold/variant 只能出现 checkpoint、validation predictions、validation
embedding、manifest 和 metrics。结果校验器会拒绝任何 test 文件或 test metric。

冻结 hypothesis 和 margin 后，才运行新的短 OOS：

```bash
bash run_short_oos_cuda.sh
```

短 OOS 以旧 V1 最后 test timestamp `1768374000000000000` 为 cutoff，使用之后尚未观察的
117 个 timestamp，冻结为 `3×39`。它可形成新的 research-only 证据，但窗口较短、统计功效弱；
若坚持 `3×126`，当前还缺 260 个新 timestamp，不能伪装成已完成。

## 回传

```bash
python tools/package_results.py \
  --root runs/phase2b-feature-pgd-v2-short-new-oos \
  --output phase2b_feature_pgd_v2_short_oos_results.tar.gz
```

回传结果包、归档 SHA-256，以及根目录的 `runtime_preflight_short_oos.json`。正式结果应有
3 folds × 3 variants 共 9 个 checkpoint、validation/test 六输出、两套 embedding snapshots、
`preregistered_contract.json`、`phase2b_oos_report.json` 和 `RESULT_VALIDATION.json`。

本包不会把 Feature-PGD 自动晋级；clean/noisy/stress、paired bootstrap 与三窗口方向 gate
仍需全部通过。ONNX 可回本机统一导出，Feature-PGD 训练分支不得进入生产推理图。

# GPU 训练移交状态

更新时间：2026-08-08

## 结论

当前 Phase 2B `Feature-PGD V2` r5 与 Phase 5 `InfoTS` v1 达到“源码、冻结配置、数据、启动脚本和输出结构校验器齐备”的外部训练移交条件，状态为：

```text
READY_FOR_EXTERNAL_GPU_TRAINING
TRAINING_COMPLETE = false
RESEARCH_GATE_PASSED = false
PROMOTION_ELIGIBLE = false
```

r5 是训练输入包，不是训练结果。只有外部 CUDA 训练、test-blind validation、冻结 hypothesis 后的短 OOS、结果回传校验及 clean/noisy/stress gate 全部完成，才能登记训练和研究判定；不能因压缩包存在而宣称 Feature-PGD 已通过。

Phase 5 InfoTS、Phase 6 Mamba 和路线图 4090 阶段的全 A 多 seed Walk-forward 当前不具备真实可运行迁移包的条件。不得创建只含 README、占位配置或伪输出的压缩包。

## 路线图 GPU 项盘点

| 施工项 | 路线图要求 | 当前证据 | 打包判定 |
|---|---|---|---|
| Phase 2B Feature-PGD | 标准化连续特征上的 mask-aware PGD；clean、压力集和三窗口 gate；4090 阶段执行 adversarial 训练 | r5 已有实现、CUDA 配置、冻结数据、test-blind validation、`3×39` 新短 OOS 和结果校验器 | `READY_FOR_EXTERNAL_GPU_TRAINING`，尚未训练 |
| 全 A 多 seed Walk-forward | 4090 阶段执行全 A 多 seed 训练 | 当前数据审计只冻结 120 只、实际 119 个 symbol，不是全 A；没有对应多 seed 配置和汇总合同 | `NOT_PACKAGEABLE_DATA_AND_CONTRACT_MISSING` |
| Phase 5 InfoTS | causal augmentations、global/local InfoNCE、information-aware selector、train-fold-only 预训练及三组等监督预算比较 | v1 已有 NumPy 合同、PyTorch Siamese trainer、冻结 CUDA 配置、pinned requirements、真实数据 runner、preflight、六输出契约；输入包 manifest/data/path 审计通过，尚无训练回传结果 | `READY_FOR_EXTERNAL_GPU_TRAINING`，尚未训练 |
| Phase 6 Mamba | 4090/Linux CUDA、`T>=256/512` 或分钟数据、`MambaResearchV1`、等参数/等 token Transformer、冻结六输出和 C++ replay | 两个可用研究数据集均为 `T=64`；无 Mamba 模型、固定依赖、配置、测试、benchmark 或预测 artifact | `NOT_PACKAGEABLE_ENTRY_GATE_NOT_MET` |

Phase 3A/3B 的 factor、GARCH/FHS/EVT/Expectile 主线和 Phase 4 posterior/policy 主线是 C++/统计与优化施工，不属于待打包的 GPU 训练。Phase 3C Attention Analysis 使用冻结模型做分析 forward/occlusion，不等于新增模型训练，也不应伪装成 GPU 训练包。

## Feature-PGD r5 输入包

- 归档：`/Users/Zhuanz/CLionProjects/quant-backtester-cpp/work/phase2b_feature_pgd_v2_gpu_training_bundle_20260808_r5.tar.gz`
- SHA-256 sidecar：`/Users/Zhuanz/CLionProjects/quant-backtester-cpp/work/phase2b_feature_pgd_v2_gpu_training_bundle_20260808_r5.tar.gz.sha256`
- 归档 SHA-256：`91541dbf5957684ff3bd23843c0680b050f06540400783e0f6dbe94e07767895`
- 解压根目录：`qbt_phase2b_feature_pgd_v2_gpu_r5/`
- 归档大小：约 708 MiB。
- 包内 manifest：`MANIFEST.sha256`，共 101 个受保护文件，全部为 LF。
- 数据：`data/research/phase1e_pit_120_dataset.npz`，SHA-256 为 `4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`。
- 数据 shape：`features=(173242,64,23)`，共 1521 个 model-ready timestamp；因此不能用于 Mamba 的 `T>=256/512` 进入门槛。
- GPU 配置：`configs/ml/phase2b_feature_pgd_v2_validation_cuda.json` 和 `configs/ml/phase2b_feature_pgd_v2_short_oos_cuda.json`，均冻结 `device=cuda`、50 epochs、单 worker、deterministic algorithms 与 validation-return-guarded checkpoint selection。
- 启动脚本：`run_validation_cuda.sh`、`run_short_oos_cuda.sh`，均使用 `set -euo pipefail` 并先执行 CUDA/data preflight。
- 输出工具：`tools/validate_results.py`、`tools/package_results.py`。

CUDA 不可用、数据 hash 不匹配或 GPU probe 失败时，`tools/gpu_preflight.py` 返回非零并停止，不允许静默回退 CPU。

## 窗口与证据边界

短 OOS 以 V1 最后 test timestamp `1768374000000000000` 为 cutoff。数据中 cutoff 后有 118 个 timestamp；冻结配置只使用前 117 个，形成三个互不重叠的 39-timestamp test 窗口：

| Fold | Test first | Test last | Count |
|---|---:|---:|---:|
| 1 | `1768460400000000000` | `1773817200000000000` | 39 |
| 2 | `1773903600000000000` | `1779087600000000000` | 39 |
| 3 | `1779174000000000000` | `1783926000000000000` | 39 |

`3×39` 只能形成统计功效较弱的 `research-only` 新 OOS 证据。若保持 `3×126`，当前需要 378 个 cutoff 后 timestamp，仍缺 260 个；不得复用 V1 已观察 test、复制窗口或把短窗口称为完整长窗口。

## 独立验证记录

以下检查均从 r5 归档的干净 staging 副本执行，而不是从源工程同名文件推断：

```text
shasum -a 256 <r5.tar.gz>
=> 91541dbf...e07767895，与 sidecar 一致

shasum -a 256 -c MANIFEST.sha256
=> 101/101 OK

bash -n run_validation_cuda.sh
bash -n run_short_oos_cuda.sh
python -m py_compile tools/gpu_preflight.py tools/validate_results.py tools/package_results.py
=> PASS

python3 -m pytest -q -p no:cacheprovider python/tests/test_phase2b_robust_training.py
=> 7 passed, 9 skipped（本机未安装 PyTorch）

python tools/gpu_preflight.py --output <temp>
=> 本机无 CUDA，exit 1，明确报错“禁止静默回退 CPU”

python tools/validate_results.py --mode validation --root <空目录>
=> exit 1，RESULT_VALIDATION.status=FAIL，并列出缺失制品
```

归档路径检查未发现绝对路径或 `..` traversal 项；两个 shell 启动脚本保留 executable bit。

本机没有执行真实 CUDA 训练，因此上述 PASS 只证明输入包完整、CPU 可运行的定向工程测试通过和失败关闭生效，不证明 GPU 兼容性、训练完成、结果质量或 gate 通过。

## 回传验收

validation 必须先执行，且 3 folds × `none/structured_missing/feature_pgd` 的每个目录必须包含 official/best/last checkpoint、`checkpoint_selection.json`、validation predictions、validation embedding/manifest 和 metrics；出现任意 test metric、prediction 或 embedding 即拒收。

冻结 hypothesis/margin 后才能执行短 OOS。短 OOS 回传至少包含：

- 9 个 checkpoint 及可复算 SHA-256；
- validation/test 六输出与两套 embedding snapshot；
- `preregistered_contract.json`，包含 implementation、完整 training config 和 dataset hash；
- `phase2b_oos_report.json`，包含 clean/noisy/stress、paired bootstrap 和三窗口 gate；
- `runtime_preflight_short_oos.json`；
- `RESULT_VALIDATION.json` 和结果目录 `MANIFEST.sha256`。

已知工具边界：当前 `tools/package_results.py` 只检查 `RESULT_VALIDATION.json` 是否存在，不检查其中 `status` 是否为 `PASS`；独立负向试验确认 `FAIL` 文件也能被归档。因此接收方必须显式确认 `RESULT_VALIDATION.status == "PASS"`、重新校验结果 `MANIFEST.sha256`，并核对报告/合同/checkpoint hash 后才可登记结果。仅有结果压缩包或 `RESULT_VALIDATION.json` 文件存在不构成验收。

即使输出结构验收通过，若 `research_gate_passed=false`，候选仍保持 research-only，不替换 frozen champion，不形成运行时 fallback，也不阻塞后续彼此独立的非 GPU 数学施工。

## Phase 5 输入包（2026-08-09 最新）

- 归档：`/Users/Zhuanz/CLionProjects/quant-backtester-cpp/work/phase5_infots_gpu_training_bundle_20260809.tar.gz`
- SHA-256 sidecar：`/Users/Zhuanz/CLionProjects/quant-backtester-cpp/work/phase5_infots_gpu_training_bundle_20260809.tar.gz.sha256`
- 归档 SHA-256：`ee9e28b0806cd5c969556228bc930133a6a31d65b9ebd049eb7dcd14a6434c2e`
- 解压根目录：`qbt_phase5_infots_gpu_v1/`
- 数据 SHA-256：`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`
- 包含：causal augmentation、InfoNCE、selector、PyTorch trainer、三组预算/OOS evaluator、跨语言 C++ Replay
  联合验收器、CUDA preflight、冻结配置、测试、pinned requirements 和数据；不包含 checkpoint 或实验结果。

## Phase 5 训练回传门槛

训练包已经生成；外部 GPU 回传前后仍必须满足：

1. preflight 通过并保留 runtime/device/data hash；
2. `from scratch`、固定增强、InfoTS 三组等监督预算 runner 和可校验输出 schema；
3. purged OOS、六输出 prediction/embedding artifact、质量/稳定性 gate 和 C++ replay provenance。

Mamba 至少需先完成：

1. 可审计的 `T>=256/512` 或分钟 PIT 数据集；
2. 固定版本的维护中 Mamba CUDA 依赖和 `MambaResearchV1`；
3. mask/variable-length/seed 测试及与 Transformer 等参数、等 token、等 seed 配置；
4. 六输出 prediction artifact、C++ precomputed replay 接口和吞吐/显存/质量报告合同。

这些条件满足前，InfoTS 只有可交付输入包而没有训练结果；Mamba 仍是路线图施工项，不能创建其训练包。

## 2026-08-06 复核

- 本轮没有新增需要 GPU 的训练实现；Phase 3C Attention/IG 使用冻结 checkpoint 的分析 forward，Phase 4A
  posterior 是 CPU 数学 reference，不应再次打包成训练任务。
- 已重新校验 r3 输入包 SHA-256：
  `df9c9258e5c9bea25a370312fc3054f49e62cebe83190adc1c25b025c959b740`，与 sidecar 一致。
- 当前工作区另有可校验副本：`work/phase2b_feature_pgd_v2_gpu_training_bundle_20260805_r3.tar.gz` 及同名
  `.sha256`；在 `work/` 目录执行 `shasum -a 256 -c ...sha256` 返回 `OK`。本轮挂载中 Downloads 副本
  不可见，不能据此臆测其仍存在。
- r3 仍是 `READY_FOR_EXTERNAL_GPU_TRAINING`，不改变 `TRAINING_COMPLETE=false`、
  `RESEARCH_GATE_PASSED=false` 和 `PROMOTION_ELIGIBLE=false`；没有凭压缩包存在宣称训练或 gate 通过。

## 2026-08-08 Phase 5 更新

- 已新增 CPU InfoTS 合同、可选 PyTorch trainer、CUDA preflight、冻结配置和外部输入包；本机没有
  PyTorch/CUDA，preflight 明确以非零退出，禁止静默回退 CPU。
- Phase 5 当前为 `READY_FOR_EXTERNAL_GPU_TRAINING`，但 `TRAINING_COMPLETE=false`、
  `phase_exit_eligible=false`、`promotion_eligible=false`；输入包存在不代表训练或 gate 通过。

## 2026-08-09 Phase 5 非 GPU 更新

- 已完成 C++ 预计算 InfoTS Replay/CVaR、跨语言输入/report hash、future/state/proxy guard 和九组 Python/C++
  联合验收；带 analytics 的 CTest 17/17、全量 Python 31 passed/1 skipped。
- 该工程闭环不改变 GPU 训练状态：仍为 `TRAINING_COMPLETE=false`、`phase_exit_eligible=false`、
  `promotion_eligible=false`。

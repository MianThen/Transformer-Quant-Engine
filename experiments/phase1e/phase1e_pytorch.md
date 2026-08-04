# Phase 1E PyTorch 环境

真实训练环境复用 `$PYTHON_PROJECT_ROOT/.venv`，不要在 C++ 回测环境中另建第二套 Torch。当前环境为 Apple Silicon、Python 3.9、PyTorch 2.8.0；CUDA 不可用，MPS backend 已编译但当前运行时不可用，训练设备固定为 `cpu`。

```bash
export PYTHONPATH=$PYTHON_PROJECT_ROOT
export QBT_PYTHON=$PYTHON_PROJECT_VENV/bin/python
$QBT_PYTHON -c 'import torch; print(torch.__version__)'
```

现有真实实现位于 `$PYTHON_PROJECT_ROOT/python/qbt_ml/training/gradient_methods.py`，测试位于 `$PYTHON_PROJECT_ROOT/python/tests/test_phase1e_gradients.py`。当前项目的 `qbt_ml.research.torch_gradient` 提供 C++ 项目侧的 autograd 观测桥；默认只读取共享参数梯度，不写入 `parameter.grad`。

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=$PYTHON_PROJECT_ROOT \
$PYTHON_PROJECT_VENV/bin/python \
-m pytest -q -p no:cacheprovider \
$PYTHON_PROJECT_ROOT/python/tests/test_phase1e_gradients.py
```

普通 C++ 回测仍不依赖 PyTorch；只有显式进入 ML 研究环境时才使用 `.[ml]` 依赖 extra。

## Challenger 编排

三窗口 challenger 使用当前项目的 `tools/run_phase1e_challenger.py`，但训练运行时复用
`$PYTHON_PROJECT_ROOT` 的真实 Torch、dataset 和 security-state 审计。
PCGrad 与 GradNorm-4 必须分开运行，并为每次运行提供新的 `hypothesis_id`：

```bash
PYTHONDONTWRITEBYTECODE=1 \
$PYTHON_PROJECT_VENV/bin/python \
tools/run_phase1e_challenger.py \
  --config $PYTHON_PROJECT_ROOT/configs/ml/phase1e_diagnostics.local.json \
  --method pcgrad \
  --hypothesis-id MTL-PCGRAD-REAL-OOS-<run-id> \
  --epochs 3 \
  --output runs/phase1e-pcgrad-real-<run-id>
```

输出目录中的 `preregistered_contract.json`、六个配对 checkpoint 和 `challenger_report.json`
只构成 report-only 研究记录；缺少 C++ Replay/CVaR 所需经济字段时，报告必须保持
`promotion_allowed=false`、`promotion_eligible=false`，不得覆盖 `FROZEN_CHAMPION`。

已有 OHLCV 与预测也可以生成 C++ proxy 回放，不必等待缺失字段补齐：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=$QBT_ROOT/build/phase1c-python/cpp_engine \
$PYTHON_PROJECT_VENV/bin/python \
tools/run_cpp_proxy_replay_cvar.py \
  --data $PYTHON_PROJECT_ROOT/data/research/phase1e_pit_120_2020plus.parquet \
  --predictions runs/phase1e-pcgrad-real-e3/fold-1/pcgrad/test_predictions.npz \
  --output runs/cpp-proxy/pcgrad-fold-1.json
```

该命令使用 bar close 作为 reference/fill proxy，并将 C++ ledger、Return Analysis 和 empirical
CVaR 写入报告；`PROXY`、缺失字段和 `promotion_eligible=false` 会被强制保留。

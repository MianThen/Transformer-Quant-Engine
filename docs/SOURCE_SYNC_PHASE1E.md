# 当前源码同步说明

日期：2026-08-02

本次同步将当前工作区的 C++ 与 Python 源码按本仓库既有目录布局合并：

| 当前工作区 | 本仓库目录 |
|---|---|
| `CMakeLists.txt`、`CMakePresets.json`、`pyproject.toml`、`setup.py` | `cpp_engine/` |
| `cpp_engine/` | `cpp_engine/cpp_engine/` |
| `engine_common/`、`ml_runtime/`、`strategy_runtime/`、`trading_engine/` | `cpp_engine/<module>/` |
| `performance_analytics/`、`portfolio_math/`、`quant_math/` | `cpp_engine/<module>/` |
| `tools/*.py` | `cpp_engine/tools/` |
| `python/qbt_ml/monitoring/`、`python/qbt_ml/research/` | `python_model/python/qbt_ml/<module>/` |

未同步 `build/`、`cmake-build-*/`、`runs/`、`dist/`、`*.pyc`、模型、数据集和本地缓存。目标仓库中
已有但当前工作区没有的文件不删除；因此历史 ML/策略辅助文件仍保留，但默认构建以当前同步后的
CMake/source list 为准。

## 验证

- C++ core（ML/ONNX 关闭）：7/7 CTest 通过。
- C++ performance analytics（15 个测试）：15/15 CTest 通过。
- Python `qbt_ml`：31 passed、7 skipped。
- `QBT_ENABLE_PORTFOLIO_MATH=ON` 的配置因 Eigen 下载网络不可用未完成；未伪造该结果。

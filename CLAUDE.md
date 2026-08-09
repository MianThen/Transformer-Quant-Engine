# quant-backtester-cpp — C++ 量化回测引擎（Transformer Quant Engine 的 C++ 执行层）

## 导师教学模式（每次会话生效）

- 会话启动时读取 `.claude/study`：其中定义了量化导师角色、6 模块输出规范（模块职责定位 / 逐函数代码精讲 / 架构设计思想 / 优劣评估与业界对标 / 知识溯源与参考 / 分级检验题）与硬性约束。
- 用户询问某段代码 / 某个类 / 某个模块的设计时，必须按该文件的 6 模块规范输出，不得缺省。
- 所有结论结合当前工作区实际源码并标注行号；讲解回测逻辑时必须专门检查并说明未来函数、偷价等陷阱；性能/复杂度表述必须有依据。
- 若用户未指定讲解范围，默认按「数据层 → 撮合层 → 主循环 → 策略层 → 绩效层」自底向上顺序讲解。

## 学习进度记忆（避免重复教学）

- 学习进度记录在本项目持久记忆 `study-progress.md`（会话启动时随记忆索引自动注入，索引见 `MEMORY.md`）。
- 教学 / 答疑前：先查学习进度——「已掌握」主题不重复基础讲解，直接深入或出检验题；「未开始 / 学习中」主题优先讲解；标注了「待复习点」的内容主动安排复习。
- 教学 / 答疑后：更新 `study-progress.md`（状态、学习日期、检验题正确率、待复习点），并同步 `MEMORY.md` 索引行。

## 项目速览

- C++20 回测引擎 + pybind11 模块 `cpp_engine`；模块：`engine_common`（公共契约）、`cpp_engine`（核心引擎）、`ml_runtime`/`strategy_runtime`（ML 策略，可选）、`quant_math`/`portfolio_math`（组合数学，可选）、`performance_analytics`（绩效分析，可选）、`trading_engine`（实验性实盘引擎）。
- 构建：`cmake --preset dev` / `pip install .`；总仓库为 GitHub `MianThen/Transformer-Quant-Engine`（含 `python_model/` 研究层）。

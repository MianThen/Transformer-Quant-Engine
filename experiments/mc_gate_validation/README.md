# MC-GATE-VALIDATION

量化模型门控（Model-Champion Gating）的验证实验：用 Monte Carlo 数据生成器（DGP）
构造已知真值的面板，检验各类 GATE 规则在「选出真冠军」这件事上的实际功效，
并为合同（contract）里的每个验收条件（AC）给出实测证据。

- 合同版本：v1.4 → v1.5.0
- 结论状态：✓34 / ◐2 / ✗4（41 个 AC）
- 证据规模：269 个网格 cell × 6 条规则 + ABL 消融

## 目录

| 位置 | 内容 |
|---|---|
| `cpp_engine/performance_analytics/include/performance_analytics/mc_gate_*.h` | 20 个头文件（header-only 实现：DGP、基线、决策、评分、重放、成本、阈值、种子树、SHA-256 等） |
| `cpp_engine/performance_analytics/apps/mc_gate_*.cpp` | 15 个驱动与探针（全网格、COSTP、FPR/功率、金丝雀、OCN、H4、不变性、真值回收等） |
| `cpp_engine/performance_analytics/tests/mc_gate/` | 20 个 header 自检（每卡一个可执行文件） |
| `cpp_engine/configs/mc/` | 核心配置、269 cell 网格表、94 张施工卡、fixtures |
| `cpp_engine/docs/mc_gate_*` | 合同、施工计划、发现、验收包、修订决策、v1.5 建议、结果 jsonl |
| `cpp_engine/tools/*mc_gate*` | 验收驱动脚本 |

## 构建

MC-GATE 是纯 header-only、**零第三方依赖**（只用 C++20 标准库），
但它挂在 `performance_analytics` 模块下，需要显式打开该模块：

```bash
cd cpp_engine
cmake -B build/mcgate -DCMAKE_BUILD_TYPE=Release \
  -DQBT_ENABLE_PERFORMANCE_ANALYTICS=ON
cmake --build build/mcgate -j 8
```

## 测试

20 个 header 自检已注册为 ctest：

```bash
cd cpp_engine/build/mcgate
ctest -R test_mc_gate
```

预期：`100% tests passed, 0 tests failed out of 20`。

驱动与探针**不注册 ctest**（它们是实验工具，需要按参数人工运行）：

```bash
cd cpp_engine/build/mcgate/performance_analytics

# 全网格（269 cell）。--limit/--reps-cap 用于小规模试跑
./mc_gate_full --limit 3 --reps-cap 5

# §5.8 成本模型实测 ĉ(T,N)
./mc_gate_costp

# AC-09 金丝雀
./mc_gate_canary

# 集成驱动：plan -> cost_profile -> metrics -> records -> replay
./mc_gate_run
```

## 一处已修的真实缺陷

`mc_gate_full.cpp` 原先直接 `fopen("/tmp/qgate/C27/aggregates.jsonl", "wb")`
且**不检查返回值**。当输出目录不存在时（干净容器、CI、脚本化运行），
`fopen` 返回 `NULL`，随后 `fprintf(NULL, ...)` 触发**段错误**。

同目录的 `mc_gate_run.cpp` 在同样位置有 `if (f)` 保护，唯独此处漏了。
移植到本仓库时已修复：先 `std::filesystem::create_directories` 建目录，
再检查 `fopen` 结果，失败时打印错误并返回非零退出码。

## 移植说明

本实验的原始开发发生在另一条独立历史（`main-mac-archive` 分支）上。
由于该分支与 `main` 无共同祖先，本次采用**仅取文件内容**的方式移植
（`git archive` 解包），不带入任何提交历史。

移植物仅限 `mc_gate_*` 前缀的文件，未触碰 `main` 上既有的
`performance_analytics` 源码。已实测：配置通过、35 个目标注册成功、
全量编译零错误、20 个自检全过、5 类驱动可实际运行。

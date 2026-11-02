# MC-GATE-VALIDATION v1.4 验收包（逐卡可判）

> 面向"由你逐卡验收（编译 / 单测 / AC 项）"。
> 全部输入（每张卡的 prompt、build_cmd、verify_cmd、验收脚本、期望样本）已归档在
> `configs/mc/cards/`，验收可重放。

---

## 1. 一条命令验收

```bash
tools/run_mc_gate_acceptance.sh
```

它做四步：

| 步骤 | 内容 | 期望 |
|---|---|---|
| A. L0 自检 | 用归档的"已知正确样本"跑 10 个验收脚本 | 10 项 PASS（证明**验收脚本本身**判得对） |
| B. 构建 | cmake 配置 + 构建全部 15 个 mc_gate 目标和 6 个 app | PASS |
| C. 单测 | `ctest -R mc_gate` | 15/15 Passed |
| D. 驱动 | 跑 smoke 驱动 + 集成驱动 v2 + 3 个诊断探针 | 5 项 PASS |

当前实测输出：**PASS 17 / 17**（A 10 + B 1 + C 1 + D 5）。

跳过构建用 `--skip-build`；换构建目录用 `--build-dir DIR`。

---

## 2. 逐卡验收表

| 卡 | 产物 | 覆盖的 AC | 验收方式 | 关键实测证据 |
|---|---|---|---|---|
| C01 | `mc_gate_rng.h` | AC-06、AC-07 | `ctest -R test_mc_gate_rng` | PCG64 golden 16/16 逐位命中；`T5VAR 1.0014`、`T5KURT 5.8388`、`SVB 0.70710678`、`SVSKEW −2.5055`、`GARCHVAR 1.0025` |
| C02 | `mc_gate_dgp.h` | AC-07 | `ctest -R test_mc_gate_dgp` | 四族方差 1.83–2.03e−4、相关系数 0.243–0.257；tau 计数 0/1/25/25/25/60 |
| C03 | `mc_gate_truth.h` | AC-08（公式部分） | `ctest -R test_mc_gate_truth` | `NORMPPF` 四位分位 1e−15 级命中；`TEFF` 9×4 列全中；`CKAPPA` 4 行；`ZCELL` 恒等 0 |
| C04 | `mc_gate_baselines.h` | AC-10、AC-11（实现部分） | `ctest -R test_mc_gate_baselines` | SR 4/4 命中；B0–B4 判定全中；B4 PBO = 2/6 = 0.3333 |
| C05 | `mc_gate_decide.h` | AC-12、AC-19、AC-28（实现部分） | `ctest -R test_mc_gate_decide` | `GATE_THR` 3/3、`DECIDE` 13/13；四模式判定分离正确；**静态检查无 truth 符号（AC-19）**、**无计时符号（AC-31）** |
| C06a | `mc_gate_adapter.h` | AC-19 | `ctest -R test_mc_gate_adapter` | `SR` 20/20、`M=0.362071`、阈值 5.325897；Full 族拒 / RawT 选 / Abort 弃权 |
| C06 | `mc_gate_runner.h` | **AC-05**、AC-18（部分） | `ctest -R test_mc_gate_runner` | 分片行数 72 / 各 18 / 36+18+18；**`0/1` 与 `0/4..3/4` 拼接、`0/3..2/3` 拼接均逐字节一致**；同 (cell,rep) 六规则指纹相同、跨 rep 互异 |
| C07 | `mc_gate_scorer.h` | **AC-20**、**AC-23**、AC-21 | `ctest -R test_mc_gate_scorer` | `WILSON` 6/6（k=0 下界恰为 0、k=n 上界恰为 1）、`MCNEMAR` 4/4；`V1/V2/VU/V3` 与 `HIGH_ABSTAIN`/`OCN_CONSTRAINED` 全中 |
| C08 | `mc_gate_output.h` | **AC-03**、AC-04（生成本身） | `ctest -R test_mc_gate_output` | `CRC 0xCBF43926`；gzip 头 10 字节恒为 `1f 8b 08 00 00 00 00 00 00 ff`（**mtime=0**）；系统 `gzip -t`/`-dc` 往返一致 |
| C09 | `mc_gate_onc.h` | AC-30（机制部分） | `ctest -R test_mc_gate_onc` | 两组结构 → `OK cc=2 assign=0011`；等相关 0.5 → `NO_VALID_PARTITION`（**复现 §5.3 机制**）；三组 6 资产择优 `cc=3`；`SEEDMOD 4 = 1 1 3 3` |
| C10 | `mc_gate_budget.py` | **AC-01**、AC-33（规则部分） | `python3 tools/mc_gate_budget.py` | 从 config 推出 269/194/617000（**源码不含这三个字面量**）；删 NEG 块 → 260/185/608000；`N=3` 桶成本 99 也不降 |
| C11 | `mc_gate_reduce.py` | AC-15（归并部分） | `python3 tools/mc_gate_reduce.py` | 乱序双 shard 归并、聚合率、重复键、倒序再归并一致 |
| C12 | `mc_gate_validation.cpp` | AC-03、AC-04（端到端） | `./build/mc-gate/performance_analytics/mc_gate_validation` | 1200 条记录；`GATE_SEL 47/7/11/8`（与独立 probe 逐一相同）；gzip 合法、mtime=0、1200 行、两次运行 md5 相同 |
| C13 | `mc_gate_manifest.h` | **AC-02**、AC-32（部分）、AC-33（规则） | `ctest -R test_mc_gate_manifest` | manifest/metrics 的四个哈希字段同源一致（负向对照已验证改一个即挂）；键序固定、无时间戳；`budget_threshold_ms` 按 `%.6f`；降档表 8000→4000/2000→1000/1000→500、`N=3` 豁免；阈值 1.0 → `planned_calls=9500`，阈值 100 → 14000（与 `tools/mc_gate_budget.py` 的 Python 实现对得上） |
| C14 | `mc_gate_sha256.h` | 支撑 AC-02/AC-34 的哈希能力 | `ctest -R test_mc_gate_sha256` | 6 个权威 golden 向量全中：`""`、`"abc"`、56 字节、1,000,000 个 `'a'`、字节序列 0..255、以及该序列重复 4 次；**构建带 `-Werror` 且零警告** |
| C15 | `mc_gate_replay.h` | **AC-34** | `ctest -R test_mc_gate_replay` | 写 3 个数据集 → 目录里恰好 3 个 `dataset_*.bin`（**不是每 rep 一个**）；文件 = 小端 f64 数据段 + 32 字节 SHA-256；三个数据段的哈希与独立工具计算逐一相同；把数据段第 15 字节翻转后 `verify` 返回 0（损坏能被检出） |
| C16 | `mc_gate_cost.h` | AC-32（cost_profile 一半）、AC-33（汇总部分） | `ctest -R test_mc_gate_cost` | 由 3 个 (T,N) 点 × 3 个权重复算出单核 `176175.600` ms（与独立 Python 一致）；分项占比 dgp 0.445533 / m 0.464928 / ocn 0.056003 / decide 0.033549；`over_threshold=1`；规范 JSON 键序、`%.3f`/`%.6f` 格式逐字符命中 |
| C17 | `mc_gate_run.cpp`（集成驱动 v2） | AC-32（cost_profile 部分）、AC-33（链路部分） | `./build/.../mc_gate_run` 或 `ctest` 之外的 app | PLAN 8000/1000/500/500、planned_calls 10000；`COST total 9046.800`、shares 0.530840/0.328359/0.101406/0.039439；`metrics.json` **含 cost_profile**；1200 条 jsonl；`replay/dataset_0.bin` 4064 字节；WALL 非零 |
| C18 | `mc_gate_wall.h` | AC-32（wall_by_rule 的序列化） | `ctest -R test_mc_gate_wall` | 六条规则键序固定为 GATE_Full/B0..B4；缺失规则补 `0.000`；非法 rule_id 被忽略；单行无空格；8 条负向对照（键序错、缺规则、浮点格式、未补 0、非法规则被输出、有空格、缺 DONE、多一行）全拦 |

**"验收方式"里的 ctest 名对应 `performance_analytics/tests/mc_gate/test_<name>.cpp`。**

---

## 3. AC 覆盖矩阵（诚实的完成度）

图例：**✓ 已实现且已验** ／ **◐ 部分实现** ／ **✗ 未实现**

| AC | 状态 | 说明 |
|---|---|---|
| AC-01 | **✓** | C10：`cells/datasets/calls` 全由 config 推导，源码无硬编码 |
| AC-02 | **✓** | C13 输出并校验四个哈希字段的同源一致；`binary_sha256` 所需的 SHA-256 实现由 C14 提供（已知文件/配置即可算真值） |
| AC-03 | **✓** | C08：无时间戳/主机名/耗时；gzip mtime=0（逐字节验证头 10 字节） |
| AC-04 | **✓** | C12：每行含 `cell_id/rep_id/rule_id/selected/index/abstain/returns_sha256` |
| AC-05 | **✓** | C06：`0/1` 与 `0/4..3/4`、`0/3..2/3` 拼接均与整体逐字节一致 |
| AC-06 | **✓** | C01：自实现 PCG64，golden 16/16 逐位命中；无 `std::normal_distribution` |
| AC-07 | **✓** | C02：`garch` 的 `omega=1−α−β`、`corr≡rho`；`shortvol` 偏度 −2.5055、方差比 1.0025 |
| AC-08 | ◐ | C03 实现了 `z_cell` 公式与 `T_eff`；但**未在整个网格上跑** τ 回收与 `Σz²≤242.8` 的判据 |
| AC-09 | ✗ | canary 单元未实现。定档建议见发现 F2：`(T=100,N=126,S=4.0)` 实测功率 0.96 |
| AC-10 | ✗ | `NEG` 的零选中率 ≥99.5% 未跑 |
| AC-11 | ✗ | 解析锚（B0/B1 的 FPR 带）未实跑比对 |
| AC-12 | ◐ | C05 的 13 个 `DECIDE` 场景覆盖了四模式分离；"5 布尔全真 ⇒ candidate==winner" 未单列断言 |
| AC-13 | ✗ | 门槛一致性（ALT 9 点）未跑 |
| AC-14 | ✗ | Python 独立复算（BH/DSR/T_eff/六规则）+ `replay/` 120 矩阵比对未做 |
| AC-15 | ◐ | C11 提供归并与聚合；"从 jsonl 重算表 1–5 与 metrics 逐格相等"未接 |
| AC-16 | ✗ | 置换 `N_hold` 后决策不变，未测 |
| AC-17 | ✗ | 置换 `tau` 后决策不变，未测 |
| AC-18 | ◐ | C06 已断言"同 rep 六规则同指纹"；与 manifest 的比对随 AC-02 一起未做 |
| AC-19 | **✓** | 静态检查：`mc_gate_decide.h`/`mc_gate_adapter.h` 不含 truth 符号 |
| AC-20 | **✓** | C07：Wilson 区间，判据用 `ci_lower ≤ p ≤ ci_upper` |
| AC-21 | **✓** | C07：`invalid→V3 / undecided→VU / dominated→V2 / else V1` + 两个 flags |
| AC-22 | ✗ | 图与"每个数字可追溯到 cell_id"未实现 |
| AC-23 | **✓** | C07：`verdict/flags/abstain_rate/Δ` 全部机械计算，无手填 |
| AC-24 | ◐ | 已实测单核 25.4 min（发现 F5）；未落成 `metrics.json` 里的正式判据 |
| AC-25 | ✗ | smoke `--strict` + 60 s 判据未实现 |
| AC-26 | **✓** | 全部头文件只依赖标准库（零第三方） |
| AC-27 | ✗ | `BYA/SWP/ABL/RET-K1` 与父数据集 `returns_sha256` 逐位相同，未测 |
| AC-28 | ✗ | `artifact_hash` 未实现 |
| AC-29 | ✗ | `收益 ≤ −1.0` 记 ABSTAIN 的处理未实现 |
| AC-30 | ◐ | C09 实现了 ONC 机制并复现 §5.3；`OCNP` 的 9 单元 × 200 reps 实跑未做 |
| AC-31 | ◐ | 静态部分 ✓（decide 所在 TU 无计时）；`wall_by_rule` 的分项落盘未实现 |
| AC-32 | ◐ | `cost_profile` 已能产出并写进 `metrics.json`（C16+C17）；`wall_by_rule` 的**序列化已就绪**（C18），但驱动尚未把六规则墙钟按规则分别累计并落盘 |
| AC-33 | ◐ | C17 已把 cost_profile → `plan_units` → `planned_calls` 写进 metrics（12000→10000）。但 smoke 实跑的 reps=50 不等于计划 reps，AC-33 的"最终 reps 必须等于推导值"要在真实 run 上才成立 |
| AC-34 | **✓** | C15：`replay/` 每数据集一文件（3 个数据集 → 恰好 3 个文件），文件内为小端 f64 + 32 字节 SHA-256，损坏可检出 |
| AC-35 | ✗ | 双轨门槛（analytic/empirical）未实现 |

**统计：✓ 14 项、◐ 8 项、✗ 13 项。**

结论：**算法核心（RNG/DGP/真值/基线/决策/适配/编排/Wilson-McNemar/ONC/gzip 落盘/预算推导）已实现并逐卡验过**；
**报告与运行层（manifest、metrics 聚合、replay、图、canary、置换检验、双轨门槛）尚未实现**。

---

## 4. 已实测的跨平台结论（影响 AC-05/18/28 的口径）

| 项目 | 结论 |
|---|---|
| 10 个 selftest | 9 个在 macOS/clang 与 Linux/g++ 上**逐字节一致** |
| `runner` 的 72 条记录 | 决策列（`selected/index/abstain/reason`）**差异 0**；`returns_sha256` **72/72 不同** |
| FPR 探针 | 两平台**逐格相同**（T=4/rho=0 都是 0.935） |

⇒ **AC-05/18/28 的确定性范围应明确写成"同机同二进制"**；跨机比对只能用决策级统计量，不能用字节哈希。
详见 `docs/mc_gate_v14_findings.md` F4。

---

## 5. 需要人工裁决的四条（见 findings）

| 编号 | 事项 | 建议 | 代价 |
|---|---|---|---|
| **F1** | §7.2 的 α 网格上界 0.2 不足 | 扩到 **0.6**（实测最大需要 0.495） | 低：只改 §7.2 的网格定义 |
| **F2** | canary `S=1.2/T=100/N=126` 只有 10% 功率 | 改成 `(100,126,4.0)`（功率 0.96） | 低：只改 AC-09 的 canary 参数 |
| **F3** | §9.1 的 `T/N` 位置颠倒 | **下一轮优先改** | 高：需重算 §9.2 九行 + C03/C05 的 golden |
| **F5** | v1.3 的 4.3 h 预算高估 10.1 倍 | 已由 v1.4 的 §5.7/§5.8 改写覆盖，无需再改 | 无 |

---

## 6. CI 兼容性（`-Werror`）

仓库的 `ci-gcc` 预设开启 `QBT_WARNINGS_AS_ERRORS=ON`。已验证：

```
cmake -B /tmp/werror -DQBT_ENABLE_PERFORMANCE_ANALYTICS=ON -DQBT_WARNINGS_AS_ERRORS=ON
cmake --build /tmp/werror -j8   → WERROR_BUILD_ALL_OK
ctest -R mc_gate                → 15/15 Passed
```

两处需注意（均已在 `performance_analytics/CMakeLists.txt` 里注释说明）：

| 位置 | 处理 | 原因 |
|---|---|---|
| 全部 mc_gate 目标 | `-Wno-pedantic` | `mc_gate_rng.h` 用 `unsigned __int128` 实现 PCG64 |
| 全部 selftest 目标 | `-Wno-unused-variable` | C01 的 selftest 内有一个未使用局部变量（模型产物）；重生成该头需连带重跑全部 golden 向量。该告警仅在定义 `MC_GATE_*_SELFTEST` 时出现，库本体不受影响 |

C14 则是**真正零警告**：它的卡在 `build_cmd` 里直接带了 `-Werror`，模型首轮即通过。

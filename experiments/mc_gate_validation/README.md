# MC-GATE-VALIDATION：Monte Carlo 门控验证实验

对一个**策略族治理门控**（`policy_family_governance`）做 Monte Carlo 验证：
在合成数据上**种下已知真值 α**，看门控能不能守住假阳性、能不能检出真信号、在什么条件下失效。

- 网格：**269 个决策单元 / 202 个独立面板 / 617,000 次调用**，单核 28 分钟
- 合同：[`docs/mc_gate_validation_contract_v1.md`](docs/mc_gate_validation_contract_v1.md)（v1.5.0）
- 结项声明：[`docs/mc_gate_closeout.md`](docs/mc_gate_closeout.md)

## 结论速览

| 假设 | 判定 | 关键数字 |
|---|---|---|
| H0-OCN：`T=4` 被 OCN 分区退化限制 | 成立（比预估更严重） | abstain **55–71%**（合同原估 15–35%） |
| H1a：门控守住名义 FPR | **通过** | 27 个 null 格 max **4.64%**、中位 1.35% |
| H2：等 FPR 下的效率差 | 统计层不亏、损失来自 OCN | `Δ_stat` **+13.1pp** / `Δ_ocn` **−17.2pp** / `Δ_total` **−4.1pp** |
| H3：假阳性差随 T 单调不减 | **通过** | 83.2 → 94.3 → 96.1pp |
| H4：实测门槛 = 解析门槛 | **通过** | 9 个 S 点 \|gap\| ≤ 0.036（容差 0.10） |
| H5：门控更少把诱饵当真 | **通过** | DEC 块 GATE 38.6% vs B0 100% |
| H6：跨族重试击穿门控 | 定性成立、定量偏小 | K=1 0.90% → K=10 **16.1%**（放大 17.9×，合同原预测 >40%） |

**最重要的一条**：`T=4` 的劣势**主要来自 OCN 弃权，不是 DSR 公式**——
统计层反而赚 +13.1pp。这印证了合同里"不得把缺陷算到 DSR 头上"的要求。

## 目录

```
code/      mc_gate_* 头文件（20）、可执行目标（13）、单测
configs/   网格配置（269 cell）、验收卡片（32 张）、fixtures
tools/     网格校验、预算、归并、绘图、种子碰撞、重算核对、一键验收
docs/      合同、发现记录、验收包、修订建议、结项声明
results/   全网格与 ABL 消融的聚合产出（带 SHA-256）
```

## 怎么构建

本实验是为 `performance_analytics` 模块新增的代码，不改动既有文件：

1. 把 `code/include/performance_analytics/mc_gate_*.h` 拷进模块的
   `include/performance_analytics/`；
2. 把 `code/apps/mc_gate_*.cpp` 拷进 `apps/`，`code/tests/mc_gate/` 拷进 `tests/`；
3. 把 `code/CMakeLists.addition.txt` 的内容**追加**到模块的 `CMakeLists.txt` 末尾；
4. 配置时需要打开该模块（默认关闭）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DQBT_ENABLE_PERFORMANCE_ANALYTICS=ON -DQBT_BUILD_PYTHON=OFF \
  -DQBT_WARNINGS_AS_ERRORS=ON
cmake --build build -j 8
ctest -R mc_gate          # 20 个自检
```

依赖：C++20，仅标准库（无第三方）。

## 怎么复现

```bash
# 1) 一键验收（L0 自检 + 构建 + 单测 + 驱动探针，35 项）
./tools/run_mc_gate_acceptance.sh --build-dir build

# 2) 全网格（269 cell / 617,000 调用，约 28 分钟单核）
./build/performance_analytics/mc_gate_full --out runs/full_$(date +%Y%m%d)

# 3) 从原始记录重算聚合，核对一致性
python3 tools/mc_gate_recompute_check.py runs/full_YYYYMMDD
```

产物：`aggregates.jsonl`（每 cell×规则一行）、`decisions.jsonl.gz`（逐条决策）。

**可复现性**：同一二进制连跑两遍，aggregates 与 decisions **逐字节一致**。
但 `returns_sha256` 因 `libm` 的 ulp 差异不可跨机钉死。

## 已知局限

1. **garch 的功率数字偏乐观约 10%**：`SR̂=μ̂/ŝ` 的有限样本正偏被波动聚集放大
   （N=126 时 +0.157，N=520 时 +0.069）。见 `docs/mc_gate_ac08_truth_recovery.md`。
2. **`D=202` 未兑现**：驱动未实现 `calls_per_rep`，实际独立面板 **193**，
   差额 9 = `RET` 的 `K=10` 辅助面板。
3. **ABL 消融只在 `T=100,N=266,rho=0.3` 上做过**（40 格）。
4. 剩余 4 项 AC 未闭合，逐条理由见 `docs/mc_gate_closeout.md` §三。

## 一个实现层面的教训

构建过程中抓到 6 个真实缺口，其中两个同类且最危险：

- ABL 的 `variant` 列没接进驱动 → **消融从未真正执行**，字段在但值是占位；
- `returns_sha256` 一律写 0 → "同 rep 六规则指纹相同"成了"零等于零"的空断言。

**字段填了占位值比字段没填更危险，因为验收器会通过。**

## 附：模型能力边界

本实验的 32 个卡片文件全部由本地模型 `qwen3.8:27B-IQ3_XXS` 施工完成，
经人工逐卡验收（编译 / 单测 / AC）。**卡作者缺陷 29 次，模型真实缺陷 1 次**
（SHA-256 padding 的循环不变量）。作业规范见 DSH 技能 `local-model-card-ops`。

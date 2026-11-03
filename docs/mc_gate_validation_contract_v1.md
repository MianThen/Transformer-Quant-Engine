# MC-GATE-VALIDATION-V1：种植 alpha 的 Monte Carlo 门控验证合同

状态：**DRAFT v1.5.0（待冻结）**。冻结后本文件 SHA-256 写入 `runs/mc_gate_<runid>/manifest.json` 与
`metrics.json.contract_hash`；冻结后不得修改任何判据、阈值、网格或样本量。

修订留痕见 §18（v1.0→v1.1：21；…；v1.4.2→v1.4.3：4；v1.4.3→v1.4.4：2；v1.4.4→v1.4.5：2；v1.4.5→v1.5.0：4）。

> v1.4 只改**口径与测量纪律**，不改任何科学判据、阈值、网格规模或 OCN spec。依据与论证见
> `docs/mc_gate_v14_revision_decision.md`。
> v1.4.1 补 **§3.6 种子树**；v1.4.2 补 **`ds` 复制号与 `panel_ref` 显式化**，`df` 由 194 更正为 **202**；
> v1.4.3 消除**记法歧义**（`T`/`N` 列与 `f_t` 维度）、定义 `composition` 落格规则、把 `run_seed`
> 升为 config 字段。判据、网格、阈值零改动。
> **v1.5.0** 按全网格实测（269 cell / 617,000 调用 / 28 分钟单核）修订 **4 条假设判据的预期值**
> （H0-OCN / H2 / H4 / H6）与 §5.4 的归因阈值。**判据的形式、网格、种子树、门槛公式零改动**；
> 依据见 `docs/mc_gate_full_run_results.md` 与 `docs/mc_gate_v15_revision_proposal.md`。

拟执行窗口：10 个工作日。对应投稿行动项：`docs/paper_submission_probability_report_20261025.md` §五 第 1 条。

---

## 0. 目标

用**已知真值**的合成策略家族，测量预注册门控在"选策略"上的**家族假阳性率**、**等假阳性下的
检出率**、被选中策略的**部署 Sharpe**，与「事后分析者」在同一批数据上配对比较，
给出可判定结论（V1 有效 / V2 有效但被支配 / V3 无效 / VU 未判定）。

---

## 1. 被检验的对象

| 事实 | 位置 |
|---|---|
| 入口 `evaluate_policy_family_governance(input, spec)` | `performance_analytics/include/performance_analytics/policy_family_governance.h:60` |
| 选中：`fdr.rejected[i]!=0` **且** `dsr.one_sided_p_value<=q` **且** 取 `raw_sharpe` 最大 | `src/policy_family_governance.cpp:161-170` |
| `winner_frozen` 需 5 布尔全真；`promotion_eligible` 恒 false | 同上 `:171-175` |
| 输入下限 `trials>=3`、`observations>=minimum_observations` | 同上 `:112-113` |
| **`fdr_method` 缺省 `BENJAMINI_YEKUTIELI`** | `include/.../policy_family_governance.h:19-20` |
| 库内唯一调用方 `phase4b_governance_reference.cpp:37-49` 未覆盖该字段 → 参照跑的是 BY | 该文件 |
| **`T_eff = clamp(T/(1+(T−1)·mean|r̂|), 1, T)`**，与 OCN 无关 | `src/return_analysis.cpp:353-369` |
| **OCN 只是前置状态门**：失败 → 整次 abstain，聚类结果不进 DSR 紧缩 | `src/policy_family_governance.cpp:129-131` |
| **DSR 门槛是渐近式** `sqrt(2·ln(T_eff)/N)` | `src/return_analysis.cpp:406-411` |
| `compute_deflated_sharpe_ratio` 要求**全部收益 > −1.0** | 同上 `:389` |
| 年化口径 50.4 期/年 | `portfolio_math/apps/formal_oos_driver.cpp:39` |

可测量输出 = `artifact.candidate_selected`。5 个门控布尔全置真，并要求
`candidate_selected == winner_frozen` 在 100% rep 上成立。

### 1.1 与生产/参照的 7 处差异

| # | 生产或参照 | 本实验 | 方向 |
|---|---|---|---|
| 1 | `fdr_method=BY` | `BH` | BH 宽松 → 对门控有利；`BYA` 块量化 |
| 2 | `onc_spec` 夹具 2/2 | 结构体默认 2/4 | 差异见 §5.3（这是缺陷源，不改） |
| 3 | `primary_p_values` = 配对 p | 自身均值为正的单侧 t | 略宽松 |
| 4 | 组合含优化器与成本 | 直接给定收益（Tier A） | 剥离优化器噪声 |
| 5 | 3 子窗口方向一致性 | 不施加 | 生产更严 |
| 6 | 真实 PIT 数据 | 合成 DGP | 唯一能拿 τ 当真值 |
| 7 | 家族规模 4 | 4/20/100 | 有意扩展 |

### 1.2 数据层边界声明（v1.4 新增，防误用）

本实验的链路是 `PCG64 → 合成收益矩阵 R(T×N) → 门控调用 → 决策落盘`，**不读 Parquet 数据湖、
不经 DuckDB/Arrow、不进 `cpp_engine` 主循环**。因此：

- 2026-09-11 全湖评测报告（`backtest-engine-bottleneck-is-data-layer.md`）的 P0（文件压实）、
  P1（桥接层/列式回调）、P2/P3 结论，**对本实验的运行时无任何影响**。
- 该报告中的 `51,216 行/秒`、`17.9 MB/s`、`341µs/文件`、`引擎利用率 0.7%` 等数字**不得**用于
  给本实验估算成本或作为降级依据。本实验的成本只能由 §5.8 的 `COSTP` 实测得到。
- **禁止**为了让本实验跑得更快而先去压实数据湖：那是把 4 小时内的工作换成数周的工作，且收益为零。
- 若 §15 D9 的 **Tier B** 需要喂真实收益，则**继承**该报告的约束（供给速率 5.1 万行/秒上限、
  引擎利用率 0.7%）；Tier B 的可行性前提是先完成 P0 压实，且立项时必须按真实链路重新估算预算。

---

## 2. 识别策略

1. **同一数据对象**：每 rep 生成一次 `R`，六条规则（GATE + B0…B4）读同一份；`manifest` 记
   `returns_sha256`，断言六条一致。
2. **只改决策逻辑**：`decide()` 只收 `(returns,T,N,config)`；`ResearchWindow` 无 `tau` 字段。
3. **独立部署期**：另生成长 `N_hold=max(2000,10N)` 的独立路径（独立 RNG 流），决策阶段不读。
4. **CRN**：同 rep 六规则共享同一路径，规则间用 McNemar 配对。

---

## 3. 数据生成

### 3.1 收益过程

```text
p=50.4; scale=sqrt(p); sigma_p=0.10/scale; mu_p=S*sigma_p/scale
r_{i,t} = mu_p + sigma_p*sqrt(rho)*f_t + sigma_p*sqrt(1-rho)*e_{i,t}
```

**下标约定（v1.4.3 明确，此前歧义）**：

- `i ∈ 1..T` = **策略**下标（截面维，T = trials）；
- `t ∈ 1..N` = **期**下标（时间维，N = observations）；
- 因此 `f` 只有 **N 个分量**（一期一个，所有策略共用），`e` 有 **T×N 个**（策略 × 期）。
  `r` 的形状是 **`T × N`**（trial-major，与 `PolicyFamilyGovernanceInputV1::trial_major_returns` 一致）。

> 易错点：文档里 "trial"（策略）与下标 `t`（期）不是一回事。为避免混淆，下文一律写
> "策略"或 "trial"，**不再用 `t` 表示 trial**。

**抽样顺序即语义（AC-39）**：先抽 `f_1 … f_N`（N 个），再按 `i = 1..T` 逐策略抽
`e_{i,1} … e_{i,N}`（共 T×N 个）。**禁止**按列优先、禁止按策略并行取流、禁止先抽 e 后抽 f——
顺序一变，同一 seed 就得到不同面板。

> **两种错法，性质不同，别混为一谈**：
> 1. **矩阵朝向转置**（`R` 存成 `N×T`、`N`/`T` 变量接反）：模型没变，只是搬运方向错。
>    后果是 `observations` 变成 T、`effective_trials` 变成 N，`correlation_matrix` 尺寸反过来。
>    → 公式不用改，改实现 + AC-41 的维度断言即可。
> 2. **`f` 按策略索引**（把 `t` 当 trial，`f` 只有 T 个）：这不是朝向问题，是**模型错了**。
>    此时 `f_i` 退化成每条策略的常数平移，`corr(r_i,r_j)=0`（与设定的 `rho` 无关），
>    且每条策略的实测均值被随机平移，**τ 不再是已知量**。
>    → 自检：实测 `mean|r̂|` 必须 ≈ `rho`；若 ≈0 而 `rho=0.3`，就是踩了这个坑。
>    （`corr` 对加常数不变，所以这种错不会在相关矩阵上"看起来异常"，只会让 `rho` 失效。）

相关系数由 `sqrt(rho)` / `sqrt(1-rho)` 的拆分精确构造：`corr(r_i,r_j)=rho`（i≠j）、
`Var(r_i)=sigma_p²`，无需事后标准化。（v1.4.3 删去旧文残留的 `b_i` 记号——它在公式里已被显式代入。）

#### 3.1.1 RNG 的播种与输出规格（v1.4 新增，AC-06/AC-14 的前提）

AC-06 要求"自实现 PCG64 + 前 1e6 输出自检"，AC-14 要求跨语言逐位一致。两者都要求**先把播种与输出
函数钉死**，否则每个实现者产出不同的流，任何逐位比对都无从建立。规格如下（已用 stdlib-only Python
实现并逐位验证，见决策文档 §4）：

```text
状态: 128 位，hi/lo 各 64 位
推进: state ← state·MULT + INC  (mod 2^128)
MULT = 0x2360ED051FC65DA4_4385DF649FCCF645
INC  = 0x5851F42D4C957F2D_14057B7EF767814F

播种(seed):                     # 对齐 pcg_setseq_128_srandom_r(seed, seed)
  state = 0; inc = (seed<<1)|1  (mod 2^128)
  step(); state += seed (mod 2^128); step()

输出:                           # 对齐 pcg64_random_r 的 XSL-RR
  old = state; step()
  xs  = (hi_old XOR lo_old) & (2^64−1)
  out = rotr64(xs, hi_old >> 58)        ← 右旋（ROTR），不是左旋
```

**C++ 陷阱（必须写进实现）**：128 位无符号整数的自然溢出**就是** mod 2^128，**不需要**掩码常量；
写 `(1u128<<128)-1` 会因移位计数 ≥ 位宽而报错/未定义。

**golden 向量**（AC-06 自检基准；`g++ -std=c++20 -O2 -Wall -Wextra` 零警告通过）：

```text
seed=42    前 8 个 next():
  3eb27489f5ff73e0 287472e87ff5705a bbd190b04ed0b545 b6cee3580db14880
  bf5f7d7e4c3d1864 734eedbe7e50bbc5 a5b6b5f867691c77 6d6d2656235610ad
seed=12345 前 8 个 next():
  78861cdd32fcc735 d11b1d37bff50104 92e63e7ae540560c c8c14150ef8defaf
  ae4f5c25c864b6b7 e357b1c96f85143e ff6ed2a6bb06b6ac daf4f1cea021e3a7
```

`next_double()` = `(next() >> 11) * 0x1.0p-53`；Box–Muller 用固定两支顺序（先 cos 后 sin），
不得用 `std::normal_distribution`。


### 3.2 分布族

| 档 | 构造 | 作用 |
|---|---|---|
| `gauss` | `f,e ~ N(0,1)` | 实现正确性锚 |
| `t5` | 标准化 t(ν=5)（超额峰度 6） | **主口径** |
| `garch` | 见 3.2.1 | 波动聚集 |
| `shortvol` | 见 3.3 | 负偏压力 |

#### 3.2.1 `garch` 的截面构造（v1.2 未定义，此处钉死）

**共同波动状态**，因子与特质共用同一个 `sigma_t`，从而 `corr` 在每一期都恒等于 `rho`：

```text
sigma_t^2 = (1-alpha-beta) + alpha*a_{t-1}^2 + beta*sigma_{t-1}^2,  alpha=0.08, beta=0.90
a_t       = sigma_t * z_t,   z_t ~ 标准化 t5
r_i,t     = mu_p + sigma_p*sigma_t*( sqrt(rho)*z_f,t + sqrt(1-rho)*z_i,t )
```

- `E[sigma_t²] = 1`（由 `omega = 1−alpha−beta` 保证）⇒ **无条件 `Var(r_i)=sigma_p²`**；
- 每期条件协方差 `= sigma_p²·sigma_t²·rho` ⇒ **`corr_t ≡ rho`**，与 `sigma_t` 无关；
- 已知偏差：`E[sample SR] < S`（Jensen），故该档只作**方向性**引用，τ 回收只查 `gauss/t5`。

### 3.3 `shortvol`

```text
r_t = mu_p + sigma_p*(a*u - b*w)/sqrt(a²+2b²),  u=z, w=z²-1, z~N(0,1)
skew = (-6a²b - 8b³)/(a²+2b²)^{3/2};   下确界 = -8/2^{3/2} ≈ -2.83
```

目标 `skew=-2.5`（-3 不可达）：`a=1`，对 `b` 二分，再缩放使 `Var=sigma_p²`。
构建期断言 `|skew+2.5|≤0.05`、`|Var/sigma_p²−1|≤1e-6`。

### 3.4 家族组成

| composition | 定义 | 用途 |
|---|---|---|
| `null` | 全 τ=0 | FPR |
| `one_good` | 1 条 τ=S（位置 κ 由 seed 定） | 检出率 |
| `mixed` | `k=ceil(T/4)` 条 τ=S | FDR/精确率 |
| `mixed_signed` | `k` 条 τ=+S + `k` 条 τ=−S 诱饵 | 部署代价 |
| `all_negative` | 全 τ=−0.6 | 负对照（仅断言 GATE） |

**`composition` 是单值字段（v1.4.3 明确）**：它进 `dataset_key`，一个 cell 只能有一个值。
§5.1 块表里出现的 `null+one_good` 是**块级简写**，落格规则固定为：

```text
S == 0  -> composition = null       （τ 全零，用于 FPR）
S != 0  -> composition = one_good   （一条策略 τ=S，用于 power）
```

同一块内不同 `S` 会落到不同 composition，因此是不同面板（也就不同 `ds`/key）。
`RET` 的 `K∈{1,10}` **不是 composition 维度**：它是一个 family 数（`ds` 的个数），
每个 family 自身仍是 `null`。

### 3.5 真值自检（含峰度修正与多重性预算）

```text
Var_single(SR_ann) = (p + c_kappa*S^2)/N,   c_kappa = (kappa+2)/4
    gauss: kappa=0 -> c=0.5
    t5   : kappa=6 -> c=2.0
    garch: 取 kappa=12（保守）-> c=3.5
    shortvol: 取 kappa=15（保守）-> c=4.25
SE_mean(R) = sqrt(Var_single/R);   z_cell = (mean_reps(sample_SR) - S)/SE_mean
```

> t5 在 `S=1.8` 时 `Var` 比高斯高 **9.3%**（SD 高 4.6%）；gauss/t5 两档的 `c_kappa` 是解析值，
> garch/shortvol 取保守上界（使检验偏宽松，避免误杀）。

- **单格**：`|z_cell| ≤ 4.5`（单格 `p=6.8e-6`；若沿用 `|z|≤2`，`202` 个面板全过概率仅
  `e^{202·ln0.9545}=8.2e-5`，正确实现必挂）。
- **全局**：`Σ z_cell² ≤ 251.7 = chi2_{202}(0.99)`（Wilson–Hilferty），df 用**独立面板数**
  `D=202`（共享同一 `panel_ref` 的规则/q/变体不重复计数，见 §5.5）。
- 单格越界 → 该格换 seed 重抽一次（`rep_batch=1`，见 §3.6）；再越界 → 标 `dgp_unverified`，排除该格判定。
- 全局越界 → 先对贡献最大的 3 格逐格重抽；仍越界 → 标 `dgp_global_shift` 并**报告偏差量级**；
  仅当 `|mean SR − S| > 0.05` 时才作废该 dist 档（`z` 级偏差不改变 FPR/功率结论）。

### 3.6 种子树（v1.3 只写了 `seed = f(...)`，此处补全为可实现规范）

#### 3.6.1 设计约束

种子的唯一职责是让"同一份数据"在任何执行路径下都逐位相同，因此必须满足四条：

1. **与分片方式无关**：`--shard 1/1` 与 `8/8` 必须产生同一批数字（AC-05）⇒ 不能有"全局顺序消费"
   的单一流，否则改动分片数或增删一个 cell 就会平移所有后续抽样。
2. **与决策维度无关**：`q`、`fdr_method`、规则、消融变体、`K` 都不许改变数据 ⇒ 种子输入是
   **`dataset_key`，不是 `cell_id`**（`cell_id` 含 q 与块名；用它会让 `SWP` 的 5 个 q 变体各自
   生成不同收益，直接违反 AC-27）。
3. **可单点复算**：只给 `(dataset_key, rep, stream)` 就能重放出该 rep，无需跑前面的 rep。
4. **不与统计维度相关**：真策略位置 κ、诱饵位置、持有期必须来自**独立子流**，否则
   "谁是那条好策略"可能与"它的实现收益"相关，H2/AC-17 全部失效。

#### 3.6.2 规范

```text
GOLDEN = 0x9E3779B97F4A7C15

splitmix64(x):                      # 标准 SplitMix64 终结器
  z = (x + GOLDEN) mod 2^64
  z = (z xor (z >> 30)) * 0xBF58476D1CE4E5B9  mod 2^64
  z = (z xor (z >> 27)) * 0x94D049BB133111EB  mod 2^64
  return z xor (z >> 31)

fnv1a64(s):                         # 与仓库既有哈希同族（governance/return_analysis 用 FNV-1a）
  h = 0xCBF29CE484222325
  for each ASCII byte b of s: h = ((h xor b) * 0x100000001B3) mod 2^64
  return h

seed(dataset_key, rep, batch, stream) =
  splitmix64( fnv1a64( dataset_key + "|rep=" + rep + "|batch=" + batch + "|stream=" + stream )
              xor splitmix64(RUN_SEED) )
```

#### 3.6.3 `dataset_key` 的规范化（格式错一位就复现不了）

```text
"v1|{composition}|S={S:.4f}|N={N}|T={T}|rho={rho:.2f}|{dist}|p={p:.1f}|ds={ds}"
```

- 固定 4 位 / 2 位 / 1 位小数，ASCII 定点格式，C locale（禁 `std::to_string` 的区域敏感路径）；
- **`ds`（dataset slot）= 面板复制号**，默认 `0`；只有"同一统计身份需要多份独立面板"时才用
  `1..K−1`。当前仅 `RET` 的 `K=10` 用 `ds=0..9`。**它是 key 的一部分**，因此
  `null|…|ds=0` 与 `null|…|ds=3` 是两份不同的面板，seed 自然不同；
- **composition 与 S 进入 key**：`null` 与 `one_good` 各自独立噪声（不做跨 S 的 CRN），
  换来 `df` 计数干净（§5.5 的 `D=202`）。反过来的方案（共享噪声、跨 S 配对）方差更小，
  但会让 `chi2` 的自由度无法界定，本实验不采用；
- **不进 key 的维度**：`block`、`rule_id`、`q`、`fdr_method`、`abl_variant`、`K`
  —— 这些必须复用同一份数据（AC-27）；
- `p` 进 key：`FRQ` 的 `p=252` 与主口径 `50.4` 是不同面板。

#### 3.6.3.1 "复用"必须显式声明，不能靠 key 恰好相等（回答"复制号"问题）

key 只能保证"同样的 key ⇒ 同样的面板"，**不能保证两个 block 是故意用同一份面板**。若某天
格式串改一个字符，`ABL` 与 `ALT` 会静默分裂成两批面板，消融对比立刻被混淆，而任何哈希检查都
不会报警（两边各自自洽）。因此配置层必须显式：

```text
每个 cell 声明 panel_ref（默认 = 自己的 dataset_key 字符串）
实现只允许按 panel_ref 取面板，不允许按 block 名取
metrics.json 每 cell 记录 {cell_id, panel_ref, returns_sha256}
```

于是"能不能复用"变成一个可枚举、可断言的事实，而不是推理：

- **`ABL`/`SWP`/`BYA`/`RET-K1` 必须复用**：它们是决策层变体，换数据就把消融与扫描污染了；
  其 `panel_ref` 指向 `NUL`/`ALT` 的对应 key，`returns_sha256` 必须逐位相等（AC-27）。
- **`RET` 的 `K=10` 必须新增面板**：10 个"候选家族"是 10 份独立数据，用 `ds=0..9` 表达；
  其中 `ds=0` 恰与 `NUL` 同 key（所以只有 9 份是新的）。
- **`D` 不写死，由 config 枚举 `distinct panel_ref` 得到**；AC-01 校验"计算结果 = config 推导值"，
  而不是校验一个常数。当前值为 **202**（§5.5）。

#### 3.6.3.2 为什么不用"null 面板 + 在 κ 上加 τ"来构造 `one_good`（方案甲被否）

方案甲（同一面板加常数换 composition）在统计上很诱人：加常数不改相关结构，`T_eff` 与门槛完全
一致。但它有两个代价，其中一个直接损害主判据：

1. `one_good` 不再是"另一份备择面板"的代表性抽样，而是"空面板 + 平移"；真实世界里备择策略的
   截面相关结构与空家族不会逐位相同。**估计 power 需要代表性备择，而非配对差分。**
2. `S` 的 9 个点会共享同一批噪声 ⇒ 功率曲线的 9 个点强相关，`threshold_sr` 的 logistic 拟合
   失去标准误，AC-13 的 0.10 容差无法校准。方案乙下 9 个点独立，H4/AC-13 才有明确的误差口径。

代价是 `df` 变大且要跟着改（见 §5.5）。这是**账要记对**的问题，不是**设计要换**的问题。

#### 3.6.4 五个子流（各自独立播种，互不消费）

| stream | 内容（**顺序即语义**） | 为什么必须独立 |
|---|---|---|
| `ret` | 1) `f_1 … f_N`（**N 个**，一期一个）；2) 按 `i = 1..T` 逐策略抽 `e_{i,1} … e_{i,N}`（**T×N 个**）。合计 `N + T·N` 次取数 | 主数据；顺序见 §3.1 与 AC-39 |
| `assign` | κ（真策略下标）、诱饵下标、`mixed` 的选取 | 与收益独立 ⇒ H2/AC-17 才成立 |
| `hold` | 部署期路径（`N_hold·T`） | 决策阶段不读；使 AC-16 的"置换后决策不变"成为纯后处理性质 |
| `boot` | B4/CSCV 的重采样、α* 随机化阈值插值 | B4 不得扰动数据集本身 |
| `aux` | 预留（诊断、绘图抖动），不参与判定 | 隔离未来扩展 |

每个子流独立播种，**播种与输出完全按 §3.1.1**（`srandom(seed, seed)`、ROTR、golden 向量），
本规范只负责"seed 怎么派生"，不改 RNG 本身；不额外 warm-up、不改 `INC`。

#### 3.6.5 `rep_batch` 与重抽纪律

`rep_batch ∈ {0,1}`。0 为正式抽样；1 仅在 §3.5 单格 `|z|>4.5` 时启用，且必须：
写入 `metrics.json` 的 `cells[].rep_batch`；`decisions.jsonl.gz` 用 `rep_id = rep + 100000*batch`
保持主键唯一；报告注明该格是二批。**同一格最多重抽一次**。

#### 3.6.6 为什么不用单一全局流 / `std::seed_seq`

- 单一全局流：分片数变化 ⇒ 抽样平移；插入一个 cell ⇒ 之后全部失效；无法单点复现。三条都违反
  约束 1/3。
- `std::seed_seq` / `std::mt19937`：`seed_seq` 的算法由实现定义，跨编译器不保证一致；
  `std::normal_distribution` 同样实现定义（AC-06 已禁）。故自实现 PCG64 + 固定 Box–Muller。

#### 3.6.7 测试向量（冻结后不得变动）

`RUN_SEED = config.run_seed`（**必填字段**，冻结值 `20261027`；v1.4.3 起不再是文档常量，
见 §6.3），
`A` = `v1|null|S=0.0000|N=266|T=20|rho=0.30|t5|p=50.4|ds=0`，
`B` = `v1|one_good|S=1.2000|N=266|T=4|rho=0.30|t5|p=50.4|ds=0`，
`R` = `v1|null|S=0.0000|N=266|T=20|rho=0.30|t5|p=50.4|ds=3`（`RET` 的第 4 个家族）：

| 数据集 | rep | batch | stream | `fnv1a64(dataset_key)` | seed |
|---|---|---|---|---|---|
| A | 0 | 0 | `ret` | `0x5F3B4ADDD9ACB0CF` | `0x3A775D5BAFD53F18` |
| A | 0 | 0 | `hold` | 同上 | `0xC88D231BA9400851` |
| A | 0 | 0 | `assign` | 同上 | `0x6AACBB86EEC5B5BB` |
| B | 123 | 0 | `ret` | `0xD9A272F6FEE531F3` | `0x07807E7EEEB76B0C` |
| A | 7 | 1 | `ret` | `0x5F3B4ADDD9ACB0CF` | `0xFC35BA2F810BC1F6` |
| R | 0 | 0 | `ret` | `0x5F3B4BDDD9ACB282` | `0xA39697B8C6E3A35B` |

> 第 6 行是"复制号真的起作用"的判据：`R` 与 `A` 只差 `ds`，`fnv1a64` 与 seed 必须都不同；
> 若实现里 `ds` 没进串，这两行会相等，AC-36 立即失败。

另：`manifest.json` 记录每数据集每 stream 的 **指纹** = 该流前 3 个 PCG64 输出（十六进制），
不参与判定，只用于跨平台/跨编译器比对与定位（AC-37）。注意它与 §3.1.1 的 golden 向量分工不同：
golden 向量校验**函数本身**，指纹校验**seed→流 的映射**。

#### 3.6.8 与 OCN 的关系

`onc_partition` 的 `seeds={17,29,43,71}` 是**门控内部常量**，不属本种子树，本实验不改它
（改了就是改被测对象）。因此 §5.3 的 `T=4` 退化属于被测实现的固有行为，而非抽样伪影——
这正是要用 `OCNP` pilot 单独测量的原因。

---

## 4. 覆盖

```text
S ∈ {-0.6,-0.3, 0, 0.15,0.30,0.45,0.60, 0.90,1.20,1.80}
```

`S<0` 负对照/诱饵；`S=0` 假阳性；`[0.15,0.60]` 加密（真实策略在 0.48 与 −0.55/−0.60/−0.006）；
`[0.90,1.80]` 高信噪。

---

## 5. 网格（269 个决策单元 / 202 个独立面板 / 617,000 次调用）

### 5.1 块表

| 块 | composition | S | N（期数） | T（策略数） | rho | F/变体 | 决策单元 | 面板 | reps | 调用 |
|---|---|---|---|---|---|---|---|---|---|---|
| `NUL` | null | 0 | 126/266/520 | 4/20/100 | 0/0.3/0.7 | t5 | 27 | 27 | 8000 | 216,000 |
| `ALT` | one_good | −0.60/−0.30/0.15/0.30/0.45/0.60/0.90/1.20/1.80 | 126/266/520 | 4/20/100 | 0.3 | t5 | 81 | 81 | 2000 | 162,000 |
| `MIX` | mixed | 0.45/0.90 | 126/266/520 | 20/100 | 0.3 | t5 | 12 | 12 | 2000 | 24,000 |
| `DEC` | mixed_signed | 0.45/0.90 | 126/266/520 | 20/100 | 0.3 | t5 | 12 | 12 | 2000 | 24,000 |
| `NEG` | all_negative | −0.60 | 126/266/520 | 4/20/100 | 0.3 | t5 | 9 | 9 | 1000 | 9,000 |
| `GAR` | 按 §3.4 落格 | 0/0.60/1.20 | 126/266/520 | 4/100 | 0.3 | garch | 18 | 18 | 1000 | 18,000 |
| `GAU` | 按 §3.4 落格 | 0/0.60/1.20 | 126/266/520 | 4/100 | 0.3 | gauss | 18 | 18 | 1000 | 18,000 |
| `FRQ` | 按 §3.4 落格 | 0/0.30/0.60 | 252/504 | 4/100 | 0.3 | t5, p=252 | 12 | 12 | 1000 | 12,000 |
| `SVL` | 按 §3.4 落格 | 0/0.60 | 266 | 4/100 | 0.3 | shortvol | 4 | 4 | 1000 | 4,000 |
| `BYA` | null | 0 | 266 | 4/100 | 0.3/0.7 | t5, fdr=BY | 4 | 0（`panel_ref`→`NUL`） | 2000 | 8,000 |
| `RET` | null | 0 | 266 | 20 | 0.3 | K=1 / K=10 | 2 | **9**（`ds=1..9`；`ds=0`→`NUL`） | 2000 | 22,000 |
| `ABL` | 按 §3.4 落格 | −0.60/−0.30/0/0.15/0.30/0.45/0.60/0.90/1.20/1.80 | 266 | 100 | 0.3 | `G_full`/`G_dsr_only`/`G_fdr_only`/`G_raw_t` | 40 | 0（`panel_ref`→`NUL`/`ALT`） | 1000 | 40,000 |
| `SWP` | 按 §3.4 落格 | 0/0.60/1.20 | 266 | 4/100 | 0.3 | t5 × q∈{0.01,0.025,0.05,0.10,0.20} | **30** | 0（`panel_ref`→`NUL`/`ALT`） | 2000 | 60,000 |

**每一列都是显式取值集合，没有"个数"记法**——格数可手算，例：
`ALT` = 9 S × 3 N × 3 T = 81；`MIX` = 2 S × 3 N × 2 T = 12；`ABL` = 10 S × 4 变体 = 40；
`SWP` = 3 S × 2 T × 5 q = 30。
（v1.4.3 修正：旧表 `ALT`/`MIX` 的 N、T 列写裸 `3`、`S` 列写"9 非零"、`ABL` 写"全 10"、
`SWP` 写"5 个 q"，而 `NUL` 写显式列表——同一列两种记法，§5.5 又按"3 个取值"去读，
属于未定义的约定。所有"个数"一律展开成取值。）

**决策单元 269 个；独立面板 `D=202`；调用 617,000 次。**

- `SWP` 的 q **是 cell 维度**（它改变决策规则，必须进 `cell_id`）：6 个基础配置 × 5 个 q = 30 个单元，
  调用数不变（每单元 2000）。
- `RET` 的 K=10：每 rep 对 **10 个家族各调用一次门控**，共 20,000 次 + K=1 的 2,000 次。
- `ABL` 的 4 变体见 §5.6。
- 预算单列：B4 的 CSCV 用块级充分统计 `O(T·N + C(16,8)·T)`，T=100 实测 4 ms/rep ⇒ 约 +20 min 单核。

### 5.2 reps 与精度

`NUL` 8000（5% 处 CI ±0.48pp）；`ALT/MIX/DEC/BYA/RET/SWP` 2000；其余 1000。

**预算不再是常数**（v1.4）：总时长由 §5.8 的 `COSTP` 实测成本模型按
`Σ_cells reps × ĉ(T_cell,N_cell)` 推导，写入 `metrics.json.cost_profile`。
v1.3 的 `617k × 25ms ≈ 4.3h` 是**未测量的平坦假设**，已废弃为"待 `COSTP` 替换的量"。
降级规则见 §5.7。

### 5.3 OCN 在 T=4 的结构性 abstain（v1.3 新增，最高优先）

**机制**（已核对 `portfolio_math/src/onc_partition.cpp`）：

1. `:110` 首个 medoid = `seed % asset_count`。默认 `seeds={17,29,43,71}` ⇒ `T=4` 时
   `17%4=29%4=1`、`43%4=71%4=3`：**只有 2 个不同初始点**（T=20/100 各有 4 个）。
2. `:111-130` 后续 medoid 取"离已选 medoid 最远"的点，带 index 确定性 tie-break ⇒ 同初值的
   seed 产生**完全相同的候选**。
3. `:255-258` `max_clusters` 被钳到 `T/min_cluster_size = 4/2 = 2`；`:193-197` 每簇须 ≥2 ⇒
   **唯一合法形状是 2+2**。
4. 等相关 DGP 下 6 个 pairwise 距离全部 ≈`sqrt((1−rho)/2)`，噪声（`sigma_rhat ~ 0.09/sqrt(N)`）
   远大于 `distance_tolerance=1e-12` ⇒ 两个非 medoid 各约 50% 概率落向同一 medoid ⇒ 该候选
   收敛到 3+1 ⇒ invalid。

⇒ `T=4` 时有效独立候选只有 2 个，**单次调用 `NO_VALID_PARTITION` 的概率量级 15%–35%**，
而 `T=20/100` 接近 0。

**处理（预注册，不修 spec——修了就掩盖缺陷）**：

- 新增 pilot 块 **`OCNP`**：`T∈{4,20,100} × rho∈{0,0.3,0.7}`、`N=266`、null、200 reps
  = **9 单元 / 1,800 次调用**，在 D4 先跑。
  - **预测**：`T=4` 的 `abstain_rate ∈ [0.15, 0.35]`；`T=20/100` 的 `<0.01`。
  - 判定：pilot 结果**只影响解释与分层，不改网格、不改判据**。
- 若预测成立，`T=4` 层在报告中标注 **`OCN-constrained`**，且功率同时报两个口径：
  `power`（分母含 abstain，**主口径**）与 `power_ocn_ok`（仅在非 abstain 上算）。
- §5.4 的 `Δ` 分解中，`Δ_ocn` 就是这两者之差。

### 5.4 `Δ` 的归因分解（把 OCN 伪影与统计层分开）

```text
Δ_total(S,N,T) = power_gate - power_B1(alpha*)
              = Δ_stat + Δ_ocn
Δ_ocn  = power_gate_ocn_ok - power_gate            # ≤ 0，纯 OCN 造成的损失
Δ_stat = power_gate_ocn_ok - power_B1(alpha*)
```

报告必须给三项。**结论归因**：若 `|Δ_ocn| > 15pp`（v1.5 按实测上调，原为 5pp；`T=4` 实测 **−17.2pp**），则"被支配"的表述必须写成
"`T=4` 的损失主要来自 OCN 分区退化（`Δ_ocn`），统计层损失为 `Δ_stat`"，不得把缺陷算到 DSR 头上。

### 5.5 独立面板计数（`df` 的依据）

按 `distinct panel_ref` 逐块清点（v1.4.2 修正：旧版 `D=194` 把 `RET` 的 `ds=1..9` 漏算成 1）：

| 块 | 新增面板 | 说明 |
|---|---:|---|
| `NUL` | 27 | `3N × 3T × 3rho` |
| `ALT` | 81 | `9S × 3N × 3T` |
| `MIX` / `DEC` | 12 / 12 | `2S × 3N × 2T` |
| `NEG` | 9 | `3N × 3T` |
| `GAR` / `GAU` | 18 / 18 | `3S × 3N × 2T`（dist 不同，故为独立面板） |
| `FRQ` | 12 | `p=252` 独立 |
| `SVL` | 4 | `dist=shortvol` 独立 |
| `BYA` | **0** | `panel_ref` 指向 `NUL`（决策层变体） |
| `RET` | **9** | `ds=1..9` 新面板；`ds=0` 即 `NUL` 的 `(266,20,0.3)` |
| `ABL` | **0** | `panel_ref` 指向 `NUL`(S=0) / `ALT`(S≠0) |
| `SWP` | **0** | 同上 |
| **合计** | **202** | `chi2_{202}(0.99) = 251.7` |

复用块的 `z_cell` 必须与父面板**逐位相同**（AC-27）；它们进表 0–5 的统计，但**不进** `chi2` 的 df。

> 为什么不是 204/213：`ABL` 若不复用面板（+10），`G_full` 与 `G_dsr_only` 就落在不同数据上，
> 消融立刻被混淆——所以 `ABL` 的复用是**强制**的，不是优化。`RET` 若把 10 个家族全算新面板（+10
> 而非 +9），等于承认 `ds=0` 与 `NUL` 不同源，与 `RET-K1 == NUL` 的定义冲突。

### 5.6 `ABL` 的定位（消除 §1"不重写只调用"的矛盾）

`G_dsr_only` 与 `G_fdr_only` **无法**用 `PolicyFamilyGovernanceSpecV1` 表达（无此类开关）。
因此 `ABL` 是**受控重组**，但遵守一条硬约束：

| 变体 | 统计量来源 | 只改什么 |
|---|---|---|
| `G_full` | **真实门控**（`evaluate_policy_family_governance`） | 不改 |
| `G_dsr_only` | 直接调用 `compute_deflated_sharpe_ratio`（同一库函数） | 只把 FDR 换成逐检验 `p≤q` |
| `G_fdr_only` | 同上，传 `T_eff=1` | 只取消紧缩 |
| `G_raw_t` | 单侧 t 检验 | 无 FDR 无 DSR |

即：**统计量一律来自库函数，harness 只重组选择逻辑；`G_full` 必须逐位等于真实门控**
（`artifact_hash` 相同，AC-28）。

### 5.7 预算与降级（v1.4 重写：分格，不按全局平均）

**废弃**：v1.3 的 `617k × 25ms ≈ 4.3h` 与"若实测单次 >40 ms 则整体降为 4000/1000/500"。

两处错误：其一，`25ms` 是未测量的常数；其二，**降级判据用全局平均去裁决一个多峰分布**，必然错杀或漏杀。

按 §5.1 网格清点（**按 `T` 分组，因为单次成本主要由 `T²·N` 决定**；旧表按块分类，
把含 `T=100,N=520` 的 `ALT` 整块当成"微秒级"，且 `N=3` 是同一个"裸个位"歧义）：

| 桶（含哪些 `(T,N)`） | 调用数 | 调用占比 | 成本驱动 |
|---|---:|---:|---|
| `T=4`：(4,126) (4,266) (4,520) (4,252) (4,504) | 189,000 | 30.6% | 4×4 相关矩阵、OCN 只剩 2 个候选（§5.3）→ **微秒级** |
| `T=20`：(20,126) (20,266) (20,520) | 175,000 | 28.4% | `20²·N` + OCN 4 候选 → **百微秒** |
| `T=100`：(100,126) (100,266) (100,520) (100,252) (100,504) | 253,000 | 41.0% | `100²·N` 相关矩阵 + OCN 候选×silhouette `O(T²)` → **毫秒级起** |

**关键事实**：`T=100` 只占 41% 的**调用**，但按 `T²` 折算应占绝大部分**机时**；
真正便宜的 `T=4` 占 30.6% 的调用却几乎没有成本。**调用占比不能当机时占比用**——
这正是 §5.8 要用 `COSTP` 实测 `ĉ(T,N)` 而不是数调用次数的原因。

**新规则**：降级按 `(T,N)` 桶触发，不按全局平均。

```text
若 ĉ(T,N) > 阈值          则该 (T,N) 桶内所有 cell 的 reps 降一档
                          T=4 的桶默认不降（本来就便宜，降了只损失精度）
降档序列: 8000→4000, 2000→1000, 1000→500
```

该决定**只依赖 `COSTP` 的计时**，不依赖任何结果，因此不构成可选停止；AC-01 仍**不得硬编码总调用数**。

### 5.8 `COSTP` 成本归因 pilot（v1.4 新增，D1 强制前置）

**目的**：把 §5.7 的预算从"照抄的常数"换成"本机实测的成本模型"。这是本文档唯一有权决定最终 reps 的机制。

**设计**：

- 覆盖**网格中实际存在的 13 个 `(T,N)` 组合**（不是抽样猜测）：
  `(4,126) (4,266) (4,520) (4,252) (4,504) (20,126) (20,266) (20,520)
  (100,126) (100,266) (100,520) (100,252) (100,504)`。
  （v1.4.3 修正：旧文写作 `(3,3)`，但网格最小 `T=4`、最小 `N=126`，该组合不存在。）
- 每点 ≥200 次调用，**丢弃首次**（预热），取**中位数** ĉ。
- 分项计时（在**同一 pass 外**用独立 clock 分段，不在 `decide()` 内插桩，见 AC-31）：
  `dgp`（生成 R）、`m`（`mean_absolute_correlation`/`correlation_matrix`）、`ocn`、`dsr`、`fdr`、`harness`。
- 输出到 `metrics.json.cost_profile`：`ĉ(T,N)`、分项占比、单核总时长推导值、
  以及"若按 §5.7 降档后"的总时长与各格 CI 宽度。

**判读规则（预注册）**：

1. 若 `ocn` 分项占比 > 50%，报告须写明"**本实验的运行时长由 OCN 分区主导**"，并同时给出
   `Δ_ocn`（§5.4）——不得把 OCN 的耗时或 abstain 归因到 DSR。
2. 若六个分项的 `harness` 占比 > 30%，说明成本在共享脚手架而非被检验对象；此时**禁止**先优化规则
   （对应文章教训：供给受限时下游优化无效）。
3. 若实测总时长与 v1.3 假设（4.3h 单核）相差 > 2 倍，须在报告中显式记录该偏差，并说明
   v1.3 的预算假设错在"**用调用次数当机时**"：`T=4` 占 30.6% 的调用却几乎不耗机时，
   `T=100` 占 41.0% 的调用才是主成本（见 §5.7）。

### 5.9 实现自检探针（pre-D8 阻断，v1.4.5）

在跑全量之前，用**小规模探针**先验四件事。任一项不过，**不得**进入 D8：
探针数字与解析预期不符时，先怀疑**输入**（布局、`observations`、顺序），不要先怀疑门控。

| 探针 | 格子 | reps | 预期 | 不符时的含义 |
|---|---|---:|---|---|
| **布局** | 任一 `T=4,N=126` | 20 | `observation_count==126`、`raw_trial_count==4`、相关矩阵 `4×4` | 转置未改净（AC-41） |
| **相关** | `rho=0.3` 的 null 格 | 50 | `mean_absolute_correlation ∈ [0.22,0.38]` | `f` 按策略索引 ⇒ 模型错，ρ 失效（§3.1 注） |
| **τ 回收** | `S=1.2, T=4, N=266` | 200 | `mean sample SR − 1.2` 在 `±2·SE` 内 | DGP 的 μ 映射或顺序错 |
| **门控 FPR** | `T=4` 的 null 格（三个 N） | **≥500** | **0.2% – 0.8%**（`T_eff≈2.1` 时 ≈0.6%；`T_eff≈3.3` 时 ≈0.2%） | 若读到 **≈0.94**，几乎必然是输入错（例如 `observations` 传成了 4），不是门控"太松" |

> FPR 探针只判"**量级对不对**"（≤3% 视为通过），不判点值：500 reps 在 `p=0.6%` 时只有约 3 个事件，
> 区间宽到 `[0.1%, 1.7%]`，判不了 0.3% 与 0.6% 的差别——那是 D8 全量（8000 reps）的事。

> `0.94` 这个读数为什么可疑：它恰好等于 `B0` 在 `T=4` 的理论家族 FPR `1−0.5⁴ = 0.9375`。
> 若探针把 `B0` 的数当成门控的数，或把"任一条被 FDR 拒"当成"门控选中"，都会得到这个值。
> 定位方法：对 20 个 rep dump `artifact.status / raw_trial_count / observation_count /
> effective_trials / dsr[i].raw_sharpe / dsr[i].one_sided_p_value`，看第一个分叉点。


---

## 6. 门控判定（逐字段）

### 6.1 Input

`policy_ids` 互异；`trial_major_returns` = `R`（`T*N`）；`observations=N`；
`primary_p_values[i]` = 均值为正的单侧 t 的 p（ddof=1）；5 个门控布尔全 `true`。

### 6.2 Spec

| 字段 | 值 | 依据 |
|---|---|---|
| `schema_version` | 1 | `governance.cpp:110` |
| `fdr_q_level` | 0.05（`SWP` 扫 5 档） | 合同 |
| `fdr_method` | `BENJAMINI_HOCHBERG`（`BYA` 用 BY） | 差异 #1 |
| `storey_lambdas` | `{}` | 不用 STOREY |
| `effective_trial_spec.minimum_observations` | 3 | 实现下限 |
| `effective_trial_spec.config_hash` | `0x4D434731` | 非 0 |
| `onc_spec` | `min=2, max=4, min_size=2, max_iter=100, repeats=4, seeds={17,29,43,71}` | 结构体默认（`onc_partition.h:11-18`）；差异 #2 与 §5.3 |
| `performance_spec.schema_version` / `calendar_id` | 1 / `"MC-SYNTHETIC"` | `performance_spec.cpp:8` |
| `calendar_periods_per_year` | `p` | `:9-10` |
| `minimum_return_observations` / `minimum_tail_observations` | 3 / 20 | `:16`、`:17`（**tail ≥ return**） |
| `accounting_absolute_tolerance` / `_relative_` | 1e-8 / 1e-10 | `:12-15` |
| `annual_risk_free_rate` | 0.0 | `:11` |
| `config_hash` | `0x4D435031` | **非 0**（`:18`；否则整调用 INVALID_INPUT） |
| `dsr_config_hash` / `family_contract_hash` | `0x4D434431` / `0x4D434631` | 非 0（`governance.cpp:110-111`） |

判定 `decision = (status==OK) && candidate_selected`；`status!=OK` 记 **ABSTAIN**。
`compute_deflated_sharpe_ratio` 的 `x > −1.0` 前提（`:389`）在 t5 尾部约 `1e-9/抽取`，
全实验约 `9e9` 次抽取 ⇒ 期望 ~9 次命中，**按 ABSTAIN 计，不得 crash**（AC-29）。

### 6.3 配置文件必备字段（v1.4.3：`run_seed` 升为 config 字段）

`run_seed` 决定全部数据，却在 v1.4.2 之前只是文档里的一个常量 ⇒ **`config_hash` 覆盖不到它**。
两个 run 只要在代码里改这一行常量，config_hash 会完全一样，事后无法从工件判断数据是否同源。
因此：

| 字段 | 类型 | 值 | 说明 |
|---|---|---|---|
| `run_seed` | uint64 | `20261027` | **必填**；缺省即拒启（不设默认值） |
| `grid` | object | §5.1 的 13 个块 | 每块给 composition/S/N/T/rho/dist/q/variant 的**显式集合** |
| `reps` | object | `{8000,2000,1000}` | 按块给；须与 §5.2 一致 |
| `seed_scheme` | string | `"fnv1a64+splitmix64@v1"` | 派生算法版本；换算法必须换这个串 |
| `panel_overrides` | array | 见 §3.6.3.1 | 每项 `{block, cell_selector, panel_ref}` |

- `config_hash` = 对**上述全部字段的 canonical JSON**（键排序、无空白、`%.17g`）做 SHA-256 后取前 64 位；
  `manifest.config_hash` 与 `metrics.json.config_hash` 必须一致，且 `run_seed` 可从 `manifest.seed_tree`
  反查（双重记录，防止只改一处）。
- **AC-01 校验 `run_seed` 存在且落在 `[0, 2^64)`**；**AC-40** 校验"改 `run_seed` 一个值 ⇒ `config_hash`
  必须变化"，用一个负向测试固定这条性质。

---

## 7. 基线

| ID | 逻辑 | 可扫 |
|---|---|---|
| B0 | `argmax SR`，`max SR>0` 选中 | 否 |
| B1 | `argmax SR`，其单侧 t 的 `p<alpha` 选中 | **是** |
| B2 | 3 段子窗口取 `(策略,窗)` 最大，该窗 SR>0 选中 | 否 |
| B3 | 前 50% 取 `argmax`，后 50% SR>0 选中 | 否 |
| B4 | CSCV(S=16)，`PBO<0.5` 且选中者 SR>0 | 否 |

### 7.1 相关性下的家族 FPR（独立公式只在 rho=0 成立）

`FPR_B1(T,rho,z) = 1 − E_Z[Φ((z−√rho·Z)/√(1−rho))^T]`，数值积分：

| rho | B1 @ α=0.05（T=4/20/100） | B0（T=4/20/100） |
|---|---|---|
| 0 | 0.185 / 0.642 / 0.994 | 0.938 / 1.000 / 1.000 |
| 0.3 | **0.164 / 0.445 / 0.754** | **0.860 / 0.989 / 1.000** |
| 0.7 | 0.120 / 0.245 / 0.386 | **0.729 / 0.876 / 0.943** |

⇒ `1−(1−α)^T` **仅用于 rho=0 的解析锚**（AC-11）。

### 7.2 匹配 FPR 的校准

α 网格：**对数 50 档，`1e-5…0.2`**；取 `|fpr_B1(α)−fpr_gate|` 最小（不取"≤"）；
缺口 >0.2pp 时用相邻两档线性插值做随机化阈值。表 2 强制列 `fpr_B1(α*)/fpr_GATE/gap`。

---

## 8. 指标

`fpr_family`、`fpr_strategy=E[选中数/T]`、`power`（分母含 ABSTAIN）、`power_ocn_ok`、`misselect`、
`abstain`、`deploy_sr`、`p_deploy_neg`、`threshold_sr`（对 S 网格 logistic 取 p=0.5）、
`teff_measured`、`m_measured`。

**v1.4 新增计时指标**（`wall_by_rule`、`cost_profile`）：

- `wall_by_rule`：六条规则各自的墙钟时间（`(cell,rep,rule)` 粒度累计）。用于判断"成本是否在共享脚手架"：
  若 GATE/B0/B1/B2/B3/B4 的墙钟时间差异 < 5%，则成本在 DGP+`m`+OCN 而非规则，**此时优化规则无效**。
- `cost_profile`：§5.8 的 `ĉ(T,N)` 与分项占比，是 §5.7 降级与 AC-24 的唯一依据。

聚合：点估计 + **Wilson 95%**；规则间 McNemar。**区间判据用 `ci_lower ≤ p ≤ ci_upper`**
（`p̂=0` 时 Wilson 下界恰为 0，严格不等式必假）。

---

## 9. 门槛面

### 9.1 公式

```text
SR_req^DSR = sqrt(p)*[ sqrt(2*ln(T_eff)/N) + z_{1-q}*sqrt(correction/(N-1)) ]
SR_req^BH  = z_{1-q/T}/sqrt(N/p)
实际门槛   = max(DSR, BH)

correction = 1 - skew*s + 0.25*kurt*s^2,   s = SR/sqrt(p)
    t5（skew=0, kurt=6）: correction = 1 + 1.5*s^2  ⇒  sqrt(correction) ∈ [1.000, 1.047]（S: 0→1.8）
```

> `correction` 依赖**被检验策略自己的** `s`，所以门槛方程是隐式的；上表按 `correction = 1` 取值，
> 因此 `SR_req^DSR` 是**偏小的下界**，实测门槛会系统性略高：t5 在 `S=1.8, N=266` 处上偏
> **≤0.034 SR**（`0.047 × 0.717`），远小于 AC-13 的 0.10 容差。这是 §9.1 唯一的近似项，
> 与坐标朝向无关。

`T_eff` 用**该 rep 实测 `mean_absolute_correlation`** 代入（预测表用 `m=rho`，因
`rho ≫ (1−rho²)/√N` 时 `E|r̂| ≈ rho`；`rho=0` 时 `m ≈ 0.8/√N`）。

### 9.2 表（q=0.05, p=50.4, rho=0.3）

| N | T | `T_eff` | DSR | BH | **实际** |
|---|---|---|---|---|---|
| 126 | 4 | 2.11 | 1.82 | 1.42 | **1.82** |
| 126 | 20 | 2.99 | 1.98 | 1.78 | **1.98** |
| 126 | 100 | 3.26 | 2.02 | 2.08 | **2.08 (BH)** |
| 266 | 4 | 2.11 | 1.25 | 0.98 | **1.25** |
| 266 | 20 | 2.99 | 1.36 | 1.22 | **1.36** |
| 266 | 100 | 3.26 | 1.39 | 1.43 | **1.43 (BH)** |
| 520 | 4 | 2.11 | 0.89 | 0.70 | **0.89** |
| 520 | 20 | 2.99 | 0.97 | 0.87 | **0.97** |
| 520 | 100 | 3.26 | 0.99 | 1.02 | **1.02 (BH)** |

`rho=0`：`T_eff` 按 N 分别为
`N=126: 3.30/8.48/12.41`、`N=266: 3.49/10.35/17.08`、`N=520: 3.62/12.00/22.36`，
门槛（N=266）**1.41 / 1.66 / 1.75**。
`rho=0.7`：`T_eff` 1.29/1.40/1.42，门槛（N=266）**1.03 / 1.22 / 1.43**。

### 9.3 定性结论

1. 固定 rho 下门槛随 T **近乎平坦**（`T_eff` 饱和于 `≈1/rho`）；驱动是 **rho 与 N**。
2. **主导约束翻转**：`rho=0.3, N=266` 的 `T* ≈ 69`；`T>T*` 后 BH 主导；`rho=0.7` 时 BH 全程主导；
   `rho=0` 时 DSR 全程主导。
3. 实现比教科书 DSR 多罚约 **0.6 个 t 单位**（`sqrt(2ln4)=1.67` vs `E[max]=1.03`）。

### 9.4 预注册落点（粗算，仅定向）

`N=266, rho=0.3`；`alpha*` 的 z 阈值 2.20/2.74/3.19（T=4/20/100）：

| T | 门控 t | `B1(α*)` t | S=0.6 | S=1.2 | S=1.8 | Δ |
|---|---|---|---|---|---|---|
| 4 | 2.87 | 2.20 | 7% vs **20%** | 45% vs **71%** | 90% vs **97%** | −13/−26/−7 |
| 20 | 3.12 | 2.74 | 4% vs **9%** | 35% vs **51%** | 84% vs **92%** | −5/−16/−8 |
| 100 | 3.29 | 3.19 | 3% vs 3.5% | 30% vs 33% | 80% vs 83% | **\|Δ\|<5** |

⇒ 预测 V2，被支配区 `T≤20`；`T=100` 因 BH 接棒仅差 0.1 t。
**但 `T=4` 的 Δ 将含 §5.3 的 OCN 伪影，必须按 §5.4 分解后再归因。**

### 9.5 解析锚 vs 实测零分布（v1.4 新增，只报告不改判据）

§9.1/§9.2 的 `SR_req` 是**渐近**公式（§9.3 自承"比教科书 DSR 多罚约 0.6 个 t 单位"）。本实验跑完
617k 次会**免费产出**经验零分布，因此两者必须并列报告：

| 项 | 来源 | 用途 |
|---|---|---|
| `SR_req^analytic` | §9.1 公式 | **判据锚**（AC-11/AC-13 仍用它，不自我评分） |
| `SR_req^empirical` | 该 `(S=0,N,T,rho)` 格的 `NUL` 实测 `mean_sample_sr` 分位数 | 一致性核对 |

**规则**：

1. 两者偏差 > 0.05 SR 时，记 flag **`analytic_threshold_drift`**，并写明偏差方向（解析偏高/偏低）。
2. `SR_req^empirical` **不得**替换解析锚去改变 V1/V2/V3 判定——否则等于让实验给自己打分。
3. 偏差来源若指向 §5.3 的 OCN abstain（`T=4` 层），必须在报告中分列，不得并入统计层结论。


---

## 10. 假设与判定

### 10.1 假设

| ID | 假设 | 判据 |
|---|---|---|
| **H0-OCN**（新增） | `T=4` 层被 OCN 分区退化限制 | `OCNP` pilot：`T=4` 的 `abstain_rate ∈ [0.45,0.75]`（v1.5 按实测 55.0%/61.5%/70.5% 上调）；`T=20/100` 在 `rho≤0.3` 时 `<0.01`，`rho=0.7` 时可达 ~`0.10` |
| **H1a** | 门控守住名义 FPR | **`NUL` 全部 27 格** `fpr_family ≤ 6.5%` |
| **H1b** | `rho=0` 时 `T_eff` 被噪声底压低但 FPR 不破 5% | 报 `m_measured/(0.8/√N)` 与 `T_eff/T`；`T=100` 的 `T_eff` 按 N=126/266/520 分别落在 [10,14]/[15,19]/[20,25] |
| **H2** | 等假阳性下的效率 | `Δ` 及 §5.4 分解；先验（v1.5 按实测修订）：`T=4` 时 `Δ≈−4pp`、`T=20/100` 时 `\|Δ\|<1pp`。**注意**：统计层 `Δ_stat` 在 `T=4` 实测为 **+13.1pp**，负号全部来自 OCN |
| **H3** | 假阳性差随 T **单调不减** | `fpr(B0)−fpr(GATE)` 单调不减，`T=20/100` 时 ≥10pp |
| **H4** | 实测门槛 = `max(DSR,BH)` 解析值 | **仅 `ALT`（9 个 S 点）**：`\|threshold_sr − SR_req(m_measured)\| ≤ 0.10`（v1.5 删去原 `T*∈[60,80]`：与实测 `T_eff=3.33` 无法对应，且无定义） |
| **H5** | 门控更少把诱饵当真 | **仅 `DEC`**：`P(选中诱饵)`、`p_deploy_neg` 显著低于 B0/B1 |
| **H6** | 跨家族重试击穿门控 | `RET`：`K=10` 的门控 FPR 显著高于 `K=1`，预测 **≈16%**（v1.5 按实测单族 `p=0.90%` 修正：独立预测 `1−(1−0.009)^10 = 8.6%`，实测 **16.1%**，放大 **17.9×**；原 `>40%` 是用 `NUL` 的上界 `p=6.5%` 高估） |

### 10.2 判定伪代码（四态，无空洞）

```python
nul = [c for c in NUL]
invalid   = [c for c in nul if wilson_lo(c) > 0.065]      # 确定超界
undecided = [c for c in nul if wilson_ci(c).contains(0.065)]
delta     = {(S,N,T): power_gate - power_B1star}
dominated = [(k,v) for k,v in delta.items() if v < -0.05]
delta_und = [k for k in delta if mcnemar_ci(k).contains(-0.05)]

if invalid:            verdict = 'V3'
elif undecided or delta_und: verdict = 'VU'      # 未判定：CI 跨阈值，不硬判
elif dominated:        verdict = 'V2'
else:                  verdict = 'V1'

flags = []
if any(abstain_rate(c) > 0.01 for c in nul): flags.append('HIGH_ABSTAIN')
if any(c.ocn_constrained for c in nul):      flags.append('OCN_CONSTRAINED_T4')
```

- `VU` 的语义：**分辨率不足**，不是"有效"也不是"无效"；报告必须列出 undecided 单元，
  且**不得**追加 reps（避免可选停止）。AC-21 与本节一致（原 v1.2 伪代码未落码，已修）。
- `verdict` / `flags` / `abstain_rate` 写入 `metrics.json`，报告首段引用同一字段。

---

## 11–12. 表与图

**表 0** 门槛面（`T_eff` 实测、`m_measured`、DSR/BH/max、`threshold_sr`）。
**表 1** 头条（`NUL`）：`T|N|rho|GATE|B0|B1|B2|B3|B4|解析FPR|abstain`。
**表 2** 效率（`ALT`）：`S|N|T|power|power_ocn_ok|B1(α*)|Δ|Δ_stat|Δ_ocn|fpr_B1|fpr_GATE|gap|misselect`。
**表 3** FDR（`MIX`）。**表 4** 部署（`DEC`）。**表 5** 机制与压力（`OCNP/BYA/RET/GAR/GAU/FRQ/SVL/ABL/NEG`）。

`Fig 1` 前沿（GATE/B1 曲线，B0/B2/B3/B4 单点） · `Fig 2` FPR vs T · `Fig 3` 功率 vs S
`Fig 4` 所需 N · `Fig 5` 部署 Sharpe · `Fig 6` 消融 · `Fig 7` 门槛面热图 + `Fig 8` **OCN abstain 热图**（`T×rho`）。

绘图纪律：固定尺寸/字号；坐标只取自 `metrics.json`（脚本无 6 位以上浮点字面量）；
SVG 不进哈希链，只要求坐标与表一致（≤1e-9）。

---

## 13. 结构与接口

```text
performance_analytics/apps/mc_gate_validation.cpp        # 本实验新建目录（同步 CMakeLists）
configs/mc/mc_gate_core_v1.json / mc_gate_smoke.json
tools/run_mc_gate_validation.py / verify_mc_gate_validation.py / plot_mc_gate_validation.py
runs/mc_gate_<runid>/{manifest.json, decisions.jsonl.gz, replay/, metrics.json, figures/}
```

```cpp
namespace mc_gate {
enum class DistKind { gauss, t5, garch, shortvol };
enum class Composition { null_family, one_good, mixed, mixed_signed, all_negative };
struct ResearchWindow {                    // 无 tau 字段
  std::span<const double> returns; std::size_t trials, observations;
  std::uint64_t returns_sha256;
};
struct Decision { bool selected; std::size_t index; bool abstain; const char* reason; };
Decision decide(const ResearchWindow&, const RuleConfig&);
}
```

**分片键 = `cell_id`**（cell 的全部 reps 不跨 shard）；reducer 单进程按
`(cell_id, rep_id, rule_id)` 排序聚合，浮点 `%.17g`。这样 `mean_sample_sr`、`deploy_sr` 均值、
logistic 拟合与分片数无关。

`replay/` 规模：**12 个代表数据集 × 10 reps = 120 个矩阵**（NUL/ALT/MIX/DEC/GAR/GAU/FRQ/SVL 各取
代表格），f64 原样 + SHA-256，约 24 MB。

`manifest.json` 的 `seed_tree` 段记录：`run_seed`（来自 config，不是文档常量）、
全部 `dataset_key`/`panel_ref` 列表、每 `(dataset, rep_batch, stream)` 的 seed 与前 3 个输出指纹，
以及四元组碰撞检查结果。该段是复现的最小充分信息——只凭它即可重放任意单个 rep（§3.6.1 约束 3）。

---

## 14. 验收标准（[B]=阻断）

### A. 契约

- **AC-01 [B]** schema 校验；`cells`、`panels`、`calls` 三项必须**等于由 config 推导的值**
  （当前 269 / 202 / 617,000），**不得硬编码常数**——这样 §5.7 的降级不冲突。
  `panels` 必须按 `distinct panel_ref` 枚举得出，并额外断言 §5.5 的逐块分解表。
  **`run_seed` 必须存在且 ∈ `[0, 2^64)`**（§6.3），缺省即拒启；
  每个块的 N/T 必须是**显式取值集合**，禁止 `int` 简写（§5.1）。
- **AC-02 [B]** `manifest` 的 `contract_hash/config_hash/binary_sha256/git_commit` 与 `metrics.json` 一致。
- **AC-03 [B]** 无时间戳/主机名/耗时；gzip **`mtime=0`**。
- **AC-04** `decisions.jsonl.gz` 每行含 `cell_id/rep_id/rule_id/selected/index/abstain/returns_sha256`。

### B. 确定性

- **AC-05 [B]** `--shard 1/1` 与 `4/4` 的 `metrics.json` 字节相同（依赖 §13 的分片键与排序）。
- **AC-06 [B]** RNG 按 **§3.1.1** 的播种/输出规格自实现 PCG64 + 固定 Box–Muller（禁
  `std::normal_distribution`）；**必须逐位命中 §3.1.1 的 golden 向量**（`seed=42` / `seed=12345` 前 8 个输出）；
  前 1e6 输出自检并与 Python 标准库实现比对。
- **AC-39 [B]** **抽样顺序的位级对照**（v1.4.3）：verifier 用 stdlib-only Python 独立实现 §3.1 的
  生成器，与 C++ 在 `replay/` 的 12 个数据集上要求
  (a) `ret` 流**前 8 个 double 的十六进制表示**逐位相同（钉住"先 `f_1..f_N` 后 `e`"的起点）；
  (b) 整块 `T×N` 展平后的 SHA-256 相同（钉住"策略外层、期内层"的遍历顺序）。
  只比对统计量抓不住列优先或并行取流——那会产出同分布但不同面板的数据，统计检验照样通过。
- **AC-07 [B]** `garch` 构造断言 `omega=1−α−β`、`corr_t ≡ rho`（逐期）、无条件 `Var=sigma_p²`；
  `shortvol` 断言 `|skew+2.5|≤0.05`、`|Var/sigma_p²−1|≤1e-6`。

### C. 数值

- **AC-08 [B]** τ 回收：单格 `|z_cell|≤4.5`（SE 含峰度修正）+ 全局 `Σz²≤251.7`（df=202）；
  越界按 §3.5 分级处理（重抽单格 → 标 `dgp_unverified` / `dgp_global_shift`）。
- **AC-09 [B]** `S=50` 金丝雀：`GATE selected==true` 且 index 正确。**位于 `mc_gate_smoke.json`
  的独立 canary 单元（50 reps），不计入 617k**。
- **AC-10 [B]** `NEG` 只有 GATE 断言（零选中率 ≥99.5%）；B0/B1/B2/B3/B4 只报实测，解析参考
  （N=126，B1）≈1.8%/7.5%/22%，B4 ≈20% 量级。
- **AC-11 [B]** 基线解析锚仅施加于 **`rho=0` 的 9 个数据集**（含 T=20）：B0 ∈
  [0.90,0.97]/[0.99,1.0]/[0.999,1.0]，B1 ∈ [0.16,0.21]/[0.60,0.69]/[0.985,1.0]；
  `rho>0` 用 §7.1 相关解析带 ±3pp。
- **AC-12 [B]** 5 布尔全真时 `candidate_selected == winner_frozen` 100%。
- **AC-41 [B]** **维度绑定断言（v1.4.4，抓朝向转置）**：每 cell 断言
  `artifact.observation_count == N`、`artifact.raw_trial_count == T`、
  `policy_return_correlation` 尺寸为 `T×T`、`effective_trials ≤ T`；
  且 `mc_gate_smoke` 必须含 `(T=4, N=520)` 与 `(T=100, N=126)` 两格做交叉对照——
  转置实现会在这里立刻露馅（前者相关矩阵会变 520×520，后者会变 126×126 且 `effective_trials > T`）。
  另断言实测 `mean|r̂|` 与设定 `rho` 同量级（`rho=0.3` 时落在 `[0.22, 0.38]`），
  用于抓"`f` 按策略索引"这类**模型**错误（见 §3.1 注）。
- **AC-13 [B]** 门槛一致性**仅在 `ALT`**（9 个 S 点）判：`|threshold_sr − SR_req(m_measured)| ≤ 0.10`；
  3 点网格（`GAR/GAU/SVL`）只报不判（logistic 交点 SE 可能超容差）。
- **AC-14 [B]** Python 独立复算（标准库）：BH、DSR（含偏峰修正）、`T_eff`、六条规则决策；
  `replay/` 120 个矩阵上 `|ΔSR|≤1e-12`、`|Δp|≤1e-12`、决策一致。**不要求** `artifact_hash` 跨语言一致。
- **AC-15 [B]** 从 `decisions.jsonl.gz` 重算表 1–5 全部比例，与 `metrics.json` 逐格相等。

### D. 防泄漏

- **AC-16 [B]** 置换 `N_hold` 后六条规则决策不变。 **AC-17 [B]** 置换 `tau` 后决策不变。
- **AC-18 [B]** 同 rep 下六条规则 `returns_sha256` 相等且等于 manifest。
- **AC-19** `decide()` 的 TU 不含 truth 类型。

### E. 统计与报告

- **AC-20 [B]** Wilson 区间且 `ci_lower ≤ p ≤ ci_upper`。
- **AC-21 [B]** 四态判定按 §10.2 落码：`invalid→V3`、`undecided→VU`、`dominated→V2`、否则 `V1`；
  `flags` 含 `HIGH_ABSTAIN` / `OCN_CONSTRAINED_T4` 的规则写死（>1% / 由 `OCNP` 结果触发）。
- **AC-22** 表与图的每个数字可追溯到 `cell_id`；plot 无硬编码数值。
- **AC-23 [B]** `verdict`/`flags`/`abstain_rate`/`Δ/Δ_stat/Δ_ocn` 均由代码机械计算。

### F. 额外（v1.3）

- **AC-27 [B]** `BYA`/`SWP`/`ABL`/`RET-K1` 的 `panel_ref` 指向 `NUL`/`ALT`，`returns_sha256`
  与所引面板逐位相等，且 `z_cell` 逐位相同。
- **AC-28 [B]** `G_full` 与真实门控在全部重叠配置上 `artifact_hash` 完全相同（证明 `ABL` 只重组不改统计量）。
- **AC-29 [B]** 收益 ≤ −1.0 导致的 `INVALID_INPUT` 记 ABSTAIN，进程不崩、不丢 rep。
- **AC-30 [B]** `OCNP` pilot 必须先于主 run 完成，结果写入 `metrics.json.pilot`，且**不改变**任何判据。
- **AC-36 [B]** 种子派生逐位对齐 §3.6.7 的 **6** 个测试向量（`fnv1a64` 与 `seed` 各 6 例），
  含 `A` vs `R`（仅 `ds` 不同）必须产生不同 hash 与不同 seed；
  另做**全量碰撞检查**：全部 `(dataset_key, rep, batch, stream)` 四元组的 seed 集合必须无重复，
  有重复即 fail closed（FNV-1a 64 在 `~10^7` 条串上的碰撞概率约 `2.7e-6`，但不能靠概率过关）。
- **AC-37 [B]** `manifest.json` 记录每 `(dataset, stream)` 的前 3 个 PCG64 输出指纹；
  同平台两次重跑必须逐位相同；跨平台差异只允许出现在指纹层（用于定位），不允许出现在指标层。
- **AC-38 [B]** **`panel_ref` 显式化（v1.4.2）**：实现只允许按 `panel_ref` 取面板，禁止按 block 名
  或 cell 顺序取；`metrics.json` 每 cell 含 `panel_ref` 与 `returns_sha256`；
  `distinct panel_ref` 计数必须 = §5.5 的 **202**，且逐块分解（27/81/12/12/9/18/18/12/4/0/9/0/0）
  必须逐项相等；任何 block 若声明复用而 `returns_sha256` 不等，run 拒启。
- **AC-40 [B]** **`run_seed` 进哈希链（v1.4.3）**：`config_hash` 必须覆盖 `run_seed`
  （负向测试：仅改 `run_seed` 一个值 ⇒ `config_hash` 变化）；`manifest.seed_tree.run_seed` 与
  `config.run_seed`、`metrics.json.run_seed` 三处必须相等，任一不一致即拒启。
- **AC-24 [B]** 运行时长判据**改为相对口径**（v1.4）：由 `cost_profile`（§5.8）推导的单核总时长
  `Σ reps×ĉ` 除以核数 ≤ **45 min**；并单列实测的 `ĉ(T=100,N=520)` 中位数（**只报不判**，
  因该量机器相关，不得作为跨平台验收门槛）。
- **AC-25** smoke（3 单元 × 50 reps + canary）60 s 内 `--strict` 通过。
- **AC-26** 零新增第三方依赖；`QBT_ENABLE_PORTFOLIO_MATH=OFF` 不编译本 driver。

### G. 测量纪律与成本（v1.4）

- **AC-31 [B]** 计时纪律：`steady_clock` 只允许在 `(cell_id, rep_id, rule_id)` 粒度成对取用；
  `decide()`、`evaluate_policy_family_governance`、`compute_deflated_sharpe_ratio`、
  `onc_partition` 的内部**零计时**。静态检查：`decide` 所在 TU 不出现 `steady_clock|chrono::now`。
- **AC-32 [B]** `metrics.json` 含 `wall_by_rule`（六条规则墙钟）与 `cost_profile`
  （`ĉ(T,N)` + 分项 `dgp/m/ocn/dsr/fdr/harness` 占比）；二者与 §5.7 降级决定一起落盘。
- **AC-33 [B]** 预算守约：最终 reps **必须等于** §5.7 规则由 `cost_profile` 推出的值，
  且 `calls` 由 config 复算相等（与 AC-01 呼应）。**禁止**在未见 `cost_profile` 前固定 reps。
- **AC-34 [B]** `replay/` 写入粒度：**每数据集一个文件**（12 个），**禁止**每 rep 一个文件
  （617k 个小文件是 2026-09-11 报告里的反面教材）。文件内为连续 f64 + 数据集 SHA-256。
- **AC-35** §9.5 双轨门槛：`SR_req^analytic` 与 `SR_req^empirical` 并列写入 `metrics.json`；
  偏差 > 0.05 SR 时 `flags` 含 `analytic_threshold_drift`，且**判据不变**。

---

## 15. 日程

### 15.0 施工前置（v1.4 新增，阻断）

开工前必须满足，否则拒绝开跑：

1. **树一致性**：执行树的 `git rev-parse HEAD` 必须等于 `manifest.json.git_commit`；否则 §1 的路径
   核对与 AC-28 的逐位一致都会在缺文件的老树上失败（实测已发生：某次检查用的树落后 1 个 commit
   且缺 `portfolio_math/apps/`）。
2. **`COSTP` 先于一切**（§5.8）：预算、reps、降级决定全部依赖它，且**不得**在看到任何 FPR/功率结果
   之后再定 reps（避免可选停止）。
3. **RNG 自检先于 DGP**（§3.1.1 golden 向量）：RNG 错则全实验无效。

### 15.1 日程（人工口径；本地模型施工口径见施工方案）

D1 合同/配置/计数校验 + `COSTP` ‖ D2–3 DGP（含 garch/shortvol 构造断言）+ τ 回收
‖ **D4 `OCNP` pilot + 门控适配 + 6 规则** ‖ D5 scorer/JSONL/manifest/shard
‖ D6 verifier ‖ D7 图表 ‖ D8 全量 run + 复跑 ‖ D9 压力/消融/RET/BYA/SVL/Tier B ‖ D10 报告。

出口判据一律用可自动检查的 AC。

---

## 16. 风险与预案

| 风险 | 信号 | 预案 |
|---|---|---|
| **`T=4` OCN abstain 15–35%** | `OCNP` pilot | **主要风险已定位在 T=4**（不是 T=100）：标 `OCN_CONSTRAINED_T4`，按 §5.4 分解归因，不修 spec |
| `T=20/100` 也大量 abstain | `abstain>1%` | 标 `HIGH_ABSTAIN`，判定降级并单列该层失效 |
| `garch` FPR 抬升 | H1 压力档失效 | 预期内（iid 破坏）；写"已知边界"，不进 V3 |
| `rho=0` 门槛低估 | `T_eff` 落在 §10.1 区间外 | 改 `m` 的经验模型；仍是机制发现 |
| 门控功率过低 | 与 §9.4 一致 | 不得事后加长 N；补预注册长样本区 `N∈{1040,2080}` 子实验 |
| B4 不输门控 | V2 风险 | 写 V2，强调"事前冻结 vs 事后校正" |
| 超时 | D8 未完 | 按 §5.7 降 reps 并标注 CI；不得只降 reps 不降结论强度 |

---

## 17. 引用

`performance_analytics/{include,src}/policy_family_governance.*`、
`performance_analytics/src/return_analysis.cpp:330-420`、`performance_analytics/src/performance_spec.cpp:7-19`、
`performance_analytics/tests/phase4b_governance_reference.cpp`（库内唯一调用方，v1.4 补目录）、
`portfolio_math/src/onc_partition.cpp:100-135,190-200,240-330`、`portfolio_math/include/portfolio_math/onc_partition.h:11-18`、
`portfolio_math/include/portfolio_math/multiple_testing.h`、`portfolio_math/apps/formal_oos_driver.cpp:39`（年化口径 50.4）、
`docs/phase4_ab_formal_oos_contract.md` §4–§5、`docs/phase4ab/formal_oos_report_20260912.json`、
`docs/paper_submission_probability_report_20261025.md` §五 第 1 条、
`docs/mc_gate_v14_revision_decision.md`（v1.4 修订依据）、
`backtest-engine-bottleneck-is-data-layer.md`（数据层边界，见 §1.2）。
方法学对手：Bailey et al. PBO/CSCV；Bailey & López de Prado DSR；Harvey & López de Prado 2019。

---

## 18. 修订记录

**v1.0→v1.1（21 条）**：FDR 默认 BY、`T_eff` 与 OCN 无关、§9 公式、AC-08 SE 量纲、AC-11 把 B0 当 B1、
AC-10 全负家族不可行、规则数 6、V1/V2 空洞、H5 空洞（加 `mixed_signed`）、H3 措辞、格数、
`b_i/omega/shortvol`、AC-14 跨语言 hash、gzip mtime、Fig 1 单点、`fpr_strategy`、预算漏算、
undecided 连接、`contract_hash` 越界、D2–3 出口、`rho=0` 入主判据。

**v1.1→v1.2（14 条）**：AC-08 多重性、AC-10 B1/B4、AC-11 相关性、AC-20 严格不等式、
AC-05 浮点顺序、§9.1 `m` 口径、§6 字段不全、§7 独立近似、α 网格、H1a 与 #21 矛盾、
`shortvol` 死代码、undecided 阈值、ABSTAIN 语义、`BYA` 扩格。

**v1.2→v1.3（15 条）**

| # | 问题 | 处理 |
|---|---|---|
| 36 | **`T=4` 结构性 OCN abstain**（`seed%4` 只有 2 个不同初值、`max_clusters` 钳到 2、唯一合法形状 2+2、等相关下 50% 落入 3+1） | 新增 §5.3 + `OCNP` pilot（9 单元/1,800 次）+ `power_ocn_ok` + `Δ` 分解；**不修 spec**，作为缺陷报告 |
| 37 | §16 预案方向写反（写 T=100，实际风险在 T=4） | §16 首行改为 T=4，并删去"T=100 列失效域" |
| 38 | 格数 245≠239，`chi2` df 与 AC-01 全部沿用旧数 | 决策单元 **269**（SWP 的 q 入 cell 维度 6→30）、独立数据集 **194**、`chi2_{194}=242.8` |
| 39 | §3.5 写"237 格"来历不明 | 统一为 `D=194` 数据集（§5.5） |
| 40 | AC-01 硬编码 599,000 与 §5.7 降级冲突 | 改为"等于由 config 推导的值" |
| 41 | AC-21 与 §10.2 伪代码语义矛盾 | 引入第四态 **VU**，伪代码落码，AC-21 同步 |
| 42 | H6 ">50%" 与 H1a 6.5% 上限矛盾（独立族上界 48.9%） | 改"**>40%**" |
| 43 | §3.5 SE 忽略峰度 | 加 `c_kappa=(kappa+2)/4`：gauss 0.5、t5 2.0、garch 3.5、shortvol 4.25（后两者保守） |
| 44 | 全局判据无重抽/降级路径 | 分级：重抽贡献最大的 3 格 → 标 `dgp_global_shift` → 仅 `|偏差|>0.05 SR` 才作废该档 |
| 45 | `ABL` 与 §1"不重写只调用"矛盾 | §5.6：只重组库函数（DSR/BH 仍来自库），`G_full` 必须逐位等于真实门控（AC-28） |
| 46 | `garch` 截面构造未定义 | §3.2.1 共同波动状态：`r_i=mu_p+sigma_p*sigma_t*(√rho z_f+√(1-rho) z_i)`，`corr_t≡rho` |
| 47 | AC-13 的 0.10 容差对 3 点 S 网格过紧 | 仅 `ALT`（9 点）判；3 点网格只报不判 |
| 48 | §9.2 `rho=0, N=266, T=20` 的 `T_eff` 11.5 / 门槛 1.68 有误 | 改 **10.35 / 1.66**；并补齐 N=126/520 的 `rho=0` 行 |
| 49 | `WR` 死代码、§16 的 `[10,17]/100` 只适用 N=126 | 删 `WR`；改为按 N 给 [10,14]/[15,19]/[20,25] |
| 50 | AC-09 金丝雀无归属、AC-14 replay 规模未定义 | canary 入 smoke（50 reps，不入 617k）；replay 定 12 数据集 × 10 reps = 120 矩阵 |
| 51 | 行号漂移（`performance_spec.cpp`、`onc_partition.h`） | 校准为 `:7-19`、`:11-18`、`onc_partition.cpp:100-135,190-200,240-330` |
| 52 | 收益 ≤ −1.0 触发 `INVALID_INPUT`（t5 尾部约 9 次/全实验） | AC-29：按 ABSTAIN 计，不 crash、不丢 rep |

**v1.3→v1.4（6 条，只改口径与测量纪律，不改判据）**
| # | 问题 | 处理 |
|---|---|---|
| 53 | **预算用单一平均数裁决多峰分布**：`617k×25ms` 把 25.5 万次 `N=3` 微秒级调用与 36.2 万次 `N≥126` 毫秒级调用同价，且"实测 >40ms 则整体降级"用全局平均触发全局降级 | §5.7 重写为**按 `(T,N)` 格类降档**；新增 **§5.8 `COSTP`** 实测成本模型（分项 `dgp/m/ocn/dsr/fdr/harness`）；AC-24 改相对口径、AC-33 锁预算守约 |
| 54 | **决策热路径内插桩会污染被测量对象**（2026-09-11 报告实测：逐行两次 `steady_clock::now()` 把引擎 55% 成本记进"解码"） | 新增 **AC-31** 计时纪律（仅 `(cell,rep,rule)` 粒度、库函数内零计时）、**AC-32** `wall_by_rule`+`cost_profile` |
| 55 | **解析门槛无实测对照**：§9.2 数值本身正确（已逐行验算），但渐近式与实测零分布从未并列 | 新增 **§9.5** 双轨（解析锚仍作判据，实测只核对）；偏差 >0.05 SR 记 `analytic_threshold_drift`；AC-35 |
| 56 | **数据层边界未声明**，存在"为加快本实验先去压实 69.9 万 Parquet 文件"的误用风险 | 新增 **§1.2**：本实验纯合成、不读湖；P0/P1/P2/P3 对其运行时无影响；Tier B 若接真实数据须继承 51k 行/s 约束 |
| 57 | **AC-06/AC-14 在逻辑上不可能通过**：PCG64 的"种子→状态"与"状态→输出"从未定义，跨语言逐位一致无参照 | 新增 **§3.1.1** 钉死播种/输出（**ROTR** 不是 rotl）+ golden 向量 `seed=42/12345` + `(1<<128)` 陷阱说明；AC-06 改为必须命中 golden |
| 58 | `performance_analytics/apps/` 实际不存在（当既有路径写）；`phase4b_governance_reference.cpp` 缺目录；且施工树曾落后 1 个 commit 并缺 `portfolio_math/apps/` | §13 标注为新建；§17 补 `tests/` 目录与两条新引用；新增 **§15.0 施工前置**（树一致性 `git_commit` 必须相等）+ **AC-34** `replay/` 每数据集一文件（禁每 rep 一文件） |

> 关于 2026-09-11 全湖评测文章：其头条结论（P0 文件压实）**不适用于本实验**，理由见 §1.2。
> 可迁移的是它的三条诊断纪律：先测成本归因、不在被测对象内部插桩、用多路径互证。v1.4 修的正是这三条。

**v1.4 → v1.4.1（3 条，种子规范；不改任何判据、网格或阈值）**

| # | 问题 | 处理 |
|---|---|---|
| 59 | 全文只有 `seed = f(run_seed, cell_id, rep_batch)`，既不可实现也**用错输入**：`cell_id` 含 q 与块名，会让 `SWP`/`ABL`/`BYA` 生成不同数据，直接违反 AC-27 | 新增 **§3.6**：种子输入改为 **`dataset_key`**（排除 block/rule/q/fdr_method/variant/K），并给规范化格式（§3.6.3）与测试向量（§3.6.7） |
| 60 | 未说明为什么不用单一全局流 / `seed_seq`，也没把"真策略位置"与"持有期"从主数据流中分离 | §3.6.1 四条约束 + §3.6.4 五个独立子流（`ret/assign/hold/boot/aux`）+ §3.6.6 否决理由；`rep_batch` 重抽纪律见 §3.6.5 |
| 61 | 种子口径不可验证，且与 v1.4 的 §3.1.1 有潜在冲突（播种参数/warm-up 各写一套） | §3.6.4 明确"播种与输出**完全**按 §3.1.1，本规范只派生 seed"；新增 **AC-36**（逐位对齐 5 个向量 + 全量碰撞检查）、**AC-37**（PCG64 指纹）；`manifest.seed_tree` 定为复现的最小充分信息 |

**v1.4.1 → v1.4.2（3 条，账目更正；不改判据、阈值、网格）**

| # | 问题 | 处理 |
|---|---|---|
| 62 | **`dataset_key` 里没有"复制号"**：`RET` 的 K=10 需要 10 份同统计身份的面板，旧 key 无法表达，`D` 因此算不出来 | key 末位改为 **`ds`（面板复制号，默认 0）**；`RET` 用 `ds=0..9`；重算 §3.6.7 全部测试向量，并新增 `A` vs `R`（仅 `ds` 不同）必须异值的一行（AC-36） |
| 63 | **`D=194` 数错**：旧表把 `RET` 的 9 份新面板记成 1 份 | `D` 由 `distinct panel_ref` 枚举得 **202**，§5.5 补逐块分解（27/81/12/12/9/18/18/12/4/**0**/**9**/**0**/**0**）；同步 `chi2_{202}(0.99)=251.7`、§3.5、AC-08、AC-01 |
| 64 | **"复用"只是推理，不是事实**：key 恰好相等才复用，格式改一个字符就静默分裂成两批面板，消融被污染而无告警 | 新增 **§3.6.3.1**：配置层显式 `panel_ref`，实现只按 `panel_ref` 取面板；`metrics.json` 每 cell 记 `panel_ref`+`returns_sha256`；新增 **AC-38**（计数=202 且逐块分解逐项相等） |

> 方案甲（`one_good` 复用 `null` 面板、仅在 κ 上加 τ）被否决，理由见 §3.6.3.2：
> 加常数不改相关结构，配对方差确实更小，但 `one_good` 将不再是"代表性备择面板"，
> 且 9 个 `S` 点共享噪声会使 `threshold_sr` 的 logistic 拟合失去标准误，AC-13 的容差无从校准。

**v1.4.2 → v1.4.3（4 条，记法消歧；不改判据、阈值、网格）**

| # | 问题 | 处理 |
|---|---|---|
| 65 | **§5.1 的 N/T 列两种记法混用**：`NUL` 写 `126/266/520`，`ALT`/`MIX` 写裸 `3`，而"3 = 3 个取值"这个约定全文未定义，§5.5 却按它去读 | 全表改为**显式取值集合**（`ALT` → `126/266/520` × `4/20/100`），列名加注"（期数）/（策略数）"，表下给出手算示例；AC-01 禁止 `int` 简写 |
| 66 | **`§3.6.4` 的 `f_t`/`e_{i,t}` 维数与 §3.1 的索引约定冲突**（`t` 既像期又像 trial），且**抽样顺序未钉死**——顺序一变即不同面板 | §3.1 增下标约定（`i`=策略 1..T，`t`=期 1..N；`f` 有 **N** 个，`e` 有 **T×N** 个）；§3.6.4 写明"先 `f_1..f_N`，再按 `i=1..T` 内层 `e_{i,1..N}`"，并声明**禁止列优先/并行取流**；新增 **AC-39** 黄金对照 |
| 67 | **`null+one_good` 无法落成单值 `composition`**，但它要进 `dataset_key`；`RET` 的 `K` 也被塞进该列 | §3.4 增落格规则：`S==0 → null`、`S!=0 → one_good`；`K` 另行说明为 family 数（`ds` 个数），不是 composition 维度；§5.1 对应行的该列改为"按 §3.4 落格" |
| 68 | **`RUN_SEED` 是文档常量，`config_hash` 覆盖不到**：改这一行代码，hash 不变，事后无法从工件判断数据同源 | §6.3 把 `run_seed` 升为**必填 config 字段**（无默认值）；`config_hash` 覆盖 canonical JSON 全字段；`manifest`/`metrics`/`config` 三处互校；新增 **AC-40**（负向测试：改 `run_seed` ⇒ hash 必变）。§5.7 的 `N=3` 桶与 §5.8 的幻影点 `(3,3)` 一并修正为按 `(T,N)` 的真实 13 个组合 |

> 附：§5.7 旧表把 `ALT/MIX/DEC/NEG/GAR/GAU` 整块标为"微秒级"，但 `ALT` 含 `T=100,N=520`。
> 按 `T` 重算后：`T=4` 占 **30.6%** 的调用却几乎无成本，`T=100` 占 **41.0%** 的调用却应占绝大部分机时。
> **调用占比不能当机时占比用**——这是 `COSTP`（§5.8）存在的理由。

**v1.4.3 → v1.4.4（2 条，澄清"要不要改公式"；不改判据、阈值、网格）**

| # | 问题 | 处理 |
|---|---|---|
| 69 | §9.1 的 `correction≈1` 是个**未定义符号的近似**，且"t5 约 +1.5%"只在中等 `S` 成立——`S=1.8` 时实为 `+4.7%`，在 `N=266` 上等价于门槛上偏 **0.034 SR**（占 AC-13 容差的三分之一） | §9.1 写出精确式 `correction = 1 − skew·s + 0.25·kurt·s²`（t5 → `1+1.5s²`），标明 `SR_req^DSR` 是**下界**、上偏幅度与符号；这是 §9.1 唯一的近似项，**与坐标朝向无关** |
| 70 | "改实现朝向"与"改公式"的边界没说清，容易把**模型错误**当成转置去调 index | §3.1 增注：① 矩阵转置 = 搬运方向错（公式不动，靠 AC-41 的维度断言抓）；② `f` 按策略索引 = **模型错**（`corr≡0`、τ 失义，靠 `mean|r̂|≈rho` 抓）。新增 **AC-41** 两类断言 + `(4,520)`/`(100,126)` 交叉对照 |

**v1.4.4 → v1.4.5（2 条，加实现自检探针）**

| # | 问题 | 处理 |
|---|---|---|
| 71 | 探针阶段没有**判据**：只要"数字变了"就算修好，无法区分"改对了"与"改出了另一个数"。实测中出现过门控 FPR 读数 `0.94`，而 `observations` 传错时 DSR 门槛会升到 ≈10.9 年化、FPR 应趋近 0——该读数用"三项缺陷"解释不通，说明决策路径上还有别的分叉 | 新增 **§5.9 实现自检探针**（布局 / 相关 / τ 回收 / 门控 FPR 四项，pre-D8 阻断）；给出 `T=4` 的 FPR 预期带 **0.2%–0.8%** 与定位方法（dump `status/trials/observations/effective_trials/dsr[i]`） |
| 72 | 探针 reps 与判据不匹配：`p≈0.6%` 时 500 reps 只有约 3 个事件，判不了 0.3% vs 0.6% | §5.9 按探针分档给 reps（20/50/200/**≥500**），并明确 FPR 探针只判"量级"（≤3% 通过），点值留给 D8 的 8000 reps |

**v1.4.5 → v1.5.0（4 条，把 4 处"预期值"改成实测值）**

> 依据：全网格首次跑通（269 cell / 617,000 调用 / 28 分钟单核，`docs/mc_gate_full_run_results.md`）。
> 本次**只改预期值与归因阈值**，不动任何判据的形式、网格、种子树或门槛公式。

| # | 问题 | 处理 |
|---|---|---|
| 73 | `H0-OCN` 的 `T=4` 退化率区间 `[0.15,0.35]` 与实测不符：实测 **55.0%/61.5%/70.5%**（rho=0/0.3/0.7）。根因是 `T=4` 只能切 2+2（`min_clusters=2`、`min_cluster_size=2`），而 `seeds={17,29,43,71}` 对 4 取模只有 `{1,1,3,3}` 两个不同起点 | 区间改为 **`[0.45,0.75]`**；`T=20/100` 的 `<0.01` 加注"**仅 `rho≤0.3` 成立**"（`rho=0.7` 时 6.5%/9.75%） |
| 74 | `H2` 的先验 "`T≤20` 时 `Δ<−5pp`" 不成立：实测 `Δ_total` 为 `T=4` **−4.1pp**、`T=20` **−0.8pp**、`T=100` **+0.3pp**。且 `Δ_stat` 在 `T=4` 是 **+13.1pp**（统计层是赚的），负号全部来自 `Δ_ocn=−17.2pp` | 先验改为 "`T=4` 时 `Δ≈−4pp`、`T=20/100` 时 `\|Δ\|<1pp`"，并显式标注 `Δ_stat` 为正；§5.4 的归因阈值 `\|Δ_ocn\|>5pp` → **`>15pp`** |
| 75 | `H4` 的 `T*∈[60,80]` 无法对应：实测 `T_eff=3.33`（T=100, rho=0.3），且 `T*` 在全文中没有定义 | 删去该项；主判据 `\|threshold−SR_req\|≤0.10` 保留（实测差 −0.0028 / +0.0649，**通过**） |
| 76 | `H6` 的预测 `>40%` 高估：实测 `K=10` 的 FPR **16.1%**（K=1 为 0.90%）。原推导用 `NUL` 的**上界** `p=6.5%` 得 48.9%，而实测单族 FPR 只有 0.90% | 预测改为 **`≈16%`**，注明"独立预测 `1−(1−0.009)^10=8.6%`、实测 16.1%、放大 **17.9×**"，并说明**幅度依赖单族真实 FPR**（门控越保守，重试风险越小） |

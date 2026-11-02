# MC-GATE-VALIDATION-V1：种植 alpha 的 Monte Carlo 门控验证合同

状态：**DRAFT v1.4（待冻结）**。冻结后本文件 SHA-256 写入 `runs/mc_gate_<runid>/manifest.json` 与
`metrics.json.contract_hash`；冻结后不得修改任何判据、阈值、网格或样本量。

修订留痕见 §18（v1.0→v1.1：21 条；v1.1→v1.2：14 条；v1.2→v1.3：15 条；v1.3→v1.4：6 条）。

> v1.4 只改**口径与测量纪律**，不改任何科学判据、阈值、网格规模或 OCN spec。依据与论证见
> `docs/mc_gate_v14_revision_decision.md`。

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
r_i,t = mu_p + sigma_p*sqrt(rho)*f_t + sigma_p*sqrt(1-rho)*e_i,t
```

`b_i ≡ sqrt(rho)` 固定，使 `corr=rho`、`Var=sigma_p²` 精确。

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

- **单格**：`|z_cell| ≤ 4.5`（单格 `p=6.8e-6`；若沿用 `|z|≤2`，`194` 个数据集全过概率仅
  `0.0455→e^{194·ln0.9545}=1.2e-4`，正确实现必挂）。
- **全局**：`Σ z_cell² ≤ 242.8 = chi2_{194}(0.99)`（Wilson–Hilferty），df 用**独立数据集数**
  `D=194`（共享 CRN 的规则/q/变体不重复计数，见 §5.5）。
- 单格越界 → 该格换 seed 重抽一次；再越界 → 标 `dgp_unverified`，排除该格判定。
- 全局越界 → 先对贡献最大的 3 格逐格重抽；仍越界 → 标 `dgp_global_shift` 并**报告偏差量级**；
  仅当 `|mean SR − S| > 0.05` 时才作废该 dist 档（`z` 级偏差不改变 FPR/功率结论）。

---

## 4. 覆盖

```text
S ∈ {-0.6,-0.3, 0, 0.15,0.30,0.45,0.60, 0.90,1.20,1.80}
```

`S<0` 负对照/诱饵；`S=0` 假阳性；`[0.15,0.60]` 加密（真实策略在 0.48 与 −0.55/−0.60/−0.006）；
`[0.90,1.80]` 高信噪。

---

## 5. 网格（269 个决策单元 / 194 个独立数据集 / 617,000 次调用）

### 5.1 块表

| 块 | composition | S | N | T | rho | F/变体 | 决策单元 | 数据集 | reps | 调用 |
|---|---|---|---|---|---|---|---|---|---|---|
| `NUL` | null | 0 | 126/266/520 | 4/20/100 | 0/0.3/0.7 | t5 | 27 | 27 | 8000 | 216,000 |
| `ALT` | one_good | 9 非零 | 3 | 3 | 0.3 | t5 | 81 | 81 | 2000 | 162,000 |
| `MIX` | mixed | 0.45/0.90 | 3 | 20/100 | 0.3 | t5 | 12 | 12 | 2000 | 24,000 |
| `DEC` | mixed_signed | 0.45/0.90 | 3 | 20/100 | 0.3 | t5 | 12 | 12 | 2000 | 24,000 |
| `NEG` | all_negative | −0.6 | 3 | 3 | 0.3 | t5 | 9 | 9 | 1000 | 9,000 |
| `GAR` | null+one_good | 0/0.6/1.2 | 3 | 4/100 | 0.3 | garch | 18 | 18 | 1000 | 18,000 |
| `GAU` | null+one_good | 0/0.6/1.2 | 3 | 4/100 | 0.3 | gauss | 18 | 18 | 1000 | 18,000 |
| `FRQ` | null+one_good | 0/0.3/0.6 | 252/504 | 4/100 | 0.3 | t5, p=252 | 12 | 12 | 1000 | 12,000 |
| `SVL` | null+one_good | 0/0.6 | 266 | 4/100 | 0.3 | shortvol | 4 | 4 | 1000 | 4,000 |
| `BYA` | null | 0 | 266 | 4/100 | 0.3/0.7 | t5, fdr=BY | 4 | 0（复用 NUL） | 2000 | 8,000 |
| `RET` | null | 0 | 266 | 20 | 0.3 | K=1 / K=10 | 2 | 1（仅 K=10） | 2000 | 22,000 |
| `ABL` | null+one_good | 全 10 | 266 | 100 | 0.3 | 4 变体 | 40 | 0（复用 NUL/ALT） | 1000 | 40,000 |
| `SWP` | null+one_good | 0/0.6/1.2 | 266 | 4/100 | 0.3 | t5 × 5 个 q | **30** | 0（复用） | 2000 | 60,000 |

**决策单元 269 个；独立数据集 `D=194`；调用 617,000 次。**

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

报告必须给三项。**结论归因**：若 `|Δ_ocn| > 5pp`（预计 `T=4` 成立），则"被支配"的表述必须写成
"`T=4` 的损失主要来自 OCN 分区退化（`Δ_ocn`），统计层损失为 `Δ_stat`"，不得把缺陷算到 DSR 头上。

### 5.5 独立数据集计数（`df` 的依据）

`D = 27+81+12+12+9+18+18+12+4+1 = 194`。`BYA`/`SWP`/`ABL`/`RET-K1` 与其父数据集共用 CRN 与同一
seed ⇒ 不新增数据集、不进入 `chi2` 的 df，但其 `z_cell` 必须与父数据集**逐位相同**（AC-27）。

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

按 §5.1 网格清点的真实构成：

| 调用类别 | 次数 | 占比 | 成本量级 |
|---|---:|---:|---|
| `N=3`（ALT/MIX/DEC/NEG/GAR/GAU） | 255,000 | 41.3% | 微秒级（`correlation_matrix` 3×3、OCN 3 点、DSR N=3） |
| `N≥126`（NUL/FRQ/SVL/BYA/RET/ABL/SWP） | 362,000 | 58.7% | 毫秒级起（`correlation_matrix` O(T²N) + OCN `repeats×max_iter`） |

**新规则**：降级按**格类**（`(T,N)` 分桶）触发，不按全局平均。

```text
若 ĉ(T,N) > 阈值          则该 (T,N) 桶内所有 cell 的 reps 降一档
                          N=3 桶（ALT/MIX/DEC/NEG/GAR/GAU）默认不降
降档序列: 8000→4000, 2000→1000, 1000→500
```

该决定**只依赖 `COSTP` 的计时**，不依赖任何结果，因此不构成可选停止；AC-01 仍**不得硬编码总调用数**。

### 5.8 `COSTP` 成本归因 pilot（v1.4 新增，D1 强制前置）

**目的**：把 §5.7 的预算从"照抄的常数"换成"本机实测的成本模型"。这是本文档唯一有权决定最终 reps 的机制。

**设计**：

- 覆盖 `(T,N)` 分桶边界各一点：`(3,3) (4,126) (4,266) (4,520) (20,266) (100,126) (100,266) (100,520) (100,504)`。
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
   v1.3 的预算假设错在"把 41% 的预算分给了微秒级调用"。


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
SR_req^DSR = sqrt(p)*[ sqrt(2*ln(T_eff)/N) + z_{1-q}/sqrt(N-1) ]     # correction≈1（t5 约 +1.5%）
SR_req^BH  = z_{1-q/T}/sqrt(N/p)
实际门槛   = max(DSR, BH)
```

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
| **H0-OCN**（新增） | `T=4` 层被 OCN 分区退化限制 | `OCNP` pilot：`T=4` 的 `abstain_rate ∈ [0.15,0.35]`、`T=20/100` 的 `<0.01` |
| **H1a** | 门控守住名义 FPR | **`NUL` 全部 27 格** `fpr_family ≤ 6.5%` |
| **H1b** | `rho=0` 时 `T_eff` 被噪声底压低但 FPR 不破 5% | 报 `m_measured/(0.8/√N)` 与 `T_eff/T`；`T=100` 的 `T_eff` 按 N=126/266/520 分别落在 [10,14]/[15,19]/[20,25] |
| **H2** | 等假阳性下的效率 | `Δ` 及 §5.4 分解；先验 `T≤20` 时 `Δ<−5pp`、`T=100` 时 `\|Δ\|<5pp` |
| **H3** | 假阳性差随 T **单调不减** | `fpr(B0)−fpr(GATE)` 单调不减，`T=20/100` 时 ≥10pp |
| **H4** | 实测门槛 = `max(DSR,BH)` 解析值 | **仅 `ALT`（9 个 S 点）**：`\|threshold_sr − SR_req(m_measured)\| ≤ 0.10`；`T*∈[60,80]` |
| **H5** | 门控更少把诱饵当真 | **仅 `DEC`**：`P(选中诱饵)`、`p_deploy_neg` 显著低于 B0/B1 |
| **H6** | 跨家族重试击穿门控 | `RET`：`K=10` 的门控 FPR 显著高于 `K=1`，预测 **>40%**（独立族上界 `1−(1−0.065)^10 = 48.9%`） |

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

---

## 14. 验收标准（[B]=阻断）

### A. 契约

- **AC-01 [B]** schema 校验；`cells`、`datasets`、`calls` 三项必须**等于由 config 推导的值**
  （当前 269 / 194 / 617,000），**不得硬编码常数**——这样 §5.7 的降级不冲突。
- **AC-02 [B]** `manifest` 的 `contract_hash/config_hash/binary_sha256/git_commit` 与 `metrics.json` 一致。
- **AC-03 [B]** 无时间戳/主机名/耗时；gzip **`mtime=0`**。
- **AC-04** `decisions.jsonl.gz` 每行含 `cell_id/rep_id/rule_id/selected/index/abstain/returns_sha256`。

### B. 确定性

- **AC-05 [B]** `--shard 1/1` 与 `4/4` 的 `metrics.json` 字节相同（依赖 §13 的分片键与排序）。
- **AC-06 [B]** RNG 按 **§3.1.1** 的播种/输出规格自实现 PCG64 + 固定 Box–Muller（禁
  `std::normal_distribution`）；**必须逐位命中 §3.1.1 的 golden 向量**（`seed=42` / `seed=12345` 前 8 个输出）；
  前 1e6 输出自检并与 Python 标准库实现比对。
- **AC-07 [B]** `garch` 构造断言 `omega=1−α−β`、`corr_t ≡ rho`（逐期）、无条件 `Var=sigma_p²`；
  `shortvol` 断言 `|skew+2.5|≤0.05`、`|Var/sigma_p²−1|≤1e-6`。

### C. 数值

- **AC-08 [B]** τ 回收：单格 `|z_cell|≤4.5`（SE 含峰度修正）+ 全局 `Σz²≤242.8`（df=194）；
  越界按 §3.5 分级处理（重抽单格 → 标 `dgp_unverified` / `dgp_global_shift`）。
- **AC-09 [B]** `S=50` 金丝雀：`GATE selected==true` 且 index 正确。**位于 `mc_gate_smoke.json`
  的独立 canary 单元（50 reps），不计入 617k**。
- **AC-10 [B]** `NEG` 只有 GATE 断言（零选中率 ≥99.5%）；B0/B1/B2/B3/B4 只报实测，解析参考
  （N=126，B1）≈1.8%/7.5%/22%，B4 ≈20% 量级。
- **AC-11 [B]** 基线解析锚仅施加于 **`rho=0` 的 9 个数据集**（含 T=20）：B0 ∈
  [0.90,0.97]/[0.99,1.0]/[0.999,1.0]，B1 ∈ [0.16,0.21]/[0.60,0.69]/[0.985,1.0]；
  `rho>0` 用 §7.1 相关解析带 ±3pp。
- **AC-12 [B]** 5 布尔全真时 `candidate_selected == winner_frozen` 100%。
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

- **AC-27 [B]** `BYA`/`SWP`/`ABL`/`RET-K1` 的 `returns_sha256` 与父数据集相同，且 `z_cell` 逐位相同。
- **AC-28 [B]** `G_full` 与真实门控在全部重叠配置上 `artifact_hash` 完全相同（证明 `ABL` 只重组不改统计量）。
- **AC-29 [B]** 收益 ≤ −1.0 导致的 `INVALID_INPUT` 记 ABSTAIN，进程不崩、不丢 rep。
- **AC-30 [B]** `OCNP` pilot 必须先于主 run 完成，结果写入 `metrics.json.pilot`，且**不改变**任何判据。
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

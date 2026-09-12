# Phase 4A/4B 正式 OOS 对照合同（预注册）

冻结时间：2026-09-12（任何 OOS 运行之前）
证据目标：把 Phase 4A/4B 从 `REFERENCE_ONLY` 推进到正式 walk-forward OOS 判定

## 1. 研究问题

1. **Q1（4A）**：在相同 PIT 数据与相同下游优化器下，Gaussian BL 与 FFV 后验更新的
   Posterior Direct 是否有净收益差？
2. **Q2（4B）**：posterior covariance 接入 NCO-MinVar（NCO-FFV）是否优于直接 capped-simplex？
3. **Q3（incumbent）**：任何后验 policy 是否在成本后稳定优于无视图的 NCO risk-only 基线？

## 2. 数据与宇宙（PIT-only）

- 来源：`phase1e_pit_120_2020plus.parquet`（PIT 120-symbol 宇宙，模型层既用的同一张源表）。
- 预备脚本按 `(timestamp, symbol)` 计算：
  - **情景收益** `r[i,t] = close(i,t) / close(i,t−1) − 1`（回看，无未来信息）；
  - **前向收益** `f[i,t] = close(i,t+5) / close(i,t) − 1`（一个调仓间隔；仅用于 OOS 度量，
    绝不进入估计）。
- 每 rebalance 的参与宇宙 = 该时点 `is_tradable` 且拥有完整 252 情景历史的 symbol 子集
  （四个 arm 使用同一子集，保持下游共享）。截面有效 symbol < 60 时该步跳过并记录。
- 预备产物：`scenarios.f64`（[M,N] 行主序）、`timestamps.i64`、`symbols.txt`、
  `forward.f64`、`prep_manifest.json`（全部 SHA-256）。symbol 缺 bar 的格点记 0 并在
  manifest 记 missing mask；tradable 掩码一并导出。

## 3. Walk-forward 设计

```text
estimation_window M = 252 个 timestamp（约一年）
rebalance          = 每 5 个 timestamp 一次（首个为第 252 个）
OOS 区间           = 估计窗之后全部 timestamp，等分为 3 个不重叠子窗口
```

每个 rebalance 时点 `t`：
1. prior：`[t−252, t)` 情景矩阵 → `PriorScenarioArtifactV1`（等权样本矩，合同既定）。
2. views（确定性、PIT、mean-equality）：截面动量 z 排序后取**最高/最低各 10%**
   （各 ⌈0.1·n⌉ 条，约 24 条）symbol 建立 mean-equality view，loading 为对应单位向量；
   FFV `max_iterations=1000`（其余 reference 默认）。该收缩在首次运行 FFV 全截面
   NUMERICAL_FAILURE 后、任何有效 OOS 产出前确定：
   - target = 20-timestamp 动量的截面 z-score，clip ±2.0，乘以截面收益波动尺度
     （`view_scale = 过去 252 个 timestamp 的 |r| 截面均值`）；
   - confidence 固定 `0.30`；`observation_variance = view_scale²`；
   - `source_artifact_hash` = 本合同 SHA-256；`confidence_mapping_hash` = 固定映射
     "identity@fixed-0.30" 的 SHA-256（`6a4d...` 由预备脚本计算并写入 manifest）。
3. 四个 policy arm（下游全部冻结）：

| arm | 后验 | 下游 |
|---|---|---|
| `risk_only` | 无（prior covariance） | NCO-MinVar（既有 phase2a solver） |
| `posterior_bl` | Gaussian BL | Posterior Direct：capped-simplex，`risk_aversion=1.0`，`max_single_weight=0.10`，`target_investment=1.0` |
| `posterior_ffv` | FFV（同 views；`min_probability` 等取 reference 默认） | 同上 Posterior Direct |
| `nco_ffv` | 同 `posterior_ffv` 的 posterior | NCO-MinVar（4B solver） |

   NCO 聚类统一为：估计窗相关距离 + complete-linkage 层次聚类（phase2a 实现既定方法）、
   固定切割距离 `0.70`，每步确定性重算。Posterior Direct 共享数值参数：`tolerance=1e-8`、
   `max_iterations=200000`（两 arm 完全一致；运行前因日频尺度下投影梯度收敛慢而放宽，
   此时仍无任何有效 OOS 产出）。
3b. **FFV 数值降级阶梯**（两引擎同视图集，逐级收缩，确定性）：两端各 10% → 各 5% →
   各 1 条 → 无视图（后验=prior 奇偶）。逐级尝试至 FFV 数值成功；该步视图层级记入报告。
   全 run 降级步（<10% 层级）占比 > 20% 时 run 作废重设计，不得丢弃子窗口。
4. 成本与度量：单边 `cost_bps = 10`，净收益
   `net_t = Σ_i w_i·f[i,t→t+5] − cost_bps·1e-4·Σ_i|w_i − w_i^prev|`（首个周期无成本项）。

## 4. 评估口径（预注册）

- 每 arm：净收益序列 → 年化 Sharpe（按 rebalance 周期折算）、累计净收益、最大回撤、平均换手。
- 配对差（challenger − `risk_only`）：block bootstrap（block=10 rebalance 周期，2000 draws，
  seed 20260912）95% CI；**3 个 OOS 子窗口点估计方向一致性**。
- 多重检验：3 组 challenger 配对比较做 FDR（BH）；存活者的 Deflated Sharpe（试验数=4 arms）。
- 全部在 C++ driver 输出 `formal_oos_report.json`（含每 arm 逐期收益、权重诊断、每步
  posterior/solver artifact hash），Python 侧独立复算 hash 与统计量。

## 5. 判定规则（先于结果冻结）

1. 任一 challenger 须同时满足：(a) 3/3 子窗口净 Sharpe 差点估计 > 0；(b) bootstrap 95% CI
   下界 > 0；(c) FDR 校正后 p < 0.05 —— 才构成"优于 incumbent"的正式证据。
2. `posterior_bl` vs `posterior_ffv` 与 `nco_ffv` vs `posterior_ffv` 的两两对照同样用
   规则 1 的 (a)(b)（不涉 incumbent）。
3. 无 challenger 通过 → 冻结 incumbent（`risk_only`），Phase 4A/4B 判定
   `formal_oos_complete=true, winner=incumbent_or_none`，家族按结果登记关闭或续期。
4. 任何 arm 在任一步失败关闭（数值、不可行、hash 不符）→ 整个 run 作废，修复后以新
   run id 重跑；不得挑选性丢弃子窗口。

## 6. 边界

- 本对照只评估 **policy 家族**，不评估预测层（view 为机械 PIT 动量，与已封闭的模型层无关）。
- 结果不改变 `eligible_for_official_risk` 的生产语义；实盘接入仍按 live V1 合同独立把关。
- 预备脚本、driver、 verifier 的实现 hash 在运行前一并冻结进 run manifest。

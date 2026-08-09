# Phase 3B 状态：Conditional Tail Risk

更新时间：2026-08-05

## 当前判定

Phase 3B 已完成固定组合经验尾部风险第一阶梯，并补齐 N≤200 同步资产向量 GARCH-FHS
reference path；阶段整体尚未退出：

```text
engineering_scaffold_complete = true
phase_exit_eligible = false
promotion_eligible = false
```

## 已交付：Phase 3B 预注册合同

- 新增并在未来 OOS 尚为空时修订 `runs/phase3b-preregistered/preregistered_contract.json`，canonical
  SHA-256 为 `4a81f722465ac3e3124e6ad8d770a6305cc04dea408393b2a3308247f0267cf7`。
- canonical 口径固定为 UTF-8、`sort_keys=true`、无空白 separators，并在计算时只移除根级
  `contract_sha256`；`tools/verify_phase3b_preregistered_contract.py` 会复算合同哈希和六份来源报告哈希。
- 已有三个 proxy 时间窗口明确标记为 `OBSERVED_BEFORE_REGISTRATION`。每个窗口对应 GradNorm/PCGrad
  两份 C++ proxy Replay，共六份报告；它们只能用于 validation 和 research diagnostic，不能充当新的
  formal OOS、阶段退出或生产晋级证据。
- `future_new_oos.registered_windows=[]`、`window_count=0`、`status=UNAVAILABLE`；至少三个未来新窗口仍是
  后续正式 gate，当前没有伪造日期或把旧窗口重新命名为新 OOS。
- 合同冻结 `alpha=0.95`、upper-tail loss FZ0、strict/auxiliary/strict-intercept ESR、Newey-West HAC lag 4、
  四点 EVT 网格、固定 PIT Expectile 特征、estimator ladder 和禁止 fallback/averaging/post-hoc 选择的政策。
- 预注册只冻结未来施工和评价口径，不改变现有 `tail_risk` 数值，也不改变
  `phase_exit_eligible=false`、`promotion_eligible=false`。

## 已交付：TAIL-EMPIRICAL-ES

- `estimate_tail_risk` 保持固定组合、portfolio-return-series 和 Rockafellar-Uryasev 经验 VaR/ES 口径；带权离散尾部质量点分配、符号和 `return_cvar` 已有 parity 测试。
- 新增 `backtest_tail_risk`，对逐期 realized return、VaR loss 和 ES loss 做时间顺序、future-data、有限性和 `ES >= VaR` 校验。
- 联合回测输出异常次数/率、ES 违约次数/率、平均 VaR 超额损失、平均 ES 超额损失、Kupiec LR/p-value、Christoffersen transition/LR/p-value、upper-loss orientation FZ0 mean score 和确定性 input/artifact hash。
- FZ0 使用 `I(L>VaR)(L-VaR)/((1-alpha)ES) + VaR/ES + log(ES) - 1`；`ES<=0`
  返回 `FZ0_DOMAIN_FAILURE`，不会对未定义的 log-domain 输出分数。
- 新增 `serialize_tail_risk_backtest_artifact`；`reference_price_quality=PROXY` 或 `ARRIVAL_PROXY` 时强制 `promotion_eligible=false`。
- 当前接口只处理固定组合收益序列，不把缺失的费用、真实滑点、reference provenance 或 PIT 因子数据伪装成观测值。

## 已交付：Regression-based ESR robust suite

- 新增 Bayer-Dimitriadis 风格 joint VaR/ES regression：strict ESR 的 quantile/ES 两部分都使用 ES
  forecast；auxiliary ESR 的 quantile 部分使用 VaR forecast、ES 部分使用 ES forecast；strict-intercept
  对 `realized_return - ES forecast` 拟合 quantile 与常数 ES 方程。
- 三种版本使用同一 `G1=0`、`G2(e)=-1/e` FZ 联合损失；strict/auxiliary 检验 ES intercept/slope
  是否为 `(0,1)`，strict-intercept 同时报 two-sided 与 underestimation one-sided p-value。
- 默认协方差升级为 `KERNEL_BREAD_EMPIRICAL_SCORE_NEWEY_WEST_HAC_V1`：核密度 bread、逐期 FZ score
  outer product、score centering、Bartlett/Newey-West lag 4；`KERNEL_IND_CORRECT_SPEC_SANDWICH_V1`
  仍作为显式 reference path，不再是默认正式推断候选。
- 时间顺序、future observation、`ES<=0`、`ES<Var`、短样本、优化不收敛和协方差奇异均失败关闭；
  三个 variant 及 suite 均有 deterministic hash，proxy artifact 继续强制 `promotion_eligible=false`。
- 合成异方差 Normal oracle 恢复 `(0,1)` 邻域；三版本、序列化、重放和失败路径由
  correct-spec/robust 分离、HAC lag hash、重放和失败路径由 `test_tail_risk_esr` 覆盖。

六份旧 proxy 的 robust research diagnostic 位于
`runs/phase3b-proxy-diagnostics-robust-v2/report.json`：每份使用 40 期 expanding history 产生 85 个预测，
计算状态全部成功，但 strict ESR 在 4/6、auxiliary ESR 在 5/6 个窗口以 5% 水平拒绝经验 VaR/ES；
strict-intercept 未拒绝。该结果只说明旧 proxy empirical baseline 校准不稳定，不能作为未来 formal OOS。

## 已交付：TAIL-GARCH-FHS-ES

- 新增确定性的 Gaussian-QMLE GARCH(1,1) 过滤：`omega>0`、`alpha>=0`、`beta>=0`、`alpha+beta<1`，固定一步 forecast 和 variance floor。
- `PORTFOLIO_RETURN_SERIES` 枚举全部有效 standardized residual；`ASSET_VECTOR_SYNCHRONIZED`
  对 N≤200 各资产独立过滤尺度，但每个场景严格复用同一历史 residual row，再按冻结组合权重流式聚合损失。
- 资产向量 artifact 输出每个 symbol 的 GARCH 参数、stationarity margin、forecast variance、standardized
  residual 均值/方差、Ljung-Box、squared-residual Ljung-Box 和最大残差。
- `synchronized_residual_rows=false`、独立逐资产抽样、非有限数据、未来时间戳、未冻结规格、N>200
  和残差诊断越界全部失败关闭；同一输入重复运行产生相同 artifact hash。
- 合成 GARCH recovery、单组合/两资产完全同步 parity、确定性 replay、proxy promotion gate 与失败路径已加入
  `test_portfolio_math`。

该 reference path 保留历史 residual-row 中的同期相关和共同极端事件，但仍不能证明 full-A
factor-specific FHS、三个真实 OOS 窗口或真实交易成本后的生产 CVaR。

## 已交付：TAIL-GARCH-FHS-EVT-ES（N≤200 reference）

- 单组合与 `ASSET_VECTOR_SYNCHRONIZED` 共用同一组合-loss POT-GPD splice；资产向量逐资产过滤 GARCH，
  再按同一 residual row 合成组合损失，不独立抽样。
- 固定阈值网格 `[0.75,0.80,0.85,0.90]`，使用带场景权重的 moments V1；原始和有效超额样本数均需
  达标，至少 3/4 个候选有效，shape spread≤0.35、relative ES spread≤0.25，选择最低有效稳定阈值。
- artifact 逐阈值记录状态、tail mass、有效样本、shape/scale、VaR/ES、连续性和概率归一误差；
  `xi=0`、`xi<0`、`xi>=1`、非等权、稳定性失败和非同质相关双资产列置换 parity 均有测试。
- 这仍是 N≤200 数学/reference path；没有极端尾部真实 OOS 或 full-A factor-specific 证据。

## 已交付：TAIL-EXPECTILE（固定 PIT 条件 ALS reference）

- 保留无条件标量二分 oracle，并新增独立 `FIXED_PIT_LINEAR_ALS`：固定 feature matrix、ridge、IRLS、
  系数、预测 Expectile、非对称 score、一阶条件残差和 deterministic hash。
- 每个训练 feature 必须严格早于对应 target，forecast feature 必须在 `decision_at` 可得；未来、错位、
  未冻结 feature/solver spec、短样本和非有限输入全部 fail closed。
- 合成条件 oracle、平移、正齐次、PIT future guard、artifact claim boundary 已由
  `test_conditional_expectile` 通过。当前不声称 CARE-SAV，也没有真实 PIT feature OOS 结果。
- `TAIL-EXPECTILE-TAYLOR-MAPPED-ES` 仍拒绝输出，直到 Taylor 映射参数/校准规则单独冻结；当前没有把直接 Expectile 冒充 ES。

实现位置：

- `portfolio_math/include/portfolio_math/tail_risk.h`
- `portfolio_math/include/portfolio_math/conditional_expectile.h`
- `portfolio_math/src/tail_risk.cpp`
- `portfolio_math/src/conditional_expectile.cpp`
- `portfolio_math/src/tail_risk_esr.cpp`
- `portfolio_math/tests/test_portfolio_math.cpp`
- `portfolio_math/tests/test_tail_risk_evt.cpp`
- `portfolio_math/tests/test_conditional_expectile.cpp`
- `portfolio_math/tests/test_tail_risk_esr.cpp`

## 验证

- `test_portfolio_math`：通过经验 VaR/ES 手算、带权质量点、平移、future guard、联合异常、FZ0
  upper-loss orientation/domain 和 proxy promotion gate。
- EVT、conditional Expectile、ESR robust 定向测试与预注册 self-check：通过；全量 C++ CTest 在本轮最终收口时重跑。

## 未完成与边界

- Taylor-mapped Expectile-ES、真实 PIT conditional Expectile/CARE-SAV、full-A factor-specific FHS 和至少三个
  未来新 purged OOS 窗口仍未完成。
- 经验 backtest 只提供研究证据；在 `RESEARCH_PROXY` 下不能解释为真实净收益 CVaR 或生产风险门禁。
- Feature-PGD 延期不阻塞本阶段；Phase 3A 真实 PIT industry/style exposure 仍是独立数据源缺口。

## 下一步

下一步先取得未参与本轮选择的真实 PIT feature validation，独立验证 conditional Expectile 的 tau/coverage
稳定性；只有通过后才实现 Taylor mapping。FACTOR-PIT-EWMA 和完整 large panel 到位后再进入 full-A
factor-specific FHS。未来出现
预注册之后的新数据时，才登记并运行至少三个新的 purged OOS 窗口；当前三个旧 proxy 窗口继续只作
validation/research diagnostic。

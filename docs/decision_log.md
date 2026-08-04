# 项目想法记忆

> 更新日期：2026-07-27  
> 作用：记录提出但不一定适合立即实施的方向，避免想法丢失或被错误地塞入当前版本。  
> 原则：想法不是当前版本需求；是否实施由依赖、验证价值、运行成本和版本目标共同决定。

## 1. 决策原则

收到新想法后按以下三类处理：

1. **当前核心能力**：直接影响正确性、可复现性或当前版本目标，纳入发布门槛。
2. **当前离线评估能力**：有助于判断模型是否可靠，但不进入生产热路径。
3. **未来候选能力**：依赖尚未满足或当前收益不明确，记录触发条件后延期。

任何研究功能都不能绕过模型制品校验、组合、订单规划和风险管理。

## 2. 已记录想法

### 2.1 Attention Analysis

用户目标：查看时间 Transformer 在预测时重点关注哪些历史交易日。

当前判断：**未来候选 / V1.1 可选研究项**。

不作为 V1.1 发布门槛，原因：

- 当前更优先的是修正标签、训练所有输出头并完成真实 ONNX 三方一致性；
- attention 权重不等于因果解释，必须同时具备 occlusion 或 gradient 对照；
- 完整 attention 矩阵开销较大，不应进入 C++ 热路径。

适合实施的触发条件：

1. TemporalTransformerV1.1 已完成训练且所有输出语义有效；
2. PyTorch、ORT Python、ORT C++ 预测一致；
3. 已有至少一个通过基础质量门槛的模型；
4. 研究排期不阻塞 Feature Ablation 和基线评估。

届时增加热力图、attention rollout、head 统计、稳定性和 top-attention occlusion；只在 Python
离线分析中运行，生产 ONNX 继续保持六输出。

### 2.2 Feature Ablation Pipeline

用户目标：分析不同特征或特征组对模型和交易结果的贡献。

当前判断：**纳入 V1.1 离线晋级评估**。

原因：

- 当前只有 23 个固定 `BAR_V1` 特征，适合先做语义分组；
- 能发现无效、重复、泄漏或只在单一时期有效的特征；
- 不改变生产模型协议和 C++ 推理接口；
- 对是否值得继续扩大 Transformer 架构有直接决策价值。

实施顺序：

1. M4 Pro 阶段完成 7 个特征组的 group-drop retrain；
2. 同时提供 inference occlusion 和 time-safe permutation 作为诊断；
3. RTX 4090 可用后，再执行逐特征 ablation、更多随机种子和完整 walk-forward 重训练。

正式结论以重新训练的 group-drop ablation 为准，不能把简单置零结果当作因果贡献。

### 2.3 多任务预测

用户目标：同时增加收益预测、波动率预测、分类等任务。

当前判断：**收益、波动率、方向分类和收益区间已经是 V1.1 核心能力**，无需再建立一套重复
模型。

当前固定任务：

| 任务 | 类型 | V1.1 状态 |
|---|---|---|
| 未来 5 日收益 | 回归 | 核心输出 |
| 持有期波动率 | 非负回归 | 核心输出，需修正现有标签 |
| 未来 5 日涨跌方向 | 二分类 | 核心输出，需概率校准 |
| q10/q90 收益区间 | 分位数回归 | 核心输出，需加入 Pinball loss |

未来可选分类：市场状态、波动率状态、趋势/反转状态、风险事件分类。

这些多分类任务暂不加入 V1.1，触发条件是：

1. 有明确的下游组合或风控用途；
2. 标签能够 point-in-time 构造且类别定义稳定；
3. 消融实验显示不会损害核心收益任务；
4. 增加输出后仍能保持 ONNX/C++ 协议清晰。

### 2.4 分钟线模型

用户计划未来增加分钟线数据。

当前判断：**未来独立版本**。

日频和分钟模型使用不同 FeatureSchema、LabelSpec、归一化参数、cadence 和模型制品。分钟线
数据准备完成后，根据数据质量决定使用 1 分钟或 5 分钟，不直接复用日频模型权重。

制品元数据统一声明 `frequency`（如 `1d`、`1m`、`5m`）和 `calendar_id`，由 C++ loader
拒绝频率缺失或不支持的制品；频率不同仍必须使用独立模型权重和窗口 cadence 实现。

### 2.5 RTX 4090 与 CUDA

用户预计一个月后可使用 RTX 4090。

当前判断：**训练加速优先，CUDA 推理后评估**。

- 当前 M1 Pro 完成 CPU/MPS 小规模训练、接口和正确性验证；
- 4090 首先用于 walk-forward、多随机种子和 ablation 重训练；
- C++ 首个可部署制品仍以 ONNX Runtime CPU 为兼容基线；
- 只有 CPU p99 不满足 cadence 时，再建立 CUDA Execution Provider 的独立性能预算。

### 2.6 横截面 Transformer V2

当前判断：**未来候选**。

进入条件：V1.1 通过软件和模型门槛，Feature Ablation 与简单截面聚合特征仍证明存在稳定剩余
增益，并且全 A 最大截面的延迟和内存预算可接受。届时优先行业分组或稀疏注意力，不默认使用
全市场 `O(N^2)` 注意力。

### 2.7 Leakage Detection

用户目标：增加数据泄漏检测，避免回测或模型评估使用未来信息。

当前判断：**纳入 V1.1 P0 强制门禁**，不是可延期的研究功能。

原因：

- 一旦存在未来特征、标签跨 split、全时期归一化或幸存者偏差，所有预测和收益指标都失效；
- Feature Ablation 和 Attention Analysis 也必须建立在无泄漏数据上；
- 当前项目已有 point-in-time 数据、lineage 和 fingerprint 基础，适合形成自动化门禁。

V1.1 强制覆盖：

1. Future Mutation Test；
2. Prefix Invariance Test；
3. 按标签时间区间执行 purge/embargo；
4. train/validation/test 主键与标签区间互斥；
5. normalizer、calibrator 和特征选择的 fit scope；
6. point-in-time 股票池、行业、ST、公司行动和复权；
7. 特征缓存 lineage；
8. 制品绑定 PASS 的 leakage report hash。

Critical/High 违规直接阻止训练和模型导出，不提供“忽略后继续”的生产开关。

### 2.8 Walk-forward + Benchmark

用户目标：增加 Walk-forward 和 Benchmark，评估模型跨时期稳定性并与其他方法公平比较。

当前判断：**纳入 V1.1 核心晋级流水线**。

Walk-forward 采用 expanding train，默认至少 3 个互不重叠 test 窗口。每个窗口独立执行：

```text
PIT 数据冻结 -> Leakage Detection -> train-only normalizer
-> 训练 -> validation 选参/校准 -> 冻结 -> test 一次 -> C++扣费回测
```

Benchmark 分为：

1. **模型质量 Benchmark**：动量、反转、等权、全现金、Ridge/Logistic、MLP、TCN、GRU、
   Transformer V1/V1.1；所有模型共用窗口、股票池、成本、组合和风险配置。
2. **工程性能 Benchmark**：数据集、特征窗口、ORT 推理、组合/风险和端到端 C++ 回测；M1 Pro
   CPU 是首个基线，RTX 4090/CUDA 后续单独报告。

模型晋级必须在多数独立窗口优于最强基线，不能只凭全时期平均收益或单次 test 结果。

## 3. 当前优先级

```text
P0  Leakage Detection + 标签/损失语义 + 真实 ONNX 三方一致性
P1  多任务 V1.1 + Walk-forward/Benchmark + Feature Ablation Pipeline
P2  原生 C++ ArtifactLoader、窗口语义、Replay 与失败关闭
P3  Attention Analysis（核心闭环完成后）
P4  分钟模型、CUDA 推理、横截面 Transformer V2
```

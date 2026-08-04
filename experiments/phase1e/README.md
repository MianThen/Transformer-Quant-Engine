# Phase 1E 实验归档

发布日期：2026-08-02
状态：研究施工完成；正式 promotion `NO-GO`

本目录归档 Phase 1E 的共享梯度诊断、PCGrad、GradNorm-4 和 C++ proxy Replay/CVaR
结果。它只包含可审计的配置、报告和摘要 artifact，不包含训练数据、模型 checkpoint、私人
路径或凭据。

## 结论

- `legacy + fixed` 继续保持 `FROZEN_CHAMPION`。
- PCGrad 与 GradNorm-4 都完成三个新的 purged OOS fold，但保留为 `report-only`。
- C++ Replay/CVaR 已生成六个 proxy artifact；由于 reference-price、费用、涨跌停、公司行动、lot
  和滑点字段不完整，全部 `promotion_eligible=false`。
- Phase 1D real drift baseline/report 不可得，因此 Phase 1E 严格退出和 challenger promotion
  均不通过。

## 文件

| 文件 | 内容 |
|---|---|
| `phase1e_experiment_report.md` | 完整实验报告、指标、限制和 Go/No-Go 判定 |
| `phase1e_pytorch.md` | 真实 PyTorch 运行环境和复现说明 |
| `pcgrad-real-e3/` | PCGrad 预注册合同与 challenger report |
| `gradnorm4-real-e3/` | GradNorm-4 预注册合同与 challenger report |
| `cpp-proxy/` | 六个 C++ proxy Replay/CVaR artifact |

## 数据与复现边界

Phase 1E 数据集不随仓库发布。数据 fingerprint 为
`4261f9b5875176dcc6badd8ab9c68d681edab42b19ad8b34456c4a44c581f554`；审计报告 hash 为
`5aedcc0bbbcd5cebda11a5e2e15be0a8420f65810c43cf7e71b77ff22c7c3c47`。真实训练使用 PyTorch
`2.8.0` CPU 环境。外部诊断 artifact hash 为
`d0e492b0c12070005bc18e69f6942a202b5eb8bcfe53bf8d70769603f89aa2aa`，本目录不复制其本地
数据和训练输出。

报告中的 absolute local path 已替换为本目录的相对 artifact path；发布 hash 以实际归档文件
为准。proxy artifact 只证明 C++ 回放和 CVaR 计算链路可运行，不代表正式净收益或经济晋级。

## Published SHA-256

```text
phase1e_experiment_report.md  758d0a9a2027c9981ecef13f548b60a7dc698dd897f74e433ca1c94b944dc683
phase1e_pytorch.md            d2074761050a35a636c43e12f757d3effb7a31d95e633c2fc9e2b1bd0f1d31d9
pcgrad challenger              37a7795a62dcd94496774883df946a24e308b02c86d9e7a35582e79f0772105b
gradnorm challenger            1cef4c1e86e82bc2530dd0a609316b9083fc95001744e642690de0a1a6f3c117
```

六个 `cpp-proxy/*.json` 的 hash 记录在发布提交的文件内容中；它们均保留
`reference_price_quality=PROXY` 和 `promotion_eligible=false`。

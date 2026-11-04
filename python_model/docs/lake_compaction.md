# 数据湖压实（Lake Compaction）

## 问题：数据层喂数远慢于引擎消费

C++ 引擎单线程可以吃下约 700 万行/秒，但数据层实际只能喂到约 5 万行/秒，端到端被卡在数据层。
根因不是 Parquet 解码慢，而是**文件数量**：真实湖里 14,105,167 行被切成了 **698,654 个文件**，
平均每个文件仅 20 行、约 4.3KB。`ArrowDatasetScanner` 的批次中位数只有 21 行 / 1.2KB，
每交付一个批次都要付一次文件打开 + 元数据解析 + 列块定位的固定开销。

在 Linux（NTFS + ntfs-3g）上实测：2024 年切片有 45,358 个文件、914,906 行，
扫描耗时 23.635 秒，即 **38,709 行/秒、1,919 文件/秒**，等效每个文件 0.52 毫秒的固定成本。

## 方案：按分区压实，不改引擎

新增四个模块 + 一个 CLI，只重排物理布局，不动 C++ 引擎、不改查询语义：

| 模块 | 职责 |
| --- | --- |
| `python/market_data/partition_planner.py` | 把 catalog 记录按布局聚成分区计划（`year_month_bucket` / `year_bucket` / `bucket` / `year`），产出目标相对路径 |
| `python/market_data/partition_merge.py` | 单分区合并：DuckDB `COPY (SELECT ... ORDER BY timestamp, symbol)`，带重复主键守卫，输出 sha256 |
| `python/market_data/catalog_publish.py` | catalog 原子替换：generation 乐观锁、`catalog.json.bak.<gen>` 备份、`os.replace` 原子落盘 |
| `python/market_data/lake_fingerprint.py` | 内容指纹：`SUM(hash(timestamp,symbol,ohlcv) % 1000000007)`，支持逐分区比对 |
| `tools/compact_lake.py` | CLI：源目录 → 新目录，压实后自动做指纹等价性校验 |

默认布局 `year_month_bucket`：保留 `(year, month)` 两级分区，因此 `scanner._windows` 的
月份窗口启发式与 `(timestamp, symbol)` 严格排序断言都无需修改，属于最保守的 P0 方案。

## 用法

```bash
# 1) 先看计划（不写文件）
python tools/compact_lake.py --source-root data/lake --target-root data/lake_compact --dry-run

# 2) 压实到新目录（源目录保持不动）
python tools/compact_lake.py \
    --source-root data/lake \
    --target-root data/lake_compact \
    --layout year_month_bucket \
    --workers 3 --memory-limit-mb 400

# 3) 切流：把 data_feed / 基准指向新目录即可，旧目录保留用于回滚
```

CLI 关键参数：`--layout`、`--row-group-size`、`--compression`、`--workers`、
`--memory-limit-mb`、`--temp-dir`、`--limit`、`--dry-run`、`--skip-fingerprint`、`--report`。

安全约束：

- 源目录与目标目录相同时直接拒绝（退出码 2），避免原地覆盖。
- catalog 更新走原子替换 + 备份，generation 不匹配时抛 `RuntimeError` 且不写备份。
- 默认对源湖和目标湖做全局 + 逐分区指纹比对，不一致退出码 1。

## 实测结果（真实湖：14,105,167 行）

压实：698,654 个文件 → **15,808 个分区文件**，耗时 434 秒（3 线程）。

等价性校验全部通过：

- 文件数 698,654 → 15,808，行数总和一致，generation 10 → 11；
- 按月聚合的（行数、最小时间戳、最大时间戳）完全一致；
- 全局逐行哈希 `SUM(hash(...))` 一致（14.1M 行逐值校验）；
- 随机 12 个分区逐值比对一致。

数据层吞吐（同一查询：2024 全年、全部标的、914,906 行）：

| 指标 | 压实前 | 压实后 | 提升 |
| --- | --- | --- | --- |
| 涉及文件数 | 45,358 | 768 | 59× |
| 迭代耗时 | 23.635 s | 0.709 s | **33.3×** |
| 行吞吐 | 38,709 行/s | 1,289,515 行/s | **33.3×** |
| 峰值内存 | 8,421 MB | 584 MB | 14.4× |

全湖（14,105,167 行、15,808 文件）：9.986 秒，**1,412,446 行/秒**。

端到端回测基准 `run_minute_replay --target-mb 256`：

| 数据湖 | 结果 |
| --- | --- |
| 压实前（698,654 文件） | 无法完成：9GB 与 13GB 内存上限下均被 OOM 杀死（13GB 运行 7 分钟后触发 cgroup OOM） |
| 压实后（15,808 文件） | 完成，全流程 84.8 秒，峰值内存 908 MB |

压实后各阶段吞吐：`arrow_decode` 2,189,405 行/s、`c_stream_replay` 728,970 行/s、
`event_replay` 550,475 行/s、`strategy_backtest` 448,706 行/s。

## 注意事项

- 压实只改变物理布局，`catalog.json` 的 `schema_version`、`bucket_count` 保持不变，
  额外写入 `layout`、`compacted_from`、`compacted_files` 三个清单级字段，供回滚与审计。
- `--limit` 属调试用：只压实前 N 个分区时目标湖是不完整的，指纹校验会如实报 FAIL（退出码 1）。
- 更粗的布局（`year_bucket` / `bucket` / `year`）文件更少，但会破坏 `scanner` 的
  `(year, month)` 月份窗口启发式，需要额外给清单打 `presorted`/`layout` 标记才能安全启用。
- 目标湖落在同一块物理盘上时，压实时会与源湖竞争 I/O；建议 `--workers` 不超过 4。

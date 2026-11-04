from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

SUPPORTED_LAYOUTS = ("year_month_bucket", "year_bucket", "bucket", "year")


def normalize_layout(name: str) -> str:
    normalized = name.strip().lower()
    if normalized in SUPPORTED_LAYOUTS:
        return normalized
    raise ValueError(f"不支持的压实布局: {name!r}")


def partition_key(record: Mapping, layout: str) -> tuple:
    layout = normalize_layout(layout)
    if layout == "year_month_bucket":
        return (int(record["year"]), int(record["month"]), int(record["bucket"]))
    if layout == "year_bucket":
        return (int(record["year"]), int(record["bucket"]))
    if layout == "bucket":
        return (int(record["bucket"]),)
    if layout == "year":
        return (int(record["year"]),)
    raise ValueError(f"不支持的压实布局: {layout!r}")


@dataclass(frozen=True)
class PartitionPlan:
    key: tuple
    layout: str
    paths: tuple[str, ...]
    rows: int
    min_timestamp: int
    max_timestamp: int
    target_relative_path: str


def _target_path(key: tuple, layout: str, *, root_prefix: str) -> str:
    if layout == "year_month_bucket":
        return f"{root_prefix}/year={key[0]}/month={key[1]}/bucket={key[2]}/part-compact-0.parquet"
    if layout == "year_bucket":
        return f"{root_prefix}/year={key[0]}/bucket={key[1]}/part-compact-0.parquet"
    if layout == "bucket":
        return f"{root_prefix}/bucket={key[0]}/part-compact-0.parquet"
    if layout == "year":
        return f"{root_prefix}/year={key[0]}/part-compact-0.parquet"
    raise ValueError(f"不支持的压实布局: {layout!r}")


def plan_partitions(
    records: Iterable[Mapping],
    layout: str,
    *,
    root_prefix: str = "bars_compact",
) -> tuple[PartitionPlan, ...]:
    layout = normalize_layout(layout)
    groups: dict[tuple, dict] = {}
    for record in records:
        key = partition_key(record, layout)
        rows = int(record["rows"])
        ts_min = int(record["min_timestamp"])
        ts_max = int(record["max_timestamp"])
        path = str(record["path"])
        if key not in groups:
            groups[key] = {
                "rows": rows,
                "min_timestamp": ts_min,
                "max_timestamp": ts_max,
                "paths": [path],
            }
        else:
            state = groups[key]
            state["rows"] += rows
            state["min_timestamp"] = min(state["min_timestamp"], ts_min)
            state["max_timestamp"] = max(state["max_timestamp"], ts_max)
            state["paths"].append(path)

    items: list[PartitionPlan] = []
    for key, state in groups.items():
        paths = tuple(sorted(state["paths"]))
        target = _target_path(key, layout, root_prefix=root_prefix)
        items.append(
            PartitionPlan(
                key=key,
                layout=layout,
                paths=paths,
                rows=state["rows"],
                min_timestamp=state["min_timestamp"],
                max_timestamp=state["max_timestamp"],
                target_relative_path=target,
            )
        )
    return tuple(sorted(items, key=lambda item: item.key))


def plan_summary(plans: Sequence[PartitionPlan]) -> dict:
    return {
        "partitions": len(plans),
        "source_files": sum(len(p.paths) for p in plans),
        "rows": sum(p.rows for p in plans),
    }

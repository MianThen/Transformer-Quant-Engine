"""数据集与查询血缘的稳定指纹。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from .config import BAR_COLUMNS, SCHEMA_VERSION


BAR_TYPES = {
    "timestamp": "int64",
    "symbol": "utf8",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",
}


@dataclass(frozen=True)
class DataLineage:
    catalog_generation: int
    schema_hash: str
    dataset_fingerprint: str
    source_fingerprints: tuple[str, ...] = ()
    query_fingerprint: str = ""

    def with_query(
        self,
        *,
        symbols: str | Iterable[str] | None,
        start: int | None,
        end: int | None,
        columns: Sequence[str],
    ) -> "DataLineage":
        normalized_symbols = (
            None if symbols is None
            else sorted({symbols} if isinstance(symbols, str) else {str(x) for x in symbols})
        )
        query = _hash_json({
            "dataset_fingerprint": self.dataset_fingerprint,
            "symbols": normalized_symbols,
            "start": start,
            "end": end,
            "columns": list(columns),
        })
        return DataLineage(
            self.catalog_generation,
            self.schema_hash,
            self.dataset_fingerprint,
            self.source_fingerprints,
            query,
        )

    def to_dict(self) -> dict:
        value = asdict(self)
        value["source_fingerprints"] = list(self.source_fingerprints)
        return value


def schema_hash() -> str:
    return _hash_json({
        "schema_version": SCHEMA_VERSION,
        "columns": [(name, BAR_TYPES[name]) for name in BAR_COLUMNS],
        "primary_key": ["timestamp", "symbol"],
        "timestamp_unit": "UTC epoch nanoseconds",
    })


def dataset_lineage(snapshot, root) -> DataLineage:
    files = []
    for item in sorted(snapshot.files, key=lambda value: value.path):
        path = root / item.path
        files.append({
            "path": item.path,
            "rows": item.rows,
            "year": item.year,
            "month": item.month,
            "bucket": item.bucket,
            "min_timestamp": item.min_timestamp,
            "max_timestamp": item.max_timestamp,
            "size": path.stat().st_size,
            "content_hash": item.content_hash,
        })
    sources = snapshot.source_fingerprints
    current_schema_hash = schema_hash()
    fingerprint = _hash_json({
        "catalog_generation": snapshot.generation,
        "schema_hash": current_schema_hash,
        "files": files,
        "sources": list(sources),
    })
    return DataLineage(
        snapshot.generation, current_schema_hash, fingerprint, sources
    )


def _hash_json(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

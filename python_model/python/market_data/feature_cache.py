"""版本化、不可变的物化特征缓存。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .config import symbol_bucket

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    version: str
    output_columns: tuple[str, ...]
    parameters: dict = field(default_factory=dict)
    code_hash: str = ""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ValueError("feature name 只能包含字母、数字、点、下划线和连字符")
        if not self.version or not self.output_columns:
            raise ValueError("feature version/output_columns 不能为空")
        if len(set(self.output_columns)) != len(self.output_columns):
            raise ValueError("feature output_columns 不能重复")
        if {"timestamp", "symbol"} & set(self.output_columns):
            raise ValueError("特征列不能覆盖 timestamp/symbol")

    @property
    def definition_hash(self) -> str:
        return _hash_json({
            "name": self.name,
            "version": self.version,
            "output_columns": list(self.output_columns),
            "parameters": self.parameters,
            "code_hash": self.code_hash,
        })


@dataclass(frozen=True)
class FeatureContext:
    calendar_fingerprint: str
    universe_fingerprint: str
    adjustment_mode: str
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (self.calendar_fingerprint and self.universe_fingerprint
                and self.adjustment_mode):
            raise ValueError("特征缓存必须声明 calendar/universe/adjustment 身份")


@dataclass(frozen=True)
class FeatureCacheEntry:
    key: str
    path: str
    rows: int
    bytes: int
    definition_hash: str
    dataset_fingerprint: str
    query_fingerprint: str
    skipped: bool = False


class MaterializedFeatureCache:
    def __init__(
        self,
        root: str | Path,
        *,
        bucket_count: int = 64,
        compression: str = "zstd",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.bucket_count = bucket_count
        self.compression = compression
        if bucket_count <= 0 or bucket_count & (bucket_count - 1):
            raise ValueError("bucket_count 必须是 2 的幂")
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()

    def key(self, definition: FeatureDefinition, lineage, context: FeatureContext) -> str:
        return _hash_json({
            "definition_hash": definition.definition_hash,
            "schema_hash": lineage.schema_hash,
            "dataset_fingerprint": lineage.dataset_fingerprint,
            "query_fingerprint": lineage.query_fingerprint,
            "context": asdict(context),
        })

    def materialize(self, definition, table, lineage, context) -> FeatureCacheEntry:
        pa, pc, dataset = _arrow_modules()
        cache_key = self.key(definition, lineage, context)
        destination = self.root / definition.name / cache_key
        manifest_path = destination / "manifest.json"
        with self._locked():
            if manifest_path.is_file():
                value = _read_json(manifest_path)
                value.pop("skipped", None)
                return FeatureCacheEntry(**value, skipped=True)
            normalized = _normalize_table(table, definition, pa, pc, self.bucket_count)
            staging = self.root / ".staging" / uuid.uuid4().hex
            data_path = staging / "data"
            staging.mkdir(parents=True, exist_ok=False)
            try:
                partition_schema = pa.schema([
                    ("year", pa.int16()), ("month", pa.int8()), ("bucket", pa.int16())
                ])
                dataset.write_dataset(
                    normalized,
                    base_dir=data_path,
                    format="parquet",
                    partitioning=dataset.partitioning(partition_schema, flavor="hive"),
                    existing_data_behavior="error",
                    file_options=dataset.ParquetFileFormat().make_write_options(
                        compression=self.compression
                    ),
                )
                stored_bytes = sum(path.stat().st_size for path in data_path.rglob("*.parquet"))
                entry = FeatureCacheEntry(
                    cache_key,
                    str(destination),
                    normalized.num_rows,
                    stored_bytes,
                    definition.definition_hash,
                    lineage.dataset_fingerprint,
                    lineage.query_fingerprint,
                )
                _write_json(staging / "manifest.json", asdict(entry))
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging, destination)
                return entry
            finally:
                shutil.rmtree(staging, ignore_errors=True)

    def lookup(self, definition, lineage, context) -> FeatureCacheEntry | None:
        cache_key = self.key(definition, lineage, context)
        manifest = self.root / definition.name / cache_key / "manifest.json"
        if not manifest.is_file():
            return None
        return FeatureCacheEntry(**_read_json(manifest))

    def scanner(
        self,
        entry: FeatureCacheEntry,
        *,
        symbols=None,
        start=None,
        end=None,
        columns: Sequence[str] | None = None,
        batch_size: int = 65_536,
    ):
        _, _, dataset = _arrow_modules()
        data_path = Path(entry.path) / "data"
        source = dataset.dataset(data_path, format="parquet", partitioning="hive")
        selected = list(columns) if columns is not None else None
        expression = None
        normalized_symbols = None
        if symbols is not None:
            values = (symbols,) if isinstance(symbols, str) else tuple(symbols)
            normalized_symbols = tuple(sorted({str(value) for value in values}))
            expression = dataset.field("symbol").isin(normalized_symbols)
        if start is not None:
            value = dataset.field("timestamp") >= start
            expression = value if expression is None else expression & value
        if end is not None:
            value = dataset.field("timestamp") <= end
            expression = value if expression is None else expression & value
        return source.scanner(
            columns=selected, filter=expression, batch_size=batch_size,
            use_threads=True,
        )

    @contextmanager
    def _locked(self):
        lock_path = self.root / ".feature-cache.lock"
        with self._thread_lock:
            with lock_path.open("a+") as file:
                if fcntl is not None:
                    fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _normalize_table(table, definition, pa, pc, bucket_count):
    required = {"timestamp", "symbol", *definition.output_columns}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError("特征表缺少字段: " + ", ".join(sorted(missing)))
    table = table.select(["timestamp", "symbol", *definition.output_columns])
    if table.num_rows == 0:
        raise ValueError("不能物化空特征表")
    if any(table[name].null_count for name in ("timestamp", "symbol")):
        raise ValueError("特征主键不能包含 null")
    table = table.set_column(
        0, "timestamp", pc.cast(table["timestamp"], pa.int64(), safe=True)
    ).set_column(
        1, "symbol", pc.utf8_trim_whitespace(pc.cast(table["symbol"], pa.string()))
    )
    indices = pc.sort_indices(
        table, sort_keys=[("timestamp", "ascending"), ("symbol", "ascending")]
    )
    table = pc.take(table, indices)
    if table.num_rows > 1:
        duplicate = pc.and_(
            pc.equal(table["timestamp"].slice(1), table["timestamp"].slice(0, table.num_rows - 1)),
            pc.equal(table["symbol"].slice(1), table["symbol"].slice(0, table.num_rows - 1)),
        )
        if pc.any(duplicate).as_py():
            raise ValueError("特征表包含重复 (timestamp, symbol)")
    timestamps = pc.cast(table["timestamp"], pa.timestamp("ns", tz="Asia/Shanghai"))
    table = table.append_column("year", pc.cast(pc.year(timestamps), pa.int16()))
    table = table.append_column("month", pc.cast(pc.month(timestamps), pa.int8()))
    table = table.append_column("bucket", pa.array([
        symbol_bucket(symbol, bucket_count) for symbol in table["symbol"].to_pylist()
    ], type=pa.int16()))
    return table


def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, sort_keys=True, separators=(",", ":"))
        file.flush()
        os.fsync(file.fileno())


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _arrow_modules():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.dataset as dataset
    except ImportError as exc:
        raise RuntimeError("特征缓存需要 pyarrow") from exc
    return pa, pc, dataset

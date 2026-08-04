from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from .config import SCHEMA_VERSION, DataLakeConfig

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


@dataclass(frozen=True)
class CatalogFile:
    path: str
    year: int
    month: int
    bucket: int
    rows: int
    min_timestamp: int
    max_timestamp: int
    content_hash: str = ""


@dataclass(frozen=True)
class CatalogSnapshot:
    schema_version: int
    bucket_count: int
    generation: int
    files: tuple[CatalogFile, ...]
    source_fingerprints: tuple[str, ...]


class DataCatalog:
    """以原子 JSON manifest 管理查询可见的 Parquet 文件。"""

    def __init__(self, config: DataLakeConfig) -> None:
        self.config = config
        self.path = config.root / "catalog.json"
        self.lock_path = config.root / ".catalog.lock"
        self._thread_lock = threading.RLock()
        config.root.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if not self.path.exists():
                self._write_unlocked(self._empty())
            self._validate_config(self._read_unlocked())

    @property
    def generation(self) -> int:
        with self._locked():
            return int(self._read_unlocked()["generation"])

    def contains_source(self, fingerprint: str) -> bool:
        with self._locked():
            return fingerprint in self._read_unlocked()["sources"]

    def files(self) -> tuple[CatalogFile, ...]:
        with self._locked():
            raw = self._read_unlocked()["files"]
        return tuple(CatalogFile(**item) for item in raw)

    def source_fingerprints(self) -> tuple[str, ...]:
        with self._locked():
            return tuple(sorted(self._read_unlocked()["sources"]))

    def snapshot(self) -> CatalogSnapshot:
        """在同一把 catalog 锁下返回一致的 generation/files/sources。"""
        with self._locked():
            manifest = self._read_unlocked()
        return CatalogSnapshot(
            schema_version=int(manifest["schema_version"]),
            bucket_count=int(manifest["bucket_count"]),
            generation=int(manifest["generation"]),
            files=tuple(CatalogFile(**item) for item in manifest["files"]),
            source_fingerprints=tuple(sorted(manifest["sources"])),
        )

    def commit(
        self,
        ingestion_id: str,
        source_fingerprint: str,
        source_path: str,
        files: list[CatalogFile],
    ) -> bool:
        """提交一个导入批次；相同源指纹重复提交时返回 False。"""
        return self.commit_many(
            ingestion_id,
            [{
                "fingerprint": source_fingerprint,
                "source_path": source_path,
                "files": len(files),
                "rows": sum(item.rows for item in files),
            }],
            files,
        )

    def commit_many(
        self,
        ingestion_id: str,
        sources: list[dict[str, object]],
        files: list[CatalogFile],
    ) -> bool:
        """Atomically publish multiple source files in one catalog generation."""
        with self._locked():
            manifest = self._read_unlocked()
            pending = [
                source for source in sources
                if str(source["fingerprint"]) not in manifest["sources"]
            ]
            if not pending:
                return False
            manifest["files"].extend(asdict(item) for item in files)
            for source in pending:
                fingerprint = str(source["fingerprint"])
                manifest["sources"][fingerprint] = {
                    "ingestion_id": ingestion_id,
                    "source_path": str(source["source_path"]),
                    "files": int(source.get("files", 0)),
                    "rows": int(source.get("rows", 0)),
                }
            manifest["generation"] += 1
            self._write_unlocked(manifest)
            return True

    def _empty(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "bucket_count": self.config.bucket_count,
            "generation": 0,
            "files": [],
            "sources": {},
        }

    def _validate_config(self, manifest: dict) -> None:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("数据目录 schema_version 与当前代码不兼容")
        if manifest.get("bucket_count") != self.config.bucket_count:
            raise RuntimeError(
                "数据目录 bucket_count 已固定，不能用不同配置打开；需要新建数据目录"
            )

    def _read_unlocked(self) -> dict:
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write_unlocked(self, value: dict) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, separators=(",", ":"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            with self.lock_path.open("a+") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

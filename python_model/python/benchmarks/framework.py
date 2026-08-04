from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from python.events import BatchEventRunner, HistoricalReplaySource
from python.market_data import ArrowDatasetScanner, PartitionAwareIterator


BAR_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class BenchmarkStage:
    name: str
    seconds: float
    rows: int = 0
    bytes: int = 0
    rows_per_second: float = 0.0
    megabytes_per_second: float = 0.0
    rss_start_bytes: int = 0
    rss_end_bytes: int = 0
    rss_delta_bytes: int = 0
    process_peak_rss_bytes: int = 0
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkReport:
    created_at: str
    lineage: dict
    query: dict
    environment: dict
    stages: tuple[BenchmarkStage, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["stages"] = [asdict(stage) for stage in self.stages]
        return value

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(destination)


@dataclass
class _Measurement:
    started: float
    rss_start_bytes: int
    seconds: float = 0.0
    rss_end_bytes: int = 0

    @classmethod
    def begin(cls) -> "_Measurement":
        rss = _current_rss_bytes()
        return cls(time.perf_counter(), rss, rss_end_bytes=rss)

    def finish(self) -> None:
        self.seconds = time.perf_counter() - self.started
        self.rss_end_bytes = _current_rss_bytes()


class _CacheController:
    def __init__(self, state: str, cold_cache_command: str | None) -> None:
        if state not in {"cold", "warm", "uncontrolled"}:
            raise ValueError("cache_state 必须是 cold、warm 或 uncontrolled")
        self.state = state
        self.cold_cache_command = cold_cache_command

    def prepare(self, files: list[str]) -> dict:
        started = time.perf_counter()
        if self.state == "uncontrolled":
            method = "none"
            prepared_bytes = 0
        elif self.state == "warm":
            method = "sequential-preread"
            prepared_bytes = _read_files(files)
        elif self.cold_cache_command:
            subprocess.run(
                shlex.split(self.cold_cache_command), check=True,
                stdout=subprocess.DEVNULL,
            )
            method = "external-command"
            prepared_bytes = 0
        elif sys.platform.startswith("linux") and hasattr(os, "posix_fadvise"):
            _drop_linux_file_cache(files)
            method = "posix_fadvise-dontneed"
            prepared_bytes = 0
        else:
            raise RuntimeError(
                "当前平台无法可靠清理文件缓存；cold benchmark 必须提供 "
                "cold_cache_command"
            )
        return {
            "requested_state": self.state,
            "method": method,
            "files": len(files),
            "prepared_bytes": prepared_bytes,
            "preparation_seconds": time.perf_counter() - started,
        }


class _EngineProbe:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.c_stream_calls = 0
        self.call_rows: list[int] = []
        self.call_batches: list[int] = []
        self.call_bytes: list[int] = []
        self.decode_seconds = 0.0
        self.execution_seconds = 0.0

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def process_arrow_stream(self, reader):
        self.c_stream_calls += 1
        stats = self.engine.process_arrow_stream(reader)
        self.call_rows.append(int(stats.rows))
        self.call_batches.append(int(stats.batches))
        self.call_bytes.append(int(stats.bytes))
        self.decode_seconds += float(stats.decode_seconds)
        self.execution_seconds += float(stats.execution_seconds)
        return stats

    def details(self) -> dict:
        return {
            "c_stream_calls": self.c_stream_calls,
            "input_batches": sum(self.call_batches),
            "call_rows_p50": _percentile(self.call_rows, 0.50),
            "call_rows_p95": _percentile(self.call_rows, 0.95),
            "decode_seconds": self.decode_seconds,
            "execution_seconds": self.execution_seconds,
        }


class MinuteReplayBenchmark:
    """分别测量存储、整段 C Stream、事件回放和完整策略回放。"""

    def __init__(
        self,
        lake,
        *,
        target_bytes: int = 256 * 1024 * 1024,
        compute_sample_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if compute_sample_bytes <= 0:
            raise ValueError("compute_sample_bytes 必须为正数")
        self.lake = lake
        self.scanner = ArrowDatasetScanner(lake)
        self.replay = PartitionAwareIterator(
            self.scanner, target_bytes=target_bytes
        )
        self.compute_sample_bytes = compute_sample_bytes

    def run(
        self,
        *,
        symbols=None,
        start=None,
        end=None,
        engine_factory=None,
        strategy_callback=None,
        strategy_name: str = "noop_cross_section",
        strategy_parameters: dict | None = None,
        cache_state: str = "uncontrolled",
        cold_cache_command: str | None = None,
    ) -> BenchmarkReport:
        query = {
            "symbols": symbols,
            "start": start,
            "end": end,
            "columns": list(BAR_COLUMNS),
            "target_bytes": self.replay.target_bytes,
            "cache_state": cache_state,
            "strategy": {
                "name": strategy_name,
                "parameters": dict(strategy_parameters or {}),
            },
        }
        cache = _CacheController(cache_state, cold_cache_command)
        files = [str(path) for path in self.lake.files_for_query(
            symbols=symbols, start=start, end=end,
            snapshot=self.scanner.snapshot,
        )]
        stages = []

        cache_details = cache.prepare(files)
        measurement = _Measurement.begin()
        io_bytes = _read_files(files)
        measurement.finish()
        stages.append(_stage(
            "storage_io", measurement, bytes_count=io_bytes,
            details={"files": len(files), "cache": cache_details},
        ))

        cache_details = cache.prepare(files)
        measurement = _Measurement.begin()
        decoded_rows = decoded_bytes = 0
        compute_sample = []
        compute_sample_size = 0
        batch_rows = []
        batch_bytes = []
        batch_latencies = []
        iterator = iter(self.scanner.iter_batches(
            symbols=symbols, start=start, end=end, columns=query["columns"]
        ))
        while True:
            batch_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            batch_latencies.append(time.perf_counter() - batch_started)
            size = int(batch.nbytes)
            batch_rows.append(batch.num_rows)
            batch_bytes.append(size)
            decoded_rows += batch.num_rows
            decoded_bytes += size
            if compute_sample_size < self.compute_sample_bytes:
                compute_sample.append(batch)
                compute_sample_size += size
        measurement.finish()
        stages.append(_stage(
            "arrow_decode", measurement, rows=decoded_rows,
            bytes_count=decoded_bytes,
            details={
                "cache": cache_details,
                "batches": len(batch_rows),
                "batch_rows_p50": _percentile(batch_rows, 0.50),
                "batch_rows_p95": _percentile(batch_rows, 0.95),
                "batch_bytes_p50": _percentile(batch_bytes, 0.50),
                "batch_bytes_p95": _percentile(batch_bytes, 0.95),
                "batch_latency_p50_ms": _percentile(batch_latencies, 0.50) * 1000,
                "batch_latency_p95_ms": _percentile(batch_latencies, 0.95) * 1000,
                "compute_sample_bytes": compute_sample_size,
            },
        ))

        import pyarrow.compute as pc

        measurement = _Measurement.begin()
        close_sum = 0.0
        compute_rows = 0
        for batch in compute_sample:
            value = pc.sum(batch["close"]).as_py()
            close_sum += float(value or 0.0)
            compute_rows += batch.num_rows
        measurement.finish()
        stages.append(_stage(
            "arrow_compute", measurement, rows=compute_rows,
            bytes_count=compute_sample_size,
            details={"close_sum": close_sum, "sampled": compute_rows < decoded_rows},
        ))
        compute_sample.clear()
        del compute_sample
        gc.collect()

        previous_engine = None
        backend_environment = {}

        def new_engine():
            nonlocal previous_engine, backend_environment
            engine = engine_factory()
            if engine is previous_engine:
                raise RuntimeError("engine_factory 每次必须返回新的引擎实例")
            if not hasattr(engine, "process_arrow_stream"):
                raise RuntimeError("当前引擎未暴露 Arrow C Stream bridge")
            if not backend_environment:
                backend_environment = _backend_environment(type(engine))
            previous_engine = engine
            return engine

        if engine_factory is not None:
            cache_details = cache.prepare(files)
            engine = new_engine()
            probe = _EngineProbe(engine)
            observed_rows: list[int] = []
            observed_bytes: list[int] = []
            reader = _instrument_reader(
                self.replay.reader(
                    symbols=symbols, start=start, end=end, columns=query["columns"]
                ),
                observed_rows,
                observed_bytes,
            )
            measurement = _Measurement.begin()
            stats = probe.process_arrow_stream(reader)
            measurement.finish()
            details = probe.details()
            details.update({
                "cache": cache_details,
                "stream_batch_rows_p50": _percentile(observed_rows, 0.50),
                "stream_batch_rows_p95": _percentile(observed_rows, 0.95),
                "stream_batch_bytes_p50": _percentile(observed_bytes, 0.50),
                "stream_batch_bytes_p95": _percentile(observed_bytes, 0.95),
                "equity": engine.get_equity(),
            })
            stages.append(_stage(
                "c_stream_replay", measurement, rows=int(stats.rows),
                bytes_count=int(stats.bytes), details=details,
            ))

            cache_details = cache.prepare(files)
            engine = new_engine()
            probe = _EngineProbe(engine)
            source = HistoricalReplaySource(
                self.replay, symbols=symbols, start=start, end=end,
                columns=query["columns"],
            )
            measurement = _Measurement.begin()
            event_stats = BatchEventRunner(probe).run(source)
            measurement.finish()
            details = probe.details()
            details.update({
                "cache": cache_details,
                "events": event_stats.events,
                "market_batches": event_stats.market_batches,
                "equity": engine.get_equity(),
            })
            stages.append(_stage(
                "event_replay", measurement, rows=event_stats.rows,
                bytes_count=sum(probe.call_bytes), details=details,
            ))

            cache_details = cache.prepare(files)
            engine = new_engine()
            probe = _EngineProbe(engine)
            callback_calls = 0

            def benchmark_strategy(batch):
                nonlocal callback_calls
                callback_calls += 1
                if strategy_callback is None:
                    return []
                return strategy_callback(batch)

            engine.set_on_cross_section(benchmark_strategy)
            source = HistoricalReplaySource(
                self.replay, symbols=symbols, start=start, end=end,
                columns=query["columns"],
            )
            measurement = _Measurement.begin()
            event_stats = BatchEventRunner(probe).run(source)
            result_summary = {
                "equity": engine.get_equity(),
                "total_return": engine.get_total_return(),
                "trades": len(engine.get_trade_history()),
                "round_trips": len(engine.get_round_trip_history()),
            }
            measurement.finish()
            details = probe.details()
            details.update({
                "cache": cache_details,
                "events": event_stats.events,
                "market_batches": event_stats.market_batches,
                "strategy_callback_calls": callback_calls,
                "result": result_summary,
            })
            stages.append(_stage(
                "strategy_backtest", measurement, rows=event_stats.rows,
                bytes_count=sum(probe.call_bytes), details=details,
            ))

        lineage = self.scanner.lineage(
            symbols=symbols, start=start, end=end, columns=query["columns"]
        )
        return BenchmarkReport(
            datetime.now(timezone.utc).isoformat(),
            lineage.to_dict(),
            query,
            {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "packages": _package_versions(),
                "rss_fields": (
                    "rss_start/end/delta are current resident memory at stage "
                    "boundaries where supported; process_peak is the process "
                    "high-water mark"
                ),
                "rss_source": _rss_source(),
                **backend_environment,
            },
            tuple(stages),
        )


def _instrument_reader(reader, observed_rows: list[int], observed_bytes: list[int]):
    import pyarrow as pa

    def batches():
        for batch in reader:
            observed_rows.append(batch.num_rows)
            observed_bytes.append(int(batch.nbytes))
            yield batch

    return pa.RecordBatchReader.from_batches(reader.schema, batches())


def _stage(
    name: str,
    measurement: _Measurement,
    *,
    rows: int = 0,
    bytes_count: int = 0,
    details: dict | None = None,
) -> BenchmarkStage:
    elapsed = measurement.seconds
    return BenchmarkStage(
        name=name,
        seconds=elapsed,
        rows=rows,
        bytes=bytes_count,
        rows_per_second=rows / elapsed if elapsed else 0.0,
        megabytes_per_second=(
            bytes_count / elapsed / 1_000_000 if elapsed else 0.0
        ),
        rss_start_bytes=measurement.rss_start_bytes,
        rss_end_bytes=measurement.rss_end_bytes,
        rss_delta_bytes=max(
            measurement.rss_end_bytes - measurement.rss_start_bytes, 0
        ),
        process_peak_rss_bytes=_peak_rss_bytes(),
        details=details or {},
    )


def _read_files(files: list[str]) -> int:
    total = 0
    buffer = bytearray(8 * 1024 * 1024)
    for filename in files:
        with open(filename, "rb", buffering=0) as file:
            while True:
                count = file.readinto(buffer)
                if not count:
                    break
                total += count
    return total


def _drop_linux_file_cache(files: list[str]) -> None:
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advice is None:
        raise RuntimeError("Python 未暴露 POSIX_FADV_DONTNEED")
    for filename in files:
        descriptor = os.open(filename, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, advice)
        finally:
            os.close(descriptor)


def _current_rss_bytes() -> int:
    if sys.platform.startswith("linux"):
        statm = Path("/proc/self/statm")
        if statm.exists():
            resident_pages = int(statm.read_text(encoding="ascii").split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    if sys.platform == "darwin":
        value = _darwin_current_rss_bytes()
        if value is not None:
            return value
    return _peak_rss_bytes()


def _darwin_current_rss_bytes() -> int | None:
    import ctypes

    class TimeValue(ctypes.Structure):
        _fields_ = [
            ("seconds", ctypes.c_int32),
            ("microseconds", ctypes.c_int32),
        ]

    class MachTaskBasicInfo(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("resident_size", ctypes.c_uint64),
            ("resident_size_max", ctypes.c_uint64),
            ("user_time", TimeValue),
            ("system_time", TimeValue),
            ("policy", ctypes.c_int32),
            ("suspend_count", ctypes.c_int32),
        ]

    try:
        system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        system.mach_task_self.restype = ctypes.c_uint32
        info = MachTaskBasicInfo()
        count = ctypes.c_uint32(
            ctypes.sizeof(MachTaskBasicInfo) // ctypes.sizeof(ctypes.c_uint32)
        )
        result = system.task_info(
            system.mach_task_self(),
            20,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        return int(info.resident_size) if result == 0 else None
    except (AttributeError, OSError):
        return None


def _rss_source() -> str:
    if sys.platform.startswith("linux") and Path("/proc/self/statm").exists():
        return "proc-self-statm"
    if sys.platform == "darwin" and _darwin_current_rss_bytes() is not None:
        return "mach-task-basic-info"
    return "process-high-water-mark-fallback"


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def validate_benchmark_backend(engine_type) -> None:
    metadata = _backend_environment(engine_type)
    if metadata["backend_build_configuration_hint"] == "Debug":
        raise RuntimeError(
            "benchmark 拒绝 Debug/dev/sanitize 扩展；请安装 Release wheel "
            "或使用 release preset"
        )


def _backend_environment(engine_type) -> dict:
    module_name = getattr(engine_type, "__module__", "")
    module = sys.modules.get(module_name)
    raw_artifact = getattr(module, "__file__", "") if module is not None else ""
    artifact = Path(raw_artifact).resolve() if raw_artifact else None
    is_file = artifact is not None and artifact.is_file()
    return {
        "backend_module": module_name,
        "backend_class": getattr(engine_type, "__qualname__", str(engine_type)),
        "backend_artifact_path": str(artifact) if artifact else "",
        "backend_artifact_sha256": _file_sha256(artifact) if is_file else "",
        "backend_artifact_size_bytes": artifact.stat().st_size if is_file else 0,
        "backend_artifact_mtime_ns": artifact.stat().st_mtime_ns if is_file else 0,
        "backend_build_configuration_hint": _build_configuration_hint(artifact),
    }


def _build_configuration_hint(artifact: Path | None) -> str:
    if artifact is None:
        return "unknown"
    parts = {part.lower() for part in artifact.parts}
    if (
        parts.intersection({"dev", "debug", "sanitize", "all-modules"})
        or any(part.startswith("cmake-build-debug") for part in parts)
    ):
        return "Debug"
    if "release" in parts:
        return "Release"
    if "site-packages" in parts:
        return "Release wheel default (unverified)"
    return "unknown"


def _package_versions() -> dict:
    versions = {}
    for name in ("duckdb", "numpy", "pyarrow", "pybind11"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = ""
    return versions


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""可序列化、可指纹化的回测运行配置。"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Mapping


RUN_SPEC_VERSION = 1


@dataclass(frozen=True)
class RunSpec:
    version: int
    backend: str
    strategy: dict[str, object]
    initial_cash: float
    fill_timing: str
    execution_config: dict[str, object]
    broker: dict[str, object]
    random_seed: int | None
    calendar_fingerprint: str
    reference_data_fingerprint: str
    data_lineage: dict[str, object] | None
    environment: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return _json_value(asdict(self))

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def capture_run_spec(
    *,
    backend: str,
    engine_type,
    strategy,
    strategy_parameters: Mapping[str, object],
    initial_cash: float,
    fill_timing: str,
    execution_config,
    broker,
    random_seed: int | None,
    calendar,
    reference_data,
    data_lineage,
) -> RunSpec:
    return RunSpec(
        version=RUN_SPEC_VERSION,
        backend=backend,
        strategy={
            "class": _qualified_name(type(strategy)),
            "code_hash": _strategy_code_hash(strategy),
            "symbols": sorted(str(symbol) for symbol in strategy.symbols),
            "parameters": _json_value(dict(strategy_parameters)),
        },
        initial_cash=float(initial_cash),
        fill_timing=str(fill_timing),
        execution_config=_execution_config(execution_config),
        broker=_json_value(broker.to_config()),
        random_seed=random_seed,
        calendar_fingerprint=_object_fingerprint(calendar),
        reference_data_fingerprint=_object_fingerprint(reference_data),
        data_lineage=(data_lineage.to_dict() if data_lineage is not None else None),
        environment=_environment(backend, engine_type),
    )


def _execution_config(config) -> dict[str, object]:
    fields = (
        "max_volume_participation",
        "slippage_bps",
        "enforce_price_limits",
        "enforce_t_plus_one",
        "allow_short",
        "enforce_board_lot",
        "enforce_cash",
        "market_order_price_buffer_bps",
    )
    return {name: getattr(config, name) for name in fields}


def _strategy_code_hash(strategy) -> str:
    cls = type(strategy)
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        source = _qualified_name(cls)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _environment(backend: str, engine_type) -> dict[str, object]:
    module_name = getattr(engine_type, "__module__", "")
    module = sys.modules.get(module_name)
    artifact = Path(getattr(module, "__file__", "")) if module is not None else None
    artifact_hash = _file_sha256(artifact) if artifact and artifact.is_file() else ""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "backend": backend,
        "backend_module": module_name,
        "backend_artifact": artifact.name if artifact else "",
        "backend_artifact_sha256": artifact_hash,
    }


def _object_fingerprint(value) -> str:
    if value is None:
        return ""
    encoded = json.dumps(
        _json_value(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value):
    if isinstance(value, Enum):
        return _json_value(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    state = getattr(value, "__dict__", None)
    if isinstance(state, dict):
        return {
            "class": _qualified_name(type(value)),
            "state": _json_value({
                key: item for key, item in state.items() if not callable(item)
            }),
        }
    return {"class": _qualified_name(type(value))}


def _qualified_name(cls) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

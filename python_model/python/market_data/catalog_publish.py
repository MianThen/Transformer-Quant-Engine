from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


def publish_replacement(
    catalog_path: str | Path,
    files: Sequence[Any],
    *,
    expected_generation: int,
    backup_dir: str | Path | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    path = Path(catalog_path)
    lock_path = path.parent / ("." + path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            current = int(manifest["generation"])
            if current != int(expected_generation):
                raise RuntimeError(
                    f"catalog generation 已变化: 期望 {expected_generation}, 实际 {current}"
                )

            backup_root = Path(backup_dir) if backup_dir is not None else path.parent
            backup_root.mkdir(parents=True, exist_ok=True)
            backup = backup_root / f"{path.name}.bak.{current}"
            if not backup.exists():
                shutil.copy2(path, backup)

            new_items = [asdict(item) for item in files]
            manifest["files"] = new_items
            manifest["generation"] = current + 1

            if extra_metadata is not None:
                for key, value in extra_metadata.items():
                    manifest[key] = value

            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, path)

            return {
                "generation": current + 1,
                "previous_generation": current,
                "files": len(new_items),
                "rows": sum(int(item["rows"]) for item in new_items),
                "backup": str(backup),
            }
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def read_generation(catalog_path: str | Path) -> int:
    manifest = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    return int(manifest["generation"])


def copy_catalog(source_path: str | Path, target_dir: str | Path) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(source_path).name
    shutil.copy2(source_path, target)
    return target

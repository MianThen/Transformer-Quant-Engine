from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    validation = root / "RESULT_VALIDATION.json"
    if not validation.is_file():
        raise RuntimeError("请先运行 tools/validate_results.py")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    )
    manifest = root / "MANIFEST.sha256"
    manifest.write_text(
        "".join(
            f"{sha256_file(path)}  ./{path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
        newline="\n",
    )
    output = Path(args.output).resolve()
    with tarfile.open(output, "w:gz") as archive:
        archive.add(root, arcname=root.name)
    print(f"{output}\nsha256={sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path


def differences(left, right, path="$", output=None):
    output = [] if output is None else output
    if type(left) is not type(right):
        output.append(f"{path}: type {type(left).__name__} != {type(right).__name__}")
    elif isinstance(left, dict):
        for key in sorted(left.keys() | right.keys()):
            if key not in left: output.append(f"{path}.{key}: missing on left")
            elif key not in right: output.append(f"{path}.{key}: missing on right")
            else: differences(left[key], right[key], f"{path}.{key}", output)
    elif isinstance(left, list):
        if len(left) != len(right): output.append(f"{path}: length {len(left)} != {len(right)}")
        for index, (lhs, rhs) in enumerate(zip(left, right)):
            differences(lhs, rhs, f"{path}[{index}]", output)
    elif left != right:
        output.append(f"{path}: {left!r} != {right!r}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare golden replay JSON field by field")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    diffs = differences(json.loads(args.left.read_text(encoding="utf-8-sig")),
                        json.loads(args.right.read_text(encoding="utf-8-sig")))
    if diffs:
        print("\n".join(diffs[:200]))
        if len(diffs) > 200: print(f"... {len(diffs) - 200} more differences")
        return 1
    print("golden replay matches field by field")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

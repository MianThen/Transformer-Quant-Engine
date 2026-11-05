#!/usr/bin/env python3
"""AC-27：复用块（BYA/SWP/ABL/RET-K1）的 panel_ref 必须指向 NUL/ALT 已有的面板。

判据：复用块的每个 panel_ref 都能在 NUL/ALT 的 panel_ref 集合里找到。
若找不到，说明该块的参数与父块不重合 —— 它其实生成了新面板，D 计数会失真。
"""
import json
import os
import sys

REUSE_BLOCKS = ("BYA", "SWP", "ABL")
PARENT_BLOCKS = ("NUL", "ALT")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "mc", "mc_gate_core_v1.json")
    with open(path) as f:
        cfg = json.load(f)

    parent = {c["panel_ref"] for c in cfg["cells"] if c["block"] in PARENT_BLOCKS}
    total = 0
    matched = 0
    missing = []
    for c in cfg["cells"]:
        # RET 的 K=1 复用 NUL；K=10 的 ds=0 也复用 NUL，ds>=1 是新的
        is_reuse = c["block"] in REUSE_BLOCKS
        if c["block"] == "RET":
            is_reuse = (c.get("K") == 1) or (c.get("K") == 10 and c.get("ds", 0) == 0)
        if not is_reuse:
            continue
        total += 1
        if c["panel_ref"] in parent:
            matched += 1
        else:
            missing.append((c.get("cell_id"), c["block"], c["panel_ref"]))

    print("PANELREF reuse_cells=%d matched=%d missing=%d" % (total, matched, len(missing)))
    for m in missing[:5]:
        print("  MISS cell=%s block=%s ref=%s" % m)
    ok = (total > 0 and matched == total)
    print("PANELREF_MATCH 1" if ok else "PANELREF_MATCH 0")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

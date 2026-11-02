#!/usr/bin/env python3
"""AC-01 的端到端验证：从 configs/mc/mc_gate_core_v1.json 重新推导
   cells / datasets / calls 三项，并与合同 §5 的 269 / 194 / 617000 比对。

   合同要求"不得硬编码常数"——本脚本不出现那三个数作为输入，
   只用它们做最后比对。
"""
import json
import os
import sys

EXPECTED = {"cells": 269, "datasets": 202, "calls": 617000}   # v1.4.5：D 由 194 更正为 202


def derive(cfg):
    cells = cfg["cells"]
    # 决策单元：真正跑调用的格子（RET 的 ds=1..9 只登记面板，calls_per_rep=0）
    executed = [c for c in cells if int(c.get("calls_per_rep", 1)) > 0]
    n_cells = len(executed)
    # 独立面板：distinct panel_ref（v1.4.5 §3.6.3.1）
    n_ds = len({c["panel_ref"] for c in cells})
    n_calls = sum(int(c["reps"]) * int(c.get("calls_per_rep", 1)) for c in cells)
    return n_cells, n_ds, n_calls


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "mc", "mc_gate_core_v1.json")
    with open(path) as f:
        cfg = json.load(f)
    c, d, k = derive(cfg)
    print("GRID cells=%d datasets=%d calls=%d" % (c, d, k))
    ok = (c == EXPECTED["cells"] and d == EXPECTED["datasets"] and k == EXPECTED["calls"])
    print("GRID expect %d / %d / %d" % (EXPECTED["cells"], EXPECTED["datasets"], EXPECTED["calls"]))
    print("GRID_MATCH 1" if ok else "GRID_MATCH 0")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

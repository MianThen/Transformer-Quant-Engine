#!/usr/bin/env python3
"""AC-15：从 decisions.jsonl.gz 重算逐格聚合，与 aggregates.jsonl 逐格比对。

本实验的聚合表就是 aggregates.jsonl（每个 cell×rule 一行：reps / selected），
所以"从原始记录重算表"等价于：按 (cell_id, rule) 重新计数并与聚合文件比对。
"""
import gzip
import json
import os
import sys
from collections import defaultdict


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qgate/C27"
    dpath = os.path.join(base, "decisions.jsonl.gz")
    apath = os.path.join(base, "aggregates.jsonl")
    for p in (dpath, apath):
        if not os.path.exists(p):
            print("RECOMPUTE missing %s" % p)
            print("RECOMPUTE_OK 0")
            return 1

    agg = {}
    with open(apath) as f:
        for line in f:
            r = json.loads(line)
            agg[(r["cell_id"], r["rule"])] = (r["reps"], r["selected"])

    cnt = defaultdict(lambda: [0, 0])          # (cell, rule) -> [reps, selected]
    n = 0
    with gzip.open(dpath, "rt") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            k = (r["cell_id"], r["rule_id"])
            cnt[k][0] += 1
            if r["selected"]:
                cnt[k][1] += 1

    bad = 0
    for k, (reps, sel) in agg.items():
        got = cnt.get(k)
        if got is None:
            print("RECOMPUTE missing_in_jsonl cell=%s rule=%s" % k)
            bad += 1
        elif got[0] != reps or got[1] != sel:
            print("RECOMPUTE mismatch cell=%s rule=%s agg=(%d,%d) jsonl=(%d,%d)"
                  % (k[0], k[1], reps, sel, got[0], got[1]))
            bad += 1
        if bad > 5:
            break
    extra = set(cnt) - set(agg)
    for k in list(extra)[:3]:
        print("RECOMPUTE extra_in_jsonl cell=%s rule=%s" % k)
        bad += 1

    print("RECOMPUTE records=%d cells=%d agg_rows=%d bad=%d" % (n, len(cnt), len(agg), bad))
    print("RECOMPUTE_OK %d" % (1 if bad == 0 and n > 0 and len(agg) > 0 else 0))
    return 0 if (bad == 0 and n > 0 and len(agg) > 0) else 1


if __name__ == "__main__":
    sys.exit(main())

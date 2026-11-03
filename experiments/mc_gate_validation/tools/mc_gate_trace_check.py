#!/usr/bin/env python3
"""AC-18：同 rep 下六条规则的 returns_sha256 必须相等，且等于 manifest.dataset_sha256。"""
import gzip
import json
import os
import sys
from collections import defaultdict


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qgate/C26"
    mpath = os.path.join(base, "manifest.json")
    dpath = os.path.join(base, "decisions.jsonl.gz")
    for p in (mpath, dpath):
        if not os.path.exists(p):
            print("TRACE missing %s" % p)
            print("TRACE_OK 0")
            return 1

    manifest = json.load(open(mpath))
    ds = manifest.get("dataset_sha256") or {}
    if not ds:
        print("TRACE manifest_has_no_dataset_sha256")
        print("TRACE_OK 0")
        return 1

    by_rep = defaultdict(set)     # (cell,rep) -> set of hashes
    rep0 = {}                     # cell -> hash of rep 0
    n = 0
    with gzip.open(dpath, "rt") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            key = (r["cell_id"], r["rep_id"])
            by_rep[key].add(r["returns_sha256"])
            if r["rep_id"] == 0:
                rep0.setdefault(r["cell_id"], r["returns_sha256"])

    bad = 0
    # 1) 同 rep 六规则同指纹
    for key, hs in by_rep.items():
        if len(hs) != 1:
            print("TRACE rep_mismatch cell=%s rep=%s hashes=%d" % (key[0], key[1], len(hs)))
            bad += 1
            if bad > 3:
                break
    # 2) rep0 的指纹等于 manifest
    for cell, h in rep0.items():
        want = ds.get(str(cell))
        if want is None:
            print("TRACE manifest_missing_cell %s" % cell)
            bad += 1
        elif want != h:
            print("TRACE manifest_mismatch cell=%s decisions=%s manifest=%s" % (cell, h, want))
            bad += 1

    print("TRACE records=%d reps=%d cells=%d bad=%d" % (n, len(by_rep), len(rep0), bad))
    print("TRACE_OK %d" % (1 if bad == 0 and n > 0 else 0))
    return 0 if (bad == 0 and n > 0) else 1


if __name__ == "__main__":
    sys.exit(main())

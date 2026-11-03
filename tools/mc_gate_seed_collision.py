#!/usr/bin/env python3
"""AC-36 的碰撞检查：全网格的 (dataset_key, rep, batch, stream) 不得产生重复 seed。

逐位对齐 6 个测试向量由 C22 负责；本工具只做碰撞检查。
默认扫每个数据集的前 REPS 个 rep × 5 个子流。
"""
import json
import os
import sys

M64 = (1 << 64) - 1
RUN_SEED = 20261027
STREAMS = ("ret", "assign", "hold", "boot", "aux")


def splitmix64(x):
    z = (x + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return (z ^ (z >> 31)) & M64


def fnv1a64(s):
    h = 0xCBF29CE484222325
    for b in s.encode("ascii"):
        h = ((h ^ b) * 0x100000001B3) & M64
    return h


def seed_of(key, rep, batch, stream, run_seed=RUN_SEED):
    full = "%s|rep=%d|batch=%d|stream=%s" % (key, rep, batch, stream)
    return splitmix64(fnv1a64(full) ^ splitmix64(run_seed))


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "configs", "mc", "mc_gate_core_v1.json")
    with open(path) as f:
        cfg = json.load(f)

    keys = sorted({c["panel_ref"] for c in cfg["cells"]})
    seen = {}
    collisions = []
    n = 0
    for key in keys:
        for rep in range(reps):
            for stream in STREAMS:
                s = seed_of(key, rep, 0, stream)
                n += 1
                if s in seen:
                    collisions.append((s, seen[s], (key, rep, stream)))
                else:
                    seen[s] = (key, rep, stream)

    print("SEEDCOLL keys=%d reps=%d streams=%d total=%d unique=%d collisions=%d"
          % (len(keys), reps, len(STREAMS), n, len(seen), len(collisions)))
    for c in collisions[:3]:
        print("  COLLIDE seed=%016x  %s  vs  %s" % c)
    ok = (n > 0 and not collisions)
    print("SEEDCOLL_OK 1" if ok else "SEEDCOLL_OK 0")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

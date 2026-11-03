#!/usr/bin/env python3
"""AC-31 计时纪律的静态检查。

规则：
  1) 被测量的核心模块（decide/adapter/onc/baselines/dgp/truth）内部**零计时**；
  2) 计时只能出现在驱动里，且成对（每次 steady_clock::now() 的调用数应为偶数）。
"""
import os
import re
import sys

CORE = ("mc_gate_decide.h", "mc_gate_adapter.h", "mc_gate_onc.h",
        "mc_gate_baselines.h", "mc_gate_dgp.h", "mc_gate_truth.h")
DRIVER = "apps/mc_gate_run.cpp"
PAT = re.compile(r"steady_clock|chrono::now|high_resolution_clock")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inc = os.path.join(root, "performance_analytics", "include", "performance_analytics")
    bad = 0
    for name in CORE:
        p = os.path.join(inc, name)
        if not os.path.exists(p):
            print("TIMING miss_header %s" % name)
            bad += 1
            continue
        n = len(PAT.findall(open(p, encoding="utf-8").read()))
        if n:
            print("TIMING core_has_timer %s count=%d" % (name, n))
            bad += 1
    dp = os.path.join(root, "performance_analytics", DRIVER)
    if not os.path.exists(dp):
        print("TIMING miss_driver")
        bad += 1
    else:
        n = len(PAT.findall(open(dp, encoding="utf-8").read()))
        if n == 0:
            print("TIMING driver_has_no_timer")
            bad += 1
        elif n % 2 != 0:
            print("TIMING driver_unpaired count=%d" % n)
            bad += 1
        else:
            print("TIMING driver_pairs %d" % (n // 2))
    print("TIMING_CORE_CLEAN 1" if bad == 0 else "TIMING_CORE_CLEAN 0")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

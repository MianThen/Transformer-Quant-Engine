#!/bin/bash
cd /tmp/qgate/C06R || exit 1
./r.out --shard 0/1 --records-only > whole.txt 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "C06R_FAIL run_rc=$rc"; exit 1; fi
./r.out --shard 0/1 --records-only > whole2.txt 2>&1
if ! cmp -s whole.txt whole2.txt; then echo "C06R_FAIL not_deterministic"; exit 1; fi
for i in 0 1 2 3; do ./r.out --shard $i/4 --records-only > s$i.txt 2>&1 || { echo "C06R_FAIL shard4_run_$i"; exit 1; }; done
cat s0.txt s1.txt s2.txt s3.txt > cat4.txt
if ! cmp -s whole.txt cat4.txt; then echo "C06R_FAIL shard4_concat_mismatch"; exit 1; fi
for i in 0 1 2; do ./r.out --shard $i/3 --records-only > t$i.txt 2>&1 || { echo "C06R_FAIL shard3_run_$i"; exit 1; }; done
cat t0.txt t1.txt t2.txt > cat3.txt
if ! cmp -s whole.txt cat3.txt; then echo "C06R_FAIL shard3_concat_mismatch"; exit 1; fi
awk -f c06r_verify.awk whole.txt

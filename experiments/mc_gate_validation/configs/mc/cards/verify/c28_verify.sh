#!/bin/bash
cd /tmp/qgate/C28 || exit 1
timeout 200 ./ocn.out > out.txt 2>&1
rc=$?
if [ $rc -eq 124 ]; then echo "C28_FAIL run_hang"; exit 1; fi
if [ $rc -ne 0 ]; then echo "C28_FAIL run_rc=$rc"; head -5 out.txt; exit 1; fi
awk -f c28_verify.awk out.txt || exit 1
echo "C28_PASS"

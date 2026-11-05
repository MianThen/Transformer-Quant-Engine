#!/bin/bash
cd /tmp/qgate/C32 || exit 1
timeout 150 ./inv.out > out.txt 2>&1
rc=$?
if [ $rc -eq 124 ]; then echo "C32_FAIL run_hang"; exit 1; fi
if [ $rc -ne 0 ]; then echo "C32_FAIL run_rc=$rc"; head -5 out.txt; exit 1; fi
awk -f c32_verify.awk out.txt || exit 1
echo "C32_PASS"

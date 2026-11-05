#!/bin/bash
cd /tmp/qgate/C27 || exit 1
rm -f aggregates.jsonl decisions.jsonl.gz
timeout 200 ./full.out --limit 8 --reps-cap 20 > out.txt 2>&1
rc=$?
if [ $rc -eq 124 ]; then echo "C27_FAIL run_hang"; exit 1; fi
if [ $rc -ne 0 ]; then echo "C27_FAIL run_rc=$rc"; head -5 out.txt; exit 1; fi
awk -f c27_verify.awk out.txt || exit 1

[ -f aggregates.jsonl ] || { echo "C27_FAIL no aggregates"; exit 1; }
na=$(wc -l < aggregates.jsonl | tr -d ' ')
[ "$na" -eq 48 ] || { echo "C27_FAIL aggregates=$na want 48"; exit 1; }
head -1 aggregates.jsonl | grep -qE '^\{"cell_id":0,"rule":"GATE_Full","reps":20,"selected":[0-9]+\}$' \
  || { echo "C27_FAIL aggregates line1 format: $(head -1 aggregates.jsonl)"; exit 1; }
for r in GATE_Full B0 B1 B2 B3 B4; do
  c=$(grep -c "\"rule\":\"$r\"" aggregates.jsonl)
  [ "$c" -eq 8 ] || { echo "C27_FAIL rule $r count=$c want 8"; exit 1; }
done

[ -f decisions.jsonl.gz ] || { echo "C27_FAIL no jsonl"; exit 1; }
gzip -t decisions.jsonl.gz 2>/dev/null || { echo "C27_FAIL gzip_t"; exit 1; }
nl=$(gzip -dc decisions.jsonl.gz | wc -l | tr -d ' ')
[ "$nl" -eq 960 ] || { echo "C27_FAIL jsonl=$nl want 960"; exit 1; }

echo "C27_PASS"

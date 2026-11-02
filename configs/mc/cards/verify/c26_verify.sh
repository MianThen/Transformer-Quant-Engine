#!/bin/bash
cd /tmp/qgate/C26 || exit 1
rm -rf replay decisions.jsonl.gz manifest.json metrics.json
timeout 300 ./drv.out > out.txt 2>&1
rc=$?
if [ $rc -eq 124 ]; then echo "C26_FAIL run_hang"; exit 1; fi
if [ $rc -ne 0 ]; then echo "C26_FAIL run_rc=$rc"; head -5 out.txt; exit 1; fi

awk -f c26_verify.awk out.txt || exit 1

[ -f decisions.jsonl.gz ] || { echo "C26_FAIL no jsonl"; exit 1; }
gzip -t decisions.jsonl.gz 2>/dev/null || { echo "C26_FAIL gzip_t"; exit 1; }
hdr=$(head -c 8 decisions.jsonl.gz | od -An -tx1 | tr -d ' \n')
[ "$hdr" = "1f8b080000000000" ] || { echo "C26_FAIL gzip_hdr=$hdr"; exit 1; }
nl=$(gzip -dc decisions.jsonl.gz | wc -l | tr -d ' ')
[ "$nl" -eq 60000 ] || { echo "C26_FAIL jsonl=$nl want 60000"; exit 1; }

[ -f metrics.json ] || { echo "C26_FAIL no metrics"; exit 1; }
grep -q '"cost_profile":{"single_core_ms":9046.800' metrics.json || { echo "C26_FAIL metrics cost_profile"; exit 1; }
grep -q '"planned_calls":10000' metrics.json || { echo "C26_FAIL metrics planned_calls"; exit 1; }
grep -q '"wall_by_rule":{"GATE_Full":[0-9]' metrics.json || { echo "C26_FAIL metrics wall_by_rule"; exit 1; }
for k in B0 B1 B2 B3 B4; do
  grep -q "\"$k\":[0-9]" metrics.json || { echo "C26_FAIL wall_by_rule missing $k"; exit 1; }
done
if grep -qE '202[0-9]-|hostname|elapsed|timestamp' metrics.json; then echo "C26_FAIL metrics has timestamp"; exit 1; fi

[ -f manifest.json ] || { echo "C26_FAIL no manifest"; exit 1; }
grep -qE '"contract_hash":"[0-9a-f]{64}"' manifest.json || { echo "C26_FAIL manifest contract_hash"; exit 1; }
grep -qE '"config_hash":"[0-9a-f]{64}"'   manifest.json || { echo "C26_FAIL manifest config_hash"; exit 1; }
grep -q '"run_seed":20261027' manifest.json || { echo "C26_FAIL manifest run_seed"; exit 1; }
grep -q '"seed_tree":{' manifest.json || { echo "C26_FAIL manifest seed_tree"; exit 1; }
for s in ret assign hold boot aux; do
  grep -q "\"$s\":\"[0-9a-f]" manifest.json || { echo "C26_FAIL seed_tree missing $s"; exit 1; }
done

[ -f replay/dataset_0.bin ] || { echo "C26_FAIL no replay"; exit 1; }
sz=$(wc -c < replay/dataset_0.bin | tr -d ' ')
[ "$sz" -eq 4064 ] || { echo "C26_FAIL replay size=$sz want 4064"; exit 1; }

echo "C26_PASS"

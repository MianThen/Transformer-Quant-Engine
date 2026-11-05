#!/bin/bash
cd /tmp/qgate/C17 || exit 1
rm -rf replay decisions.jsonl.gz manifest.json metrics.json
./drv.out > out.txt 2>&1
rc=$?
if [ $rc -eq 124 ]; then echo "C17_FAIL run_hang"; exit 1; fi
if [ $rc -ne 0 ]; then echo "C17_FAIL run_rc=$rc"; head -5 out.txt; exit 1; fi

awk -f c17_verify.awk out.txt || exit 1

# 决策记录
[ -f decisions.jsonl.gz ] || { echo "C17_FAIL no jsonl.gz"; exit 1; }
gzip -t decisions.jsonl.gz 2>/dev/null || { echo "C17_FAIL gzip_t"; exit 1; }
hdr=$(head -c 8 decisions.jsonl.gz | od -An -tx1 | tr -d ' \n')
[ "$hdr" = "1f8b080000000000" ] || { echo "C17_FAIL gzip_hdr=$hdr"; exit 1; }
nl=$(gzip -dc decisions.jsonl.gz | wc -l)
[ "$nl" -eq 1200 ] || { echo "C17_FAIL jsonl lines=$nl want 1200"; exit 1; }

# manifest
[ -f manifest.json ] || { echo "C17_FAIL no manifest"; exit 1; }
grep -qE '"contract_hash":"[0-9a-f]{64}"'  manifest.json || { echo "C17_FAIL manifest contract_hash"; exit 1; }
grep -qE '"config_hash":"[0-9a-f]{64}"'    manifest.json || { echo "C17_FAIL manifest config_hash"; exit 1; }
grep -qE '"binary_sha256":"[0-9a-f]{64}"'  manifest.json || { echo "C17_FAIL manifest binary_sha256"; exit 1; }
grep -qE '"git_commit":"([0-9a-f]{40}|unknown)"' manifest.json || { echo "C17_FAIL manifest git_commit"; exit 1; }

# metrics（AC-32：必须含 cost_profile）
[ -f metrics.json ] || { echo "C17_FAIL no metrics"; exit 1; }
grep -qE '"contract_hash":"[0-9a-f]{64}"'  metrics.json || { echo "C17_FAIL metrics contract_hash"; exit 1; }
grep -qE '"config_hash":"[0-9a-f]{64}"'    metrics.json || { echo "C17_FAIL metrics config_hash"; exit 1; }
grep -qE '"binary_sha256":"[0-9a-f]{64}"'  metrics.json || { echo "C17_FAIL metrics binary_sha256"; exit 1; }
grep -qE '"git_commit":"([0-9a-f]{40}|unknown)"' metrics.json || { echo "C17_FAIL metrics git_commit"; exit 1; }
grep -q '"cost_profile":{"single_core_ms":9046.800' metrics.json || { echo "C17_FAIL metrics cost_profile"; exit 1; }
grep -q '"planned_calls":10000' metrics.json || { echo "C17_FAIL metrics planned_calls"; exit 1; }
grep -q '"over_threshold":2' metrics.json || { echo "C17_FAIL metrics over_threshold"; exit 1; }
if grep -qE '202[0-9]|hostname|elapsed|timestamp' metrics.json; then echo "C17_FAIL metrics has timestamp/host"; exit 1; fi

# replay：1 个数据集，4*126*8 + 32 = 4064 字节
[ -f replay/dataset_0.bin ] || { echo "C17_FAIL no replay file"; exit 1; }
sz=$(wc -c < replay/dataset_0.bin)
[ "$sz" -eq 4064 ] || { echo "C17_FAIL replay size=$sz want 4064"; exit 1; }
nrep=$(ls replay/dataset_*.bin 2>/dev/null | wc -l | tr -d ' ')
[ "$nrep" -eq 1 ] || { echo "C17_FAIL replay files=$nrep want 1"; exit 1; }

echo "C17_PASS"

#!/bin/bash
cd /tmp/qgate/C08 || exit 1
./o.out > out.txt 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "C08_FAIL run_rc=$rc"; head -5 out.txt; exit 1; fi
awk -f c08_verify.awk out.txt || exit 1
if ! gzip -t out.jsonl.gz 2>/dev/null; then echo "C08_FAIL gzip_t"; exit 1; fi
cat > expected.jsonl <<'GZEOF'
{"cell_id":101,"rep_id":0,"rule_id":"GATE_Full","selected":false,"index":1,"abstain":false,"returns_sha256":"92b4f2a290c24fe4","reason":"below_threshold"}
{"cell_id":101,"rep_id":0,"rule_id":"B0","selected":true,"index":1,"abstain":false,"returns_sha256":"92b4f2a290c24fe4","reason":"baseline"}
{"cell_id":205,"rep_id":2,"rule_id":"B4","selected":true,"index":4,"abstain":true,"returns_sha256":"0123456789abcdef","reason":"ocn_abstain"}
GZEOF
if ! gzip -dc out.jsonl.gz > dec.jsonl 2>/dev/null; then echo "C08_FAIL gzip_dc"; exit 1; fi
if ! cmp -s dec.jsonl expected.jsonl; then echo "C08_FAIL roundtrip_mismatch"; exit 1; fi
sz=$(wc -c < out.jsonl.gz)
if [ "$sz" -ne 460 ]; then echo "C08_FAIL file_size got $sz want 460"; exit 1; fi
echo "C08_PASS"

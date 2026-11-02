#!/bin/bash
cd /tmp/qgate/C12 || exit 1
./drv.out > out.txt 2>&1
[ $? -eq 0 ] || { echo "C12_FAIL run_rc=$?"; head -5 out.txt; exit 1; }

awk '
BEGIN{ bad=0; ns=0; ng=0; nc=0; nm=0; done=0; sel1004=-1 }
/^SMOKE records=/{
  ns++
  if($2!="records=1200"){printf "C12_FAIL records got %s\n",$2; bad++}
  if($3!="cells=4"){printf "C12_FAIL cells got %s\n",$3; bad++}
  if($4!="reps=50"){printf "C12_FAIL reps got %s\n",$4; bad++}
}
/^SMOKE gzip_size=/{ ns++ }
/^GATE_SEL /{ ng++; if($4+0!=50){printf "C12_FAIL GATE_SEL n got %s\n",$4; bad++} if($2+0==1004) sel1004=$3+0 }
/^CANARY_RATE /{
  nc++; got=$2
  if(got !~ /^[0-9]+\/50$/){printf "C12_FAIL canary_rate format %s\n",got; bad++}
  else { split(got,a,"/"); if(sel1004>=0 && a[1]+0!=sel1004){printf "C12_FAIL canary_rate mismatch\n"; bad++} }
}
/^METRICS_WRITTEN /{ nm++; if($2!="1"){printf "C12_FAIL metrics_written %s\n",$2; bad++} }
/^SMOKE_DONE$/{done=1}
END{
  if(!done){printf "C12_FAIL missing SMOKE_DONE\n"; bad++}
  if(ns!=2){printf "C12_FAIL SMOKE lines=%d want=2\n",ns; bad++}
  if(ng!=4){printf "C12_FAIL GATE_SEL lines=%d want=4\n",ng; bad++}
  if(nc!=1){printf "C12_FAIL CANARY lines=%d want=1\n",nc; bad++}
  if(nm!=1){printf "C12_FAIL METRICS lines=%d want=1\n",nm; bad++}
  if(bad==0){printf "C12_AWK_PASS\n"} else {printf "C12_FAIL total=%d\n",bad; exit 1}
}' out.txt || exit 1

[ -f decisions.jsonl.gz ] || { echo "C12_FAIL no jsonl.gz"; exit 1; }
gzip -t decisions.jsonl.gz 2>/dev/null || { echo "C12_FAIL gzip_t"; exit 1; }
hdr=$(head -c 8 decisions.jsonl.gz | od -An -tx1 | tr -d ' \n')
[ "$hdr" = "1f8b080000000000" ] || { echo "C12_FAIL gzip_hdr=$hdr"; exit 1; }

python3 - <<'PY' || exit 1
import gzip, json, os, re, sys
bad = []
out = open("out.txt").read().splitlines()
gzip_size = None; gate_sel = {}; canary = None
for ln in out:
    if ln.startswith("SMOKE gzip_size="): gzip_size = int(ln.split("=",1)[1])
    elif ln.startswith("GATE_SEL "):
        p = ln.split(); gate_sel[int(p[1])] = int(p[2])
    elif ln.startswith("CANARY_RATE "): canary = ln.split()[1]
path = "decisions.jsonl.gz"
if gzip_size != os.path.getsize(path):
    bad.append("gzip_size mismatch")
raw = gzip.open(path, "rb").read()
lines = raw.decode("utf-8").splitlines()
if len(lines) != 1200: bad.append("jsonl lines=%d want 1200" % len(lines))
if not raw.endswith(b"\n"): bad.append("jsonl missing trailing newline")
RULE_ORDER = ["GATE_Full","B0","B1","B2","B3","B4"]
pat = re.compile(r'^\{"cell_id":(\d+),"rep_id":(\d+),"rule_id":"([A-Za-z0-9_]+)",'
                 r'"selected":(true|false),"index":(\d+),"abstain":(true|false),'
                 r'"returns_sha256":"([0-9a-f]{16})","reason":"([a-z_]+)"\}$')
buckets = {}; gate_true = {}; last = None; order_ok = True
for i, ln in enumerate(lines):
    m = pat.match(ln)
    if not m:
        bad.append("bad_json_line %d" % (i+1))
        if len(bad) > 6: break
        continue
    cell=int(m.group(1)); rep=int(m.group(2)); rule=m.group(3)
    sel=m.group(4)=="true"; sha=m.group(7)
    buckets.setdefault((cell,rep), []).append((rule, sha))
    if rule=="GATE_Full" and sel: gate_true[cell]=gate_true.get(cell,0)+1
    cur=(cell,rep,RULE_ORDER.index(rule) if rule in RULE_ORDER else 999)
    if last is not None and cur<last: order_ok=False
    last=cur
if not order_ok: bad.append("records not ordered by (cell,rep,rule)")
want=set((c,r) for c in (1001,1002,1003,1004) for r in range(50))
if set(buckets)!=want: bad.append("(cell,rep) set mismatch")
for key,items in buckets.items():
    rules=[r for r,_ in items]
    if sorted(rules)!=sorted(RULE_ORDER): bad.append("key %s rule set wrong" % (key,)); continue
    if len(set(s for _,s in items))!=1: bad.append("key %s fp not shared" % (key,))
for cell in (1001,1002,1003,1004):
    n=len(set(buckets[(cell,r)][0][1] for r in range(50) if (cell,r) in buckets))
    if n!=50: bad.append("cell %d distinct fps=%d" % (cell,n))
for cell,printed in gate_sel.items():
    if printed != gate_true.get(cell,0):
        bad.append("GATE_SEL %d printed %d jsonl %d" % (cell,printed,gate_true.get(cell,0)))
if canary != "%d/50" % gate_sel.get(1004,-1): bad.append("CANARY_RATE inconsistent")
if bad:
    for b in bad[:10]: print("C12_FAIL "+b)
    sys.exit(1)
print("C12_CROSS_PASS")
PY

[ -f metrics.json ] || { echo "C12_FAIL no metrics.json"; exit 1; }
grep -q '"records":1200' metrics.json || { echo "C12_FAIL metrics records"; exit 1; }
grep -q '"cells":4' metrics.json || { echo "C12_FAIL metrics cells"; exit 1; }
grep -q '"gzip_mtime":0' metrics.json || { echo "C12_FAIL metrics mtime"; exit 1; }
grep -qE '202[0-9]|hostname|elapsed|timestamp' metrics.json && { echo "C12_FAIL metrics has timestamp/host"; exit 1; }
echo "C12_PASS"

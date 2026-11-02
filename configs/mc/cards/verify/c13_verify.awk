BEGIN{
  n=0; bad=0; done=0
  want_manifest = "{\"contract_hash\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"config_hash\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\",\"binary_sha256\":\"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\",\"git_commit\":\"dddddddddddddddddddddddddddddddddddddddd\"}"
  want_metrics  = "{\"contract_hash\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"config_hash\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\",\"binary_sha256\":\"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\",\"git_commit\":\"dddddddddddddddddddddddddddddddddddddddd\",\"records\":1200,\"cells\":4,\"reps_per_cell\":50,\"rules\":6,\"gzip_mtime\":0,\"budget_threshold_ms\":1.000000,\"planned_calls\":9500}"
  want_ladder   = "8000->4000 2000->1000 1000->500 4000->4000 500->500"
}
/^MANIFEST /{
  n++
  p=index($0," "); got=substr($0,p+1)
  if(got!=want_manifest){printf "C13_FAIL manifest\n  got  [%s]\n  want [%s]\n",got,want_manifest; bad++}
}
/^LADDER /{
  n++
  p=index($0," "); got=substr($0,p+1)
  if(got!=want_ladder){printf "C13_FAIL ladder got [%s]\n",got; bad++}
}
/^PLAN /{
  n++
  got=$2" "$3" "$4" "$5" "$6
  if(got!="4000 2000 2000 500 1000"){printf "C13_FAIL plan got [%s]\n",got; bad++}
}
/^PLAN2 /{
  n++
  got=$2" "$3" "$4" "$5" "$6
  if(got!="8000 2000 2000 1000 1000"){printf "C13_FAIL plan2 got [%s]\n",got; bad++}
}
/^CALLS /{ n++; if($2+0!=9500){printf "C13_FAIL calls got %s want 9500\n",$2; bad++} }
/^CALLS2 /{ n++; if($2+0!=14000){printf "C13_FAIL calls2 got %s want 14000\n",$2; bad++} }
/^METRICS /{
  n++
  p=index($0," "); got=substr($0,p+1)
  if(got!=want_metrics){printf "C13_FAIL metrics\n  got  [%s]\n  want [%s]\n",got,want_metrics; bad++}
}
/^MANIFEST_DONE$/{done=1}
END{
  if(!done){printf "C13_FAIL missing MANIFEST_DONE\n"; bad++}
  if(n!=7){printf "C13_FAIL lines=%d want=7\n",n; bad++}
  if(bad==0){printf "C13_PASS\n"} else {printf "C13_FAIL total=%d\n",bad; exit 1}
}

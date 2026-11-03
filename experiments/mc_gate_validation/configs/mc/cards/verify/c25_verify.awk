BEGIN{ n=0; bad=0; done=0
  F["ret"]="c849fb5a38a25ca2 9d4fc1d81aeff5ac 464963e575d48a7a"; F["assign"]="853aca6fd63317e8 02d6e0fe6c9acb58 57e0ae21d946b4da"; F["hold"]="60a83a5e1b236d92 bb349715cf615955 e1a80f8e76eb077c"; F["boot"]="0bd20b9aab82efa3 a76a13e5b9b5a0cc 828bc176edb79059"; F["aux"]="d3b54ecaed619428 0c16b2f891f24d51 08e7cc642673b005"
  CANON="{\"contract_version\":\"1.4.5\",\"grid\":\"mc_gate_core_v1\",\"run_seed\":20261027,\"q\":0.050000,\"alpha\":0.050000,\"reps_scale\":1}"
  CH="64a6f1bfa3d6a955ef9b64b542795e6ff24f1cbc0ea1409f9dba0910c3090ea6"; TH="c1e40f3dc36d2f1f62c43fb0daa48b16bcbd45ad3871358a44e6c65b9428d07c"
}
/^SEEDREC_BEGIN$/{ n++ }
/^FP /{
  n++
  s=$2
  if(!(s in F)){printf "C25_FAIL unknown stream %s\n",s; bad++; next}
  got=$3" "$4" "$5
  if(got!=F[s]){printf "C25_FAIL FP %s\n  got  [%s]\n  want [%s]\n",s,got,F[s]; bad++}
}
/^CANON /{
  n++
  got=substr($0, length("CANON ")+1)
  if(got!=CANON){printf "C25_FAIL canon\n  got  [%s]\n  want [%s]\n",got,CANON; bad++}
}
/^CONFIG_HASH /{ n++; if($2!=CH){printf "C25_FAIL config_hash %s\n",$2; bad++} }
/^TAMPER_HASH /{ n++; if($2!=TH){printf "C25_FAIL tamper_hash %s\n",$2; bad++} }
/^SEEDREC_DONE$/{done=1}
$0 !~ /^(FP|CANON|CONFIG_HASH|TAMPER_HASH) / && $0 != "SEEDREC_DONE" && $0 != "SEEDREC_BEGIN" {
  printf "C25_FAIL unexpected [%s]\n",$0; bad++
}
END{
  if(!done){printf "C25_FAIL missing SEEDREC_DONE\n"; bad++}
  if(n!=9){printf "C25_FAIL counted=%d want=9\n",n; bad++}
  if(bad==0){printf "C25_PASS\n"} else {printf "C25_FAIL total=%d\n",bad; exit 1}
}

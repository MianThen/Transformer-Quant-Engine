BEGIN{
  gt["S1"]="2.105263 1.816170 1.417588"
  gt["S2"]="3.257329 2.081112 2.081112"
  gt["S3"]="2.105263 1.816170 1.417588"
  de["S1 Full"]="0 0 0 fdr_rejected"
  de["S1 DsrOnly"]="1 0 0 selected"
  de["S1 FdrOnly"]="0 0 0 fdr_rejected"
  de["S1 RawT"]="1 0 0 selected"
  de["S2 Full"]="1 0 0 selected"
  de["S2 DsrOnly"]="1 0 0 selected"
  de["S2 FdrOnly"]="1 0 0 selected"
  de["S2 RawT"]="1 0 0 selected"
  de["S3 Full"]="0 0 0 below_threshold"
  de["S3 DsrOnly"]="0 0 0 below_threshold"
  de["S3 FdrOnly"]="0 0 0 below_threshold"
  de["S3 RawT"]="0 0 0 raw_t_rejected"
  de["S1_Abort Full"]="0 0 1 ocn_abstain"
  n=0; bad=0
}
/^GATE_THR /{
  n++
  if(!($2 in gt)){printf "C05_FAIL unknown GATE_THR %s\n",$2; bad++; next}
  split(gt[$2],e," ")
  for(j=1;j<=3;j++){ d=$(2+j)-e[j]; d=(d<0?-d:d); if(d>1e-4){printf "C05_FAIL GATE_THR %s col%d got %s want %s\n",$2,j,$(2+j),e[j]; bad++} }
}
/^DECIDE /{
  n++
  k=$2" "$3
  if(!(k in de)){printf "C05_FAIL unknown DECIDE %s\n",k; bad++; next}
  g=$4" "$5" "$6" "$7
  if(g!=de[k]){printf "C05_FAIL DECIDE %s got [%s] want [%s]\n",k,g,de[k]; bad++}
}
/^DECIDE_DONE$/{done=1}
END{
  if(!done){printf "C05_FAIL missing DECIDE_DONE\n"; bad++}
  if(n!=16){printf "C05_FAIL line count %d want 16\n",n; bad++}
  if(bad==0){printf "C05_PASS\n"} else {printf "C05_FAIL total %d\n",bad; exit(1)}
}

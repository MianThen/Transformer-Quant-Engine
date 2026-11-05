BEGIN{ n=0; bad=0; done=0 }
/^MERGE /{
  n++
  if($2!=12){printf "C11_FAIL merge n got %s want 12\n",$2; bad++}
  if($3!="first=101,0,GATE_Full"){printf "C11_FAIL first got %s\n",$3; bad++}
  if($4!="last=205,2,B0"){printf "C11_FAIL last got %s\n",$4; bad++}
}
/^AGG /{
  n++
  k=$2
  if(k=="B0") w="n=6 selected=2 abstain=0"
  else if(k=="GATE_Full") w="n=6 selected=3 abstain=1"
  else {printf "C11_FAIL unknown rule %s\n",k; bad++; next}
  got=$3" "$4" "$5
  if(got!=w){printf "C11_FAIL %s got [%s] want [%s]\n",k,got,w; bad++}
  if(k=="B0") wr=0.333333; else wr=0.5
  r=$6; sub("rate=","",r); d=r-wr; d=(d<0?-d:d)
  if(d>1e-6){printf "C11_FAIL %s rate got %s want %.6f\n",k,r,wr; bad++}
}
/^DUP /{ n++; if($2!=0){printf "C11_FAIL dup got %s want 0\n",$2; bad++} }
/^SHUFFLE_EQUALS /{ n++; if($2!=1){printf "C11_FAIL shuffle got %s want 1\n",$2; bad++} }
/^REDUCE_DONE$/{done=1}
END{
  if(!done){printf "C11_FAIL missing REDUCE_DONE\n"; bad++}
  if(n!=5){printf "C11_FAIL lines=%d want=5\n",n; bad++}
  if(bad==0){printf "C11_PASS\n"} else {printf "C11_FAIL total=%d\n",bad; exit 1}
}

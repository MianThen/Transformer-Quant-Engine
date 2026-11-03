BEGIN{ n=0; bad=0; done=0; hits=0 }
/^CANARY /{
  n++
  if(NF!=5){printf "C29_FAIL fields=%d want 5\n",NF; bad++; next}
  rep=$2+0; sel=$3+0; idx=$4+0; kap=$5+0
  if(sel!=1){printf "C29_FAIL rep %d not selected (Gold alpha S=50 must be selected)\n",rep; bad++}
  else if(idx!=kap){printf "C29_FAIL rep %d index=%d but kappa=%d\n",rep,idx,kap; bad++}
  else hits++
  if(idx<0||idx>=100){printf "C29_FAIL rep %d index out of range %d\n",rep,idx; bad++}
  if(kap<0||kap>=100){printf "C29_FAIL rep %d kappa out of range %d\n",rep,kap; bad++}
}
/^CANARY_OK /{
  n++
  split($2,a,"/")
  if(a[1]+0!=20||a[2]+0!=20){printf "C29_FAIL CANARY_OK %s want 20/20\n",$2; bad++}
  if(hits!=20){printf "C29_FAIL counted hits=%d want 20\n",hits; bad++}
}
/^CANARY_DONE$/{done=1}
$0 !~ /^(CANARY|CANARY_OK) / && $0 != "CANARY_DONE" { printf "C29_FAIL unexpected [%s]\n",$0; bad++ }
END{
  if(!done){printf "C29_FAIL missing CANARY_DONE\n"; bad++}
  if(n!=21){printf "C29_FAIL lines=%d want 21\n",n; bad++}
  if(bad==0){printf "C29_AWK_PASS\n"} else {printf "C29_FAIL total=%d\n",bad; exit 1}
}

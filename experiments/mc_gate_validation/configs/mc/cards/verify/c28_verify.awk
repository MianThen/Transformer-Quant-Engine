BEGIN{ n=0; bad=0; done=0; nT4=0; nT100=0 }
/^OCNROW /{
  n++
  if(NF!=7){printf "C28_FAIL fields=%d want 7\n",NF; bad++; next}
  T=$2+0; comp=$3+0; rho10=$4+0; ab=$5+0; sall=$6+0; scond=$7+0
  if(ab<0||ab>1){printf "C28_FAIL abstain out of range %s\n",$5; bad++}
  if(sall<0||sall>1){printf "C28_FAIL sel_all out of range %s\n",$6; bad++}
  if(scond<0||scond>1){printf "C28_FAIL sel_cond out of range %s\n",$7; bad++}
  if(scond+1e-9 < sall){printf "C28_FAIL T=%d comp=%d: sel_cond(%s) < sel_all(%s)\n",T,comp,$7,$6; bad++}
  if(comp==0 && sall>0.05+1e-9){printf "C28_FAIL null FPR too high T=%d rho=%d: %s\n",T,rho10,$6; bad++}
  if(T==4 && comp==0){ nT4++; if(ab<0.40||ab>0.80){printf "C28_FAIL T=4 null abstain %s not in [0.40,0.80]\n",$5; bad++} }
  if(T==100 && rho10==0){ nT100++; if(ab>1e-9){printf "C28_FAIL T=100 rho=0 abstain %s should be 0\n",$5; bad++} }
}
/^OCN_DONE$/{done=1}
$0 !~ /^OCNROW / && $0 != "OCN_DONE" { printf "C28_FAIL unexpected [%s]\n",$0; bad++ }
END{
  if(!done){printf "C28_FAIL missing OCN_DONE\n"; bad++}
  if(n!=18){printf "C28_FAIL rows=%d want 18\n",n; bad++}
  if(nT4!=3){printf "C28_FAIL T=4 null rows=%d want 3\n",nT4; bad++}
  if(nT100!=2){printf "C28_FAIL T=100 rho=0 rows=%d want 2\n",nT100; bad++}
  if(bad==0){printf "C28_AWK_PASS\n"} else {printf "C28_FAIL total=%d\n",bad; exit 1}
}

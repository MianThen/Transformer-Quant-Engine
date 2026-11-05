BEGIN{ n=0; bad=0; done=0 }
/^PLOT /{
  n++
  if(NF!=4){printf "C20_FAIL PLOT fields=%d want 4\n",NF; bad++}
  k=$2+0
  if(k==0){ if($3!="fpr_vs_t"){printf "C20_FAIL spec0 name %s\n",$3; bad++} if($4+0!=3){printf "C20_FAIL spec0 points %s\n",$4; bad++} }
  else if(k==1){ if($3!="abstain_vs_t"){printf "C20_FAIL spec1 name %s\n",$3; bad++} if($4+0!=3){printf "C20_FAIL spec1 points %s\n",$4; bad++} }
  else { printf "C20_FAIL unknown spec idx %d\n",k; bad++ }
}
/^TRACE_OK /{ n++; if(NF!=2){bad++} if($2+0!=1){printf "C20_FAIL TRACE_OK got %s want 1\n",$2; bad++} }
/^TRACE_BAD /{
  n++
  if(NF!=3){printf "C20_FAIL TRACE_BAD fields=%d want 3\n",NF; bad++}
  if($2+0!=0){printf "C20_FAIL TRACE_BAD got %s want 0\n",$2; bad++}
  if($3!="fpr_vs_t"){printf "C20_FAIL TRACE_BAD name %s want fpr_vs_t\n",$3; bad++}
}
/^PLOT_DONE$/{done=1}
$0 !~ /^(PLOT|TRACE_OK|TRACE_BAD) / && $0 != "PLOT_DONE" {
  printf "C20_FAIL unexpected line [%s]\n",$0; bad++
}
END{
  if(!done){printf "C20_FAIL missing PLOT_DONE\n"; bad++}
  if(n!=4){printf "C20_FAIL counted=%d want=4\n",n; bad++}
  if(bad==0){printf "C20_PASS\n"} else {printf "C20_FAIL total=%d\n",bad; exit 1}
}

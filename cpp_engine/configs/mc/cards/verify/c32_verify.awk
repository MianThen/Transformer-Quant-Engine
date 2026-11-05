BEGIN{ n=0; bad=0; done=0; np=0; nh=0 }
/^PERM /{ n++; np++
  if(NF!=5){printf "C32_FAIL PERM fields=%d\n",NF; bad++}
  if($3+0!=7){printf "C32_FAIL PERM rep %d i0=%s want 7 (kappa)\n",$2,$3; bad++}
  if($4+0!=92){printf "C32_FAIL PERM rep %d i1=%s want 92 (T-1-kappa)\n",$2,$4; bad++}
  if($5+0!=1){printf "C32_FAIL PERM rep %d ok=%s\n",$2,$5; bad++} }
/^HOLD /{ n++; nh++
  if(NF!=4){printf "C32_FAIL HOLD fields=%d\n",NF; bad++}
  if($3+0!=1){printf "C32_FAIL HOLD rep %d bytes differ after hold generation\n",$2; bad++}
  if($4+0!=12600){printf "C32_FAIL HOLD size %s want 12600\n",$4; bad++} }
/^AC17_OK /{ n++; if($2!="10/10"){printf "C32_FAIL AC17_OK %s\n",$2; bad++} }
/^AC16_OK /{ n++; if($2!="10/10"){printf "C32_FAIL AC16_OK %s\n",$2; bad++} }
/^INVAR_DONE$/{done=1}
$0 !~ /^(PERM|HOLD|AC17_OK|AC16_OK) / && $0 != "INVAR_DONE" { printf "C32_FAIL unexpected [%s]\n",$0; bad++ }
END{
  if(!done){printf "C32_FAIL missing INVAR_DONE\n"; bad++}
  if(np!=10||nh!=10){printf "C32_FAIL PERM=%d HOLD=%d want 10/10\n",np,nh; bad++}
  if(n!=22){printf "C32_FAIL counted=%d want=22\n",n; bad++}
  if(bad==0){printf "C32_AWK_PASS\n"} else {printf "C32_FAIL total=%d\n",bad; exit 1}
}

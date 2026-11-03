BEGIN{ n=0; bad=0; done=0; have1=0; have10=0 }
/^RET1 /{ n++; have1=1
  if($2!~/^fpr=/){printf "C30_FAIL RET1 field2 [%s]\n",$2; bad++}
  else { v=substr($2,5)+0; if(v<0||v>1){printf "C30_FAIL RET1 fpr out of range %s\n",$2; bad++} }
  if($3!="reps=1000"){printf "C30_FAIL RET1 reps %s\n",$3; bad++}
  k1=substr($2,5)+0 }
/^RET10 /{ n++; have10=1
  v=substr($2,6)+0; if(v<0||v>1){printf "C30_FAIL RET10 fpr out of range %s\n",$2; bad++}
  if($3!="reps=1000"){printf "C30_FAIL RET10 reps %s\n",$3; bad++}
  k10=v }
/^PRED /{ n++; if(NF<3){printf "C30_FAIL PRED fields %d\n",NF; bad++} }
/^RATIO /{ n++; r=substr($2,1)+0
  if(r<1.0){printf "C30_FAIL RATIO %s should be > 1 (retry must increase FPR)\n",$2; bad++} }
/^RETRY_DONE$/{done=1}
$0 !~ /^(RET1|RET10|PRED|RATIO) / && $0 != "RETRY_DONE" { printf "C30_FAIL unexpected [%s]\n",$0; bad++ }
END{
  if(!done){printf "C30_FAIL missing RETRY_DONE\n"; bad++}
  if(n!=4){printf "C30_FAIL counted=%d want=4\n",n; bad++}
  if(!have1||!have10){printf "C30_FAIL missing RET1/RET10\n"; bad++}
  else if(k10<=k1){printf "C30_FAIL K=10 (%.4f) must exceed K=1 (%.4f)\n",k10,k1; bad++}
  if(bad==0){printf "C30_AWK_PASS\n"} else {printf "C30_FAIL total=%d\n",bad; exit 1}
}

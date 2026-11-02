BEGIN{ n=0; bad=0; done=0; P[0]=8000; P[1]=1000; P[2]=500; P[3]=500 }
/^PLAN [0-9]+ /{
  n++; k=$2+0
  if(k<0||k>3){printf "C26_FAIL plan idx %d\n",k; bad++; next}
  if($3+0!=P[k]){printf "C26_FAIL plan %d got %s want %d\n",k,$3,P[k]; bad++}
}
/^CALLS /{ n++; if($2!="orig=12000"){printf "C26_FAIL orig %s\n",$2;bad++}
           if($3!="planned=10000"){printf "C26_FAIL planned %s\n",$3;bad++} }
/^COST total /{ n++; d=$3-9046.800; d=(d<0?-d:d)
                if(d>1e-3){printf "C26_FAIL total %s\n",$3;bad++}
                if($5+0!=2){printf "C26_FAIL over %s\n",$5;bad++} }
/^RECORDS /{ n++; if($2+0!=60000){printf "C26_FAIL records %s want 60000\n",$2;bad++} }
/^WALL /{
  n++
  if(NF!=7){printf "C26_FAIL wall fields %d\n",NF; bad++}
  s=0; for(i=2;i<=7;i++){ if($i+0<0){printf "C26_FAIL wall neg %s\n",$i;bad++} s+=$i+0 }
  if(s<=0){printf "C26_FAIL wall all zero\n";bad++}
}
/^FILES /{ n++; if(NF!=5){bad++} for(i=2;i<=5;i++) if($i+0!=1){printf "C26_FAIL files col %d\n",i;bad++} }
/^RUN_DONE$/{done=1}
$0 !~ /^(PLAN|CALLS|COST|RECORDS|WALL|FILES) / && $0 != "RUN_DONE" { printf "C26_FAIL unexpected [%s]\n",$0; bad++ }
END{
  if(!done){printf "C26_FAIL missing RUN_DONE\n";bad++}
  if(n!=9){printf "C26_FAIL counted=%d want=9\n",n;bad++}
  if(bad==0){printf "C26_AWK_PASS\n"} else {printf "C26_FAIL total=%d\n",bad; exit 1}
}

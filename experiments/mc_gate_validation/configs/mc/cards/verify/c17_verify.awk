BEGIN{
  n=0; bad=0; done=0
  pw[0]=8000; pw[1]=1000; pw[2]=500; pw[3]=500
  sw[1]=0.530840; sw[2]=0.328359; sw[3]=0.101406; sw[4]=0.039439
}
/^PLAN [0-9]+ /{
  n++; k=$2+0
  if(k<0 || k>3){printf "C17_FAIL plan idx %d\n",k; bad++; next}
  if($3+0!=pw[k]){printf "C17_FAIL plan %d got %s want %d\n",k,$3,pw[k]; bad++}
}
/^CALLS /{
  n++
  if($2!="orig=12000"){printf "C17_FAIL orig %s\n",$2; bad++}
  if($3!="planned=10000"){printf "C17_FAIL planned %s\n",$3; bad++}
}
/^COST total /{
  n++
  d=$3-9046.800; d=(d<0?-d:d)
  if(d>1e-3){printf "C17_FAIL total got %s want 9046.800\n",$3; bad++}
  if($5+0!=2){printf "C17_FAIL over got %s want 2\n",$5; bad++}
}
/^COST shares /{
  n++
  if(NF!=6){printf "C17_FAIL shares fields %d want 6\n",NF; bad++}
  for(i=1;i<=4;i++){
    d=$(i+2)-sw[i]; d=(d<0?-d:d)
    if(d>1e-6){printf "C17_FAIL share %d got %s want %.6f\n",i,$(i+2),sw[i]; bad++}
  }
}
/^RECORDS /{ n++; if($2+0!=1200){printf "C17_FAIL records got %s want 1200\n",$2; bad++} }
/^WALL /{
  n++
  if(NF!=7){printf "C17_FAIL wall fields %d want 7\n",NF; bad++}
  ws=0
  for(i=2;i<=7;i++){ if($i+0<0){printf "C17_FAIL wall neg %s\n",$i; bad++} ws+=$i+0 }
  if(ws<=0){printf "C17_FAIL wall all zero\n"; bad++}
}
/^FILES /{
  n++
  if(NF!=5){printf "C17_FAIL files fields %d want 5\n",NF; bad++}
  for(i=2;i<=5;i++){ if($i+0!=1){printf "C17_FAIL files col %d got %s want 1\n",i,$i; bad++} }
}
/^COST /{
  if($2!="total" && $2!="shares"){printf "C17_FAIL unknown COST key %s\n",$2; bad++}
}
$0 !~ /^(PLAN|CALLS|COST|RECORDS|WALL|FILES) / && $0 != "RUN_DONE" {
  printf "C17_FAIL unexpected line [%s]\n",$0; bad++
}
/^RUN_DONE$/{done=1}
END{
  if(!done){printf "C17_FAIL missing RUN_DONE\n"; bad++}
  if(n!=10){printf "C17_FAIL counted lines=%d want=10\n",n; bad++}
  if(bad==0){printf "C17_AWK_PASS\n"} else {printf "C17_FAIL total=%d\n",bad; exit 1}
}

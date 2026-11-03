BEGIN{ n=0; bad=0; done=0
  W[1]="bf7d80c56930e0e1"; W[2]="3fa3fab0fae971a3"; W[3]="bf825d46e6270511"; W[4]="3f68c1c71d3c948a"
  W[5]="bf82804f21bc51d7"; W[6]="3f433722c91aa67b"; W[7]="3f7c9402ec673160"; W[8]="bf21f951d206fe72"
  K[0]=0; K[1]=0; K[2]=0; K[3]=3
}
/^KAPPA /{
  n++
  if(NF!=3){printf "C24_FAIL KAPPA fields %d\n",NF; bad++; next}
  r=$2+0
  if(!(r in K)){printf "C24_FAIL unknown rep %d\n",r; bad++; next}
  if($3+0!=K[r]){printf "C24_FAIL KAPPA %d got %s want %d\n",r,$3,K[r]; bad++}
}
/^HOLD8 /{
  n++
  if(NF!=9){printf "C24_FAIL HOLD8 fields %d want 9\n",NF; bad++}
  for(i=1;i<=8;i++) if($(i+1)!=W[i]){printf "C24_FAIL HOLD8[%d] got %s want %s\n",i,$(i+1),W[i]; bad++}
}
/^HOLD_SHA /{ n++; if($2!="5de704f8ef97cff1bda1ce8d2ef3945439243b1b1b4877cc348697d66ff25628"){printf "C24_FAIL HOLD_SHA %s\n",$2; bad++} }
/^STREAMS_DONE$/{done=1}
$0 !~ /^(KAPPA|HOLD8|HOLD_SHA) / && $0 != "STREAMS_DONE" { printf "C24_FAIL unexpected [%s]\n",$0; bad++ }
END{
  if(!done){printf "C24_FAIL missing STREAMS_DONE\n"; bad++}
  if(n!=6){printf "C24_FAIL counted=%d want=6\n",n; bad++}
  if(bad==0){printf "C24_PASS\n"} else {printf "C24_FAIL total=%d\n",bad; exit 1}
}

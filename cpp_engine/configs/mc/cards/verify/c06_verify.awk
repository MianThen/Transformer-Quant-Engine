BEGIN{
  sr[0]=3.011472714649464
  sr[1]=0.000000000000000
  sr[2]=0.045671624108428
  sr[3]=-0.103607319982201
  sr[4]=-0.057632379010077
  sr[5]=0.183468866871128
  sr[6]=0.034259033392405
  sr[7]=-0.113967089050806
  sr[8]=-0.068531742002118
  sr[9]=-0.023063893610305
  sr[10]=0.023063893610305
  sr[11]=0.068531742002118
  sr[12]=0.113967089050806
  sr[13]=-0.034259033392405
  sr[14]=-0.183468866871128
  sr[15]=0.057632379010077
  sr[16]=0.103607319982201
  sr[17]=-0.045671624108428
  sr[18]=0.000000000000000
  sr[19]=0.045671624108428
  M=0.362070870867891; THF=5.325896864348375; THD=5.325896864348375
  de["Full"]="0 0 0 below_threshold"
  de["DsrOnly"]="0 0 0 below_threshold"
  de["FdrOnly"]="0 0 0 below_threshold"
  de["RawT"]="1 0 0 selected"
  de["Abort"]="0 0 1 ocn_abstain"
  n=0; bad=0; TOL=1e-3
}
/^SR /{
  n++; i=$2+0
  if(!(i in sr)){printf "C06_FAIL unknown SR idx %d\n",i; bad++; next}
  d=$3-sr[i]; d=(d<0?-d:d)
  if(d>TOL){printf "C06_FAIL SR %d got %s want %.6f\n",i,$3,sr[i]; bad++}
}
/^M /{
  n++; d=$2-M; d=(d<0?-d:d)
  if(d>TOL){printf "C06_FAIL M got %s want %.6f\n",$2,M; bad++}
}
/^THR /{
  n++; d1=$2-THF; d1=(d1<0?-d1:d1); d2=$3-THD; d2=(d2<0?-d2:d2)
  if(d1>TOL||d2>TOL){printf "C06_FAIL THR got %s %s want %.6f %.6f\n",$2,$3,THF,THD; bad++}
}
/^DECIDE /{
  n++; k=$2; g=$3" "$4" "$5" "$6
  if(!(k in de)){printf "C06_FAIL unknown DECIDE %s\n",k; bad++; next}
  if(g!=de[k]){printf "C06_FAIL DECIDE %s got [%s] want [%s]\n",k,g,de[k]; bad++}
}
/^ADAPTER_DONE$/{done=1}
END{
  if(!done){printf "C06_FAIL missing ADAPTER_DONE\n"; bad++}
  if(n!=27){printf "C06_FAIL line count %d want 27\n",n; bad++}
  if(bad==0){printf "C06_PASS\n"} else {printf "C06_FAIL total %d\n",bad; exit(1)}
}

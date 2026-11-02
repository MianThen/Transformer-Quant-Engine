BEGIN{
  w["100_1000"]="0.082909443593096 0.120151963195348"
  w["40_1000"]="0.029510973685453 0.054009644121258"
  w["60_1000"]="0.046896472675932 0.076471074791357"
  w["65_1000"]="0.051323770569062 0.082005509313371"
  w["0_1000"]="0.000000000000000 0.003826758485555"
  w["1000_1000"]="0.996173241514445 1.000000000000000"
  mc["150_50_1000"]="-0.127016239115241 -0.072983760884759"
  mc["55_45_1000"]="-0.029589837574297 0.009589837574297"
  mc["10_10_1000"]="-0.008765225405766 0.008765225405766"
  mc["500_400_1000"]="-0.158471346405036 -0.041528653594964"
  vd["S1"]="V3 0 0"
  vd["S2"]="V2 0 0"
  vd["S3"]="VU 0 0"
  vd["S4"]="V1 0 0"
  vd["S5"]="V1 1 1"
  n=0; bad=0; TOL=2e-6
}
/^WILSON /{
  n++; k=$2"_"$3
  if(!(k in w)){printf "C07_FAIL unknown WILSON %s\n",k; bad++; next}
  split(w[k],e," ")
  d1=$4-e[1]; d1=(d1<0?-d1:d1); d2=$5-e[2]; d2=(d2<0?-d2:d2)
  if(d1>TOL||d2>TOL){printf "C07_FAIL WILSON %s got %s %s want %s %s\n",k,$4,$5,e[1],e[2]; bad++}
}
/^MCNEMAR /{
  n++; k=$2"_"$3"_"$4
  if(!(k in mc)){printf "C07_FAIL unknown MCNEMAR %s\n",k; bad++; next}
  split(mc[k],e," ")
  d1=$5-e[1]; d1=(d1<0?-d1:d1); d2=$6-e[2]; d2=(d2<0?-d2:d2)
  if(d1>TOL||d2>TOL){printf "C07_FAIL MCNEMAR %s got %s %s want %s %s\n",k,$5,$6,e[1],e[2]; bad++}
}
/^VERDICT /{
  n++; k=$2; g=$3" "$4" "$5
  if(!(k in vd)){printf "C07_FAIL unknown VERDICT %s\n",k; bad++; next}
  if(g!=vd[k]){printf "C07_FAIL VERDICT %s got [%s] want [%s]\n",k,g,vd[k]; bad++}
}
/^SCORER_DONE$/{done=1}
END{
  if(!done){printf "C07_FAIL missing SCORER_DONE\n"; bad++}
  if(n!=15){printf "C07_FAIL line count %d want 15\n",n; bad++}
  if(bad==0){printf "C07_PASS\n"} else {printf "C07_FAIL total %d\n",bad; exit(1)}
}

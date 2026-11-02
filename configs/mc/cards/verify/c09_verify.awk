BEGIN{
  n=0; bad=0; done=0
  st["A"]="OK 2 0011 4 4"; st["B"]="NO_VALID_PARTITION 0 - 4 0"
  st["C"]="INVALID_INPUT 0 - 0 0"; st["D"]="OK 3 001122 8 8"
  q["A"]=0.666667; q["B"]=0.0; q["C"]=0.0; q["D"]=0.666667
  sm["4"]="1 1 3 3"; sm["20"]="17 9 3 11"
  TOL=1e-5
}
/^CASE /{
  n++
  k=$2
  got=$3" "$5" "$7" "$9" "$11
  # 先把 $3=status, $4=cc=, $5=cc, $6=assign=, $7=assign ... 重新按 key=val 解析
  delete f
  for(i=3;i<=NF;i++){ p=index($i,"="); if(p>0){ f[substr($i,1,p-1)]=substr($i,p+1) } }
  g=f["status"]" "f["cc"]" "f["assign"]" "f["cand"]" "f["valid"]
  if(!(k in st)){ printf "C09_FAIL unknown case %s\n",k; bad++; next }
  if(g!=st[k]){ printf "C09_FAIL %s got [%s] want [%s]\n",k,g,st[k]; bad++ }
  d=f["quality"]-q[k]; d=(d<0?-d:d)
  if(d>TOL){ printf "C09_FAIL %s quality got %s want %.6f\n",k,f["quality"],q[k]; bad++ }
}
/^SEEDMOD /{
  n++
  k=$2; got=$3" "$4" "$5" "$6
  if(!(k in sm)){ printf "C09_FAIL unknown SEEDMOD %s\n",k; bad++; next }
  if(got!=sm[k]){ printf "C09_FAIL SEEDMOD %s got [%s] want [%s]\n",k,got,sm[k]; bad++ }
}
/^ONC_DONE$/{done=1}
END{
  if(!done){ printf "C09_FAIL missing ONC_DONE\n"; bad++ }
  if(n!=6){ printf "C09_FAIL lines=%d want=6\n",n; bad++ }
  if(bad==0){ printf "C09_PASS\n" } else { printf "C09_FAIL total=%d\n",bad; exit 1 }
}

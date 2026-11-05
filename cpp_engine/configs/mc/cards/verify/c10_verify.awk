BEGIN{
  n=0; bad=0; done=0
}
/^DERIVE /{
  n++
  k=$2; got=$3" "$4" "$5
  if(k=="real") w="cells=269 datasets=194 calls=617000"
  else if(k=="mutated") w="cells=260 datasets=185 calls=608000"
  else { printf "C10_FAIL unknown derive %s\n",k; bad++; next }
  if(got!=w){ printf "C10_FAIL derive %s got [%s] want [%s]\n",k,got,w; bad++ }
}
/^LADDER /{
  n++
  got=$2" "$3" "$4" "$5" "$6
  w="8000->4000 2000->1000 1000->500 4000->4000 500->500"
  if(got!=w){ printf "C10_FAIL ladder got [%s] want [%s]\n",got,w; bad++ }
}
/^PLAN /{
  n++; if($2!="u1"&&$2!="u2"&&$2!="u3"&&$2!="u4"&&$2!="u5"){printf "C10_FAIL plan key %s\n",$2; bad++}
  pl[$2]=$3
}
/^PLAN2 /{
  n++; if($2!="u1"&&$2!="u2"&&$2!="u3"&&$2!="u4"&&$2!="u5"){printf "C10_FAIL plan2 key %s\n",$2; bad++}
  p2[$2]=$3
}
/^TOTAL /{ n++; t1=$2 }
/^TOTAL2 /{ n++; t2=$2 }
/^BUDGET_DONE$/{done=1}
END{
  if(!done){printf "C10_FAIL missing BUDGET_DONE\n"; bad++}
  exp1["u1"]=4000; exp1["u2"]=2000; exp1["u3"]=2000; exp1["u4"]=500; exp1["u5"]=1000
  exp2["u1"]=8000; exp2["u2"]=2000; exp2["u3"]=2000; exp2["u4"]=1000; exp2["u5"]=1000
  for(k in exp1){ if(!(k in pl)){printf "C10_FAIL plan missing %s\n",k; bad++} else if(pl[k]+0!=exp1[k]){printf "C10_FAIL plan %s got %s want %d\n",k,pl[k],exp1[k]; bad++} }
  for(k in exp2){ if(!(k in p2)){printf "C10_FAIL plan2 missing %s\n",k; bad++} else if(p2[k]+0!=exp2[k]){printf "C10_FAIL plan2 %s got %s want %d\n",k,p2[k],exp2[k]; bad++} }
  if(t1+0!=9500){printf "C10_FAIL total got %s want 9500\n",t1; bad++}
  if(t2+0!=14000){printf "C10_FAIL total2 got %s want 14000\n",t2; bad++}
  if(n!=15){printf "C10_FAIL lines=%d want=15\n",n; bad++}
  if(bad==0){printf "C10_PASS\n"} else {printf "C10_FAIL total=%d\n",bad; exit 1}
}

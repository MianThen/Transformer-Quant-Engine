BEGIN{ n=0; bad=0; done=0 }
/^KEY /{
  n++
  want["A"]="v1|null|S=0.0000|N=266|T=20|rho=0.30|t5|p=50.4|ds=0"
  want["B"]="v1|one_good|S=1.2000|N=266|T=4|rho=0.30|t5|p=50.4|ds=0"
  want["R"]="v1|null|S=0.0000|N=266|T=20|rho=0.30|t5|p=50.4|ds=3"
  got=substr($0, length("KEY ")+3)
  lbl=$2
  if(!(lbl in want)){printf "C22_FAIL unknown KEY %s\n",lbl; bad++; next}
  if(got!=want[lbl]){printf "C22_FAIL KEY %s\n  got  [%s]\n  want [%s]\n",lbl,got,want[lbl]; bad++}
}
/^FNV /{ n++; if($4!=$3){} mid[$2]=$3; next }
/^SEED /{
  n++
  h[$2 "|" $3 "|" $4 "|" $5]=$6
}
/^SEED_TREE_DONE$/{done=1}
$0 !~ /^(KEY|FNV|SEED) / && $0 != "SEED_TREE_DONE" {
  printf "C22_FAIL unexpected line [%s]\n",$0; bad++
}
END{
  if(mid["A"]!="5f3b4addd9acb0cf"){printf "C22_FAIL FNV A %s\n",mid["A"]; bad++}
  if(mid["B"]!="d9a272f6fee531f3"){printf "C22_FAIL FNV B %s\n",mid["B"]; bad++}
  if(mid["R"]!="5f3b4bddd9acb282"){printf "C22_FAIL FNV R %s\n",mid["R"]; bad++}
  if(h["A|0|0|ret"]!="3a775d5bafd53f18"){printf "C22_FAIL SEED A ret %s\n",h["A|0|0|ret"]; bad++}
  if(h["A|0|0|hold"]!="c88d231ba9400851"){printf "C22_FAIL SEED A hold %s\n",h["A|0|0|hold"]; bad++}
  if(h["A|0|0|assign"]!="6aacbb86eec5b5bb"){printf "C22_FAIL SEED A assign %s\n",h["A|0|0|assign"]; bad++}
  if(h["B|123|0|ret"]!="07807e7eeeb76b0c"){printf "C22_FAIL SEED B ret %s\n",h["B|123|0|ret"]; bad++}
  if(h["A|7|1|ret"]!="fc35ba2f810bc1f6"){printf "C22_FAIL SEED A batch1 %s\n",h["A|7|1|ret"]; bad++}
  if(h["R|0|0|ret"]!="a39697b8c6e3a35b"){printf "C22_FAIL SEED R ret %s\n",h["R|0|0|ret"]; bad++}
  if(!done){printf "C22_FAIL missing SEED_TREE_DONE\n"; bad++}
  if(n!=12){printf "C22_FAIL counted=%d want=12\n",n; bad++}
  if(bad==0){printf "C22_PASS\n"} else {printf "C22_FAIL total=%d\n",bad; exit 1}
}

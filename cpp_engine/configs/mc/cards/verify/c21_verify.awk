BEGIN{ n=0; bad=0; done=0; TOL=1e-6 }
function near(a,b,tol){ d=a-b; if(d<0)d=-d; return d<=tol }
/^RECHECK n=/{
  n++
  if(index($0,"n=504")==0){printf "C21_FAIL n\n"; bad++}
  if(index($0,"sha_ok=1")==0){printf "C21_FAIL sha_ok\n"; bad++}
}
/^RECHECK m=/{
  n++
  v=substr($0,index($0,"m=")+2)+0
  if(!near(v,0.503583985525,1e-9)){printf "C21_FAIL m got %s\n",v; bad++}
}
/^RECHECK teff=/{
  n++
  v=substr($0,index($0,"teff=")+5)+0
  if(!near(v,1.593148215827,1e-9)){printf "C21_FAIL teff got %s\n",v; bad++}
}
/^RECHECK thr=/{
  n++
  v=substr($0,index($0,"thr=")+4)+0
  if(!near(v,1.654834515,1e-6)){printf "C21_FAIL thr got %s\n",v; bad++}
}
/^RECHECK maxidx=/{
  n++
  i=substr($0,index($0,"maxidx=")+7)+0
  s=substr($0,index($0,"maxsr=")+6)+0
  if(i!=115){printf "C21_FAIL maxidx got %d\n",i; bad++}
  if(!near(s,22.260256,1e-6)){printf "C21_FAIL maxsr got %s\n",s; bad++}
}
/^RECHECK above=/{ n++; v=substr($0,index($0,"above=")+6)+0; if(v!=53){printf "C21_FAIL above got %d\n",v; bad++} }
/^MATCH /{
  n++
  if(NF!=6){printf "C21_FAIL MATCH fields=%d want 6\n",NF; bad++}
  if($2!=1001||$3!=0||$4!="GATE_Full"){printf "C21_FAIL MATCH key [%s %s %s]\n",$2,$3,$4; bad++}
  if($5+0!=1){printf "C21_FAIL MATCH selected %s want 1\n",$5; bad++}
  if($6+0!=115){printf "C21_FAIL MATCH index %s want 115\n",$6; bad++}
}
/^RECHECK_DONE$/{done=1}
/^RECHECK /{
  k = $2
  p = index(k, "=")
  if (p > 0) k = substr(k, 1, p - 1)
  if (k != "n" && k != "m" && k != "teff" && k != "thr" && k != "maxidx" && k != "above") {
    printf "C21_FAIL unknown RECHECK key [%s]\n", k; bad++
  }
}
$0 !~ /^RECHECK / && $0 !~ /^MATCH / && $0 != "RECHECK_DONE" {
  printf "C21_FAIL unexpected line [%s]\n",$0; bad++
}
END{
  if(!done){printf "C21_FAIL missing RECHECK_DONE\n"; bad++}
  if(n!=7){printf "C21_FAIL counted=%d want=7\n",n; bad++}
  if(bad==0){printf "C21_PASS\n"} else {printf "C21_FAIL total=%d\n",bad; exit 1}
}

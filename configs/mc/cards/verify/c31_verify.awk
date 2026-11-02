BEGIN{ n=0; bad=0; done=0 }
/^H4 /{
  n++
  if(NF!=5){printf "C31_FAIL H4 fields=%d want 5\n",NF; bad++; next}
  lo=substr($2,8)+0; hi=substr($3,10)+0; ns=substr($4,6)+0
  if($5!="reps=2000"){printf "C31_FAIL reps %s\n",$5; bad++}
  if(ns<1||ns>2000){printf "C31_FAIL nsel %d\n",ns; bad++}
  klo=lo; khi=hi; kns=ns
}
/^THR /{
  n++
  tm=substr($2,6)+0; tem=substr($3,11)+0
  if(tm<=0||tm>20){printf "C31_FAIL thr_mean %.4f out of range\n",tm; bad++}
  if(tem<1||tem>100){printf "C31_FAIL teff_mean %.4f out of range\n",tem; bad++}
  ktm=tm; ktem=tem
}
/^GAP /{
  n++
  lg=substr($2,8)+0; hg=substr($3,8)+0
  # 门槛逐 rep 随 m 变化，所以 lo_sel 与 hi_nosel 会重叠；只要求两者都落在 thr 的 ±0.30 内
  if(lg < -0.30){printf "C31_FAIL lo_gap %.4f too negative (>0.30 below thr)\n",lg; bad++}
  if(hg >  0.30){printf "C31_FAIL hi_gap %.4f too positive (>0.30 above thr)\n",hg; bad++}
}
/^H4_DONE$/{done=1}
$0 !~ /^(H4|THR|GAP) / && $0 != "H4_DONE" { printf "C31_FAIL unexpected [%s]\n",$0; bad++ }
END{
  if(!done){printf "C31_FAIL missing H4_DONE\n"; bad++}
  if(n!=3){printf "C31_FAIL counted=%d want=3\n",n; bad++}
  if(bad==0){printf "C31_AWK_PASS\n"} else {printf "C31_FAIL total=%d\n",bad; exit 1}
}

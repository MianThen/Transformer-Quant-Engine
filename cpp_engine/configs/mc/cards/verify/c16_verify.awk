BEGIN{
  n=0; bad=0; done=0
  want_json = "{\"single_core_ms\":176175.600,\"over_threshold\":1,\"buckets\":[{\"T\":3,\"N\":3,\"ch_ms\":0.027800},{\"T\":4,\"N\":126,\"ch_ms\":0.137500},{\"T\":100,\"N\":266,\"ch_ms\":7.015500}],\"shares\":{\"dgp\":0.445533,\"m\":0.464928,\"ocn\":0.056003,\"decide\":0.033549}}"
}
/^COST total /{
  n++; d=$3-176175.600; d=(d<0?-d:d)
  if(d>1e-3){printf "C16_FAIL total got %s want 176175.600\n",$3; bad++}
}
/^COST over /{ n++; if($3+0!=1){printf "C16_FAIL over got %s want 1\n",$3; bad++} }
/^COST unmatched /{ n++; if($3+0!=0){printf "C16_FAIL unmatched got %s want 0\n",$3; bad++} }
/^COST json /{
  n++
  got=substr($0, length("COST json ")+1)
  if(got!=want_json){printf "C16_FAIL json\n  got  [%s]\n  want [%s]\n",got,want_json; bad++}
}
/^COST /{
  if($2!="total" && $2!="over" && $2!="unmatched" && $2!="json"){
    printf "C16_FAIL unexpected COST key [%s]\n",$2; bad++
  }
}
$0 !~ /^COST / && $0 != "COST_DONE" {
  printf "C16_FAIL unexpected line [%s]\n",$0; bad++
}
/^COST_DONE$/{done=1}
END{
  if(!done){printf "C16_FAIL missing COST_DONE\n"; bad++}
  if(n!=4){printf "C16_FAIL lines=%d want=4\n",n; bad++}
  if(bad==0){printf "C16_PASS\n"} else {printf "C16_FAIL total=%d\n",bad; exit 1}
}

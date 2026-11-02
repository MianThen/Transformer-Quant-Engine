BEGIN{ n=0; bad=0; done=0; nc=0 }
/^GRID total=/{ n++
  if($2!="total=269"){printf "C27_FAIL total %s\n",$2;bad++}
  if($3!="limit=8"){printf "C27_FAIL limit %s\n",$3;bad++} }
/^CELL /{ n++; nc++
  if(NF!=4){printf "C27_FAIL CELL fields %d\n",NF;bad++}
  if($2+0!=(nc-1)){printf "C27_FAIL CELL order: got %s want %d\n",$2,nc-1; bad++}
  if($3+0!=20){printf "C27_FAIL CELL reps %s\n",$3;bad++}
  if($4+0!=120){printf "C27_FAIL CELL records %s\n",$4;bad++} }
/^AGG_CELLS /{ n++; if($2+0!=48){printf "C27_FAIL AGG_CELLS %s\n",$2;bad++} }
/^RECORDS /{ n++; if($2+0!=960){printf "C27_FAIL RECORDS %s\n",$2;bad++} }
/^GRID_DONE$/{done=1}
$0 !~ /^(GRID|CELL|AGG_CELLS|RECORDS) / && $0 != "GRID_DONE" { printf "C27_FAIL unexpected [%s]\n",$0; bad++ }
END{
  if(!done){printf "C27_FAIL missing GRID_DONE\n";bad++}
  if(nc!=8){printf "C27_FAIL CELL count=%d want=8\n",nc;bad++}
  if(n!=11){printf "C27_FAIL counted=%d want=11\n",n;bad++}
  if(bad==0){printf "C27_AWK_PASS\n"} else {printf "C27_FAIL total=%d\n",bad; exit 1}
}

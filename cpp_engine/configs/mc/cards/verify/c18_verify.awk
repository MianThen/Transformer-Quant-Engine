BEGIN{
  n=0; bad=0; done=0
  w1="{\"GATE_Full\":12.345,\"B0\":1.234,\"B1\":2.345,\"B2\":3.456,\"B3\":4.567,\"B4\":5.678}"
  w2="{\"GATE_Full\":0.000,\"B0\":9.999,\"B1\":0.000,\"B2\":0.000,\"B3\":0.000,\"B4\":0.000}"
}
/^WALL_JSON /{
  n++
  got=substr($0, length("WALL_JSON ")+1)
  if(got!=w1){printf "C18_FAIL full\n  got  [%s]\n  want [%s]\n",got,w1; bad++}
}
/^WALL_JSON_PARTIAL /{
  n++
  got=substr($0, length("WALL_JSON_PARTIAL ")+1)
  if(got!=w2){printf "C18_FAIL partial\n  got  [%s]\n  want [%s]\n",got,w2; bad++}
}
/^WALL_DONE$/{done=1}
$0 !~ /^WALL_JSON / && $0 !~ /^WALL_JSON_PARTIAL / && $0 != "WALL_DONE" {
  printf "C18_FAIL unexpected line [%s]\n",$0; bad++
}
END{
  if(!done){printf "C18_FAIL missing WALL_DONE\n"; bad++}
  if(n!=2){printf "C18_FAIL lines=%d want=2\n",n; bad++}
  if(bad==0){printf "C18_PASS\n"} else {printf "C18_FAIL total=%d\n",bad; exit 1}
}

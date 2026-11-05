BEGIN{
  rn=0; fn=0; jn=0; en=0; bad=0; done=0
  want_gz=460; want_ps=437
  json[1]="{\"cell_id\":101,\"rep_id\":0,\"rule_id\":\"GATE_Full\",\"selected\":false,\"index\":1,\"abstain\":false,\"returns_sha256\":\"92b4f2a290c24fe4\",\"reason\":\"below_threshold\"}"
  json[2]="{\"cell_id\":101,\"rep_id\":0,\"rule_id\":\"B0\",\"selected\":true,\"index\":1,\"abstain\":false,\"returns_sha256\":\"92b4f2a290c24fe4\",\"reason\":\"baseline\"}"
  json[3]="{\"cell_id\":205,\"rep_id\":2,\"rule_id\":\"B4\",\"selected\":true,\"index\":4,\"abstain\":true,\"returns_sha256\":\"0123456789abcdef\",\"reason\":\"ocn_abstain\"}"
  want_esc="q\\\"w\\\\e"
}
/^CRC /{ rn++; if($2!="0xCBF43926"){printf "C08_FAIL crc got %s\n",$2; bad++} }
/^GZIP_HDR /{
  fn++
  g=$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "$10" "$11
  if(g!="1f 8b 08 00 00 00 00 00 00 ff"){printf "C08_FAIL gzip_hdr got [%s]\n",g; bad++}
}
/^GZIP_SIZE /{ if($2+0!=want_gz){printf "C08_FAIL gzip_size got %s want %d\n",$2,want_gz; bad++} }
/^PAYLOAD_SIZE /{ if($2+0!=want_ps){printf "C08_FAIL payload_size got %s want %d\n",$2,want_ps; bad++} }
/^JSONL /{
  jn++
  p=index($0," "); got=substr($0,p+1)
  if(jn>3){printf "C08_FAIL extra_jsonl\n"; bad++; next}
  if(got!=json[jn]){printf "C08_FAIL jsonl %d\n  got  [%s]\n  want [%s]\n",jn,got,json[jn]; bad++}
}
/^ESC /{
  en++
  p=index($0," "); got=substr($0,p+1)
  if(got!=want_esc){printf "C08_FAIL esc got [%s] want [%s]\n",got,want_esc; bad++}
}
/^OUTPUT_DONE$/{done=1}
END{
  if(!done){printf "C08_FAIL missing OUTPUT_DONE\n"; bad++}
  if(rn!=1){printf "C08_FAIL crc_lines=%d\n",rn; bad++}
  if(fn!=1){printf "C08_FAIL hdr_lines=%d\n",fn; bad++}
  if(jn!=3){printf "C08_FAIL jsonl_lines=%d\n",jn; bad++}
  if(en!=1){printf "C08_FAIL esc_lines=%d\n",en; bad++}
  if(bad==0){printf "C08_AWK_PASS\n"} else {printf "C08_FAIL total=%d\n",bad; exit 1}
}

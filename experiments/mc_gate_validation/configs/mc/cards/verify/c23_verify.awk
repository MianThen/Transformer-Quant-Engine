BEGIN{ n=0; bad=0; done=0 }
/^SEED /{ n++; if($2!="61bcf27ecc342c9f"){printf "C23_FAIL seed %s\n",$2; bad++} }
/^HEAD8 /{
  n++
  if(NF!=9){printf "C23_FAIL head fields %d want 9\n",NF; bad++}
  want[1]="bf824c383478604c"; want[2]="3f61ee38928e900f"; want[3]="3f8c5b6e1084cfec"
  want[4]="3f935539024e4a85"; want[5]="3f86f3a57f49da23"; want[6]="bf543b9e4c9e22ba"
  want[7]="bf912d44f6befa15"; want[8]="3f1066642e65416b"
  for(i=1;i<=8;i++) if($(i+1)!=want[i]){printf "C23_FAIL head %d got %s want %s\n",i,$(i+1),want[i]; bad++}
}
/^SHA256 /{ n++; if($2!="9b356e4cc62a98252e387fc92d72544e42d306248fe18ff1ca542de4d9c1a8c3"){printf "C23_FAIL sha %s\n",$2; bad++} }
/^PANEL_DONE$/{done=1}
$0 !~ /^(SEED|HEAD8|SHA256) / && $0 != "PANEL_DONE" { printf "C23_FAIL unexpected [%s]\n",$0; bad++ }
END{
  if(!done){printf "C23_FAIL missing PANEL_DONE\n"; bad++}
  if(n!=3){printf "C23_FAIL counted=%d want=3\n",n; bad++}
  if(bad==0){printf "C23_PASS\n"} else {printf "C23_FAIL total=%d\n",bad; exit 1}
}

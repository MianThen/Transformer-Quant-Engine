BEGIN{
  n=0; bad=0; done=0
  h[0]="65a3b7032623038ab05e5761941456ae1ec0288a9364175c48db0ab49dc945d4"; h[1]="fc62429c3e69001d65972cdeb94fb9aa18a7d9c16bc449e1e474e7e41bb95a7d"; h[2]="9cb2b0781202d34b99680abef0d0c8797b5ceb0a7a87bcf1c5c7bd0b0d68bbe0"
  sz[0]=96; sz[1]=48; sz[2]=72
}
/^REPLAY files /{ n++; if($3+0!=3){printf "C15_FAIL files=%s\n",$3; bad++} }
/^REPLAY [0-9]+ sha /{
  n++; k=$2+0
  if(!(k in h)){printf "FAIL unknown idx %d\n",k; bad++; next}
  if($4!=h[k]){printf "FAIL sha %d\n  got  %s\n  want %s\n",k,$4,h[k]; bad++}
}
/^REPLAY [0-9]+ bytes /{
  n++; k=$2+0
  if(!(k in sz)){bad++; next}
  if($4+0!=sz[k]){printf "FAIL bytes %d got %s want %d\n",k,$4,sz[k]; bad++}
}
/^REPLAY extra /{ n++; if($3+0!=0){printf "FAIL extra files=%s\n",$3; bad++} }
/^VERIFY [0-9]+ /{
  n++; k=$2+0
  if(k!=0 && k!=1 && k!=2){printf "FAIL unknown verify idx %d\n",k; bad++}
  if($3+0!=1){printf "FAIL verify %d got %s want 1\n",k,$3; bad++}
}
/^VERIFY corrupt /{ n++; if($3+0!=0){printf "FAIL corrupt verify got %s want 0\n",$3; bad++} }
/^REPLAY_DONE$/{done=1}
END{
  if(!done){printf "FAIL missing REPLAY_DONE\n"; bad++}
  if(n!=12){printf "FAIL lines=%d want=12\n",n; bad++}
  if(bad==0){printf "C15_PASS\n"} else {printf "C15_FAIL total=%d\n",bad; exit 1}
}

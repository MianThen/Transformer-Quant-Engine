BEGIN{ n=0; bad=0; done=0
  v["empty"]="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"; v["abc"]="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"; v["448"]="248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
  v["million_a"]="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"; v["bytes256"]="40aff2e9d2d8922e47afd4648e6967497158785fbd1da870e7110266bf944880"; v["bytes256x4"]="785b0751fc2c53dc14a4ce3d800e69ef9ce1009eb327ccf458afe09c242c26c9"
}
/^SHA /{
  n++; k=$2
  if(!(k in v)){printf "C14_FAIL unknown case %s\n",k; bad++; next}
  if($3!=v[k]){printf "C14_FAIL %s\n  got  %s\n  want %s\n",k,$3,v[k]; bad++}
}
/^SHA256_DONE$/{done=1}
END{
  if(!done){printf "C14_FAIL missing SHA256_DONE\n"; bad++}
  if(n!=6){printf "C14_FAIL lines=%d want=6\n",n; bad++}
  if(bad==0){printf "C14_PASS\n"} else {printf "C14_FAIL total=%d\n",bad; exit 1}
}

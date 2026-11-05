BEGIN{
  nc = split("101 205 307 412", cl, " ")
  split("GATE_Full B0 B1 B2 B3 B4", rl, " ")
  nr = 6; reps = 3
  for (i = 1; i <= nc; ++i) cellN[cl[i]] = 6
  k = 0
  for (i = 1; i <= nc; ++i)
    for (rep = 0; rep < reps; ++rep)
      for (j = 1; j <= nr; ++j)
        seq[++k] = cl[i] " " rep " " rl[j]
  total = k
  n = 0; bad = 0
}
/^REC /{
  n++
  if (NF != 9) { printf "C06R_FAIL fields=%d line=%d\n", NF, n; bad++; next }
  key = $2 " " $3 " " $4
  if (n > total) { printf "C06R_FAIL extra line [%s]\n", key; bad++; next }
  if (key != seq[n]) { printf "C06R_FAIL order line=%d got=[%s] want=[%s]\n", n, key, seq[n]; bad++ }
  if ($5 != "0" && $5 != "1") { printf "C06R_FAIL sel=%s\n", $5; bad++ }
  if ($7 != "0" && $7 != "1") { printf "C06R_FAIL abst=%s\n", $7; bad++ }
  if ($8 !~ /^[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]$/) { printf "C06R_FAIL fp=%s\n", $8; bad++ }
  if ($9 == "") { printf "C06R_FAIL empty_reason\n"; bad++ }
  if ($4 ~ /^GATE_/) {
    if ($9 !~ /^(selected|below_threshold|fdr_rejected|raw_t_rejected|ocn_abstain)$/) { printf "C06R_FAIL gate_reason=%s\n", $9; bad++ }
  } else {
    if ($9 != "baseline") { printf "C06R_FAIL base_reason=%s\n", $9; bad++ }
  }
  i2 = $6 + 0
  if (i2 < 0 || i2 >= cellN[$2]) { printf "C06R_FAIL index=%s cell=%s\n", $6, $2; bad++ }
  tk = $2 " " $3
  if (tk in fp1) { if (fp1[tk] != $8) { printf "C06R_FAIL fp_mismatch [%s]\n", tk; bad++ } }
  else { fp1[tk] = $8 }
}
END{
  if (n != total) { printf "C06R_FAIL count=%d want=%d\n", n, total; bad++ }
  for (i = 1; i <= nc; ++i) {
    split("", seen); dup = 0; cnt = 0
    for (rep = 0; rep < reps; ++rep) {
      tk = cl[i] " " rep
      if (tk in fp1) { cnt++; f = fp1[tk]; if (f in seen) dup = 1; seen[f] = 1 }
    }
    if (cnt != reps) { printf "C06R_FAIL cell=%s missing_reps\n", cl[i]; bad++ }
    if (dup) { printf "C06R_FAIL cell=%s dup_fp_across_reps\n", cl[i]; bad++ }
  }
  for (rep = 0; rep < reps; ++rep) {
    split("", seen2); dup2 = 0; cnt2 = 0
    for (i = 1; i <= nc; ++i) {
      tk = cl[i] " " rep
      if (tk in fp1) { cnt2++; f = fp1[tk]; if (f in seen2) dup2 = 1; seen2[f] = 1 }
    }
    if (cnt2 != nc) { printf "C06R_FAIL rep=%d missing_cells\n", rep; bad++ }
    if (dup2) { printf "C06R_FAIL rep=%d dup_fp_across_cells\n", rep; bad++ }
  }
  if (bad == 0) { print "C06R_PASS" } else { printf "C06R_FAIL total=%d\n", bad; exit 1 }
}

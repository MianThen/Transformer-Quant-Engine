function val(field, prefix,   v) {
  if (index(field, prefix) != 1) return -999999
  return substr(field, length(prefix) + 1) + 0
}
function chk(label, want_a, want_e, want_d, want_f,   a, e, d, f) {
  if (NF != 6) { printf "C19_FAIL %s fields=%d want 6\n", label, NF; bad++; return }
  if (index($3, "analytic=") != 1) { printf "C19_FAIL %s field3 must be analytic=\n", label; bad++; return }
  if (index($4, "empirical=") != 1) { printf "C19_FAIL %s field4 must be empirical=\n", label; bad++; return }
  if (index($5, "drift=") != 1) { printf "C19_FAIL %s field5 must be drift=\n", label; bad++; return }
  if (index($6, "flagged=") != 1) { printf "C19_FAIL %s field6 must be flagged=\n", label; bad++; return }
  a = val($3, "analytic="); e = val($4, "empirical="); d = val($5, "drift="); f = val($6, "flagged=")
  if ((a - want_a) > TOL || (want_a - a) > TOL) { printf "C19_FAIL %s analytic got %s want %.6f\n", label, a, want_a; bad++ }
  if ((e - want_e) > TOL || (want_e - e) > TOL) { printf "C19_FAIL %s empirical got %s want %.6f\n", label, e, want_e; bad++ }
  if ((d - want_d) > TOL || (want_d - d) > TOL) { printf "C19_FAIL %s drift got %s want %.6f\n", label, d, want_d; bad++ }
  if (f != want_f) { printf "C19_FAIL %s flagged got %s want %d\n", label, f, want_f; bad++ }
}
BEGIN { n = 0; bad = 0; done = 0; TOL = 1e-5 }
/^DUAL A /{ n++; chk("A", 1.657640, 18.0, 16.342360, 1) }
/^DUAL B /{ n++; chk("B", 1.657640, 1.7, 0.042360, 0) }
/^THRESHOLD_DONE$/{ done = 1 }
$0 !~ /^DUAL [AB] / && $0 != "THRESHOLD_DONE" { printf "C19_FAIL unexpected line [%s]\n", $0; bad++ }
END {
  if (!done) { printf "C19_FAIL missing THRESHOLD_DONE\n"; bad++ }
  if (n != 2) { printf "C19_FAIL lines=%d want=2\n", n; bad++ }
  if (bad == 0) { printf "C19_PASS\n" } else { printf "C19_FAIL total=%d\n", bad; exit 1 }
}

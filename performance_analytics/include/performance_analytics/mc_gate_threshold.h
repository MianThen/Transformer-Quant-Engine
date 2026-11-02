// mc_gate_threshold.h
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>

#include "mc_gate_decide.h"

namespace mc_gate {

struct ThresholdPair {
  double analytic;
  double empirical;
  double drift;
  bool flagged;
};

inline ThresholdPair dual_track(int T, int N, double m,
                                double q_alpha,
                                std::vector<double> max_sr,
                                double q_quantile,
                                double tol) {
  ThresholdPair out;
  const double teff = gate_teff(static_cast<double>(T), m);
  out.analytic = gate_threshold_sr(teff, static_cast<double>(T),
                                   static_cast<double>(N), q_alpha);

  const std::size_t n = max_sr.size();
  if (n == 0) {
    out.empirical = 0.0;
  } else {
    std::sort(max_sr.begin(), max_sr.end());
    const double idx_d = q_quantile * static_cast<double>(n - 1);
    const std::size_t idx = static_cast<std::size_t>(std::floor(idx_d));
    out.empirical = max_sr[idx];
  }

  out.drift = out.empirical - out.analytic;
  out.flagged = std::fabs(out.drift) > tol;
  return out;
}

} // namespace mc_gate

#ifdef MC_GATE_THRESHOLD_SELFTEST
int main() {
  using namespace mc_gate;

  {
    const int T = 4;
    const int N = 126;
    const double m = 0.5;
    const double q_alpha = 0.05;
    std::vector<double> max_sr;
    max_sr.reserve(20);
    for (int i = 0; i < 20; ++i) {
      max_sr.push_back(static_cast<double>(i));
    }
    const double q_quantile = 0.95;
    const double tol = 0.05;

    const ThresholdPair a = dual_track(T, N, m, q_alpha, max_sr, q_quantile, tol);
    std::printf("DUAL A analytic=%.6f empirical=%.6f drift=%.6f flagged=%d\n",
                a.analytic, a.empirical, a.drift, a.flagged ? 1 : 0);
  }

  {
    const int T = 4;
    const int N = 126;
    const double m = 0.5;
    const double q_alpha = 0.05;
    std::vector<double> max_sr = {1.6, 1.65, 1.7, 1.75};
    const double q_quantile = 0.95;
    const double tol = 0.05;

    const ThresholdPair b = dual_track(T, N, m, q_alpha, max_sr, q_quantile, tol);
    std::printf("DUAL B analytic=%.6f empirical=%.6f drift=%.6f flagged=%d\n",
                b.analytic, b.empirical, b.drift, b.flagged ? 1 : 0);
  }

  std::printf("THRESHOLD_DONE\n");
  return 0;
}
#endif

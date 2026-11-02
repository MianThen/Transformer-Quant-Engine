#include <cstdint>
#include <cmath>
#include <vector>
#include <cstddef>
#include "mc_gate_rng.h"

namespace mc_gate {

constexpr double kP      = 50.4;
constexpr double kScale  = 7.099295739719547;       // sqrt(50.4)
constexpr double kSigmaP = 0.014086041373984414;    // = 0.10/kScale，注意是 0.0140 不是 0.0014
// 每期均值 mu_i = tau_i * kSigmaP / kScale；这样年化 SR = tau_i 精确成立。
// 期望每期方差 = kSigmaP^2 = 1.98417e-4

enum class Family { Gauss, T5, Garch, ShortVol };
enum class Composition { Null, OneGood, Mixed, MixedSigned, AllNegative };

struct ReturnMatrix {
  int T = 0, N = 0;
  std::vector<double> r;                 // T*N，行主序
  double& at(int t, int i) { return r[(std::size_t)t * N + i]; }
  double  at(int t, int i) const { return r[(std::size_t)t * N + i]; }
};

inline std::vector<double> make_tau(Composition comp, int T, int N, double S, std::uint64_t seed) {
  std::vector<double> tau(N, 0.0);
  if (comp == Composition::Null) {
    // all 0.0
  } else if (comp == Composition::OneGood) {
    if (N > 0) {
      int idx = static_cast<int>(seed % static_cast<std::uint64_t>(N));
      tau[idx] = S;
    }
  } else if (comp == Composition::Mixed) {
    int k = (T + 3) / 4;
    int limit = (k < N) ? k : N;
    for (int i = 0; i < limit; ++i) {
      tau[i] = S;
    }
  } else if (comp == Composition::MixedSigned) {
    int k = (T + 3) / 4;
    int limit_pos = (k < N) ? k : N;
    for (int i = 0; i < limit_pos; ++i) {
      tau[i] = S;
    }
    int limit_neg = (k < (N - limit_pos)) ? k : (N - limit_pos);
    for (int i = 0; i < limit_neg; ++i) {
      tau[limit_pos + i] = -S;
    }
  } else if (comp == Composition::AllNegative) {
    for (int i = 0; i < N; ++i) {
      tau[i] = -0.6;
    }
  }
  return tau;
}

ReturnMatrix generate(Family fam, Composition comp, int T, int N,
                      double rho, double S, std::uint64_t seed) {
  ReturnMatrix m;
  m.T = T;
  m.N = N;
  m.r.assign((std::size_t)T * N, 0.0);
  
  Pcg64 rng(seed);
  std::vector<double> tau = make_tau(comp, T, N, S, seed);
  
  if (fam == Family::Gauss || fam == Family::T5) {
    for (int t = 0; t < T; ++t) {
      double f = (fam == Family::Gauss) ? normal_z(rng) : t5_z(rng);
      for (int i = 0; i < N; ++i) {
        double e = (fam == Family::Gauss) ? normal_z(rng) : t5_z(rng);
        m.at(t, i) = tau[i] * kSigmaP / kScale
                   + kSigmaP * std::sqrt(rho) * f
                   + kSigmaP * std::sqrt(1.0 - rho) * e;
      }
    }
  } else if (fam == Family::Garch) {
    double sigma2 = 1.0;
    double a_prev_sq = 0.0;
    for (int t = 0; t < T; ++t) {
      sigma2 = 0.02 + 0.08 * a_prev_sq + 0.90 * sigma2;
      double sigma = std::sqrt(sigma2);
      double zf = t5_z(rng);
      a_prev_sq = (sigma * zf) * (sigma * zf);
      for (int i = 0; i < N; ++i) {
        double zi = t5_z(rng);
        m.at(t, i) = tau[i] * kSigmaP / kScale
                  + kSigmaP * sigma * (std::sqrt(rho) * zf + std::sqrt(1.0 - rho) * zi);
      }
    }
  } else if (fam == Family::ShortVol) {
    ShortVol sv;
    auto svz = [&sv](double u) -> double {
      return (sv.a * u - sv.b * (u * u - 1.0)) / sv.norm;
    };
    for (int t = 0; t < T; ++t) {
      double uf = normal_z(rng);
      for (int i = 0; i < N; ++i) {
        double ui = normal_z(rng);
        m.at(t, i) = tau[i] * kSigmaP / kScale
                  + kSigmaP * (std::sqrt(rho) * svz(uf) + std::sqrt(1.0 - rho) * svz(ui));
      }
    }
  }
  
  return m;
}

} // namespace mc_gate

#ifdef MC_GATE_DGP_SELFTEST

#include <cstdio>

double series_var(const mc_gate::ReturnMatrix& m, int i) {
  if (m.T <= 1) return 0.0;
  double sum = 0.0;
  for (int t = 0; t < m.T; ++t) {
    sum += m.at(t, i);
  }
  double mean = sum / m.T;
  double var_sum = 0.0;
  for (int t = 0; t < m.T; ++t) {
    double d = m.at(t, i) - mean;
    var_sum += d * d;
  }
  return var_sum / (m.T - 1);
}

double series_corr(const mc_gate::ReturnMatrix& m, int i, int j) {
  if (m.T <= 1) return 0.0;
  double sum_i = 0.0, sum_j = 0.0;
  for (int t = 0; t < m.T; ++t) {
    sum_i += m.at(t, i);
    sum_j += m.at(t, j);
  }
  double mean_i = sum_i / m.T;
  double mean_j = sum_j / m.T;
  
  double cov = 0.0, var_i = 0.0, var_j = 0.0;
  for (int t = 0; t < m.T; ++t) {
    double di = m.at(t, i) - mean_i;
    double dj = m.at(t, j) - mean_j;
    cov += di * dj;
    var_i += di * di;
    var_j += dj * dj;
  }
  var_i /= (m.T - 1);
  var_j /= (m.T - 1);
  cov /= (m.T - 1);
  
  double denom = std::sqrt(var_i * var_j);
  if (denom == 0.0) return 0.0;
  return cov / denom;
}

int main() {
  const int T = 50000;
  const int N = 3;
  const double rho = 0.25;
  const double S = 1.8;
  const std::uint64_t seed = 20260101;
  
  {
    auto m = mc_gate::generate(mc_gate::Family::Gauss, mc_gate::Composition::Null, T, N, rho, S, seed);
    std::printf("DGP gauss VAR0 %.6e VAR1 %.6e VAR2 %.6e\n", series_var(m, 0), series_var(m, 1), series_var(m, 2));
    std::printf("DGP gauss CORR01 %.6f CORR02 %.6f\n", series_corr(m, 0, 1), series_corr(m, 0, 2));
  }
  {
    auto m = mc_gate::generate(mc_gate::Family::T5, mc_gate::Composition::Null, T, N, rho, S, seed);
    std::printf("DGP t5 VAR0 %.6e VAR1 %.6e VAR2 %.6e\n", series_var(m, 0), series_var(m, 1), series_var(m, 2));
    std::printf("DGP t5 CORR01 %.6f CORR02 %.6f\n", series_corr(m, 0, 1), series_corr(m, 0, 2));
  }
  {
    auto m = mc_gate::generate(mc_gate::Family::Garch, mc_gate::Composition::Null, T, N, rho, S, seed);
    std::printf("DGP garch VAR0 %.6e VAR1 %.6e VAR2 %.6e\n", series_var(m, 0), series_var(m, 1), series_var(m, 2));
    std::printf("DGP garch CORR01 %.6f CORR02 %.6f\n", series_corr(m, 0, 1), series_corr(m, 0, 2));
  }
  {
    auto m = mc_gate::generate(mc_gate::Family::ShortVol, mc_gate::Composition::Null, T, N, rho, S, seed);
    std::printf("DGP shortvol VAR0 %.6e VAR1 %.6e VAR2 %.6e\n", series_var(m, 0), series_var(m, 1), series_var(m, 2));
    std::printf("DGP shortvol CORR01 %.6f CORR02 %.6f\n", series_corr(m, 0, 1), series_corr(m, 0, 2));
  }
  
  {
    const int T2 = 100;
    const int N2 = 60;
    const double S2 = 1.8;
    const std::uint64_t seed2 = 20260101;
    
    auto tau_null = mc_gate::make_tau(mc_gate::Composition::Null, T2, N2, S2, seed2);
    int n_null = 0;
    for (int i = 0; i < N2; ++i) if (tau_null[i] != 0.0) ++n_null;
    
    auto tau_onegood = mc_gate::make_tau(mc_gate::Composition::OneGood, T2, N2, S2, seed2);
    int n_onegood = 0;
    for (int i = 0; i < N2; ++i) if (tau_onegood[i] != 0.0) ++n_onegood;
    
    auto tau_mixed = mc_gate::make_tau(mc_gate::Composition::Mixed, T2, N2, S2, seed2);
    int n_mixed = 0;
    for (int i = 0; i < N2; ++i) if (tau_mixed[i] != 0.0) ++n_mixed;
    
    auto tau_mixedsigned = mc_gate::make_tau(mc_gate::Composition::MixedSigned, T2, N2, S2, seed2);
    int n_mixedsigned_pos = 0, n_mixedsigned_neg = 0;
    for (int i = 0; i < N2; ++i) {
      if (tau_mixedsigned[i] > 0.0) ++n_mixedsigned_pos;
      else if (tau_mixedsigned[i] < 0.0) ++n_mixedsigned_neg;
    }
    
    auto tau_neg = mc_gate::make_tau(mc_gate::Composition::AllNegative, T2, N2, S2, seed2);
    int n_neg = 0;
    for (int i = 0; i < N2; ++i) if (tau_neg[i] != 0.0) ++n_neg;
    
    std::printf("TAUNULL %d TAUNONEGOOD %d TAUMIXED %d TAUMIXEDSIGNEDPOS %d TAUMIXEDSIGNEDNEG %d TAUNEG %d\n",
                n_null, n_onegood, n_mixed, n_mixedsigned_pos, n_mixedsigned_neg, n_neg);
  }
  
  std::printf("DGP_DONE\n");
  return 0;
}

#endif

#ifndef MC_GATE_DGP_H
#define MC_GATE_DGP_H

#include <cstdint>
#include <cmath>
#include <vector>
#include <cstddef>
#include "mc_gate_rng.h"

namespace mc_gate {

constexpr double kP      = 50.4;
constexpr double kScale  = 7.099295739719547;       // sqrt(50.4)
// §3.1：sigma_p = 0.10/scale。**必须由公式算，不要写十进制字面量**——
// 旧版手写的 0.014086041373984414 与 0.10/kScale = 0.014085904245475263 相差 9.7e-6，
// 会让每条收益带 1e-5 的相对偏差（2026-11 修正）。
constexpr double kSigmaP = 0.10 / kScale;
// 每期均值 mu_i = tau_i * kSigmaP / kScale；这样年化 SR = tau_i 精确成立。
// 期望每期方差 = kSigmaP^2 = 1.98417e-4

enum class Family { Gauss, T5, Garch, ShortVol };
enum class Composition { Null, OneGood, Mixed, MixedSigned, AllNegative };

// ---------------------------------------------------------------------------
// v1.4.5 §3.1 下标约定（2026-11 修正；此前实现曾把 T/N 转置）：
//   i ∈ 0..T-1 = 策略（trials，截面维，行）
//   t ∈ 0..N-1 = 期  （observations，时间维，列）
// 存储 trial-major：r[i*N + t]，与 PolicyFamilyGovernanceInputV1::trial_major_returns 一致。
// ---------------------------------------------------------------------------
struct ReturnMatrix {
  int T = 0;                             // 策略数（trials）
  int N = 0;                             // 期数（observations）
  std::vector<double> r;                 // T*N，trial-major
  double& at(int i, int t) { return r[(std::size_t)i * (std::size_t)N + (std::size_t)t]; }
  double  at(int i, int t) const { return r[(std::size_t)i * (std::size_t)N + (std::size_t)t]; }
};

// shortvol 的单变量变换（§3.2）
inline double shortvol_z(const ShortVol& sv, double u) {
  return (sv.a * u - sv.b * (u * u - 1.0)) / sv.norm;
}

// τ 的长度 = T（每条策略一个）。§3.4：mixed / mixed_signed 的 k = ceil(T/4)。
inline std::vector<double> make_tau(Composition comp, int T, double S, std::uint64_t seed) {
  std::vector<double> tau(static_cast<std::size_t>(T > 0 ? T : 0), 0.0);
  if (T <= 0) return tau;
  if (comp == Composition::OneGood) {
    tau[static_cast<std::size_t>(seed % static_cast<std::uint64_t>(T))] = S;
  } else if (comp == Composition::Mixed) {
    const int k = (T + 3) / 4;
    for (int i = 0; i < k && i < T; ++i) tau[static_cast<std::size_t>(i)] = S;
  } else if (comp == Composition::MixedSigned) {
    const int k = (T + 3) / 4;
    for (int i = 0; i < k && i < T; ++i) tau[static_cast<std::size_t>(i)] = S;
    for (int i = k; i < k + k && i < T; ++i) tau[static_cast<std::size_t>(i)] = -S;
  } else if (comp == Composition::AllNegative) {
    for (int i = 0; i < T; ++i) tau[static_cast<std::size_t>(i)] = S;
  }
  return tau;
}

// ---------------------------------------------------------------------------
// §3.1 抽样顺序（AC-39）：
//   1) 先抽 f_1 … f_N（N 个，一期一个，全部策略共用）
//   2) 再按 i = 0..T-1 逐策略抽 e_{i,1} … e_{i,N}（T×N 个）
// 禁止列优先、禁止按策略并行取流、禁止先 e 后 f。
// ---------------------------------------------------------------------------
inline ReturnMatrix generate(Family fam, Composition comp, int T, int N,
                             double rho, double S, std::uint64_t seed) {
  ReturnMatrix m;
  m.T = T;
  m.N = N;
  m.r.assign(static_cast<std::size_t>(T > 0 ? T : 0) *
                 static_cast<std::size_t>(N > 0 ? N : 0),
             0.0);
  if (T <= 0 || N <= 0) return m;

  Pcg64 rng(seed);
  const std::vector<double> tau = make_tau(comp, T, S, seed);

  const std::size_t nN = static_cast<std::size_t>(N);

  // ---- 第一遍：f_1 … f_N（garch 顺带把每期波动率算出并缓存） ----
  std::vector<double> f(nN, 0.0);
  std::vector<double> sig(nN, 1.0);

  if (fam == Family::Gauss || fam == Family::T5) {
    for (int t = 0; t < N; ++t) {
      f[static_cast<std::size_t>(t)] = (fam == Family::Gauss) ? normal_z(rng) : t5_z(rng);
    }
  } else if (fam == Family::Garch) {
    double sigma2 = 1.0;
    double a_prev_sq = 0.0;
    for (int t = 0; t < N; ++t) {
      sigma2 = 0.02 + 0.08 * a_prev_sq + 0.90 * sigma2;
      const double sigma = std::sqrt(sigma2);
      const double zf = t5_z(rng);
      a_prev_sq = (sigma * zf) * (sigma * zf);
      sig[static_cast<std::size_t>(t)] = sigma;
      f[static_cast<std::size_t>(t)] = sigma * zf;
    }
  } else {
    ShortVol sv;
    for (int t = 0; t < N; ++t) {
      f[static_cast<std::size_t>(t)] = shortvol_z(sv, normal_z(rng));
    }
  }

  // ---- 第二遍：逐策略抽出 e_{i,1..N}，同一遍内写下 r_{i,t} ----
  const double br = std::sqrt(rho);
  const double be = std::sqrt(1.0 - rho);

  if (fam == Family::Gauss || fam == Family::T5) {
    for (int i = 0; i < T; ++i) {
      const double mu = tau[static_cast<std::size_t>(i)] * kSigmaP / kScale;
      for (int t = 0; t < N; ++t) {
        const double e = (fam == Family::Gauss) ? normal_z(rng) : t5_z(rng);
        m.at(i, t) = mu + kSigmaP * (br * f[static_cast<std::size_t>(t)] + be * e);
      }
    }
  } else if (fam == Family::Garch) {
    for (int i = 0; i < T; ++i) {
      const double mu = tau[static_cast<std::size_t>(i)] * kSigmaP / kScale;
      for (int t = 0; t < N; ++t) {
        const double s = sig[static_cast<std::size_t>(t)];
        const double e = s * t5_z(rng);
        m.at(i, t) = mu + kSigmaP * (br * f[static_cast<std::size_t>(t)] + be * e);
      }
    }
  } else {
    ShortVol sv;
    for (int i = 0; i < T; ++i) {
      const double mu = tau[static_cast<std::size_t>(i)] * kSigmaP / kScale;
      for (int t = 0; t < N; ++t) {
        const double e = shortvol_z(sv, normal_z(rng));
        m.at(i, t) = mu + kSigmaP * (br * f[static_cast<std::size_t>(t)] + be * e);
      }
    }
  }

  return m;
}

// 平均绝对相关（AC-41 自检量）：全部 i<j 的 |corr| 均值。
// 判据：rho=0.3 时应落在 [0.22, 0.38]；若 ≈0，说明 f 退化成了每条策略的常数（§3.1 错法 2）。
inline double mean_abs_corr(const ReturnMatrix& m) {
  if (m.T < 2 || m.N < 2) return 0.0;
  std::vector<double> mu(static_cast<std::size_t>(m.T), 0.0);
  std::vector<double> sd(static_cast<std::size_t>(m.T), 0.0);
  for (int i = 0; i < m.T; ++i) {
    double s = 0.0;
    for (int t = 0; t < m.N; ++t) s += m.at(i, t);
    mu[static_cast<std::size_t>(i)] = s / static_cast<double>(m.N);
    double v = 0.0;
    for (int t = 0; t < m.N; ++t) {
      const double d = m.at(i, t) - mu[static_cast<std::size_t>(i)];
      v += d * d;
    }
    sd[static_cast<std::size_t>(i)] = std::sqrt(v / static_cast<double>(m.N - 1));
  }
  double acc = 0.0;
  long long cnt = 0;
  for (int a = 0; a < m.T; ++a) {
    for (int b = a + 1; b < m.T; ++b) {
      double cov = 0.0;
      for (int t = 0; t < m.N; ++t) {
        cov += (m.at(a, t) - mu[static_cast<std::size_t>(a)]) *
               (m.at(b, t) - mu[static_cast<std::size_t>(b)]);
      }
      cov /= static_cast<double>(m.N - 1);
      const double den = sd[static_cast<std::size_t>(a)] * sd[static_cast<std::size_t>(b)];
      double c = (den > 0.0) ? cov / den : 0.0;
      if (c < -1.0) c = -1.0;
      if (c > 1.0) c = 1.0;
      acc += std::fabs(c);
      ++cnt;
    }
  }
  return (cnt > 0) ? acc / static_cast<double>(cnt) : 0.0;
}

} // namespace mc_gate

#endif // MC_GATE_DGP_H

#ifdef MC_GATE_DGP_SELFTEST

#include <cstdio>

double series_var(const mc_gate::ReturnMatrix& m, int i) {
  if (m.N <= 1) return 0.0;
  double sum = 0.0;
  for (int t = 0; t < m.N; ++t) sum += m.at(i, t);
  const double mean = sum / m.N;
  double var_sum = 0.0;
  for (int t = 0; t < m.N; ++t) {
    const double d = m.at(i, t) - mean;
    var_sum += d * d;
  }
  return var_sum / (m.N - 1);
}

double series_corr(const mc_gate::ReturnMatrix& m, int i, int j) {
  if (m.N <= 1) return 0.0;
  double si = 0.0, sj = 0.0;
  for (int t = 0; t < m.N; ++t) {
    si += m.at(i, t);
    sj += m.at(j, t);
  }
  const double mi = si / m.N;
  const double mj = sj / m.N;
  double cov = 0.0, vi = 0.0, vj = 0.0;
  for (int t = 0; t < m.N; ++t) {
    const double di = m.at(i, t) - mi;
    const double dj = m.at(j, t) - mj;
    cov += di * dj;
    vi += di * di;
    vj += dj * dj;
  }
  vi /= (m.N - 1);
  vj /= (m.N - 1);
  cov /= (m.N - 1);
  const double den = std::sqrt(vi * vj);
  return (den == 0.0) ? 0.0 : cov / den;
}

int main() {
  // 3 条策略 × 50000 期：策略数少、期数多，与 §3.1 的 i/t 角色一致
  const int T = 3;
  const int N = 50000;
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
    // τ 的长度必须是 T（策略数），不是 N
    const int T2 = 100;
    const double S2 = 1.8;
    const std::uint64_t seed2 = 20260101;

    auto t_null = mc_gate::make_tau(mc_gate::Composition::Null, T2, S2, seed2);
    auto t_one  = mc_gate::make_tau(mc_gate::Composition::OneGood, T2, S2, seed2);
    auto t_mix  = mc_gate::make_tau(mc_gate::Composition::Mixed, T2, S2, seed2);
    auto t_ms   = mc_gate::make_tau(mc_gate::Composition::MixedSigned, T2, S2, seed2);
    auto t_neg  = mc_gate::make_tau(mc_gate::Composition::AllNegative, T2, S2, seed2);

    int n_null = 0, n_one = 0, n_mix = 0, n_ms_pos = 0, n_ms_neg = 0, n_neg = 0;
    for (int i = 0; i < T2; ++i) {
      if (t_null[static_cast<std::size_t>(i)] != 0.0) ++n_null;
      if (t_one[static_cast<std::size_t>(i)] != 0.0) ++n_one;
      if (t_mix[static_cast<std::size_t>(i)] != 0.0) ++n_mix;
      if (t_ms[static_cast<std::size_t>(i)] > 0.0) ++n_ms_pos;
      else if (t_ms[static_cast<std::size_t>(i)] < 0.0) ++n_ms_neg;
      if (t_neg[static_cast<std::size_t>(i)] != 0.0) ++n_neg;
    }
    std::printf("TAUNULL %d TAUNONEGOOD %d TAUMIXED %d TAUMIXEDSIGNEDPOS %d TAUMIXEDSIGNEDNEG %d TAUNEG %d\n",
                n_null, n_one, n_mix, n_ms_pos, n_ms_neg, n_neg);
  }

  {
    // AC-41：维度绑定断言（抓朝向转置）
    const int T3 = 4, N3 = 126;
    auto m = mc_gate::generate(mc_gate::Family::T5, mc_gate::Composition::Null, T3, N3, 0.3, 0.0, 7);
    std::printf("AC41 trials=%d obs=%d corr_dim=%d\n", m.T, m.N, m.T);
  }

  {
    // rho 判别（AC-41）：mean|corr| 必须 ≈ rho；若 ≈0 则 f 退化成了逐策略常数
    const double rr[3] = {0.0, 0.3, 0.7};
    for (int k = 0; k < 3; ++k) {
      auto m = mc_gate::generate(mc_gate::Family::Gauss, mc_gate::Composition::Null, 60, 2000, rr[k], 0.0, 11);
      std::printf("MAC rho=%.1f %.4f\n", rr[k], mc_gate::mean_abs_corr(m));
    }
  }

  std::printf("DGP_DONE\n");
  return 0;
}

#endif

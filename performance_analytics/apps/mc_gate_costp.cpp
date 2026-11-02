// MC-GATE-VALIDATION §5.8 COSTP：在【目标机】上实测 ĉ(T,N) 与分项占比。
// 纪律（AC-31）：steady_clock 只在"一次完整调用"的外层成对取用；
//                 decide()/make_window_stats()/onc_partition() 内部零计时。
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "mc_gate_baselines.h"
#include "mc_gate_adapter.h"
#include "mc_gate_onc.h"

namespace {

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

volatile double g_sink = 0.0;   // 防优化

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// 对 T 个时间点、每个时间点有 N 个策略收益，构造 T×T 相关矩阵
std::vector<double> time_correlation(const mc_gate::ReturnMatrix& m) {
  const int T = m.T, N = m.N;
  std::vector<double> mean(T, 0.0), sd(T, 0.0);
  for (int t = 0; t < T; ++t) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) s += m.at(t, i);
    mean[t] = s / N;
    double v = 0.0;
    for (int i = 0; i < N; ++i) { double d = m.at(t, i) - mean[t]; v += d * d; }
    sd[t] = std::sqrt(v / (N > 1 ? N - 1 : 1));
  }
  std::vector<double> c((std::size_t)T * T, 0.0);
  for (int a = 0; a < T; ++a) {
    c[(std::size_t)a * T + a] = 1.0;
    for (int b = a + 1; b < T; ++b) {
      double s = 0.0;
      for (int i = 0; i < N; ++i)
        s += (m.at(a, i) - mean[a]) * (m.at(b, i) - mean[b]);
      double denom = (N > 1 ? N - 1 : 1) * sd[a] * sd[b];
      double r = denom > 0.0 ? s / denom : 0.0;
      if (r > 1.0) r = 1.0;
      if (r < -1.0) r = -1.0;
      c[(std::size_t)a * T + b] = r;
      c[(std::size_t)b * T + a] = r;
    }
  }
  return c;
}

template <class F>
double median_ms(F&& f, int reps, int warmup) {
  for (int i = 0; i < warmup; ++i) f(i);
  std::vector<double> v;
  v.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    auto t0 = Clock::now();
    f(i);
    auto t1 = Clock::now();
    v.push_back(Ms(t1 - t0).count());
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main() {
  using namespace mc_gate;

  struct Pt { int T; int N; };
  const Pt pts[] = {{3, 3},   {4, 3},     {20, 3},   {100, 3},
                    {4, 126},  {4, 266},   {4, 520},  {20, 126},
                    {20, 266}, {20, 520},  {100, 126}, {100, 266}, {100, 520}, {100, 504}};

  const int REPS = 220;      // §5.8：每点 ≥200 次
  const int WARMUP = 20;     // 丢弃首次（预热）

  std::printf("COSTP_BEGIN reps=%d warmup=%d\n", REPS, WARMUP);
  for (const Pt& p : pts) {
    const int T = p.T, N = p.N;
    const double rho = 0.3, S = 0.6;
    OncSpec spec;  // 默认 min=2 max=4 min_size=2 max_iter=100 repeats=4

    // --- 分量 1：dgp（只生成矩阵） ---
    double t_dgp = median_ms([&](int i) {
      ReturnMatrix m = generate(Family::T5, Composition::OneGood, T, N, rho, S,
                                splitmix64(90001 + (std::uint64_t)i));
      g_sink += m.r[0];
    }, REPS, WARMUP);

    // --- 分量 2：dgp + m（+ make_window_stats 里的均值/方差/平均绝对相关） ---
    double t_dgp_m = median_ms([&](int i) {
      ReturnMatrix m = generate(Family::T5, Composition::OneGood, T, N, rho, S,
                                splitmix64(90001 + (std::uint64_t)i));
      std::vector<double> flat((std::size_t)N * T);
      for (int t = 0; t < T; ++t)
        for (int c = 0; c < N; ++c) flat[(std::size_t)c * T + t] = m.at(t, c);
      ResearchWindow w{std::span<const double>(flat.data(), flat.size()),
                       (std::size_t)N, (std::size_t)T, 0};
      WindowStats st = make_window_stats(w, false);
      g_sink += st.m;
    }, REPS, WARMUP);

    // --- 分量 3：dgp + m + decide（= 一个完整 GATE 调用） ---
    double t_gate = median_ms([&](int i) {
      ReturnMatrix m = generate(Family::T5, Composition::OneGood, T, N, rho, S,
                                splitmix64(90001 + (std::uint64_t)i));
      std::vector<double> flat((std::size_t)N * T);
      for (int t = 0; t < T; ++t)
        for (int c = 0; c < N; ++c) flat[(std::size_t)c * T + t] = m.at(t, c);
      ResearchWindow w{std::span<const double>(flat.data(), flat.size()),
                       (std::size_t)N, (std::size_t)T, 0};
      WindowStats st = make_window_stats(w, false);
      Decision d = decide_from_stats(st, RuleConfig{GateMode::Full, kGateQ, kGateAlpha});
      g_sink += d.selected ? 1.0 : 0.0;
    }, REPS, WARMUP);

    // --- 分量 4：OCN 分区（对 T×T 时间相关矩阵；矩阵构造不计入） ---
    ReturnMatrix m0 = generate(Family::T5, Composition::OneGood, T, N, rho, S, 4242);
    std::vector<double> corr = time_correlation(m0);
    double t_ocn = median_ms([&](int) {
      OncResult r = onc_partition(corr, (std::size_t)T, spec);
      g_sink += static_cast<int>(r.status);
    }, REPS, WARMUP);

    // 分项拆解
    double c_m = t_dgp_m - t_dgp;
    double c_dec = t_gate - t_dgp_m;
    if (c_m < 0.0) c_m = 0.0;
    if (c_dec < 0.0) c_dec = 0.0;

    std::printf("COSTP %d %d dgp=%.4f m=%.4f decide=%.4f ocn=%.4f gate=%.4f "
                "gate_plus_ocn=%.4f\n",
                T, N, t_dgp, c_m, c_dec, t_ocn, t_gate, t_gate + t_ocn);
  }
  std::printf("COSTP_DONE sink=%.3f\n", static_cast<double>(g_sink));
  return 0;
}

// 测 GATE 在 Null 组合下的实际 FPR：T(策略数) × rho 网格，N=126 期，200 reps
#include <cstdio>
#include <cstdint>
#include <vector>
#include "mc_gate_baselines.h"
#include "mc_gate_adapter.h"

static std::uint64_t splitmix64(std::uint64_t x){
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
int main(){
  using namespace mc_gate;
  // 合同 AC：T=4 的 null 格（三个 N）验 0.2%-0.8%。解析预期 ≈0.28%，
  // 因此这里用 2000 reps（期望命中 ~5.6 次），低于此数则 0 命中无法区分。
  const int REPS4 = 2000;
  std::printf("FPR_BEGIN reps4=%d null\n", REPS4);
  for (int N : {126, 266, 520}) {
    for (double rho : {0.0, 0.3, 0.7}) {
      int sel = 0; double msum = 0.0;
      for (int rep = 0; rep < REPS4; ++rep) {
        std::uint64_t seed = splitmix64(999000ULL ^ ((std::uint64_t)N << 44)
                                        ^ ((std::uint64_t)(rho*1000) << 20)
                                        ^ (std::uint64_t)rep);
        ReturnMatrix m = generate(Family::T5, Composition::Null, 4, N, rho, 0.0, seed);
        std::vector<double> flat((std::size_t)4 * (std::size_t)N);
        for (int i = 0; i < 4; ++i)
          for (int t = 0; t < N; ++t) flat[(std::size_t)i * (std::size_t)N + t] = m.at(i, t);
        ResearchWindow w{ std::span<const double>(flat.data(), flat.size()), 4, (std::size_t)N, 0 };
        WindowStats st = make_window_stats(w, false);
        Decision d = decide_from_stats(st, RuleConfig{ GateMode::Full, kGateQ, kGateAlpha });
        if (d.selected) ++sel;
        msum += st.m;
      }
      std::printf("FPR4 %d %.1f %.4f %.4f\n", N, rho, (double)sel / REPS4, msum / REPS4);
    }
  }
  const int REPS = 200;
  for (int T : {4, 20, 100}) {
    for (double rho : {0.0, 0.3, 0.7}) {
      int sel = 0; double msum = 0.0;
      for (int rep = 0; rep < REPS; ++rep) {
        std::uint64_t seed = splitmix64(777000ULL ^ ((std::uint64_t)T << 40)
                                        ^ ((std::uint64_t)(rho*1000) << 20)
                                        ^ (std::uint64_t)rep);
        ReturnMatrix m = generate(Family::T5, Composition::Null, T, 126, rho, 0.0, seed);
        // trial-major：flat[i * 126 + t]
        std::vector<double> flat((std::size_t)T * 126);
        for (int i = 0; i < T; ++i)
          for (int t = 0; t < 126; ++t) flat[(std::size_t)i * 126 + t] = m.at(i, t);
        ResearchWindow w{ std::span<const double>(flat.data(), flat.size()), (std::size_t)T, 126, 0 };
        WindowStats st = make_window_stats(w, false);
        Decision d = decide_from_stats(st, RuleConfig{ GateMode::Full, kGateQ, kGateAlpha });
        if (d.selected) ++sel;
        msum += st.m;
      }
      std::printf("FPR %d %.1f %.6f %.6f\n", T, rho, (double)sel / REPS, msum / REPS);
    }
  }
  std::printf("FPR_DONE\n");
  return 0;
}

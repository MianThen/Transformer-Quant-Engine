// 测 GATE 在 Null 组合下的实际 FPR：T×rho 网格，N=126，200 reps
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
  const int REPS = 200;
  std::printf("FPR_BEGIN reps=%d N=126 null\n", REPS);
  for (int T : {4, 20, 100}) {
    for (double rho : {0.0, 0.3, 0.7}) {
      int sel = 0; double msum = 0.0;
      for (int rep = 0; rep < REPS; ++rep) {
        std::uint64_t seed = splitmix64(777000ULL ^ ((std::uint64_t)T << 40)
                                        ^ ((std::uint64_t)(rho*1000) << 20)
                                        ^ (std::uint64_t)rep);
        ReturnMatrix m = generate(Family::T5, Composition::Null, T, 126, rho, 0.0, seed);
        std::vector<double> flat((std::size_t)126 * T);
        for (int t = 0; t < T; ++t)
          for (int i = 0; i < 126; ++i) flat[(std::size_t)i * T + t] = m.at(t, i);
        ResearchWindow w{ std::span<const double>(flat.data(), flat.size()), 126, (std::size_t)T, 0 };
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

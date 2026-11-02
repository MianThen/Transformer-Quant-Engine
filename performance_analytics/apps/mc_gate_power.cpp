// 测 GATE 功率随信号 S 的变化（one_good 组合），用于 canary 定档
#include <cstdio>
#include <cstdint>
#include <vector>
#include "mc_gate_baselines.h"
#include "mc_gate_adapter.h"
static std::uint64_t splitmix64(std::uint64_t x){
  x += 0x9E3779B97F4A7C15ULL; x=(x^(x>>30))*0xBF58476D1CE4E5B9ULL;
  x=(x^(x>>27))*0x94D049BB133111EBULL; return x^(x>>31);
}
int main(){
  using namespace mc_gate;
  const int REPS=100;
  std::printf("POWER_BEGIN reps=%d one_good rho=0\n", REPS);
  struct Cfg{int T;int N;};
  for (Cfg c : {Cfg{20,126}, Cfg{100,126}, Cfg{100,50}, Cfg{100,266}}) {
    for (double S : {1.2, 1.8, 2.5, 3.0, 4.0, 5.0}) {
      int sel=0;
      for (int rep=0; rep<REPS; ++rep){
        std::uint64_t seed=splitmix64(555000ULL ^ ((std::uint64_t)c.T<<40)
                     ^ ((std::uint64_t)c.N<<20) ^ ((std::uint64_t)(S*100)<<8) ^ (std::uint64_t)rep);
        ReturnMatrix m=generate(Family::T5, Composition::OneGood, c.T, c.N, 0.0, S, seed);
        std::vector<double> flat((std::size_t)c.N*c.T);
        for(int t=0;t<c.T;++t) for(int i=0;i<c.N;++i) flat[(std::size_t)i*c.T+t]=m.at(t,i);
        ResearchWindow w{std::span<const double>(flat.data(),flat.size()),(std::size_t)c.N,(std::size_t)c.T,0};
        Decision d=decide_from_stats(make_window_stats(w,false), RuleConfig{GateMode::Full,kGateQ,kGateAlpha});
        if(d.selected) ++sel;
      }
      std::printf("POWER %d %d %.1f %.3f\n", c.T, c.N, S, (double)sel/REPS);
    }
  }
  std::printf("POWER_DONE\n");
  return 0;
}

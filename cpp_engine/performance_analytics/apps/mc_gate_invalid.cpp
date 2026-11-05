// AC-29：收益 ≤ -1.0 ⇒ INVALID_INPUT 记 ABSTAIN，不崩、不丢 rep
#include <cstdio>
#include <cstddef>
#include <vector>
#include <span>
#include "mc_gate_adapter.h"
int main(){
  using namespace mc_gate;
  const int T=4, N=126;
  // 正常收益
  std::vector<double> ok((std::size_t)T*N, 0.01);
  for (int i=0;i<T;i++) for(int t=0;t<N;t++) ok[(std::size_t)i*N+t] = 0.01 + ((t*37+i*11)%7)*0.001;
  // 注入一条 ≤ -1.0 的收益（策略 2，第 50 期）
  std::vector<double> bad = ok;
  bad[(std::size_t)2*N+50] = -1.5;
  ResearchWindow wa{std::span<const double>(ok.data(),ok.size()),(std::size_t)T,(std::size_t)N,0};
  ResearchWindow wb{std::span<const double>(bad.data(),bad.size()),(std::size_t)T,(std::size_t)N,0};
  WindowStats sa = make_window_stats(wa, false);
  WindowStats sb = make_window_stats(wb, false);
  Decision da = decide_from_stats(sa, RuleConfig{GateMode::Full,0.05,0.05});
  Decision db = decide_from_stats(sb, RuleConfig{GateMode::Full,0.05,0.05});
  std::printf("NORMAL abstain=%d selected=%d\n", sa.abstain?1:0, da.selected?1:0);
  std::printf("EDGE   abstain=%d selected=%d reason=%s\n", sb.abstain?1:0, db.selected?1:0, db.reason?db.reason:"-");
  int pass = (!sa.abstain) && sb.abstain && (!db.selected);
  std::printf("AC29 %d\n", pass?1:0);
  std::printf("INVALID_DONE\n");
  return pass?0:1;
}

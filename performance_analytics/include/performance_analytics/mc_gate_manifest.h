#ifndef MC_GATE_MANIFEST_H
#define MC_GATE_MANIFEST_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>
#include <numeric>
#include <fstream>
#include <map>

namespace mc_gate {

struct Manifest {
  std::string contract_hash;
  std::string config_hash;
  std::string binary_sha256;
  std::string git_commit;
};

struct BudgetUnit {
  int T;
  int N;
  int reps;
  int calls_per_rep;
};

inline int next_rep_step(int reps) {
  if (reps == 8000) return 4000;
  if (reps == 2000) return 1000;
  if (reps == 1000) return 500;
  return reps;
}

inline std::vector<BudgetUnit> plan_units(const std::vector<BudgetUnit>& units,
                                          const std::vector<double>& cost_ms,
                                          double threshold_ms) {
  if (units.size() != cost_ms.size()) {
    return units;
  }
  std::vector<BudgetUnit> out = units;
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (out[i].N == 3) {
      continue;
    }
    if (cost_ms[i] > threshold_ms) {
      out[i].reps = next_rep_step(out[i].reps);
    }
  }
  return out;
}

inline long long total_calls(const std::vector<BudgetUnit>& units) {
  long long sum = 0;
  for (const auto& u : units) {
    sum += static_cast<long long>(u.reps) * static_cast<long long>(u.calls_per_rep);
  }
  return sum;
}

inline std::string manifest_json(const Manifest& m) {
  std::string s;
  s.reserve(256);
  s += "{\"contract_hash\":\"";
  s += m.contract_hash;
  s += "\",\"config_hash\":\"";
  s += m.config_hash;
  s += "\",\"binary_sha256\":\"";
  s += m.binary_sha256;
  s += "\",\"git_commit\":\"";
  s += m.git_commit;
  s += "\"}";
  return s;
}

inline std::string metrics_json(const Manifest& m, long long records, long long cells,
                               long long reps_per_cell, long long rules, long long gzip_mtime,
                               double threshold_ms, const std::vector<BudgetUnit>& planned) {
  std::string s;
  s.reserve(512);
  s += "{\"contract_hash\":\"";
  s += m.contract_hash;
  s += "\",\"config_hash\":\"";
  s += m.config_hash;
  s += "\",\"binary_sha256\":\"";
  s += m.binary_sha256;
  s += "\",\"git_commit\":\"";
  s += m.git_commit;
  s += "\",\"records\":";
  s += std::to_string(records);
  s += ",\"cells\":";
  s += std::to_string(cells);
  s += ",\"reps_per_cell\":";
  s += std::to_string(reps_per_cell);
  s += ",\"rules\":";
  s += std::to_string(rules);
  s += ",\"gzip_mtime\":";
  s += std::to_string(gzip_mtime);
  s += ",\"budget_threshold_ms\":";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6f", threshold_ms);
  s += buf;
  s += ",\"planned_calls\":";
  s += std::to_string(total_calls(planned));
  s += "}";
  return s;
}

} // namespace mc_gate

#ifdef MC_GATE_MANIFEST_SELFTEST

#include <cstdio>

int main() {
  using namespace mc_gate;

  Manifest mf;
  mf.contract_hash = std::string(64, 'a');
  mf.config_hash   = std::string(64, 'b');
  mf.binary_sha256 = std::string(64, 'c');
  mf.git_commit    = std::string(40, 'd');

  std::printf("MANIFEST %s\n", manifest_json(mf).c_str());

  {
    int steps[] = {8000, 2000, 1000, 4000, 500};
    std::string ladder;
    for (std::size_t i = 0; i < 5; ++i) {
      if (i != 0) ladder += ' ';
      ladder += std::to_string(steps[i]);
      ladder += "->";
      ladder += std::to_string(next_rep_step(steps[i]));
    }
    std::printf("LADDER %s\n", ladder.c_str());
  }

  std::vector<BudgetUnit> units = {
    { 4,   126, 8000, 1 },
    { 3,   3,   2000, 1 },
    { 100, 266, 2000, 1 },
    { 20,  266, 1000, 1 },
    { 100, 520, 1000, 1 }
  };
  std::vector<double> cost = { 10.0, 99.0, 0.5, 50.0, 0.5 };

  auto planned1 = plan_units(units, cost, 1.0);
  {
    std::string line = "PLAN";
    for (const auto& u : planned1) {
      line += ' ';
      line += std::to_string(u.reps);
    }
    std::printf("%s\n", line.c_str());
    std::printf("CALLS %lld\n", total_calls(planned1));
  }

  auto planned2 = plan_units(units, cost, 100.0);
  {
    std::string line = "PLAN2";
    for (const auto& u : planned2) {
      line += ' ';
      line += std::to_string(u.reps);
    }
    std::printf("%s\n", line.c_str());
    std::printf("CALLS2 %lld\n", total_calls(planned2));
  }

  std::printf("METRICS %s\n", metrics_json(mf, 1200, 4, 50, 6, 0, 1.0, planned1).c_str());

  std::printf("MANIFEST_DONE\n");
  return 0;
}

#endif // MC_GATE_MANIFEST_SELFTEST

#endif // MC_GATE_MANIFEST_H

#ifndef MC_GATE_RUNNER_H
#define MC_GATE_RUNNER_H

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <span>
#include <utility>
#include <string_view>
#include <cstring>
#include <string>

#include "mc_gate_baselines.h"
#include "mc_gate_adapter.h"

namespace mc_gate {

struct CellSpec {
  int cell_id;
  Family family;
  Composition comp;
  int T;
  int N;
  double rho;
  double S;
};

enum class RuleKind { Gate, B0, B1, B2, B3, B4 };

struct DecisionRecord {
  int cell_id;
  int rep_id;
  const char* rule_id;
  bool selected;
  int index;
  bool abstain;
  std::uint64_t fingerprint;
  const char* reason;
};

inline std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

inline std::uint64_t fnv1a64_bytes(const void* p, std::size_t nbytes) {
  const unsigned char* b = static_cast<const unsigned char*>(p);
  std::uint64_t h = 0xCBF29CE484222325ULL;
  for (std::size_t i = 0; i < nbytes; ++i) {
    h ^= static_cast<std::uint64_t>(b[i]);
    h *= 0x100000001B3ULL;
  }
  return h;
}

struct RuleSpec { RuleKind kind; const char* id; };
inline const RuleSpec* default_rules() {
  static const RuleSpec r[6] = {
    {RuleKind::Gate, "GATE_Full"},
    {RuleKind::B0,   "B0"},
    {RuleKind::B1,   "B1"},
    {RuleKind::B2,   "B2"},
    {RuleKind::B3,   "B3"},
    {RuleKind::B4,   "B4"}
  };
  return r;
}

inline void run_one_task(const CellSpec& c, int rep, std::uint64_t base_seed,
                         std::vector<DecisionRecord>& out) {
  std::uint64_t seed = splitmix64(base_seed ^ (static_cast<std::uint64_t>(c.cell_id) << 32)
                                            ^ static_cast<std::uint64_t>(rep));
  ReturnMatrix m = generate(c.family, c.comp, c.T, c.N, c.rho, c.S, seed);
  std::uint64_t fp = fnv1a64_bytes(m.r.data(), m.r.size() * sizeof(double));
  // v1.4.5：c.T = 策略数（trials），c.N = 期数（observations）
  // ResearchWindow 采用 trial-major：flat[i * N + t]
  std::vector<double> flat(static_cast<std::size_t>(c.T) * static_cast<std::size_t>(c.N));
  for (int i = 0; i < c.T; ++i)
    for (int t = 0; t < c.N; ++t)
      flat[static_cast<std::size_t>(i) * static_cast<std::size_t>(c.N) + static_cast<std::size_t>(t)] = m.at(i, t);
  ResearchWindow w{ std::span<const double>(flat.data(), flat.size()),
                    static_cast<std::size_t>(c.T), static_cast<std::size_t>(c.N), fp };
  const RuleSpec* rs = default_rules();
  for (int k = 0; k < 6; ++k) {
    DecisionRecord d{};
    d.cell_id = c.cell_id;
    d.rep_id = rep;
    d.rule_id = rs[k].id;
    d.fingerprint = fp;
    if (rs[k].kind == RuleKind::Gate) {
      Decision dec = decide(w, RuleConfig{ GateMode::Full, kGateQ, kGateAlpha });
      d.selected = dec.selected;
      d.index = static_cast<int>(dec.index);
      d.abstain = dec.abstain;
      d.reason = dec.reason;
    } else {
      RuleResult r{};
      if (rs[k].kind == RuleKind::B0) r = rule_b0(m);
      else if (rs[k].kind == RuleKind::B1) r = rule_b1(m, 0.05);
      else if (rs[k].kind == RuleKind::B2) r = rule_b2(m);
      else if (rs[k].kind == RuleKind::B3) r = rule_b3(m);
      else r = rule_b4_cscv(m, 4, 0.5);
      d.selected = r.selected;
      d.index = r.index;
      d.abstain = false;
      d.reason = "baseline";
    }
    out.push_back(d);
  }
}

inline void run_shard(std::vector<CellSpec> cells, int reps, int shard_index, int shard_count,
                      std::uint64_t base_seed) {
  if (shard_count <= 0) return;
  if (shard_index < 0 || shard_index >= shard_count) return;
  std::sort(cells.begin(), cells.end(),
            [](const CellSpec& a, const CellSpec& b) { return a.cell_id < b.cell_id; });
  const std::size_t ncell = cells.size();
  const std::size_t sc = static_cast<std::size_t>(shard_count);
  const std::size_t si = static_cast<std::size_t>(shard_index);
  const std::size_t base = ncell / sc;
  const std::size_t rem = ncell % sc;
  const std::size_t start = si * base + (si < rem ? si : rem);
  const std::size_t count = base + (si < rem ? 1 : 0);

  std::vector<DecisionRecord> recs;
  recs.reserve(count * static_cast<std::size_t>(reps) * 6);
  for (std::size_t ci = start; ci < start + count; ++ci)
    for (int rep = 0; rep < reps; ++rep)
      run_one_task(cells[ci], rep, base_seed, recs);

  for (const DecisionRecord& d : recs)
    std::printf("REC %d %d %s %d %d %d %016llx %s\n",
                d.cell_id, d.rep_id, d.rule_id,
                d.selected ? 1 : 0, d.index, d.abstain ? 1 : 0,
                static_cast<unsigned long long>(d.fingerprint), d.reason);
}

} // namespace mc_gate

#ifdef MC_GATE_RUNNER_SELFTEST
int main(int argc, char** argv) {
  using namespace mc_gate;

  std::vector<CellSpec> cells = {
    {101, Family::Gauss,    Composition::Null,       126, 6, 0.00, 0.00},
    {205, Family::T5,       Composition::Mixed,      126, 6, 0.25, 0.40},
    {307, Family::Garch,    Composition::MixedSigned,126, 6, 0.40, 0.55},
    {412, Family::ShortVol, Composition::OneGood,    126, 6, 0.00, 0.60}
  };

  int shard_index = 0;
  int shard_count = 1;

  for (int i = 0; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--shard" && i + 1 < argc) {
      std::string_view arg = argv[i + 1];
      auto slash = arg.find('/');
      if (slash != std::string_view::npos) {
        int si = 0, scv = 0;
        for (std::size_t j = 0; j < slash; ++j)
          if (arg[j] >= '0' && arg[j] <= '9') si = si * 10 + (arg[j] - '0');
        for (std::size_t j = slash + 1; j < arg.size(); ++j)
          if (arg[j] >= '0' && arg[j] <= '9') scv = scv * 10 + (arg[j] - '0');
        if (scv > 0 && si >= 0 && si < scv) { shard_index = si; shard_count = scv; }
      }
    }
  }

  run_shard(cells, 3, shard_index, shard_count, 42);
  return 0;
}
#endif

#endif // MC_GATE_RUNNER_H

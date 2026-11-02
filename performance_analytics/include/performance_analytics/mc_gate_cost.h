#ifndef MC_GATE_COST_H
#define MC_GATE_COST_H

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
#include <filesystem>
#include <sstream>

namespace mc_gate {

struct CostPoint {
  int T;
  int N;
  double ch_ms;
  double dgp_ms;
  double m_ms;
  double ocn_ms;
  double decide_ms;
};

struct CostWeight {
  int T;
  int N;
  long long calls;
};

struct CostProfile {
  double single_core_ms = 0.0;
  int over_threshold = 0;
  int unmatched = 0;
  std::string json;
};

namespace detail {

inline void append_fixed(std::string& out, const char* fmt, double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), fmt, v);
  out += buf;
}

inline void append_int(std::string& out, long long v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%lld", v);
  out += buf;
}

} // namespace detail

inline CostProfile summarise_cost(const std::vector<CostPoint>& pts,
                                  const std::vector<CostWeight>& weights,
                                  double threshold_ms) {
  CostProfile prof;
  prof.single_core_ms = 0.0;
  prof.over_threshold = 0;
  prof.unmatched = 0;

  // Count over_threshold over pts
  for (const auto& p : pts) {
    if (p.ch_ms > threshold_ms) {
      prof.over_threshold++;
    }
  }

  // For each weight, find matching point
  double sum_dgp = 0.0;
  double sum_m = 0.0;
  double sum_ocn = 0.0;
  double sum_decide = 0.0;

  for (const auto& w : weights) {
    bool found = false;
    for (const auto& p : pts) {
      if (p.T == w.T && p.N == w.N) {
        found = true;
        prof.single_core_ms += static_cast<double>(w.calls) * p.ch_ms;
        sum_dgp += static_cast<double>(w.calls) * p.dgp_ms;
        sum_m += static_cast<double>(w.calls) * p.m_ms;
        sum_ocn += static_cast<double>(w.calls) * p.ocn_ms;
        sum_decide += static_cast<double>(w.calls) * p.decide_ms;
        break;
      }
    }
    if (!found) {
      prof.unmatched++;
    }
  }

  // Build JSON
  std::string json;
  json.reserve(512);

  // {"single_core_ms":<%.3f>,"over_threshold":<int>,"buckets":[
  json += "{\"single_core_ms\":";
  detail::append_fixed(json, "%.3f", prof.single_core_ms);
  json += ",\"over_threshold\":";
  detail::append_int(json, static_cast<long long>(prof.over_threshold));
  json += ",\"buckets\":[";

  // Sort pts by (T asc, N asc)
  std::vector<const CostPoint*> sorted;
  sorted.reserve(pts.size());
  for (const auto& p : pts) {
    sorted.push_back(&p);
  }
  std::sort(sorted.begin(), sorted.end(), [](const CostPoint* a, const CostPoint* b) {
    if (a->T != b->T) return a->T < b->T;
    return a->N < b->N;
  });

  for (size_t i = 0; i < sorted.size(); ++i) {
    const auto* p = sorted[i];
    if (i != 0) json += ",";
    json += "{\"T\":";
    detail::append_int(json, static_cast<long long>(p->T));
    json += ",\"N\":";
    detail::append_int(json, static_cast<long long>(p->N));
    json += ",\"ch_ms\":";
    detail::append_fixed(json, "%.6f", p->ch_ms);
    json += "}";
  }
  json += "],\"shares\":{";

  // shares
  double sh_dgp = 0.0, sh_m = 0.0, sh_ocn = 0.0, sh_decide = 0.0;
  if (prof.single_core_ms != 0.0) {
    sh_dgp = sum_dgp / prof.single_core_ms;
    sh_m = sum_m / prof.single_core_ms;
    sh_ocn = sum_ocn / prof.single_core_ms;
    sh_decide = sum_decide / prof.single_core_ms;
  }

  json += "\"dgp\":";
  detail::append_fixed(json, "%.6f", sh_dgp);
  json += ",\"m\":";
  detail::append_fixed(json, "%.6f", sh_m);
  json += ",\"ocn\":";
  detail::append_fixed(json, "%.6f", sh_ocn);
  json += ",\"decide\":";
  detail::append_fixed(json, "%.6f", sh_decide);
  json += "}}";

  prof.json = std::move(json);
  return prof;
}

} // namespace mc_gate

#ifdef MC_GATE_COST_SELFTEST

int main() {
  using namespace mc_gate;

  std::vector<CostPoint> pts = {
    { 4,   126, 0.137500, 0.060900, 0.047100, 0.002500, 0.027000 },
    { 100, 266, 7.015500, 3.198800, 3.365100, 0.408600, 0.043100 },
    { 3,   3,   0.027800, 0.001600, 0.000100, 0.000000, 0.026100 },
  };

  std::vector<CostWeight> weights = {
    { 4,   126, 24000 },
    { 100, 266, 24000 },
    { 3,   3,   162000 },
  };

  double threshold_ms = 1.0;
  CostProfile prof = summarise_cost(pts, weights, threshold_ms);

  std::printf("COST total %.3f\n", prof.single_core_ms);
  std::printf("COST over %d\n", prof.over_threshold);
  std::printf("COST unmatched %d\n", prof.unmatched);
  std::printf("COST json %s\n", prof.json.c_str());
  std::printf("COST_DONE\n");

  return 0;
}

#endif // MC_GATE_COST_SELFTEST

#endif // MC_GATE_COST_H

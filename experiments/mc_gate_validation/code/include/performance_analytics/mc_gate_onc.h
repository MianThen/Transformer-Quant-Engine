#ifndef MC_GATE_ONC_H
#define MC_GATE_ONC_H

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <numeric>

namespace mc_gate {

struct OncSpec {
  std::uint32_t min_clusters = 2;
  std::uint32_t max_clusters = 4;
  std::uint32_t min_cluster_size = 2;
  std::uint32_t max_iterations = 100;
  std::uint32_t repeats = 4;
  double distance_tolerance = 1e-12;
  std::vector<std::uint64_t> seeds = {17, 29, 43, 71};
};

enum class OncStatus { OK, INVALID_INPUT, NO_VALID_PARTITION };

struct OncCandidate {
  std::uint32_t cluster_count;
  std::uint64_t seed;
  bool valid;
  double quality;
};

struct OncResult {
  OncStatus status = OncStatus::INVALID_INPUT;
  std::vector<std::uint32_t> assignment;
  std::uint32_t cluster_count = 0;
  double quality = 0.0;
  std::vector<OncCandidate> candidates;
};

namespace detail {

struct CandidateResult {
  bool valid = false;
  std::vector<std::uint32_t> assignment;
  double quality = 0.0;
};

inline CandidateResult run_candidate(const std::vector<double>& d,
                                     std::size_t n,
                                     const OncSpec& spec,
                                     std::uint32_t cc,
                                     std::uint64_t seed) {
  const double tol = spec.distance_tolerance;
  const double inf = std::numeric_limits<double>::infinity();

  std::vector<std::uint32_t> medoids;
  medoids.reserve(cc);

  // (a) first medoid
  medoids.push_back(static_cast<std::uint32_t>(seed % static_cast<std::uint64_t>(n)));

  // (b) fill medoids to cc
  while (medoids.size() < cc) {
    double best = -1.0;
    std::uint32_t bi = 0;
    bool found = false;
    for (std::size_t idx = 0; idx < n; ++idx) {
      bool in_medoids = false;
      for (std::uint32_t m : medoids) {
        if (static_cast<std::size_t>(m) == idx) { in_medoids = true; break; }
      }
      if (in_medoids) continue;
      double sc = inf;
      for (std::uint32_t m : medoids) {
        sc = std::min(sc, d[idx * n + m]);
      }
      if (!found || sc > best + tol || (std::abs(sc - best) <= tol && idx < bi)) {
        best = sc;
        bi = static_cast<std::uint32_t>(idx);
        found = true;
      }
    }
    if (!found) return CandidateResult{false, {}, 0.0};
    medoids.push_back(bi);
  }

  // (c) iterations
  std::vector<std::uint32_t> assignment(n, 0);
  for (std::uint32_t iter = 0; iter < spec.max_iterations; ++iter) {
    std::vector<std::uint32_t> next(n, 0);
    for (std::size_t idx = 0; idx < n; ++idx) {
      double bd = inf;
      std::uint32_t bc = 0;
      for (std::uint32_t c = 0; c < cc; ++c) {
        double v = d[idx * n + medoids[c]];
        if (v < bd - tol || (std::abs(v - bd) <= tol && medoids[c] < medoids[bc])) {
          bd = v;
          bc = c;
        }
      }
      next[idx] = bc;
    }
    std::vector<std::uint32_t> next_medoids = medoids;
    for (std::uint32_t c = 0; c < cc; ++c) {
      std::vector<std::size_t> members;
      for (std::size_t idx = 0; idx < n; ++idx) {
        if (next[idx] == c) members.push_back(idx);
      }
      if (members.empty()) return CandidateResult{false, {}, 0.0};
      double bestcost = inf;
      std::uint32_t bm = static_cast<std::uint32_t>(members.front());
      for (std::size_t m : members) {
        double cost = 0.0;
        for (std::size_t o : members) cost += d[m * n + o];
        if (cost < bestcost - tol || (std::abs(cost - bestcost) <= tol && m < bm)) {
          bestcost = cost;
          bm = static_cast<std::uint32_t>(m);
        }
      }
      next_medoids[c] = bm;
    }
    if (next == assignment && next_medoids == medoids) {
      medoids = next_medoids;
      break;
    }
    assignment = next;
    medoids = next_medoids;
  }

  // (d) canonicalize
  std::vector<std::uint32_t> canon(n, 0);
  std::vector<std::uint32_t> remap(cc, 0);
  std::vector<bool> used(cc, false);
  std::uint32_t next_id = 0;
  for (std::size_t idx = 0; idx < n; ++idx) {
    std::uint32_t a = assignment[idx];
    if (!used[a]) {
      used[a] = true;
      remap[a] = next_id;
      ++next_id;
    }
    canon[idx] = remap[a];
  }
  std::uint32_t used_clusters = next_id;

  // (e) validate cluster count and sizes
  if (used_clusters != cc) return CandidateResult{false, {}, 0.0};
  std::vector<std::uint32_t> sizes(cc, 0);
  for (std::size_t idx = 0; idx < n; ++idx) {
    ++sizes[canon[idx]];
  }
  for (std::uint32_t c = 0; c < cc; ++c) {
    if (sizes[c] < spec.min_cluster_size) return CandidateResult{false, {}, 0.0};
  }

  // (f) silhouette
  std::vector<double> sil(n, 0.0);
  for (std::size_t idx = 0; idx < n; ++idx) {
    std::uint32_t own = canon[idx];
    double own_sum = 0.0;
    std::uint32_t own_count = 0;
    std::vector<double> other_sum(cc, 0.0);
    std::vector<std::uint32_t> other_count(cc, 0);
    for (std::size_t other = 0; other < n; ++other) {
      if (other == idx) continue;
      double dd = d[idx * n + other];
      if (canon[other] == own) {
        own_sum += dd;
        ++own_count;
      } else {
        other_sum[canon[other]] += dd;
        ++other_count[canon[other]];
      }
    }
    double a = (own_count == 0) ? 0.0 : own_sum / static_cast<double>(own_count);
    double b = inf;
    for (std::uint32_t c = 0; c < cc; ++c) {
      if (c == own || other_count[c] == 0) continue;
      b = std::min(b, other_sum[c] / static_cast<double>(other_count[c]));
    }
    sil[idx] = (!std::isfinite(b) || std::max(a, b) <= tol) ? 0.0 : (b - a) / std::max(a, b);
  }
  double quality = 0.0;
  for (std::size_t i = 0; i < n; ++i) quality += sil[i];
  quality /= static_cast<double>(n);
  bool valid = std::isfinite(quality);
  return CandidateResult{valid, canon, quality};
}

} // namespace detail

inline OncResult onc_partition(const std::vector<double>& corr,
                               std::size_t n,
                               const OncSpec& spec) {
  OncResult result;
  result.status = OncStatus::INVALID_INPUT;

  // Step 0: parameter validation
  if (n == 0 ||
      spec.min_clusters < 2 ||
      spec.max_clusters < spec.min_clusters ||
      spec.min_cluster_size == 0 ||
      spec.max_iterations == 0 ||
      spec.repeats == 0 ||
      spec.seeds.size() < spec.repeats) {
    return result;
  }

  // Step 1: clamp max clusters
  std::uint32_t mc = spec.max_clusters;
  if (static_cast<std::size_t>(mc) > n / spec.min_cluster_size) {
    mc = static_cast<std::uint32_t>(n / spec.min_cluster_size);
  }
  if (mc < spec.min_clusters) {
    return result;
  }

  // Step 2: distance matrix
  std::vector<double> d(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      double v = std::sqrt(std::max(0.0, (1.0 - corr[i * n + j]) / 2.0));
      d[i * n + j] = v;
      d[j * n + i] = v;
    }
  }

  // Step 3: enumeration
  OncResult best;
  bool has_best = false;

  for (std::uint32_t cc = spec.min_clusters; cc <= mc; ++cc) {
    for (std::uint32_t r = 0; r < spec.repeats; ++r) {
      std::uint64_t seed = spec.seeds[r];
      detail::CandidateResult candidate = detail::run_candidate(d, n, spec, cc, seed);

      result.candidates.push_back(OncCandidate{cc, seed, candidate.valid, candidate.quality});

      if (!candidate.valid) continue;

      bool better = !has_best
                 || candidate.quality > best.quality + spec.distance_tolerance
                 || (std::abs(candidate.quality - best.quality) <= spec.distance_tolerance
                     && (cc < best.cluster_count
                         || (cc == best.cluster_count
                             && std::lexicographical_compare(candidate.assignment.begin(),
                                                             candidate.assignment.end(),
                                                             best.assignment.begin(),
                                                             best.assignment.end()))));
      if (better) {
        best.status = OncStatus::OK;
        best.assignment = candidate.assignment;
        best.cluster_count = cc;
        best.quality = candidate.quality;
        has_best = true;
      }
    }
  }

  if (!has_best) {
    result.status = OncStatus::NO_VALID_PARTITION;
    result.assignment.clear();
    result.cluster_count = 0;
    result.quality = 0.0;
  } else {
    result.status = OncStatus::OK;
    result.assignment = best.assignment;
    result.cluster_count = best.cluster_count;
    result.quality = best.quality;
  }

  return result;
}

} // namespace mc_gate

#ifdef MC_GATE_ONC_SELFTEST

int main() {
  using namespace mc_gate;

  // Case A: n=4, two groups
  {
    std::vector<double> corr = {
      1.0, 0.9, 0.1, 0.1,
      0.9, 1.0, 0.1, 0.1,
      0.1, 0.1, 1.0, 0.9,
      0.1, 0.1, 0.9, 1.0
    };
    OncSpec spec;
    spec.min_clusters = 2;
    spec.max_clusters = 2;
    OncResult res = onc_partition(corr, 4, spec);
    std::string assign_str;
    if (res.status == OncStatus::OK) {
      for (std::uint32_t v : res.assignment) assign_str += static_cast<char>('0' + v);
    } else {
      assign_str = "-";
    }
    std::size_t valid_count = 0;
    for (const auto& c : res.candidates) if (c.valid) ++valid_count;
    const char* status_str = "INVALID_INPUT";
    if (res.status == OncStatus::OK) status_str = "OK";
    else if (res.status == OncStatus::NO_VALID_PARTITION) status_str = "NO_VALID_PARTITION";
    std::printf("CASE A status=%s cc=%u assign=%s cand=%zu valid=%zu quality=%.6f\n",
                status_str, res.cluster_count, assign_str.c_str(),
                res.candidates.size(), valid_count, res.quality);
  }

  // Case B: n=4, all off-diagonal 0.5
  {
    std::vector<double> corr(16, 0.5);
    for (std::size_t i = 0; i < 4; ++i) corr[i * 4 + i] = 1.0;
    OncSpec spec;
    spec.min_clusters = 2;
    spec.max_clusters = 2;
    OncResult res = onc_partition(corr, 4, spec);
    std::string assign_str;
    if (res.status == OncStatus::OK) {
      for (std::uint32_t v : res.assignment) assign_str += static_cast<char>('0' + v);
    } else {
      assign_str = "-";
    }
    std::size_t valid_count = 0;
    for (const auto& c : res.candidates) if (c.valid) ++valid_count;
    const char* status_str = "INVALID_INPUT";
    if (res.status == OncStatus::OK) status_str = "OK";
    else if (res.status == OncStatus::NO_VALID_PARTITION) status_str = "NO_VALID_PARTITION";
    std::printf("CASE B status=%s cc=%u assign=%s cand=%zu valid=%zu quality=%.6f\n",
                status_str, res.cluster_count, assign_str.c_str(),
                res.candidates.size(), valid_count, res.quality);
  }

  // Case C: n=3, triggers clamp lower bound
  {
    std::vector<double> corr = {
      1.0, 0.2, 0.2,
      0.2, 1.0, 0.2,
      0.2, 0.2, 1.0
    };
    OncSpec spec;
    spec.min_clusters = 2;
    spec.max_clusters = 4;
    OncResult res = onc_partition(corr, 3, spec);
    std::string assign_str;
    if (res.status == OncStatus::OK) {
      for (std::uint32_t v : res.assignment) assign_str += static_cast<char>('0' + v);
    } else {
      assign_str = "-";
    }
    std::size_t valid_count = 0;
    for (const auto& c : res.candidates) if (c.valid) ++valid_count;
    const char* status_str = "INVALID_INPUT";
    if (res.status == OncStatus::OK) status_str = "OK";
    else if (res.status == OncStatus::NO_VALID_PARTITION) status_str = "NO_VALID_PARTITION";
    std::printf("CASE C status=%s cc=%u assign=%s cand=%zu valid=%zu quality=%.6f\n",
                status_str, res.cluster_count, assign_str.c_str(),
                res.candidates.size(), valid_count, res.quality);
  }

  // Case D: n=6, three groups of 2
  {
    std::vector<double> corr(36, 0.1);
    for (std::size_t i = 0; i < 6; ++i) {
      corr[i * 6 + i] = 1.0;
      for (std::size_t j = 0; j < 6; ++j) {
        if (i / 2 == j / 2 && i != j) corr[i * 6 + j] = 0.9;
      }
    }
    OncSpec spec;
    spec.min_clusters = 2;
    spec.max_clusters = 4;
    OncResult res = onc_partition(corr, 6, spec);
    std::string assign_str;
    if (res.status == OncStatus::OK) {
      for (std::uint32_t v : res.assignment) assign_str += static_cast<char>('0' + v);
    } else {
      assign_str = "-";
    }
    std::size_t valid_count = 0;
    for (const auto& c : res.candidates) if (c.valid) ++valid_count;
    const char* status_str = "INVALID_INPUT";
    if (res.status == OncStatus::OK) status_str = "OK";
    else if (res.status == OncStatus::NO_VALID_PARTITION) status_str = "NO_VALID_PARTITION";
    std::printf("CASE D status=%s cc=%u assign=%s cand=%zu valid=%zu quality=%.6f\n",
                status_str, res.cluster_count, assign_str.c_str(),
                res.candidates.size(), valid_count, res.quality);
  }

  std::printf("SEEDMOD 4 %llu %llu %llu %llu\n",
              static_cast<unsigned long long>(17 % 4),
              static_cast<unsigned long long>(29 % 4),
              static_cast<unsigned long long>(43 % 4),
              static_cast<unsigned long long>(71 % 4));
  std::printf("SEEDMOD 20 %llu %llu %llu %llu\n",
              static_cast<unsigned long long>(17 % 20),
              static_cast<unsigned long long>(29 % 20),
              static_cast<unsigned long long>(43 % 20),
              static_cast<unsigned long long>(71 % 20));

  std::printf("ONC_DONE\n");
  return 0;
}

#endif // MC_GATE_ONC_SELFTEST

#endif // MC_GATE_ONC_H

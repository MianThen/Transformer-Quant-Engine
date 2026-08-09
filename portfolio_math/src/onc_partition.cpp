#include "portfolio_math/onc_partition.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>

namespace portfolio_math {
namespace {

using quant_math::DenseMatrix;

DenseMatrix copy_matrix(quant_math::MatrixView source) {
  DenseMatrix result(static_cast<Eigen::Index>(source.rows),
                     static_cast<Eigen::Index>(source.cols));
  for (std::size_t row = 0; row < source.rows; ++row) {
    for (std::size_t col = 0; col < source.cols; ++col) {
      result(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          source(row, col);
    }
  }
  return result;
}

std::uint64_t append_hash(std::uint64_t hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8)) & 0xffU;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::uint64_t matrix_hash(const DenseMatrix& matrix) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash = append_hash(hash, static_cast<std::uint64_t>(matrix.rows()));
  hash = append_hash(hash, static_cast<std::uint64_t>(matrix.cols()));
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      hash = append_hash(hash, std::bit_cast<std::uint64_t>(matrix(row, col)));
    }
  }
  return hash;
}

std::uint64_t partition_hash(const std::vector<std::uint32_t>& assignment,
                             std::uint32_t cluster_count) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash = append_hash(hash, cluster_count);
  for (const auto cluster : assignment) hash = append_hash(hash, cluster);
  return hash;
}

OncPartitionResult failed(OncStatus status) {
  OncPartitionResult result;
  result.status = status;
  return result;
}

struct Candidate {
  bool valid{false};
  std::uint32_t cluster_count{0};
  std::uint64_t seed{0};
  std::vector<std::uint32_t> assignment;
  std::vector<double> silhouette;
  double quality{-std::numeric_limits<double>::infinity()};
};

std::vector<std::uint32_t> canonicalize_assignment(
    const std::vector<std::uint32_t>& assignment,
    std::uint32_t cluster_count) {
  std::vector<std::uint32_t> order(cluster_count);
  std::iota(order.begin(), order.end(), 0U);
  std::vector<std::uint32_t> minimum(cluster_count,
                                     std::numeric_limits<std::uint32_t>::max());
  for (std::size_t index = 0; index < assignment.size(); ++index) {
    minimum[assignment[index]] = std::min(
        minimum[assignment[index]], static_cast<std::uint32_t>(index));
  }
  std::sort(order.begin(), order.end(), [&](std::uint32_t left,
                                            std::uint32_t right) {
    if (minimum[left] != minimum[right]) return minimum[left] < minimum[right];
    return left < right;
  });
  std::vector<std::uint32_t> remap(cluster_count);
  for (std::uint32_t index = 0; index < cluster_count; ++index) {
    remap[order[index]] = index;
  }
  std::vector<std::uint32_t> canonical(assignment.size());
  for (std::size_t index = 0; index < assignment.size(); ++index) {
    canonical[index] = remap[assignment[index]];
  }
  return canonical;
}

bool lexicographically_precedes(const std::vector<std::uint32_t>& left,
                                const std::vector<std::uint32_t>& right) {
  return std::lexicographical_compare(left.begin(), left.end(), right.begin(),
                                       right.end());
}

Candidate run_candidate(const std::vector<double>& distances,
                        std::size_t asset_count, const OncSpec& spec,
                        std::uint32_t cluster_count, std::uint64_t seed) {
  Candidate candidate;
  candidate.cluster_count = cluster_count;
  candidate.seed = seed;
  candidate.assignment.assign(asset_count, 0U);
  std::vector<std::uint32_t> medoids;
  medoids.push_back(static_cast<std::uint32_t>(seed % asset_count));
  while (medoids.size() < cluster_count) {
    double best_score = -1.0;
    std::uint32_t best_index = 0;
    bool found = false;
    for (std::uint32_t index = 0; index < asset_count; ++index) {
      if (std::find(medoids.begin(), medoids.end(), index) != medoids.end()) {
        continue;
      }
      double score = std::numeric_limits<double>::infinity();
      for (const auto medoid : medoids) {
        score = std::min(score,
                         distances[static_cast<std::size_t>(index) * asset_count +
                                   medoid]);
      }
      if (!found || score > best_score + spec.distance_tolerance ||
          (std::abs(score - best_score) <= spec.distance_tolerance &&
           index < best_index)) {
        best_score = score;
        best_index = index;
        found = true;
      }
    }
    if (!found) return candidate;
    medoids.push_back(best_index);
  }

  for (std::uint32_t iteration = 0; iteration < spec.max_iterations;
       ++iteration) {
    std::vector<std::uint32_t> next_assignment(asset_count, 0U);
    for (std::uint32_t index = 0; index < asset_count; ++index) {
      double best_distance = std::numeric_limits<double>::infinity();
      std::uint32_t best_cluster = 0;
      for (std::uint32_t cluster = 0; cluster < cluster_count; ++cluster) {
        const double distance = distances[static_cast<std::size_t>(index) *
                                               asset_count + medoids[cluster]];
        if (distance < best_distance - spec.distance_tolerance ||
            (std::abs(distance - best_distance) <= spec.distance_tolerance &&
             medoids[cluster] < medoids[best_cluster])) {
          best_distance = distance;
          best_cluster = cluster;
        }
      }
      next_assignment[index] = best_cluster;
    }

    std::vector<std::uint32_t> next_medoids = medoids;
    for (std::uint32_t cluster = 0; cluster < cluster_count; ++cluster) {
      std::vector<std::uint32_t> members;
      for (std::uint32_t index = 0; index < asset_count; ++index) {
        if (next_assignment[index] == cluster) members.push_back(index);
      }
      if (members.empty()) return candidate;
      double best_cost = std::numeric_limits<double>::infinity();
      std::uint32_t best_medoid = members.front();
      for (const auto member : members) {
        double cost = 0.0;
        for (const auto other : members) {
          cost += distances[static_cast<std::size_t>(member) * asset_count +
                            other];
        }
        if (cost < best_cost - spec.distance_tolerance ||
            (std::abs(cost - best_cost) <= spec.distance_tolerance &&
             member < best_medoid)) {
          best_cost = cost;
          best_medoid = member;
        }
      }
      next_medoids[cluster] = best_medoid;
    }
    if (next_assignment == candidate.assignment && next_medoids == medoids) {
      medoids = std::move(next_medoids);
      break;
    }
    candidate.assignment = std::move(next_assignment);
    medoids = std::move(next_medoids);
  }

  if (candidate.assignment.empty()) return candidate;
  candidate.assignment =
      canonicalize_assignment(candidate.assignment, cluster_count);
  std::vector<std::uint32_t> sizes(cluster_count, 0U);
  for (const auto cluster : candidate.assignment) ++sizes[cluster];
  if (std::any_of(sizes.begin(), sizes.end(), [&](std::uint32_t size) {
        return size < spec.min_cluster_size;
      })) {
    return candidate;
  }

  candidate.silhouette.assign(asset_count, 0.0);
  for (std::uint32_t index = 0; index < asset_count; ++index) {
    const auto own_cluster = candidate.assignment[index];
    double own_sum = 0.0;
    std::uint32_t own_count = 0;
    std::vector<double> other_sum(cluster_count, 0.0);
    std::vector<std::uint32_t> other_count(cluster_count, 0U);
    for (std::uint32_t other = 0; other < asset_count; ++other) {
      if (other == index) continue;
      const double distance = distances[static_cast<std::size_t>(index) *
                                         asset_count + other];
      if (candidate.assignment[other] == own_cluster) {
        own_sum += distance;
        ++own_count;
      } else {
        other_sum[candidate.assignment[other]] += distance;
        ++other_count[candidate.assignment[other]];
      }
    }
    const double a = own_count == 0 ? 0.0 : own_sum / own_count;
    double b = std::numeric_limits<double>::infinity();
    for (std::uint32_t cluster = 0; cluster < cluster_count; ++cluster) {
      if (cluster == own_cluster || other_count[cluster] == 0) continue;
      b = std::min(b, other_sum[cluster] / other_count[cluster]);
    }
    candidate.silhouette[index] =
        !std::isfinite(b) || std::max(a, b) <= spec.distance_tolerance
            ? 0.0
            : (b - a) / std::max(a, b);
  }
  candidate.quality = std::accumulate(candidate.silhouette.begin(),
                                      candidate.silhouette.end(), 0.0) /
                      static_cast<double>(asset_count);
  candidate.valid = std::isfinite(candidate.quality);
  return candidate;
}

}  // namespace

bool valid_onc_spec(const OncSpec& spec) noexcept {
  return spec.min_clusters >= 2 && spec.max_clusters >= spec.min_clusters &&
         spec.min_cluster_size > 0 && spec.max_iterations > 0 &&
         spec.repeats > 0 && spec.seeds.size() >= spec.repeats &&
         std::isfinite(spec.distance_tolerance) &&
         spec.distance_tolerance >= 0.0 &&
         std::isfinite(spec.symmetry_tolerance) &&
         spec.symmetry_tolerance >= 0.0 &&
         std::isfinite(spec.psd_tolerance) && spec.psd_tolerance > 0.0;
}

OncPartitionResult onc_partition(quant_math::MatrixView input, OncSpec spec) {
  if (!valid_onc_spec(spec) || input.rows == 0 || input.rows != input.cols ||
      !quant_math::validate_symmetric(input, spec.symmetry_tolerance).ok) {
    return failed(OncStatus::INVALID_INPUT);
  }
  const std::size_t asset_count = input.rows;
  if (spec.max_clusters > asset_count / spec.min_cluster_size) {
    spec.max_clusters = static_cast<std::uint32_t>(
        asset_count / spec.min_cluster_size);
  }
  if (spec.max_clusters < spec.min_clusters) {
    return failed(OncStatus::INVALID_INPUT);
  }

  DenseMatrix correlation = copy_matrix(input);
  const auto diagonal = correlation.diagonal();
  for (Eigen::Index index = 0;
       index < static_cast<Eigen::Index>(asset_count); ++index) {
    if (!std::isfinite(diagonal(index)) || diagonal(index) <= 0.0) {
      return failed(OncStatus::INVALID_INPUT);
    }
  }
  for (Eigen::Index row = 0;
       row < static_cast<Eigen::Index>(asset_count); ++row) {
    for (Eigen::Index col = 0;
         col < static_cast<Eigen::Index>(asset_count); ++col) {
      correlation(row, col) /= std::sqrt(diagonal(row) * diagonal(col));
      if (correlation(row, col) < -1.0 - spec.symmetry_tolerance ||
          correlation(row, col) > 1.0 + spec.symmetry_tolerance) {
        return failed(OncStatus::INVALID_INPUT);
      }
      correlation(row, col) = std::clamp(correlation(row, col), -1.0, 1.0);
    }
  }
  correlation = 0.5 * (correlation + correlation.transpose()).eval();
  for (Eigen::Index index = 0;
       index < static_cast<Eigen::Index>(asset_count); ++index) {
    correlation(index, index) = 1.0;
  }
  if (!quant_math::is_positive_semidefinite(correlation,
                                             spec.psd_tolerance)) {
    return failed(OncStatus::INVALID_INPUT);
  }

  std::vector<double> distances(asset_count * asset_count, 0.0);
  for (std::size_t row = 0; row < asset_count; ++row) {
    for (std::size_t col = row + 1; col < asset_count; ++col) {
      const double distance = std::sqrt(
          std::max(0.0, (1.0 - correlation(static_cast<Eigen::Index>(row),
                                            static_cast<Eigen::Index>(col))) /
                             2.0));
      if (!std::isfinite(distance)) return failed(OncStatus::NUMERICAL_FAILURE);
      distances[row * asset_count + col] = distance;
      distances[col * asset_count + row] = distance;
    }
  }

  OncPartitionResult result;
  result.diagnostics.input_correlation_hash = matrix_hash(correlation);
  Candidate best;
  Candidate second;
  for (std::uint32_t cluster_count = spec.min_clusters;
       cluster_count <= spec.max_clusters; ++cluster_count) {
    for (std::uint32_t repeat = 0; repeat < spec.repeats; ++repeat) {
      const auto seed = spec.seeds[repeat];
      auto candidate = run_candidate(distances, asset_count, spec,
                                     cluster_count, seed);
      OncCandidateDiagnostic diagnostic;
      diagnostic.cluster_count = cluster_count;
      diagnostic.seed = seed;
      diagnostic.valid = candidate.valid;
      if (candidate.valid) {
        diagnostic.quality = candidate.quality;
        diagnostic.minimum_silhouette =
            *std::min_element(candidate.silhouette.begin(),
                              candidate.silhouette.end());
        diagnostic.maximum_silhouette =
            *std::max_element(candidate.silhouette.begin(),
                              candidate.silhouette.end());
      }
      result.diagnostics.candidates.push_back(diagnostic);
      if (!candidate.valid) continue;
      const bool better =
          !best.valid || candidate.quality > best.quality + spec.distance_tolerance ||
          (std::abs(candidate.quality - best.quality) <=
               spec.distance_tolerance &&
           (candidate.cluster_count < best.cluster_count ||
            (candidate.cluster_count == best.cluster_count &&
             lexicographically_precedes(candidate.assignment,
                                         best.assignment))));
      if (better) {
        second = std::move(best);
        best = std::move(candidate);
      } else if (!second.valid || candidate.quality > second.quality) {
        second = std::move(candidate);
      }
    }
  }
  if (!best.valid) return failed(OncStatus::NO_VALID_PARTITION);

  result.cluster_id_by_symbol = best.assignment;
  result.diagnostics.selected_cluster_count = best.cluster_count;
  result.diagnostics.selected_quality = best.quality;
  result.diagnostics.selected_silhouette = best.silhouette;
  result.diagnostics.second_best_quality =
      second.valid ? second.quality : best.quality;
  result.diagnostics.best_second_gap =
      best.quality - result.diagnostics.second_best_quality;
  result.diagnostics.partition_hash =
      partition_hash(result.cluster_id_by_symbol, best.cluster_count);
  result.quasi_diagonal_order.resize(asset_count);
  std::iota(result.quasi_diagonal_order.begin(),
            result.quasi_diagonal_order.end(), 0U);
  std::stable_sort(result.quasi_diagonal_order.begin(),
                   result.quasi_diagonal_order.end(),
                   [&](std::uint32_t left, std::uint32_t right) {
                     if (result.cluster_id_by_symbol[left] !=
                         result.cluster_id_by_symbol[right]) {
                       return result.cluster_id_by_symbol[left] <
                              result.cluster_id_by_symbol[right];
                     }
                     return left < right;
                   });
  result.diagnostics.eligible_for_official_risk = false;
  result.status = OncStatus::OK;
  return result;
}

}  // namespace portfolio_math

#include "portfolio_math/hierarchical_linkage.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>

#include <Eigen/Eigenvalues>

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

std::uint64_t linkage_hash(
    const std::vector<HierarchicalMerge>& merge_tree,
    const std::vector<std::uint32_t>& order) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash = append_hash(hash, static_cast<std::uint64_t>(merge_tree.size()));
  for (const auto& merge : merge_tree) {
    hash = append_hash(hash, merge.left);
    hash = append_hash(hash, merge.right);
    hash = append_hash(hash, merge.cluster_size);
    hash = append_hash(hash, std::bit_cast<std::uint64_t>(merge.distance));
  }
  hash = append_hash(hash, static_cast<std::uint64_t>(order.size()));
  for (const auto symbol : order) {
    hash = append_hash(hash, symbol);
  }
  return hash;
}

HierarchicalLinkageResult failed(HierarchicalLinkageStatus status) {
  HierarchicalLinkageResult result;
  result.status = status;
  return result;
}

bool merge_precedes(std::uint32_t left, std::uint32_t right,
                    std::uint32_t best_left, std::uint32_t best_right) {
  if (left != best_left) return left < best_left;
  return right < best_right;
}

void append_quasi_order(
    std::uint32_t cluster_id, std::size_t asset_count,
    const std::vector<HierarchicalMerge>& merge_tree,
    std::vector<std::uint32_t>& order) {
  if (cluster_id < asset_count) {
    order.push_back(cluster_id);
    return;
  }
  const auto& merge = merge_tree[cluster_id - asset_count];
  append_quasi_order(merge.left, asset_count, merge_tree, order);
  append_quasi_order(merge.right, asset_count, merge_tree, order);
}

}  // namespace

bool valid_hierarchical_linkage_spec(
    const HierarchicalLinkageSpec& spec) noexcept {
  return spec.method == LinkageMethod::COMPLETE &&
         std::isfinite(spec.distance_tolerance) &&
         spec.distance_tolerance >= 0.0 &&
         std::isfinite(spec.symmetry_tolerance) &&
         spec.symmetry_tolerance >= 0.0 &&
         std::isfinite(spec.psd_tolerance) && spec.psd_tolerance > 0.0;
}

HierarchicalLinkageResult hierarchical_linkage(
    quant_math::MatrixView input, HierarchicalLinkageSpec spec) {
  if (!valid_hierarchical_linkage_spec(spec) || input.rows == 0 ||
      input.rows != input.cols ||
      !quant_math::validate_symmetric(input, spec.symmetry_tolerance).ok) {
    return failed(HierarchicalLinkageStatus::INVALID_INPUT);
  }

  const std::size_t asset_count = input.rows;
  DenseMatrix correlation = copy_matrix(input);
  for (Eigen::Index index = 0;
       index < static_cast<Eigen::Index>(asset_count); ++index) {
    if (!std::isfinite(correlation(index, index)) ||
        correlation(index, index) <= 0.0) {
      return failed(HierarchicalLinkageStatus::INVALID_INPUT);
    }
  }
  const auto diagonal = correlation.diagonal();
  for (Eigen::Index row = 0;
       row < static_cast<Eigen::Index>(asset_count); ++row) {
    for (Eigen::Index col = 0;
         col < static_cast<Eigen::Index>(asset_count); ++col) {
      correlation(row, col) /=
          std::sqrt(diagonal(row) * diagonal(col));
      if (correlation(row, col) < -1.0 - spec.symmetry_tolerance ||
          correlation(row, col) > 1.0 + spec.symmetry_tolerance) {
        return failed(HierarchicalLinkageStatus::INVALID_INPUT);
      }
      correlation(row, col) =
          std::clamp(correlation(row, col), -1.0, 1.0);
    }
  }
  correlation = 0.5 * (correlation + correlation.transpose()).eval();
  for (Eigen::Index index = 0;
       index < static_cast<Eigen::Index>(asset_count); ++index) {
    correlation(index, index) = 1.0;
  }
  if (!quant_math::is_positive_semidefinite(correlation,
                                             spec.psd_tolerance)) {
    return failed(HierarchicalLinkageStatus::INVALID_INPUT);
  }

  HierarchicalLinkageResult result;
  auto& diagnostics = result.diagnostics;
  diagnostics.method = spec.method;
  diagnostics.input_correlation_hash = matrix_hash(correlation);
  diagnostics.pairwise_distances.assign(asset_count * asset_count, 0.0);
  for (std::size_t row = 0; row < asset_count; ++row) {
    for (std::size_t col = row + 1; col < asset_count; ++col) {
      const double distance = std::sqrt(
          std::max(0.0, (1.0 - correlation(static_cast<Eigen::Index>(row),
                                            static_cast<Eigen::Index>(col))) /
                             2.0));
      if (!std::isfinite(distance)) {
        return failed(HierarchicalLinkageStatus::NUMERICAL_FAILURE);
      }
      diagnostics.pairwise_distances[row * asset_count + col] = distance;
      diagnostics.pairwise_distances[col * asset_count + row] = distance;
    }
  }

  if (asset_count > 1) {
    std::vector<std::vector<std::uint32_t>> members(asset_count);
    for (std::size_t index = 0; index < asset_count; ++index) {
      members[index].push_back(static_cast<std::uint32_t>(index));
    }
    std::vector<std::uint32_t> active(asset_count);
    std::iota(active.begin(), active.end(), 0U);
    for (std::size_t merge_index = 0; merge_index + 1 < asset_count;
         ++merge_index) {
      double best_distance = std::numeric_limits<double>::infinity();
      std::uint32_t best_left = 0;
      std::uint32_t best_right = 0;
      bool found = false;
      for (std::size_t left_index = 0; left_index < active.size();
           ++left_index) {
        for (std::size_t right_index = left_index + 1;
             right_index < active.size(); ++right_index) {
          const auto left = active[left_index];
          const auto right = active[right_index];
          double distance = 0.0;
          for (const auto left_member : members[left]) {
            for (const auto right_member : members[right]) {
              distance = std::max(
                  distance,
                  diagnostics.pairwise_distances[
                      static_cast<std::size_t>(left_member) * asset_count +
                      static_cast<std::size_t>(right_member)]);
            }
          }
          const auto ordered_left = std::min(left, right);
          const auto ordered_right = std::max(left, right);
          const bool better =
              !found || distance < best_distance - spec.distance_tolerance ||
              (std::abs(distance - best_distance) <=
                   spec.distance_tolerance &&
               merge_precedes(ordered_left, ordered_right, best_left,
                              best_right));
          if (better) {
            best_distance = distance;
            best_left = ordered_left;
            best_right = ordered_right;
            found = true;
          }
        }
      }
      if (!found || !std::isfinite(best_distance)) {
        return failed(HierarchicalLinkageStatus::NUMERICAL_FAILURE);
      }
      std::vector<std::uint32_t> merged_members = members[best_left];
      merged_members.insert(merged_members.end(), members[best_right].begin(),
                            members[best_right].end());
      std::sort(merged_members.begin(), merged_members.end());
      const auto merged_id = static_cast<std::uint32_t>(members.size());
      members.push_back(std::move(merged_members));
      diagnostics.merge_tree.push_back(
          {best_left, best_right,
           static_cast<std::uint32_t>(members[merged_id].size()),
           best_distance});
      active.erase(std::remove(active.begin(), active.end(), best_left),
                   active.end());
      active.erase(std::remove(active.begin(), active.end(), best_right),
                   active.end());
      active.push_back(merged_id);
      std::sort(active.begin(), active.end());
    }
  }

  if (asset_count == 1) {
    diagnostics.quasi_diagonal_order = {0};
  } else {
    append_quasi_order(static_cast<std::uint32_t>(2 * asset_count - 2),
                       asset_count, diagnostics.merge_tree,
                       diagnostics.quasi_diagonal_order);
  }
  diagnostics.linkage_tree_hash =
      linkage_hash(diagnostics.merge_tree, diagnostics.quasi_diagonal_order);
  diagnostics.eligible_for_official_risk = false;
  result.status = HierarchicalLinkageStatus::OK;
  return result;
}

}  // namespace portfolio_math

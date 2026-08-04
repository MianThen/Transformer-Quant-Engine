#include "portfolio_math/hrp_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace portfolio_math {
namespace {

using quant_math::DenseMatrix;

DenseMatrix copy_and_symmetrize(quant_math::MatrixView source) {
  DenseMatrix result(static_cast<Eigen::Index>(source.rows),
                     static_cast<Eigen::Index>(source.cols));
  for (std::size_t row = 0; row < source.rows; ++row) {
    for (std::size_t col = 0; col < source.cols; ++col) {
      result(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          source(row, col);
    }
  }
  return 0.5 * (result + result.transpose()).eval();
}

std::uint64_t append_hash(std::uint64_t hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8)) & 0xffU;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::uint64_t order_hash(std::span<const std::uint32_t> order) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash = append_hash(hash, static_cast<std::uint64_t>(order.size()));
  for (const auto value : order) hash = append_hash(hash, value);
  return hash;
}

HrpPolicyResult failed(OptimizationStatus status) {
  HrpPolicyResult result;
  result.diagnostics.status = status;
  return result;
}

bool valid_order(std::span<const std::uint32_t> order, std::size_t asset_count) {
  if (order.size() != asset_count) return false;
  std::vector<bool> seen(asset_count, false);
  for (const auto index : order) {
    if (index >= asset_count || seen[index]) return false;
    seen[index] = true;
  }
  return true;
}

double inverse_variance_cluster(const DenseMatrix& covariance,
                                std::span<const std::uint32_t> order,
                                std::size_t begin,
                                std::size_t end,
                                double diagonal_tolerance) {
  double inverse_diagonal_sum = 0.0;
  for (std::size_t position = begin; position < end; ++position) {
    const auto asset = static_cast<Eigen::Index>(order[position]);
    const double diagonal = covariance(asset, asset);
    if (!(diagonal > diagonal_tolerance) || !std::isfinite(diagonal)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    inverse_diagonal_sum += 1.0 / diagonal;
  }
  if (!(inverse_diagonal_sum > 0.0) || !std::isfinite(inverse_diagonal_sum)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double variance = 0.0;
  for (std::size_t row_position = begin; row_position < end; ++row_position) {
    const auto row = static_cast<Eigen::Index>(order[row_position]);
    const double row_weight =
        (1.0 / covariance(row, row)) / inverse_diagonal_sum;
    for (std::size_t col_position = begin; col_position < end; ++col_position) {
      const auto col = static_cast<Eigen::Index>(order[col_position]);
      const double col_weight =
          (1.0 / covariance(col, col)) / inverse_diagonal_sum;
      variance += row_weight * covariance(row, col) * col_weight;
    }
  }
  return variance;
}

}  // namespace

bool valid_hrp_policy_options(const HrpPolicyOptions& options) noexcept {
  return std::isfinite(options.symmetry_tolerance) &&
         options.symmetry_tolerance >= 0.0 &&
         std::isfinite(options.psd_tolerance) && options.psd_tolerance > 0.0 &&
         std::isfinite(options.diagonal_tolerance) &&
         options.diagonal_tolerance >= 0.0 &&
         std::isfinite(options.target_investment) &&
         options.target_investment > 0.0 && options.target_investment <= 1.0;
}

HrpPolicyResult solve_hrp_policy(
    quant_math::MatrixView official_covariance,
    std::span<const std::uint32_t> quasi_diagonal_order,
    HrpPolicyOptions options) {
  if (!valid_hrp_policy_options(options) || official_covariance.rows == 0 ||
      official_covariance.rows != official_covariance.cols ||
      !quant_math::validate_symmetric(
           official_covariance, options.symmetry_tolerance)
           .ok ||
      !valid_order(quasi_diagonal_order, official_covariance.rows)) {
    return failed(OptimizationStatus::INVALID_INPUT);
  }

  const DenseMatrix covariance = copy_and_symmetrize(official_covariance);
  if (!quant_math::is_positive_semidefinite(covariance, options.psd_tolerance)) {
    return failed(OptimizationStatus::NON_PSD_RISK_MODEL);
  }
  for (Eigen::Index index = 0; index < covariance.rows(); ++index) {
    if (!(covariance(index, index) > options.diagonal_tolerance) ||
        !std::isfinite(covariance(index, index))) {
      return failed(OptimizationStatus::INVALID_INPUT);
    }
  }

  HrpPolicyResult result;
  auto& diagnostics = result.diagnostics;
  diagnostics.quasi_diagonal_order_hash = order_hash(quasi_diagonal_order);
  result.weights.assign(official_covariance.rows, 1.0);
  std::vector<std::pair<std::size_t, std::size_t>> clusters{{0, official_covariance.rows}};

  while (!clusters.empty()) {
    std::vector<std::pair<std::size_t, std::size_t>> next_clusters;
    next_clusters.reserve(clusters.size() * 2);
    for (const auto [begin, end] : clusters) {
      if (end - begin <= 1) continue;
      const std::size_t midpoint = begin + (end - begin) / 2;
      const double left_variance = inverse_variance_cluster(
          covariance, quasi_diagonal_order, begin, midpoint,
          options.diagonal_tolerance);
      const double right_variance = inverse_variance_cluster(
          covariance, quasi_diagonal_order, midpoint, end,
          options.diagonal_tolerance);
      const double denominator = left_variance + right_variance;
      if (!(left_variance > options.psd_tolerance) ||
          !(right_variance > options.psd_tolerance) ||
          !(denominator > options.psd_tolerance) ||
          !std::isfinite(denominator)) {
        return failed(OptimizationStatus::NUMERICAL_FAILURE);
      }
      const double left_allocation = right_variance / denominator;
      const double right_allocation = left_variance / denominator;
      if (!std::isfinite(left_allocation) || !std::isfinite(right_allocation) ||
          left_allocation < 0.0 || right_allocation < 0.0) {
        return failed(OptimizationStatus::NUMERICAL_FAILURE);
      }
      for (std::size_t position = begin; position < midpoint; ++position) {
        result.weights[quasi_diagonal_order[position]] *= left_allocation;
      }
      for (std::size_t position = midpoint; position < end; ++position) {
        result.weights[quasi_diagonal_order[position]] *= right_allocation;
      }
      diagnostics.bisection_steps.push_back(
          {static_cast<std::uint32_t>(begin), static_cast<std::uint32_t>(midpoint),
           static_cast<std::uint32_t>(midpoint), static_cast<std::uint32_t>(end),
           left_variance, right_variance, left_allocation, right_allocation});
      next_clusters.push_back({begin, midpoint});
      next_clusters.push_back({midpoint, end});
    }
    clusters = std::move(next_clusters);
  }

  const double sum = std::accumulate(result.weights.begin(), result.weights.end(), 0.0);
  if (!(sum > 0.0) || !std::isfinite(sum)) return failed(OptimizationStatus::NUMERICAL_FAILURE);
  for (double& weight : result.weights) {
    weight *= options.target_investment / sum;
    if (!std::isfinite(weight) || weight < 0.0) {
      return failed(OptimizationStatus::NUMERICAL_FAILURE);
    }
  }
  double variance = 0.0;
  for (std::size_t row = 0; row < result.weights.size(); ++row) {
    for (std::size_t col = 0; col < result.weights.size(); ++col) {
      variance += result.weights[row] * covariance(static_cast<Eigen::Index>(row),
                                                   static_cast<Eigen::Index>(col)) *
                  result.weights[col];
    }
  }
  const double variance_scale = std::max(1.0, covariance.diagonal().cwiseAbs().maxCoeff());
  if (!std::isfinite(variance) || variance < -options.psd_tolerance * variance_scale) {
    return failed(OptimizationStatus::NUMERICAL_FAILURE);
  }
  diagnostics.predicted_risk = std::sqrt(std::max(0.0, variance));
  diagnostics.weight_sum = std::accumulate(result.weights.begin(), result.weights.end(), 0.0);
  diagnostics.recursive_bisection_steps =
      static_cast<std::uint32_t>(diagnostics.bisection_steps.size());
  diagnostics.status = OptimizationStatus::OK;
  diagnostics.eligible_for_official_risk = false;
  return result;
}

}  // namespace portfolio_math

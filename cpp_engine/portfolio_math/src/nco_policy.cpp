#include "portfolio_math/nco_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include "portfolio_math/risk_budget.h"

namespace portfolio_math {
namespace {

using quant_math::DenseMatrix;
using quant_math::DenseVector;

struct SimplexSolve {
  bool ok{false};
  std::vector<double> weights;
  std::uint32_t iterations{0};
  double residual{0.0};
};

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

DenseMatrix extract_submatrix(const DenseMatrix& source,
                              const std::vector<std::size_t>& indices) {
  DenseMatrix result(static_cast<Eigen::Index>(indices.size()),
                     static_cast<Eigen::Index>(indices.size()));
  for (std::size_t row = 0; row < indices.size(); ++row) {
    for (std::size_t col = 0; col < indices.size(); ++col) {
      result(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          source(static_cast<Eigen::Index>(indices[row]),
                 static_cast<Eigen::Index>(indices[col]));
    }
  }
  return result;
}

DenseVector project_simplex(const DenseVector& values, double target) {
  std::vector<double> sorted(values.data(), values.data() + values.size());
  std::sort(sorted.begin(), sorted.end(), std::greater<double>());
  double cumulative = 0.0;
  Eigen::Index rho = -1;
  for (Eigen::Index index = 0; index < values.size(); ++index) {
    cumulative += sorted[static_cast<std::size_t>(index)];
    const double threshold =
        (cumulative - target) / static_cast<double>(index + 1);
    if (sorted[static_cast<std::size_t>(index)] > threshold) rho = index;
  }
  if (rho < 0) return DenseVector::Constant(values.size(), target / values.size());
  cumulative = 0.0;
  for (Eigen::Index index = 0; index <= rho; ++index) {
    cumulative += sorted[static_cast<std::size_t>(index)];
  }
  const double theta =
      (cumulative - target) / static_cast<double>(rho + 1);
  return (values.array() - theta).max(0.0);
}

SimplexSolve solve_minvar(const DenseMatrix& covariance,
                          double target,
                          std::uint32_t max_iterations,
                          double tolerance) {
  if (covariance.rows() == 0 || covariance.rows() != covariance.cols() ||
      !(target > 0.0) || !std::isfinite(target)) {
    return {};
  }
  const Eigen::Index count = covariance.rows();
  if (count == 1) return {true, {target}, 1, 0.0};
  double lipschitz = 0.0;
  for (Eigen::Index row = 0; row < count; ++row) {
    double row_sum = 0.0;
    for (Eigen::Index col = 0; col < count; ++col) {
      row_sum += std::abs(covariance(row, col));
    }
    lipschitz = std::max(lipschitz, row_sum);
  }
  if (!(lipschitz > 0.0) || !std::isfinite(lipschitz)) return {};
  const double step = 1.0 / lipschitz;
  DenseVector weights = DenseVector::Constant(count, target / count);
  double residual = std::numeric_limits<double>::infinity();
  for (std::uint32_t iteration = 1; iteration <= max_iterations; ++iteration) {
    const DenseVector gradient = covariance * weights;
    if (!std::isfinite(gradient.squaredNorm())) return {};
    const DenseVector projected = project_simplex(weights - step * gradient, target);
    residual = (projected - weights).cwiseAbs().maxCoeff();
    weights = projected;
    if (residual <= tolerance * std::max(1.0, weights.maxCoeff())) {
      return {true, std::vector<double>(weights.data(), weights.data() + count),
              iteration, residual};
    }
  }
  return {false, {}, max_iterations, residual};
}

NcoPolicyResult failed(OptimizationStatus status) {
  NcoPolicyResult result;
  result.diagnostics.status = status;
  return result;
}

bool valid_partition(std::span<const std::uint32_t> cluster_ids,
                     std::uint32_t cluster_count,
                     std::vector<std::vector<std::size_t>>& members) {
  if (cluster_count == 0 || cluster_ids.empty()) return false;
  members.assign(cluster_count, {});
  for (std::size_t index = 0; index < cluster_ids.size(); ++index) {
    const auto cluster = cluster_ids[index];
    if (cluster >= cluster_count) return false;
    members[cluster].push_back(index);
  }
  return std::all_of(members.begin(), members.end(),
                     [](const auto& values) { return !values.empty(); });
}

bool valid_covariance(const DenseMatrix& covariance, const NcoPolicyOptions& options) {
  if (!quant_math::validate_symmetric(quant_math::view(covariance),
                                      options.symmetry_tolerance)
           .ok ||
      !quant_math::is_positive_semidefinite(covariance, options.psd_tolerance)) {
    return false;
  }
  for (Eigen::Index index = 0; index < covariance.rows(); ++index) {
    if (!(covariance(index, index) > options.psd_tolerance) ||
        !std::isfinite(covariance(index, index))) {
      return false;
    }
  }
  return true;
}

DenseMatrix cluster_covariance(const DenseMatrix& covariance,
                               const std::vector<std::vector<std::size_t>>& members,
                               const std::vector<std::vector<double>>& local_weights) {
  const auto asset_count = static_cast<std::size_t>(covariance.rows());
  const auto cluster_count = members.size();
  DenseMatrix portfolios = DenseMatrix::Zero(
      static_cast<Eigen::Index>(asset_count),
      static_cast<Eigen::Index>(cluster_count));
  for (std::size_t cluster = 0; cluster < cluster_count; ++cluster) {
    for (std::size_t index = 0; index < members[cluster].size(); ++index) {
      portfolios(static_cast<Eigen::Index>(members[cluster][index]),
                 static_cast<Eigen::Index>(cluster)) = local_weights[cluster][index];
    }
  }
  return portfolios.transpose() * covariance * portfolios;
}

double portfolio_risk(const DenseMatrix& covariance,
                      std::span<const double> weights) {
  const Eigen::Map<const DenseVector> vector(
      weights.data(), static_cast<Eigen::Index>(weights.size()));
  const double variance = vector.dot(covariance * vector);
  if (!std::isfinite(variance) || variance < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::sqrt(variance);
}

NcoPolicyResult solve_impl(quant_math::MatrixView official_covariance,
                           std::span<const std::uint32_t> cluster_ids,
                           std::uint32_t cluster_count,
                           std::span<const double> cluster_risk_budgets,
                           bool risk_budget_mode,
                           NcoPolicyOptions options) {
  if (!valid_nco_policy_options(options) || official_covariance.rows == 0 ||
      official_covariance.rows != official_covariance.cols ||
      official_covariance.rows != cluster_ids.size()) {
    return failed(OptimizationStatus::INVALID_INPUT);
  }
  if (risk_budget_mode && cluster_risk_budgets.size() != cluster_count) {
    return failed(OptimizationStatus::INVALID_INPUT);
  }
  std::vector<std::vector<std::size_t>> members;
  if (!valid_partition(cluster_ids, cluster_count, members)) {
    return failed(OptimizationStatus::INVALID_INPUT);
  }
  if (risk_budget_mode) {
    double budget_sum = 0.0;
    for (const double value : cluster_risk_budgets) {
      if (!std::isfinite(value) || value < 0.0) return failed(OptimizationStatus::INVALID_INPUT);
      budget_sum += value;
    }
    if (!(budget_sum > 0.0) ||
        std::abs(budget_sum - 1.0) > options.tolerance * 10.0) {
      return failed(OptimizationStatus::INVALID_INPUT);
    }
  }
  const DenseMatrix covariance = copy_matrix(official_covariance);
  if (!valid_covariance(covariance, options)) {
    return failed(OptimizationStatus::NON_PSD_RISK_MODEL);
  }

  NcoPolicyResult result;
  auto& diagnostics = result.diagnostics;
  diagnostics.cluster_count = cluster_count;
  diagnostics.cluster_sizes.reserve(cluster_count);
  for (const auto& cluster : members) {
    diagnostics.cluster_sizes.push_back(static_cast<std::uint32_t>(cluster.size()));
  }
  std::vector<std::vector<double>> local_weights(cluster_count);
  for (std::size_t cluster = 0; cluster < cluster_count; ++cluster) {
    const DenseMatrix local_covariance = extract_submatrix(covariance, members[cluster]);
    if (risk_budget_mode) {
      std::vector<double> budgets(members[cluster].size(),
                                  1.0 / static_cast<double>(members[cluster].size()));
      RiskBudgetOptions budget_options;
      budget_options.max_iterations = options.max_iterations;
      budget_options.tolerance = options.tolerance;
      budget_options.covariance_tolerance = options.psd_tolerance;
      const auto local = solve_long_only_risk_budget(
          quant_math::view(local_covariance), budgets, {}, {}, budget_options);
      if (local.diagnostics.status != OptimizationStatus::OK) {
        return failed(local.diagnostics.status);
      }
      diagnostics.intra_cluster_iterations = std::max(
          diagnostics.intra_cluster_iterations, local.diagnostics.iterations);
      diagnostics.max_risk_budget_error = std::max(
          diagnostics.max_risk_budget_error, local.diagnostics.max_risk_budget_error);
      local_weights[cluster] = local.weights;
    } else {
      const auto local = solve_minvar(local_covariance, 1.0,
                                      options.max_iterations, options.tolerance);
      if (!local.ok) {
        return failed(OptimizationStatus::MAX_ITERATIONS);
      }
      diagnostics.intra_cluster_iterations =
          std::max(diagnostics.intra_cluster_iterations, local.iterations);
      diagnostics.kkt_residual = std::max(diagnostics.kkt_residual, local.residual);
      local_weights[cluster] = local.weights;
    }
  }

  const DenseMatrix inter_covariance =
      cluster_covariance(covariance, members, local_weights);
  std::vector<double> cluster_weights;
  if (risk_budget_mode) {
    RiskBudgetOptions budget_options;
    budget_options.max_iterations = options.max_iterations;
    budget_options.tolerance = options.tolerance;
    budget_options.covariance_tolerance = options.psd_tolerance;
    budget_options.target_investment = options.target_investment;
    const auto inter = solve_long_only_risk_budget(
        quant_math::view(inter_covariance), cluster_risk_budgets, {}, {},
        budget_options);
    if (inter.diagnostics.status != OptimizationStatus::OK) {
      return failed(inter.diagnostics.status);
    }
    diagnostics.inter_cluster_iterations = inter.diagnostics.iterations;
    diagnostics.max_risk_budget_error = std::max(
        diagnostics.max_risk_budget_error, inter.diagnostics.max_risk_budget_error);
    cluster_weights = inter.weights;
  } else {
    const auto inter = solve_minvar(inter_covariance, options.target_investment,
                                    options.max_iterations, options.tolerance);
    if (!inter.ok) return failed(OptimizationStatus::MAX_ITERATIONS);
    diagnostics.inter_cluster_iterations = inter.iterations;
    diagnostics.kkt_residual = std::max(diagnostics.kkt_residual, inter.residual);
    cluster_weights = inter.weights;
  }
  diagnostics.cluster_weights = cluster_weights;
  result.weights.assign(cluster_ids.size(), 0.0);
  for (std::size_t cluster = 0; cluster < cluster_count; ++cluster) {
    for (std::size_t index = 0; index < members[cluster].size(); ++index) {
      result.weights[members[cluster][index]] =
          cluster_weights[cluster] * local_weights[cluster][index];
    }
  }
  diagnostics.weight_sum = std::accumulate(result.weights.begin(), result.weights.end(), 0.0);
  diagnostics.predicted_risk = portfolio_risk(covariance, result.weights);
  if (!(diagnostics.weight_sum > 0.0) ||
      !std::isfinite(diagnostics.weight_sum) ||
      !std::isfinite(diagnostics.predicted_risk)) {
    return failed(OptimizationStatus::NUMERICAL_FAILURE);
  }
  diagnostics.status = OptimizationStatus::OK;
  diagnostics.eligible_for_official_risk = false;
  return result;
}

}  // namespace

bool valid_nco_policy_options(const NcoPolicyOptions& options) noexcept {
  return options.max_iterations > 0 && std::isfinite(options.tolerance) &&
         options.tolerance > 0.0 && std::isfinite(options.symmetry_tolerance) &&
         options.symmetry_tolerance >= 0.0 && std::isfinite(options.psd_tolerance) &&
         options.psd_tolerance > 0.0 && std::isfinite(options.target_investment) &&
         options.target_investment > 0.0 && options.target_investment <= 1.0;
}

NcoPolicyResult solve_nco_minvar(
    quant_math::MatrixView official_covariance,
    std::span<const std::uint32_t> cluster_id_by_symbol,
    std::uint32_t cluster_count,
    NcoPolicyOptions options) {
  return solve_impl(official_covariance, cluster_id_by_symbol, cluster_count,
                    {}, false, options);
}

NcoPolicyResult solve_nco_risk_budget(
    quant_math::MatrixView official_covariance,
    std::span<const std::uint32_t> cluster_id_by_symbol,
    std::uint32_t cluster_count,
    std::span<const double> cluster_risk_budgets,
    NcoPolicyOptions options) {
  return solve_impl(official_covariance, cluster_id_by_symbol, cluster_count,
                    cluster_risk_budgets, true, options);
}

}  // namespace portfolio_math

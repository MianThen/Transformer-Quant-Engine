#include "portfolio_math/posterior_direct.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <Eigen/Eigenvalues>

namespace portfolio_math {
namespace {

bool finite_nonnegative(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
}

bool project_capped_simplex(std::span<const double> values, double target,
                            double cap, std::vector<double>& projected) {
  if (values.empty() || !std::isfinite(target) || !std::isfinite(cap) ||
      target <= 0.0 || cap <= 0.0 ||
      target > cap * static_cast<double>(values.size()) + 1e-12) {
    return false;
  }
  double left = *std::min_element(values.begin(), values.end()) - cap;
  double right = *std::max_element(values.begin(), values.end());
  for (int iteration = 0; iteration < 128; ++iteration) {
    const double shift = 0.5 * (left + right);
    double sum = 0.0;
    for (double value : values) {
      sum += std::clamp(value - shift, 0.0, cap);
    }
    if (sum > target) {
      left = shift;
    } else {
      right = shift;
    }
  }
  const double shift = 0.5 * (left + right);
  projected.resize(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    projected[index] = std::clamp(values[index] - shift, 0.0, cap);
  }
  const double sum = std::accumulate(projected.begin(), projected.end(), 0.0);
  if (!std::isfinite(sum) || std::abs(sum - target) > 1e-10) return false;
  return finite_nonnegative(projected);
}

}  // namespace

bool valid_posterior_direct_options(
    const PosteriorDirectOptions& options) noexcept {
  return options.max_iterations > 0 && std::isfinite(options.tolerance) &&
         options.tolerance > 0.0 && std::isfinite(options.risk_aversion) &&
         options.risk_aversion > 0.0 && std::isfinite(options.target_investment) &&
         options.target_investment > 0.0 && options.target_investment <= 1.0 &&
         std::isfinite(options.max_single_weight) &&
         options.max_single_weight > 0.0 && options.max_single_weight <= 1.0;
}

PosteriorDirectResult solve_posterior_direct(
    const PosteriorScenarioArtifactV1& posterior,
    PosteriorDirectOptions options) {
  PosteriorDirectResult result;
  result.diagnostics.posterior_artifact_hash = posterior.artifact_hash;
  if (!valid_posterior_direct_options(options) ||
      !valid_posterior_scenario_artifact(posterior) ||
      posterior.posterior_mean.empty() ||
      options.target_investment >
          options.max_single_weight *
              static_cast<double>(posterior.asset_count) + 1e-12) {
    return result;
  }
  const Eigen::Index dimensions = static_cast<Eigen::Index>(posterior.asset_count);
  const Eigen::VectorXd mean = Eigen::Map<const Eigen::VectorXd>(
      posterior.posterior_mean.data(), dimensions);
  const Eigen::MatrixXd covariance = posterior.posterior_covariance;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(covariance);
  if (eigen_solver.info() != Eigen::Success ||
      !eigen_solver.eigenvalues().allFinite()) {
    return result;
  }
  const double maximum_eigenvalue =
      std::max(0.0, eigen_solver.eigenvalues().maxCoeff());
  const double lipschitz =
      std::max(1.0, options.risk_aversion * maximum_eigenvalue);
  const double step = 1.0 / lipschitz;
  std::vector<double> initial(posterior.asset_count,
                              options.target_investment /
                                  static_cast<double>(posterior.asset_count));
  std::vector<double> weights;
  if (!project_capped_simplex(initial, options.target_investment,
                              options.max_single_weight, weights)) {
    return result;
  }
  double maximum_change = std::numeric_limits<double>::infinity();
  for (std::uint32_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
    const Eigen::VectorXd current = Eigen::Map<const Eigen::VectorXd>(
        weights.data(), dimensions);
    const Eigen::VectorXd gradient = options.risk_aversion * covariance * current - mean;
    if (!gradient.allFinite()) return result;
    std::vector<double> unconstrained(weights.size());
    for (std::size_t index = 0; index < weights.size(); ++index) {
      unconstrained[index] = weights[index] - step * gradient(
          static_cast<Eigen::Index>(index));
    }
    std::vector<double> projected;
    if (!project_capped_simplex(unconstrained, options.target_investment,
                                options.max_single_weight, projected)) {
      return result;
    }
    maximum_change = 0.0;
    for (std::size_t index = 0; index < weights.size(); ++index) {
      maximum_change = std::max(maximum_change,
                                std::abs(projected[index] - weights[index]));
    }
    weights = std::move(projected);
    result.diagnostics.iterations = iteration;
    if (maximum_change <= options.tolerance) break;
  }
  if (!std::isfinite(maximum_change) || maximum_change > options.tolerance) {
    result.diagnostics.status = OptimizationStatus::MAX_ITERATIONS;
    return result;
  }
  const Eigen::VectorXd final_weights = Eigen::Map<const Eigen::VectorXd>(
      weights.data(), dimensions);
  const double expected_return = mean.dot(final_weights);
  const double variance = (final_weights.transpose() * covariance * final_weights)(0, 0);
  const double objective = expected_return -
                           0.5 * options.risk_aversion * variance;
  if (!std::isfinite(expected_return) || !std::isfinite(variance) || variance < -1e-12 ||
      !std::isfinite(objective)) {
    return result;
  }
  result.weights = std::move(weights);
  result.diagnostics.status = OptimizationStatus::OK;
  result.diagnostics.kkt_residual = maximum_change;
  result.diagnostics.weight_sum = std::accumulate(
      result.weights.begin(), result.weights.end(), 0.0);
  result.diagnostics.expected_return = expected_return;
  result.diagnostics.variance = std::max(0.0, variance);
  result.diagnostics.objective = objective;
  result.diagnostics.eligible_for_official_risk = false;
  return result;
}

PosteriorDirectPolicyComparison compare_posterior_direct_policies(
    const PosteriorScenarioArtifactV1& gaussian_bl,
    const PosteriorScenarioArtifactV1& fully_flexible_views,
    PosteriorDirectOptions options) {
  PosteriorDirectPolicyComparison comparison;
  comparison.gaussian_artifact_hash = gaussian_bl.artifact_hash;
  comparison.fully_flexible_artifact_hash = fully_flexible_views.artifact_hash;
  if (gaussian_bl.asset_count != fully_flexible_views.asset_count ||
      gaussian_bl.asset_count == 0 ||
      gaussian_bl.scenario_count != fully_flexible_views.scenario_count) {
    return comparison;
  }
  comparison.gaussian_bl = solve_posterior_direct(gaussian_bl, options);
  comparison.fully_flexible_views =
      solve_posterior_direct(fully_flexible_views, options);
  if (comparison.gaussian_bl.diagnostics.status != OptimizationStatus::OK ||
      comparison.fully_flexible_views.diagnostics.status != OptimizationStatus::OK ||
      comparison.gaussian_bl.weights.size() != comparison.fully_flexible_views.weights.size()) {
    comparison.status = OptimizationStatus::NUMERICAL_FAILURE;
    return comparison;
  }
  comparison.expected_return_delta =
      comparison.fully_flexible_views.diagnostics.expected_return -
      comparison.gaussian_bl.diagnostics.expected_return;
  comparison.variance_delta =
      comparison.fully_flexible_views.diagnostics.variance -
      comparison.gaussian_bl.diagnostics.variance;
  comparison.objective_delta =
      comparison.fully_flexible_views.diagnostics.objective -
      comparison.gaussian_bl.diagnostics.objective;
  for (std::size_t index = 0; index < comparison.gaussian_bl.weights.size(); ++index) {
    comparison.weight_l1_distance += std::abs(
        comparison.fully_flexible_views.weights[index] -
        comparison.gaussian_bl.weights[index]);
  }
  if (!std::isfinite(comparison.expected_return_delta) ||
      !std::isfinite(comparison.variance_delta) ||
      !std::isfinite(comparison.objective_delta) ||
      !std::isfinite(comparison.weight_l1_distance)) {
    comparison.status = OptimizationStatus::NUMERICAL_FAILURE;
    return comparison;
  }
  comparison.status = OptimizationStatus::OK;
  comparison.winner_selected = false;
  return comparison;
}

}  // namespace portfolio_math

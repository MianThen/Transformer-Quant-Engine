#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "portfolio_math/nco_policy.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

bool near(double left, double right, double tolerance = 1e-8) {
  return std::abs(left - right) <= tolerance;
}

quant_math::DenseMatrix block_covariance() {
  quant_math::DenseMatrix covariance = quant_math::DenseMatrix::Zero(4, 4);
  covariance.diagonal() << 1.0, 4.0, 1.0, 4.0;
  covariance(0, 1) = covariance(1, 0) = 0.2;
  covariance(2, 3) = covariance(3, 2) = 0.2;
  covariance(0, 2) = covariance(2, 0) = 0.05;
  covariance(0, 3) = covariance(3, 0) = 0.02;
  covariance(1, 2) = covariance(2, 1) = 0.02;
  covariance(1, 3) = covariance(3, 1) = 0.05;
  return covariance;
}

bool test_minvar_and_cluster_structure() {
  const auto covariance = block_covariance();
  const std::vector<std::uint32_t> clusters{0, 0, 1, 1};
  const auto result = portfolio_math::solve_nco_minvar(
      quant_math::view(covariance), clusters, 2);
  bool ok = check(result.diagnostics.status == portfolio_math::OptimizationStatus::OK,
                  "NCO minvar status");
  ok &= check(result.weights.size() == 4 && result.diagnostics.cluster_count == 2,
              "NCO minvar shape");
  ok &= check(near(result.diagnostics.weight_sum, 1.0) &&
                  std::all_of(result.weights.begin(), result.weights.end(),
                              [](double value) { return value >= 0.0; }),
              "NCO minvar simplex");
  ok &= check(near(result.weights[0], result.weights[2]) &&
                  near(result.weights[1], result.weights[3]),
              "symmetric cluster composition");
  ok &= check(result.diagnostics.predicted_risk > 0.0 &&
                  !result.diagnostics.eligible_for_official_risk,
              "NCO minvar research boundary");
  return ok;
}

bool test_risk_budget_and_contributions() {
  const auto covariance = block_covariance();
  const std::vector<std::uint32_t> clusters{0, 0, 1, 1};
  const std::vector<double> budgets{0.25, 0.75};
  const auto result = portfolio_math::solve_nco_risk_budget(
      quant_math::view(covariance), clusters, 2, budgets);
  bool ok = check(result.diagnostics.status == portfolio_math::OptimizationStatus::OK,
                  "NCO risk budget status");
  ok &= check(near(result.diagnostics.weight_sum, 1.0) &&
                  result.diagnostics.max_risk_budget_error < 1e-6,
              "NCO risk budget diagnostics");
  ok &= check(near(result.diagnostics.cluster_weights[0] +
                       result.diagnostics.cluster_weights[1],
                   1.0) &&
                  result.diagnostics.cluster_weights[1] >
                      result.diagnostics.cluster_weights[0],
              "NCO cluster budget allocation");
  return ok;
}

bool test_permutation_and_failures() {
  const auto covariance = block_covariance();
  const std::vector<std::uint32_t> clusters{0, 0, 1, 1};
  const auto original = portfolio_math::solve_nco_minvar(
      quant_math::view(covariance), clusters, 2);
  const std::vector<std::size_t> permutation{2, 0, 3, 1};
  quant_math::DenseMatrix permuted(4, 4);
  for (std::size_t row = 0; row < permutation.size(); ++row) {
    for (std::size_t col = 0; col < permutation.size(); ++col) {
      permuted(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          covariance(static_cast<Eigen::Index>(permutation[row]),
                     static_cast<Eigen::Index>(permutation[col]));
    }
  }
  std::vector<std::uint32_t> permuted_clusters;
  for (const auto index : permutation) permuted_clusters.push_back(clusters[index]);
  const auto permuted_result = portfolio_math::solve_nco_minvar(
      quant_math::view(permuted), permuted_clusters, 2);
  bool ok = check(original.diagnostics.status == portfolio_math::OptimizationStatus::OK &&
                      permuted_result.diagnostics.status ==
                          portfolio_math::OptimizationStatus::OK,
                  "NCO permutation status");
  for (std::size_t index = 0; index < permutation.size(); ++index) {
    ok &= check(near(permuted_result.weights[index], original.weights[permutation[index]]),
                "NCO permutation equivariance");
  }
  const std::vector<std::uint32_t> missing_cluster{0, 0, 0, 0};
  ok &= check(portfolio_math::solve_nco_minvar(
                  quant_math::view(covariance), missing_cluster, 2)
                  .diagnostics.status == portfolio_math::OptimizationStatus::INVALID_INPUT,
              "empty cluster fails closed");
  const std::vector<double> invalid_budgets{0.5, 0.4};
  ok &= check(portfolio_math::solve_nco_risk_budget(
                  quant_math::view(covariance), clusters, 2, invalid_budgets)
                  .diagnostics.status == portfolio_math::OptimizationStatus::INVALID_INPUT,
              "invalid budget sum fails closed");
  auto non_psd = covariance;
  non_psd(0, 0) = 0.01;
  non_psd(0, 1) = non_psd(1, 0) = 1.0;
  ok &= check(portfolio_math::solve_nco_minvar(
                  quant_math::view(non_psd), clusters, 2)
                  .diagnostics.status == portfolio_math::OptimizationStatus::NON_PSD_RISK_MODEL,
              "non-PSD covariance fails closed");
  return ok;
}

}  // namespace

int main() {
  if (!(test_minvar_and_cluster_structure() &&
        test_risk_budget_and_contributions() && test_permutation_and_failures())) {
    return 1;
  }
  std::printf("test_nco_policy: all checks passed\n");
  return 0;
}

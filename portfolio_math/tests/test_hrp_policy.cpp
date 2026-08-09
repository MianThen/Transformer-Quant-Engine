#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <vector>

#include "portfolio_math/hrp_policy.h"
#include "portfolio_math/hierarchical_linkage.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

bool near(double left, double right, double tolerance = 1e-10) {
  return std::abs(left - right) <= tolerance;
}

quant_math::DenseMatrix diagonal_covariance() {
  quant_math::DenseMatrix covariance = quant_math::DenseMatrix::Zero(4, 4);
  covariance.diagonal() << 1.0, 4.0, 9.0, 16.0;
  return covariance;
}

bool test_diagonal_oracle_and_composition() {
  const auto covariance = diagonal_covariance();
  const std::vector<std::uint32_t> order{0, 1, 2, 3};
  const auto result = portfolio_math::solve_hrp_policy(
      quant_math::view(covariance), order);
  const double inverse_sum = 1.0 + 0.25 + 1.0 / 9.0 + 1.0 / 16.0;
  const std::vector<double> expected{
      1.0 / inverse_sum, 0.25 / inverse_sum,
      (1.0 / 9.0) / inverse_sum, (1.0 / 16.0) / inverse_sum};
  bool ok = true;
  ok &= check(result.diagnostics.status == portfolio_math::OptimizationStatus::OK,
              "diagonal HRP status");
  ok &= check(result.weights.size() == expected.size() &&
                  result.diagnostics.recursive_bisection_steps == 3,
              "recursive bisection shape");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    ok &= check(near(result.weights[index], expected[index]),
                "diagonal inverse-variance oracle");
  }
  ok &= check(near(result.diagnostics.weight_sum, 1.0) &&
                  near(std::accumulate(result.weights.begin(), result.weights.end(), 0.0),
                       1.0) &&
                  std::all_of(result.weights.begin(), result.weights.end(),
                              [](double weight) { return weight >= 0.0; }),
              "long-only full investment");
  ok &= check(result.diagnostics.bisection_steps[0].left_begin == 0 &&
                  result.diagnostics.bisection_steps[0].left_end == 2 &&
                  result.diagnostics.bisection_steps[0].right_begin == 2 &&
                  result.diagnostics.bisection_steps[0].right_end == 4 &&
                  near(result.diagnostics.bisection_steps[0].left_allocation +
                           result.diagnostics.bisection_steps[0].right_allocation,
                       1.0),
              "root split composition");
  ok &= check(std::isfinite(result.diagnostics.predicted_risk) &&
                  result.diagnostics.predicted_risk > 0.0 &&
                  !result.diagnostics.eligible_for_official_risk,
              "HRP risk diagnostic boundary");
  return ok;
}

bool test_two_block_oracle() {
  quant_math::DenseMatrix covariance = quant_math::DenseMatrix::Zero(4, 4);
  covariance.diagonal() << 1.0, 1.0, 4.0, 4.0;
  covariance(0, 1) = covariance(1, 0) = 0.8;
  covariance(2, 3) = covariance(3, 2) = 2.0;
  const std::vector<std::uint32_t> order{0, 1, 2, 3};
  const auto result = portfolio_math::solve_hrp_policy(
      quant_math::view(covariance), order);
  bool ok = true;
  ok &= check(result.diagnostics.status == portfolio_math::OptimizationStatus::OK,
              "two-block status");
  ok &= check(near(result.weights[0], result.weights[1]) &&
                  near(result.weights[2], result.weights[3]),
              "within-block equal allocation");
  const double left_variance = 0.9;
  const double right_variance = 3.0;
  const double left_sum = right_variance / (left_variance + right_variance);
  ok &= check(near(result.weights[0] + result.weights[1], left_sum) &&
                  near(result.weights[2] + result.weights[3], 1.0 - left_sum),
              "independent block allocation oracle");
  return ok;
}

bool test_permutation_invariance() {
  const auto covariance = diagonal_covariance();
  const std::vector<std::uint32_t> order{0, 1, 2, 3};
  const auto original = portfolio_math::solve_hrp_policy(
      quant_math::view(covariance), order);
  const std::vector<std::uint32_t> permutation{2, 0, 3, 1};
  std::vector<std::uint32_t> inverse(permutation.size());
  for (std::size_t index = 0; index < permutation.size(); ++index) {
    inverse[permutation[index]] = static_cast<std::uint32_t>(index);
  }
  quant_math::DenseMatrix permuted(4, 4);
  for (std::size_t row = 0; row < permutation.size(); ++row) {
    for (std::size_t col = 0; col < permutation.size(); ++col) {
      permuted(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          covariance(static_cast<Eigen::Index>(permutation[row]),
                     static_cast<Eigen::Index>(permutation[col]));
    }
  }
  std::vector<std::uint32_t> permuted_order;
  for (const auto old_index : order) permuted_order.push_back(inverse[old_index]);
  const auto permuted_result = portfolio_math::solve_hrp_policy(
      quant_math::view(permuted), permuted_order);
  bool ok = check(original.diagnostics.status ==
                      portfolio_math::OptimizationStatus::OK &&
                      permuted_result.diagnostics.status ==
                          portfolio_math::OptimizationStatus::OK,
                  "permutation status");
  for (std::size_t index = 0; index < permutation.size(); ++index) {
    ok &= check(near(permuted_result.weights[index], original.weights[permutation[index]]),
                "permutation equivariance");
  }
  return ok;
}

bool test_linkage_order_contract() {
  quant_math::DenseMatrix covariance =
      quant_math::DenseMatrix::Constant(4, 4, 0.1);
  covariance.diagonal().setOnes();
  covariance(0, 1) = covariance(1, 0) = 0.9;
  covariance(2, 3) = covariance(3, 2) = 0.8;
  const auto linkage = portfolio_math::hierarchical_linkage(
      quant_math::view(covariance));
  bool ok = check(linkage.status ==
                      portfolio_math::HierarchicalLinkageStatus::OK &&
                      linkage.diagnostics.quasi_diagonal_order.size() == 4,
                  "linkage produces quasi-diagonal order");
  if (!ok) return false;
  const auto result = portfolio_math::solve_hrp_policy(
      quant_math::view(covariance), linkage.diagnostics.quasi_diagonal_order);
  ok &= check(result.diagnostics.status == portfolio_math::OptimizationStatus::OK &&
                  result.diagnostics.quasi_diagonal_order_hash != 0,
              "HRP consumes linkage order");
  return ok;
}

bool test_failures() {
  const auto covariance = diagonal_covariance();
  const std::vector<std::uint32_t> valid_order{0, 1, 2, 3};
  const std::vector<std::uint32_t> duplicate_order{0, 1, 1, 3};
  const std::vector<std::uint32_t> short_order{0, 1};
  bool ok = true;
  ok &= check(portfolio_math::solve_hrp_policy(
                      quant_math::view(covariance), duplicate_order)
                      .diagnostics.status ==
                  portfolio_math::OptimizationStatus::INVALID_INPUT,
              "duplicate order fails closed");

  auto nonsymmetric = covariance;
  nonsymmetric(0, 1) = 0.5;
  ok &= check(portfolio_math::solve_hrp_policy(
                      quant_math::view(nonsymmetric), valid_order)
                      .diagnostics.status ==
                  portfolio_math::OptimizationStatus::INVALID_INPUT,
              "non-symmetric covariance fails closed");

  quant_math::DenseMatrix non_psd(2, 2);
  non_psd << 1.0, 2.0, 2.0, 1.0;
  ok &= check(portfolio_math::solve_hrp_policy(
                      quant_math::view(non_psd), short_order)
                      .diagnostics.status ==
                  portfolio_math::OptimizationStatus::NON_PSD_RISK_MODEL,
              "non-PSD covariance fails closed");

  auto zero_diagonal = covariance;
  zero_diagonal(0, 0) = 0.0;
  ok &= check(portfolio_math::solve_hrp_policy(
                      quant_math::view(zero_diagonal), valid_order)
                      .diagnostics.status ==
                  portfolio_math::OptimizationStatus::INVALID_INPUT,
              "zero diagonal fails closed");

  auto nonfinite = covariance;
  nonfinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
  ok &= check(portfolio_math::solve_hrp_policy(
                      quant_math::view(nonfinite), valid_order)
                      .diagnostics.status ==
                  portfolio_math::OptimizationStatus::INVALID_INPUT,
              "non-finite covariance fails closed");

  auto invalid_options = portfolio_math::HrpPolicyOptions{};
  invalid_options.target_investment = 0.0;
  ok &= check(!portfolio_math::valid_hrp_policy_options(invalid_options),
              "invalid HRP options fail closed");
  return ok;
}

}  // namespace

int main() {
  if (!(test_diagonal_oracle_and_composition() && test_two_block_oracle() &&
        test_permutation_invariance() && test_linkage_order_contract() &&
        test_failures())) {
    return 1;
  }
  std::printf("test_hrp_policy: all checks passed\n");
  return 0;
}

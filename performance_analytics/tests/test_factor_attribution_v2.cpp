#include "performance_analytics/factor_attribution_v2.h"

#include <cmath>
#include <iostream>
#include <array>

int main() {
  const std::array<double, 6> exposures{1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  const std::vector<double> asset_returns{0.011, -0.018, 0.004};
  const std::vector<double> factor_returns{0.01, -0.02};
  const std::vector<double> weights{0.2, 0.3, 0.5};
  const double accounting = 0.2 * asset_returns[0] +
      0.3 * asset_returns[1] + 0.5 * asset_returns[2];
  performance_analytics::FactorAttributionProblemV2 problem{
      {exposures.data(), 3, 2, 2},
      asset_returns, factor_returns, weights,
      accounting, 10, 10, 1,
      {1e-10, 11, 19, 23},
  };
  const auto result = performance_analytics::compute_factor_attribution_v2(problem);
  bool ok = result.status == performance_analytics::FactorAttributionStatus::OK &&
      result.artifact_hash != 0 &&
      std::abs(result.reconciliation_residual) < 1e-12 &&
      std::abs(result.factor_contribution_total + result.specific_contribution - accounting) < 1e-12;
  problem.accounting_portfolio_return += 0.01;
  const auto reconciliation_failure = performance_analytics::compute_factor_attribution_v2(problem);
  ok = ok && reconciliation_failure.status ==
      performance_analytics::FactorAttributionStatus::RECONCILIATION_FAILURE;
  problem.accounting_portfolio_return = accounting;
  problem.decision_at = 9;
  const auto future = performance_analytics::compute_factor_attribution_v2(problem);
  ok = ok && future.status == performance_analytics::FactorAttributionStatus::FUTURE_DATA;
  std::cout << (ok ? "test_factor_attribution_v2: all checks passed\n"
                   : "test_factor_attribution_v2: failed\n");
  return ok ? 0 : 1;
}

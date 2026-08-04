#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "portfolio_math/reconciler.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

portfolio_math::SinglePeriodReconcilerOptions options() {
  portfolio_math::SinglePeriodReconcilerOptions result;
  result.reconciler_spec_hash = 99;
  result.costs_available = true;
  return result;
}

bool test_zero_cost_anchor_parity() {
  const std::vector<double> anchor{0.6, 0.3, 0.1};
  const std::vector<double> current{0.3, 0.3, 0.0};
  const std::vector<double> penalty{1.0, 1.0, 1.0};
  const std::vector<double> zero{0.0, 0.0, 0.0};
  const auto result = portfolio_math::reconcile_single_period(
      anchor, current, penalty, zero, zero, options());
  bool ok = check(result.diagnostics.status == portfolio_math::OptimizationStatus::OK,
                  "zero-cost status");
  ok &= check(result.target_weights == anchor, "zero-cost anchor parity");
  ok &= check(result.diagnostics.turnover <= 1.0 &&
                  result.diagnostics.predicted_cost == 0.0,
              "zero-cost diagnostics");
  return ok;
}

bool test_constraints_and_costs() {
  auto config = options();
  config.max_single_weight = 0.6;
  config.turnover_cap = 0.1;
  const std::vector<double> anchor{0.9, 0.1, 0.0};
  const std::vector<double> current{0.3, 0.3, 0.0};
  const std::vector<double> penalty{1.0, 1.0, 1.0};
  const std::vector<double> linear{0.01, 0.02, 0.03};
  const std::vector<double> quadratic{0.1, 0.1, 0.1};
  const auto result = portfolio_math::reconcile_single_period(
      anchor, current, penalty, linear, quadratic, config);
  bool ok = check(result.diagnostics.status == portfolio_math::OptimizationStatus::OK,
                  "constrained status");
  ok &= check(result.target_weights.size() == 3 &&
                  std::all_of(result.target_weights.begin(), result.target_weights.end(),
                              [](double value) { return value >= 0.0 && value <= 0.6; }),
              "single-name cap");
  ok &= check(result.diagnostics.turnover <= 0.1 + 1e-9 &&
                  result.diagnostics.max_constraint_violation <= 1e-9,
              "turnover constraint");
  ok &= check(std::isfinite(result.diagnostics.predicted_cost) &&
                  result.diagnostics.predicted_cost >= 0.0,
              "cost diagnostic");
  ok &= check(result.diagnostics.predicted_cost ==
                  result.diagnostics.predicted_linear_cost +
                      result.diagnostics.predicted_quadratic_cost &&
                  result.diagnostics.predicted_linear_cost >= 0.0 &&
                  result.diagnostics.predicted_quadratic_cost >= 0.0,
              "cost decomposition");
  return ok;
}

bool test_group_caps() {
  auto config = options();
  const std::vector<double> anchor{0.7, 0.2, 0.05, 0.05};
  const std::vector<double> current{0.2, 0.2, 0.3, 0.3};
  const std::vector<double> penalty(4, 1.0);
  const std::vector<double> linear(4, 0.01);
  const std::vector<double> quadratic(4, 0.1);
  const std::vector<std::uint32_t> group_ids{0, 0, 1, 1};
  const std::vector<double> group_caps{0.5, 0.8};
  const portfolio_math::SinglePeriodConstraintView constraints{
      group_ids, group_caps, {}};
  const auto result = portfolio_math::reconcile_single_period(
      anchor, current, penalty, linear, quadratic, constraints, config);
  const double first_group = result.target_weights[0] + result.target_weights[1];
  const double second_group = result.target_weights[2] + result.target_weights[3];
  bool ok = check(result.diagnostics.status == portfolio_math::OptimizationStatus::OK,
                  "group cap status");
  ok &= check(result.target_weights.size() == 4 && first_group <= 0.5 + 1e-8 &&
                  second_group <= 0.8 + 1e-8,
              "group caps");
  ok &= check(result.diagnostics.max_constraint_violation <= 1e-8,
              "group cap diagnostics");
  return ok;
}

bool test_max_trade_weights() {
  auto config = options();
  const std::vector<double> anchor{0.8, 0.1, 0.1};
  const std::vector<double> current{0.3, 0.4, 0.3};
  const std::vector<double> penalty(3, 1.0);
  const std::vector<double> linear(3, 0.01);
  const std::vector<double> quadratic(3, 0.1);
  const std::vector<double> max_trade{0.05, 0.05, 0.05};
  const portfolio_math::SinglePeriodConstraintView constraints{
      {}, {}, max_trade};
  const auto result = portfolio_math::reconcile_single_period(
      anchor, current, penalty, linear, quadratic, constraints, config);
  bool ok = check(result.diagnostics.status == portfolio_math::OptimizationStatus::OK,
                  "max trade status");
  ok &= check(result.target_weights.size() == 3,
              "max trade target size");
  for (std::size_t index = 0; index < result.target_weights.size(); ++index) {
    ok &= check(std::abs(result.target_weights[index] - current[index]) <=
                    max_trade[index] + 1e-8,
                "max trade weight");
  }
  ok &= check(result.diagnostics.max_constraint_violation <= 1e-8,
              "max trade diagnostics");
  return ok;
}

bool test_invalid_constraint_views() {
  const std::vector<double> anchor{0.5, 0.5};
  const std::vector<double> current{0.5, 0.5};
  const std::vector<double> penalty{1.0, 1.0};
  const std::vector<double> cost{0.01, 0.01};
  const std::vector<double> impact{0.1, 0.1};
  const std::vector<std::uint32_t> bad_group_ids{0, 2};
  const std::vector<double> group_caps{0.5, 0.5};
  const portfolio_math::SinglePeriodConstraintView bad_group{
      bad_group_ids, group_caps, {}};
  bool ok = check(portfolio_math::reconcile_single_period(
                      anchor, current, penalty, cost, impact, bad_group, options())
                      .diagnostics.status == portfolio_math::OptimizationStatus::INVALID_INPUT,
                  "invalid group mapping");
  const std::vector<double> short_max_trade{0.1};
  const portfolio_math::SinglePeriodConstraintView bad_participation{
      {}, {}, short_max_trade};
  ok &= check(portfolio_math::reconcile_single_period(
                  anchor, current, penalty, cost, impact, bad_participation, options())
                  .diagnostics.status == portfolio_math::OptimizationStatus::INVALID_INPUT,
              "invalid participation shape");
  return ok;
}

bool test_joint_constraints() {
  auto config = options();
  config.turnover_cap = 0.1;
  const std::vector<double> anchor{0.7, 0.2, 0.05, 0.05};
  const std::vector<double> current{0.2, 0.2, 0.3, 0.3};
  const std::vector<double> penalty(4, 1.0);
  const std::vector<double> linear(4, 0.01);
  const std::vector<double> quadratic(4, 0.1);
  const std::vector<std::uint32_t> group_ids{0, 0, 1, 1};
  const std::vector<double> group_caps{0.45, 0.8};
  const std::vector<double> max_trade(4, 0.1);
  const portfolio_math::SinglePeriodConstraintView constraints{
      group_ids, group_caps, max_trade};
  const auto result = portfolio_math::reconcile_single_period(
      anchor, current, penalty, linear, quadratic, constraints, config);
  const double first_group = result.target_weights[0] + result.target_weights[1];
  bool ok = check(result.diagnostics.status == portfolio_math::OptimizationStatus::OK,
                  "joint constraint status");
  ok &= check(first_group <= 0.45 + 1e-8,
              "joint group cap");
  ok &= check(result.diagnostics.turnover <= 0.1 + 1e-8,
              "joint turnover cap");
  ok &= check(result.diagnostics.max_constraint_violation <= 1e-8,
              "joint diagnostics");
  for (std::size_t index = 0; index < result.target_weights.size(); ++index) {
    ok &= check(std::abs(result.target_weights[index] - current[index]) <=
                    max_trade[index] + 1e-8,
                "joint max trade");
  }
  return ok;
}

bool test_failures() {
  const std::vector<double> anchor{0.5, 0.5};
  const std::vector<double> current{0.5, 0.5};
  const std::vector<double> penalty{1.0, 1.0};
  const std::vector<double> zero{0.0, 0.0};
  auto unavailable = options();
  unavailable.costs_available = false;
  bool ok = check(portfolio_math::reconcile_single_period(
                      anchor, current, penalty, zero, zero, unavailable)
                      .diagnostics.status == portfolio_math::OptimizationStatus::INVALID_INPUT,
                  "missing costs fail closed");
  auto invalid = options();
  invalid.max_single_weight = 0.0;
  ok &= check(!portfolio_math::valid_single_period_reconciler_options(invalid),
              "invalid reconciler options");
  const std::vector<double> bad_current{0.8, 0.8};
  ok &= check(portfolio_math::reconcile_single_period(
                  anchor, bad_current, penalty, zero, zero, options())
                  .diagnostics.status == portfolio_math::OptimizationStatus::INFEASIBLE,
              "infeasible current weights");
  return ok;
}

}  // namespace

int main() {
  if (!(test_zero_cost_anchor_parity() && test_constraints_and_costs() &&
        test_group_caps() && test_max_trade_weights() &&
        test_invalid_constraint_views() && test_joint_constraints() &&
        test_failures())) {
    return 1;
  }
  std::printf("test_reconciler: all checks passed\n");
  return 0;
}

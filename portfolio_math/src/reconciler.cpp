#include "portfolio_math/reconciler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace portfolio_math {
namespace {

bool finite_nonnegative(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
}

std::vector<double> project_boxed_simplex(std::span<const double> values,
                                          std::span<const double> lower,
                                          std::span<const double> upper,
                                          double target) {
  std::vector<double> projected(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    projected[index] = std::clamp(values[index], lower[index], upper[index]);
  }
  const double sum = std::accumulate(projected.begin(), projected.end(), 0.0);
  if (sum <= target) return projected;
  double left = 0.0;
  double right = 0.0;
  for (std::size_t index = 0; index < values.size(); ++index) {
    right = std::max(right, values[index] - lower[index]);
  }
  for (int iteration = 0; iteration < 100; ++iteration) {
    const double shift = 0.5 * (left + right);
    double shifted_sum = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
      shifted_sum += std::clamp(values[index] - shift, lower[index], upper[index]);
    }
    if (shifted_sum > target) {
      left = shift;
    } else {
      right = shift;
    }
  }
  projected.clear();
  projected.reserve(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    projected.push_back(std::clamp(values[index] - 0.5 * (left + right),
                                   lower[index], upper[index]));
  }
  return projected;
}

double turnover(std::span<const double> left, std::span<const double> right) {
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result += std::abs(left[index] - right[index]);
  }
  return 0.5 * result;
}

std::vector<double> project_constraints(std::span<const double> candidate,
                                        std::span<const double> current,
                                        SinglePeriodConstraintView constraints,
                                        const SinglePeriodReconcilerOptions& options) {
  std::vector<double> lower(candidate.size(), 0.0);
  std::vector<double> upper(candidate.size(), options.max_single_weight);
  if (!constraints.max_trade_weights.empty()) {
    for (std::size_t index = 0; index < candidate.size(); ++index) {
      lower[index] = std::max(0.0, current[index] - constraints.max_trade_weights[index]);
      upper[index] = std::min(options.max_single_weight,
                              current[index] + constraints.max_trade_weights[index]);
    }
  }
  auto projected = project_boxed_simplex(candidate, lower, upper,
                                          options.target_investment);
  for (int iteration = 0; iteration < 64; ++iteration) {
    std::vector<double> previous = projected;
    if (!constraints.group_ids.empty()) {
      for (std::size_t group = 0; group < constraints.group_caps.size(); ++group) {
        double group_sum = 0.0;
        for (std::size_t index = 0; index < projected.size(); ++index) {
          if (constraints.group_ids[index] == group) group_sum += projected[index];
        }
        if (group_sum > constraints.group_caps[group] && group_sum > 0.0) {
          const double ratio = constraints.group_caps[group] / group_sum;
          for (std::size_t index = 0; index < projected.size(); ++index) {
            if (constraints.group_ids[index] == group) projected[index] *= ratio;
          }
        }
      }
    }
    projected = project_boxed_simplex(projected, lower, upper,
                                      options.target_investment);
    double change = 0.0;
    for (std::size_t index = 0; index < projected.size(); ++index) {
      change = std::max(change, std::abs(projected[index] - previous[index]));
    }
    if (change <= options.tolerance) break;
  }
  const double candidate_turnover = turnover(projected, current);
  if (candidate_turnover > options.turnover_cap && candidate_turnover > 0.0) {
    const double ratio = options.turnover_cap / candidate_turnover;
    for (std::size_t index = 0; index < projected.size(); ++index) {
      projected[index] = current[index] + ratio * (projected[index] - current[index]);
    }
  }
  return projected;
}

double max_violation(std::span<const double> weights,
                     std::span<const double> current,
                     SinglePeriodConstraintView constraints,
                     const SinglePeriodReconcilerOptions& options) {
  double violation = 0.0;
  double sum = 0.0;
  for (const double value : weights) {
    sum += value;
    violation = std::max(violation, std::max(-value, value - options.max_single_weight));
  }
  violation = std::max(violation, sum - options.target_investment);
  violation = std::max(violation, turnover(weights, current) - options.turnover_cap);
  if (!constraints.group_ids.empty()) {
    for (std::size_t group = 0; group < constraints.group_caps.size(); ++group) {
      double group_sum = 0.0;
      for (std::size_t index = 0; index < weights.size(); ++index) {
        if (constraints.group_ids[index] == group) group_sum += weights[index];
      }
      violation = std::max(violation, group_sum - constraints.group_caps[group]);
    }
  }
  if (!constraints.max_trade_weights.empty()) {
    for (std::size_t index = 0; index < weights.size(); ++index) {
      violation = std::max(violation,
                           std::abs(weights[index] - current[index]) -
                               constraints.max_trade_weights[index]);
    }
  }
  return std::max(0.0, violation);
}

bool valid_constraints(SinglePeriodConstraintView constraints,
                       std::size_t size) {
  if (!constraints.group_ids.empty() &&
      (constraints.group_ids.size() != size || constraints.group_caps.empty())) {
    return false;
  }
  if (constraints.group_ids.empty() && !constraints.group_caps.empty()) return false;
  if (!constraints.group_ids.empty()) {
    for (const auto group : constraints.group_ids) {
      if (group >= constraints.group_caps.size()) return false;
    }
    if (!finite_nonnegative(constraints.group_caps)) return false;
    if (std::any_of(constraints.group_caps.begin(), constraints.group_caps.end(),
                    [](double value) { return value > 1.0; })) {
      return false;
    }
  }
  return constraints.max_trade_weights.empty() ||
         (constraints.max_trade_weights.size() == size &&
          finite_nonnegative(constraints.max_trade_weights) &&
          std::all_of(constraints.max_trade_weights.begin(),
                      constraints.max_trade_weights.end(),
                      [](double value) { return value <= 1.0; }));
}

struct TradingCostBreakdown {
  double linear{0.0};
  double quadratic{0.0};
};

TradingCostBreakdown trading_cost(std::span<const double> weights,
                                  std::span<const double> current,
                                  std::span<const double> linear_cost,
                                  std::span<const double> quadratic_impact) {
  TradingCostBreakdown result;
  for (std::size_t index = 0; index < weights.size(); ++index) {
    const double trade_delta = weights[index] - current[index];
    result.linear += linear_cost[index] * std::abs(trade_delta);
    result.quadratic += 0.5 * quadratic_impact[index] * trade_delta * trade_delta;
  }
  return result;
}

}  // namespace

bool valid_single_period_reconciler_options(
    const SinglePeriodReconcilerOptions& options) noexcept {
  return options.max_iterations > 0 && std::isfinite(options.tolerance) &&
         options.tolerance > 0.0 && std::isfinite(options.target_investment) &&
         options.target_investment > 0.0 && options.target_investment <= 1.0 &&
         std::isfinite(options.max_single_weight) &&
         options.max_single_weight > 0.0 && options.max_single_weight <= 1.0 &&
         std::isfinite(options.turnover_cap) && options.turnover_cap >= 0.0 &&
         options.turnover_cap <= 1.0 && options.reconciler_spec_hash != 0;
}

SinglePeriodReconcilerResult reconcile_single_period(
    std::span<const double> anchor_weights,
    std::span<const double> current_weights,
    std::span<const double> anchor_penalty,
    std::span<const double> linear_cost,
    std::span<const double> quadratic_impact,
    SinglePeriodReconcilerOptions options) {
  return reconcile_single_period(anchor_weights, current_weights, anchor_penalty,
                                 linear_cost, quadratic_impact, {}, options);
}

SinglePeriodReconcilerResult reconcile_single_period(
    std::span<const double> anchor_weights,
    std::span<const double> current_weights,
    std::span<const double> anchor_penalty,
    std::span<const double> linear_cost,
    std::span<const double> quadratic_impact,
    SinglePeriodConstraintView constraints,
    SinglePeriodReconcilerOptions options) {
  SinglePeriodReconcilerResult result;
  if (!valid_single_period_reconciler_options(options) ||
      anchor_weights.empty() || current_weights.size() != anchor_weights.size() ||
      anchor_penalty.size() != anchor_weights.size() ||
      linear_cost.size() != anchor_weights.size() ||
      quadratic_impact.size() != anchor_weights.size() || !options.costs_available ||
      !finite_nonnegative(anchor_weights) || !finite_nonnegative(current_weights) ||
      !finite_nonnegative(anchor_penalty) || !finite_nonnegative(linear_cost) ||
      !finite_nonnegative(quadratic_impact) ||
      !valid_constraints(constraints, anchor_weights.size()) ||
      std::any_of(anchor_penalty.begin(), anchor_penalty.end(),
                  [](double value) { return value <= 0.0; })) {
    return result;
  }
  const auto feasible = [&](std::span<const double> weights) {
    return max_violation(weights, current_weights, constraints, options) <= options.tolerance &&
           std::accumulate(weights.begin(), weights.end(), 0.0) <=
               options.target_investment + options.tolerance;
  };
  if (!feasible(current_weights)) {
    result.diagnostics.status = OptimizationStatus::INFEASIBLE;
    return result;
  }

  bool zero_trade_cost = true;
  for (std::size_t index = 0; index < linear_cost.size(); ++index) {
    zero_trade_cost = zero_trade_cost && linear_cost[index] == 0.0 &&
                      quadratic_impact[index] == 0.0;
  }
  std::vector<double> weights = project_constraints(anchor_weights, current_weights,
                                                    constraints, options);
  if (zero_trade_cost && feasible(anchor_weights) &&
      turnover(anchor_weights, current_weights) <= options.turnover_cap + options.tolerance) {
    weights.assign(anchor_weights.begin(), anchor_weights.end());
  }
  double maximum_change = std::numeric_limits<double>::infinity();
  double lipschitz = 1.0;
  for (std::size_t index = 0; index < anchor_penalty.size(); ++index) {
    lipschitz = std::max(lipschitz, anchor_penalty[index] + quadratic_impact[index]);
  }
  const double step = 1.0 / lipschitz;
  for (std::uint32_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
    if (zero_trade_cost && feasible(anchor_weights) &&
        turnover(anchor_weights, current_weights) <= options.turnover_cap + options.tolerance) {
      maximum_change = 0.0;
      result.diagnostics.iterations = iteration;
      break;
    }
    std::vector<double> proximal(weights.size());
    for (std::size_t index = 0; index < weights.size(); ++index) {
      const double anchor_delta = weights[index] - anchor_weights[index];
      const double trade_delta = weights[index] - current_weights[index];
      const double smooth_gradient = anchor_penalty[index] * anchor_delta +
                                     quadratic_impact[index] * trade_delta;
      const double raw_delta = weights[index] - step * smooth_gradient - current_weights[index];
      const double threshold = step * linear_cost[index];
      const double shrunk_delta = raw_delta > threshold
                                      ? raw_delta - threshold
                                      : raw_delta < -threshold ? raw_delta + threshold : 0.0;
      proximal[index] = current_weights[index] + shrunk_delta;
    }
    auto projected = project_constraints(proximal, current_weights, constraints, options);
    maximum_change = 0.0;
    for (std::size_t index = 0; index < weights.size(); ++index) {
      maximum_change = std::max(maximum_change, std::abs(projected[index] - weights[index]));
    }
    weights = std::move(projected);
    result.diagnostics.iterations = iteration;
    if (maximum_change <= options.tolerance) break;
  }
  result.diagnostics.kkt_residual = maximum_change;
  if (maximum_change > options.tolerance) {
    result.diagnostics.status = OptimizationStatus::MAX_ITERATIONS;
    return result;
  }
  result.diagnostics.turnover = turnover(weights, current_weights);
  result.diagnostics.anchor_distance = turnover(weights, anchor_weights);
  result.diagnostics.max_constraint_violation = max_violation(
      weights, current_weights, constraints, options);
  const auto cost = trading_cost(weights, current_weights, linear_cost,
                                 quadratic_impact);
  result.diagnostics.predicted_linear_cost = cost.linear;
  result.diagnostics.predicted_quadratic_cost = cost.quadratic;
  result.diagnostics.predicted_cost = cost.linear + cost.quadratic;
  for (const double value : weights) {
    if (value <= options.tolerance ||
        std::abs(value - options.max_single_weight) <= options.tolerance) {
      ++result.diagnostics.active_constraint_count;
    }
  }
  if (!constraints.group_ids.empty()) {
    for (std::size_t group = 0; group < constraints.group_caps.size(); ++group) {
      double group_sum = 0.0;
      for (std::size_t index = 0; index < weights.size(); ++index) {
        if (constraints.group_ids[index] == group) group_sum += weights[index];
      }
      if (std::abs(group_sum - constraints.group_caps[group]) <= options.tolerance) {
        ++result.diagnostics.active_constraint_count;
      }
    }
  }
  if (!constraints.max_trade_weights.empty()) {
    for (std::size_t index = 0; index < weights.size(); ++index) {
      if (std::abs(std::abs(weights[index] - current_weights[index]) -
                   constraints.max_trade_weights[index]) <= options.tolerance) {
        ++result.diagnostics.active_constraint_count;
      }
    }
  }
  if (std::abs(std::accumulate(weights.begin(), weights.end(), 0.0) -
               options.target_investment) <= options.tolerance) {
    ++result.diagnostics.active_constraint_count;
  }
  if (std::abs(result.diagnostics.turnover - options.turnover_cap) <= options.tolerance) {
    ++result.diagnostics.active_constraint_count;
  }
  if (!std::isfinite(result.diagnostics.predicted_cost) ||
      !std::isfinite(result.diagnostics.predicted_linear_cost) ||
      !std::isfinite(result.diagnostics.predicted_quadratic_cost) ||
      result.diagnostics.max_constraint_violation > options.tolerance) {
    result.diagnostics.status = OptimizationStatus::NUMERICAL_FAILURE;
    return result;
  }
  result.target_weights = std::move(weights);
  result.diagnostics.status = OptimizationStatus::OK;
  result.diagnostics.eligible_for_official_risk = false;
  return result;
}

}  // namespace portfolio_math

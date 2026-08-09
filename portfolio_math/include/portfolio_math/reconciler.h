#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "portfolio_math/types.h"

namespace portfolio_math {

struct SinglePeriodReconcilerOptions {
  std::uint32_t max_iterations{10'000};
  double tolerance{1e-10};
  double target_investment{1.0};
  double max_single_weight{1.0};
  double turnover_cap{1.0};
  std::uint64_t reconciler_spec_hash{0};
  bool costs_available{false};
};

struct SinglePeriodConstraintView {
  std::span<const std::uint32_t> group_ids;
  std::span<const double> group_caps;
  std::span<const double> max_trade_weights;
};

struct SinglePeriodReconcilerDiagnostics {
  OptimizationStatus status{OptimizationStatus::INVALID_INPUT};
  std::uint32_t iterations{0};
  std::uint32_t active_constraint_count{0};
  double anchor_distance{0.0};
  double max_constraint_violation{0.0};
  double predicted_cost{0.0};
  double predicted_linear_cost{0.0};
  double predicted_quadratic_cost{0.0};
  double turnover{0.0};
  double kkt_residual{0.0};
  bool eligible_for_official_risk{false};
};

struct SinglePeriodReconcilerResult {
  std::vector<double> target_weights;
  SinglePeriodReconcilerDiagnostics diagnostics;
};

[[nodiscard]] bool valid_single_period_reconciler_options(
    const SinglePeriodReconcilerOptions& options) noexcept;

[[nodiscard]] SinglePeriodReconcilerResult reconcile_single_period(
    std::span<const double> anchor_weights,
    std::span<const double> current_weights,
    std::span<const double> anchor_penalty,
    std::span<const double> linear_cost,
    std::span<const double> quadratic_impact,
    SinglePeriodReconcilerOptions options = {});

[[nodiscard]] SinglePeriodReconcilerResult reconcile_single_period(
    std::span<const double> anchor_weights,
    std::span<const double> current_weights,
    std::span<const double> anchor_penalty,
    std::span<const double> linear_cost,
    std::span<const double> quadratic_impact,
    SinglePeriodConstraintView constraints,
    SinglePeriodReconcilerOptions options = {});

}  // namespace portfolio_math

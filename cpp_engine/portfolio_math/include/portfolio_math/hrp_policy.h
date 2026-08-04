#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "portfolio_math/types.h"
#include "quant_math/matrix.h"

namespace portfolio_math {

struct HrpPolicyOptions {
  double symmetry_tolerance{1e-10};
  double psd_tolerance{1e-10};
  double diagonal_tolerance{1e-12};
  double target_investment{1.0};
};

struct HrpBisectionStep {
  std::uint32_t left_begin{0};
  std::uint32_t left_end{0};
  std::uint32_t right_begin{0};
  std::uint32_t right_end{0};
  double left_variance{0.0};
  double right_variance{0.0};
  double left_allocation{0.0};
  double right_allocation{0.0};
};

struct HrpPolicyDiagnostics {
  OptimizationStatus status{OptimizationStatus::INVALID_INPUT};
  double predicted_risk{0.0};
  double weight_sum{0.0};
  std::uint32_t recursive_bisection_steps{0};
  std::vector<HrpBisectionStep> bisection_steps;
  std::uint64_t quasi_diagonal_order_hash{0};
  bool eligible_for_official_risk{false};
};

struct HrpPolicyResult {
  std::vector<double> weights;
  HrpPolicyDiagnostics diagnostics;
};

[[nodiscard]] bool valid_hrp_policy_options(
    const HrpPolicyOptions& options) noexcept;

[[nodiscard]] HrpPolicyResult solve_hrp_policy(
    quant_math::MatrixView official_covariance,
    std::span<const std::uint32_t> quasi_diagonal_order,
    HrpPolicyOptions options = {});

}  // namespace portfolio_math

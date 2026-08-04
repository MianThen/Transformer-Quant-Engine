#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "portfolio_math/types.h"
#include "quant_math/matrix.h"

namespace portfolio_math {

struct NcoPolicyOptions {
  std::uint32_t max_iterations{10'000};
  double tolerance{1e-10};
  double symmetry_tolerance{1e-10};
  double psd_tolerance{1e-10};
  double target_investment{1.0};
};

struct NcoPolicyDiagnostics {
  OptimizationStatus status{OptimizationStatus::INVALID_INPUT};
  std::uint32_t cluster_count{0};
  std::uint32_t intra_cluster_iterations{0};
  std::uint32_t inter_cluster_iterations{0};
  double kkt_residual{0.0};
  double max_risk_budget_error{0.0};
  double predicted_risk{0.0};
  double weight_sum{0.0};
  std::vector<std::uint32_t> cluster_sizes;
  std::vector<double> cluster_weights;
  bool eligible_for_official_risk{false};
};

struct NcoPolicyResult {
  std::vector<double> weights;
  NcoPolicyDiagnostics diagnostics;
};

[[nodiscard]] bool valid_nco_policy_options(
    const NcoPolicyOptions& options) noexcept;

[[nodiscard]] NcoPolicyResult solve_nco_minvar(
    quant_math::MatrixView official_covariance,
    std::span<const std::uint32_t> cluster_id_by_symbol,
    std::uint32_t cluster_count,
    NcoPolicyOptions options = {});

[[nodiscard]] NcoPolicyResult solve_nco_risk_budget(
    quant_math::MatrixView official_covariance,
    std::span<const std::uint32_t> cluster_id_by_symbol,
    std::uint32_t cluster_count,
    std::span<const double> cluster_risk_budgets,
    NcoPolicyOptions options = {});

}  // namespace portfolio_math

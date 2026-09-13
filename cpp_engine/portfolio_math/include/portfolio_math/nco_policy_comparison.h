#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "portfolio_math/nco_ffv.h"
#include "portfolio_math/posterior_direct.h"
#include "portfolio_math/reconciler.h"

namespace portfolio_math {

struct NcoPolicyComparisonOptions {
  PosteriorDirectOptions posterior_direct;
  NcoFfvPolicyOptions nco_ffv;
  SinglePeriodReconcilerOptions reconciler;
};

struct NcoPolicyComparisonSnapshot {
  OptimizationStatus anchor_status{OptimizationStatus::INVALID_INPUT};
  OptimizationStatus reconciler_status{OptimizationStatus::INVALID_INPUT};
  std::uint64_t anchor_artifact_hash{0};
  std::vector<double> anchor_weights;
  std::vector<double> target_weights;
  double evaluated_expected_return{0.0};
  double evaluated_variance{0.0};
  SinglePeriodReconcilerDiagnostics reconciler;
};

struct NcoPolicyComparisonArtifactV1 {
  std::uint32_t schema_version{1};
  OptimizationStatus status{OptimizationStatus::INVALID_INPUT};
  std::uint64_t posterior_artifact_hash{0};
  std::uint64_t cluster_spec_hash{0};
  NcoPolicyComparisonSnapshot posterior_direct;
  NcoPolicyComparisonSnapshot nco_ffv;
  NcoPolicyComparisonSnapshot nco_risk_only;
  bool winner_selected{false};
  bool eligible_for_official_risk{false};
  std::uint64_t artifact_hash{0};
};

[[nodiscard]] NcoPolicyComparisonArtifactV1 compare_nco_policy_family(
    const PosteriorScenarioArtifactV1& posterior,
    std::span<const std::uint32_t> cluster_id_by_symbol,
    std::uint32_t cluster_count,
    std::span<const double> current_weights,
    std::span<const double> anchor_penalty,
    std::span<const double> linear_cost,
    std::span<const double> quadratic_impact,
    NcoPolicyComparisonOptions options = {});

[[nodiscard]] std::uint64_t nco_policy_comparison_artifact_hash(
    const NcoPolicyComparisonArtifactV1& comparison) noexcept;

[[nodiscard]] std::string serialize_nco_policy_comparison(
    const NcoPolicyComparisonArtifactV1& comparison);

}  // namespace portfolio_math

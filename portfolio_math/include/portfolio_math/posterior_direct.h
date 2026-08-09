#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "portfolio_math/posterior.h"
#include "portfolio_math/types.h"

namespace portfolio_math {

struct PosteriorDirectOptions {
  std::uint32_t max_iterations{10'000};
  double tolerance{1e-10};
  double risk_aversion{1.0};
  double target_investment{1.0};
  double max_single_weight{1.0};
};

struct PosteriorDirectDiagnostics {
  OptimizationStatus status{OptimizationStatus::INVALID_INPUT};
  std::uint32_t iterations{0};
  double kkt_residual{0.0};
  double weight_sum{0.0};
  double expected_return{0.0};
  double variance{0.0};
  double objective{0.0};
  std::uint64_t posterior_artifact_hash{0};
  bool eligible_for_official_risk{false};
};

struct PosteriorDirectResult {
  std::vector<double> weights;
  PosteriorDirectDiagnostics diagnostics;
};

struct PosteriorDirectPolicyComparison {
  OptimizationStatus status{OptimizationStatus::INVALID_INPUT};
  PosteriorDirectResult gaussian_bl;
  PosteriorDirectResult fully_flexible_views;
  double expected_return_delta{0.0};
  double variance_delta{0.0};
  double objective_delta{0.0};
  double weight_l1_distance{0.0};
  std::uint64_t gaussian_artifact_hash{0};
  std::uint64_t fully_flexible_artifact_hash{0};
  bool winner_selected{false};
};

[[nodiscard]] bool valid_posterior_direct_options(
    const PosteriorDirectOptions& options) noexcept;

[[nodiscard]] PosteriorDirectResult solve_posterior_direct(
    const PosteriorScenarioArtifactV1& posterior,
    PosteriorDirectOptions options = {});

[[nodiscard]] PosteriorDirectPolicyComparison compare_posterior_direct_policies(
    const PosteriorScenarioArtifactV1& gaussian_bl,
    const PosteriorScenarioArtifactV1& fully_flexible_views,
    PosteriorDirectOptions options = {});

[[nodiscard]] std::uint64_t posterior_direct_policy_comparison_hash(
    const PosteriorDirectPolicyComparison& comparison) noexcept;

[[nodiscard]] std::string serialize_posterior_direct_policy_comparison(
    const PosteriorDirectPolicyComparison& comparison);

}  // namespace portfolio_math

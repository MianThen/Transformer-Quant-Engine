#pragma once

#include <cstdint>
#include <span>
#include <string>

#include "portfolio_math/nco_policy.h"
#include "portfolio_math/posterior.h"

namespace portfolio_math {

struct NcoFfvPolicyOptions {
  NcoPolicyOptions nco;
  bool require_valid_posterior{true};
};

struct NcoFfvPolicyResult {
  OptimizationStatus status{OptimizationStatus::INVALID_INPUT};
  NcoPolicyResult nco;
  std::uint64_t posterior_artifact_hash{0};
  std::uint64_t cluster_spec_hash{0};
  std::uint64_t artifact_hash{0};
  bool eligible_for_official_risk{false};
};

[[nodiscard]] NcoFfvPolicyResult solve_nco_ffv_minvar(
    const PosteriorScenarioArtifactV1& posterior,
    std::span<const std::uint32_t> cluster_id_by_symbol,
    std::uint32_t cluster_count,
    NcoFfvPolicyOptions options = {});

[[nodiscard]] std::uint64_t nco_ffv_policy_artifact_hash(
    const NcoFfvPolicyResult& result) noexcept;

[[nodiscard]] std::string serialize_nco_ffv_policy_result(
    const NcoFfvPolicyResult& result);

}  // namespace portfolio_math

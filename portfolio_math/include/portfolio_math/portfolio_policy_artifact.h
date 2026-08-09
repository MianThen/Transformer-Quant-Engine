#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "portfolio_math/types.h"

namespace portfolio_math {

enum class PortfolioPolicyKind : std::uint8_t {
  TOPK_EQUAL_WEIGHT,
  HRP,
  RISK_BUDGET,
  POSTERIOR_DIRECT,
  NCO_MIN_VARIANCE,
  NCO_RISK_BUDGET,
  NCO_FFV,
};

struct PortfolioPolicyArtifact {
  std::uint32_t schema_version{1};
  PortfolioPolicyKind policy{PortfolioPolicyKind::TOPK_EQUAL_WEIGHT};
  std::string policy_id;
  std::uint64_t policy_config_hash{0};
  std::string official_risk_model_sha256;
  std::string cluster_model_sha256;
  std::string posterior_scenario_sha256;
  std::uint64_t intra_cluster_objective_hash{0};
  std::uint64_t inter_cluster_objective_hash{0};
  std::uint64_t reconciler_spec_hash{0};
  std::string anchor_weights_sha256;
  std::string target_weights_sha256;
  OptimizationStatus anchor_status{OptimizationStatus::INVALID_INPUT};
  OptimizationStatus reconciler_status{OptimizationStatus::INVALID_INPUT};
  std::vector<double> anchor_weights;
  std::vector<double> target_weights;
  double anchor_distance{0.0};
  double max_constraint_violation{0.0};
  double predicted_cost{0.0};
  double predicted_linear_cost{0.0};
  double predicted_quadratic_cost{0.0};
  double turnover{0.0};
  double kkt_residual{0.0};
  std::uint32_t active_constraint_count{0};
  bool eligible_for_official_risk{false};
  std::uint64_t artifact_hash{0};
};

[[nodiscard]] bool valid_portfolio_policy_artifact(
    const PortfolioPolicyArtifact& artifact) noexcept;

[[nodiscard]] bool finalize_portfolio_policy_artifact(
    PortfolioPolicyArtifact& artifact) noexcept;

[[nodiscard]] std::uint64_t portfolio_policy_artifact_hash(
    const PortfolioPolicyArtifact& artifact) noexcept;

[[nodiscard]] std::string serialize_portfolio_policy_artifact(
    const PortfolioPolicyArtifact& artifact);

}  // namespace portfolio_math

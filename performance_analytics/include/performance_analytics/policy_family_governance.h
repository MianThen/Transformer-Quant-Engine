#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "performance_analytics/return_analysis.h"
#include "portfolio_math/multiple_testing.h"
#include "portfolio_math/onc_partition.h"
#include "quant_math/matrix.h"

namespace performance_analytics {

struct PolicyFamilyGovernanceSpecV1 {
  std::uint32_t schema_version{1};
  double fdr_q_level{0.05};
  portfolio_math::FdrMethod fdr_method{
      portfolio_math::FdrMethod::BENJAMINI_YEKUTIELI};
  std::vector<double> storey_lambdas;
  portfolio_math::StoreyAggregation storey_aggregation{
      portfolio_math::StoreyAggregation::MEAN_PRE_REGISTERED_LAMBDAS};
  EffectiveTrialSpec effective_trial_spec;
  portfolio_math::OncSpec onc_spec;
  PerformanceSpecV1 performance_spec;
  std::uint64_t dsr_config_hash{0};
  std::uint64_t family_contract_hash{0};
};

struct PolicyFamilyGovernanceInputV1 {
  std::vector<std::string> policy_ids;
  std::span<const double> trial_major_returns;
  std::size_t observations{0};
  std::span<const double> primary_p_values;
  bool registered_before_evaluation{false};
  bool formal_oos{false};
  bool economic_gate_passed{false};
  bool winner_freeze_requested{false};
  bool final_untouched_unseen{true};
};

struct PolicyFamilyGovernanceArtifactV1 {
  AnalysisStatus status{AnalysisStatus::INVALID_INPUT};
  quant_math::DenseMatrix policy_return_correlation;
  portfolio_math::OncPartitionResult policy_clusters;
  EffectiveTrialResult effective_trials;
  portfolio_math::FdrResult fdr;
  std::vector<DsrResult> dsr_by_policy;
  std::size_t raw_trial_count{0};
  std::size_t observation_count{0};
  std::size_t cluster_count{0};
  std::size_t selected_policy_index{0};
  bool candidate_selected{false};
  bool winner_frozen{false};
  bool promotion_eligible{false};
  std::uint64_t artifact_hash{0};
};

[[nodiscard]] PolicyFamilyGovernanceArtifactV1 evaluate_policy_family_governance(
    const PolicyFamilyGovernanceInputV1 &,
    const PolicyFamilyGovernanceSpecV1 &);

[[nodiscard]] std::string serialize_policy_family_governance(
    const PolicyFamilyGovernanceArtifactV1 &,
    const PolicyFamilyGovernanceInputV1 &);

}  // namespace performance_analytics

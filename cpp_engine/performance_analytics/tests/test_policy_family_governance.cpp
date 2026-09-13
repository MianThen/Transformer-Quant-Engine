#include <array>
#include <cstdio>

#include "performance_analytics/policy_family_governance.h"

namespace {
bool check(bool value, const char *message) {
  if (!value) std::fprintf(stderr, "FAILED: %s\n", message);
  return value;
}
}

int main() {
  using namespace performance_analytics;
  constexpr std::size_t observations = 12;
  const std::array<double, 48> returns{
      .020,-.010,.015,-.020,.025,-.005,.018,-.012,.010,-.015,.022,-.008,
      .019,-.009,.014,-.019,.024,-.004,.017,-.011,.009,-.014,.021,-.007,
      -.015,.020,-.010,.018,-.020,.015,-.012,.022,-.008,.017,-.018,.021,
      -.014,.019,-.009,.017,-.019,.014,-.011,.021,-.007,.016,-.017,.020};
  const std::array<double, 4> p_values{0.001, 0.002, 0.7, 0.8};
  PolicyFamilyGovernanceSpecV1 spec;
  spec.fdr_q_level = 0.05;
  spec.effective_trial_spec = EffectiveTrialSpec{6, 101};
  spec.performance_spec.calendar_id = "CN-EQUITY";
  spec.performance_spec.calendar_periods_per_year = 252.0;
  spec.performance_spec.minimum_return_observations = 6;
  spec.performance_spec.minimum_tail_observations = 6;
  spec.performance_spec.config_hash = 102;
  spec.dsr_config_hash = 103;
  spec.family_contract_hash = 104;
  spec.onc_spec.min_clusters = 2;
  spec.onc_spec.max_clusters = 2;
  spec.onc_spec.min_cluster_size = 2;
  PolicyFamilyGovernanceInputV1 input{
      {"posterior-direct", "nco-ffv", "nco-risk-1", "nco-risk-2"},
      returns, observations, p_values};
  const auto first = evaluate_policy_family_governance(input, spec);
  const auto second = evaluate_policy_family_governance(input, spec);
  bool ok = check(first.status == AnalysisStatus::OK, "governance status");
  ok &= check(first.cluster_count == 2 && first.effective_trials.effective_trials < 4.0,
              "correlation clustering and effective trials");
  ok &= check(!first.winner_frozen && !first.promotion_eligible,
              "reference evidence cannot freeze or promote");
  ok &= check(first.artifact_hash == second.artifact_hash &&
                  serialize_policy_family_governance(first, input) ==
                      serialize_policy_family_governance(second, input),
              "deterministic replay");
  const auto serialized = serialize_policy_family_governance(first, input);
  ok &= check(serialized.find("\"policy_return_correlation\"") != std::string::npos &&
                  serialized.find("\"cluster_id_by_policy\"") != std::string::npos &&
                  serialized.find("\"fdr\"") != std::string::npos &&
                  serialized.find("\"dsr_by_policy\"") != std::string::npos &&
                  serialized.find("\"gate_inputs\"") != std::string::npos,
              "auditable governance payload");
  auto renamed = input;
  renamed.policy_ids[0] = "posterior-direct-renamed";
  ok &= check(evaluate_policy_family_governance(renamed, spec).artifact_hash !=
                  first.artifact_hash,
              "policy identity enters artifact hash");
  auto invalid = input;
  invalid.policy_ids[1] = invalid.policy_ids[0];
  ok &= check(evaluate_policy_family_governance(invalid, spec).status ==
                  AnalysisStatus::INVALID_INPUT,
              "duplicate policy id fails closed");
  auto premature = input;
  premature.winner_freeze_requested = true;
  premature.economic_gate_passed = true;
  ok &= check(!evaluate_policy_family_governance(premature, spec).winner_frozen,
              "missing registered formal OOS blocks winner freeze");
  if (!ok) return 1;
  std::puts("test_policy_family_governance: all checks passed");
  return 0;
}

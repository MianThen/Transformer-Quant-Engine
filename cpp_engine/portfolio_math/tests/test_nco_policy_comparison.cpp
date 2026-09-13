#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include "portfolio_math/nco_policy_comparison.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

portfolio_math::PosteriorScenarioArtifactV1 make_posterior() {
  const std::vector<double> values{
      0.02, 0.00, 0.01, 0.00,
      0.01, 0.01, 0.00, 0.01,
      0.00, 0.02, 0.01, 0.02,
      0.03, 0.01, 0.02, 0.01,
      -0.01, 0.00, -0.01, 0.00,
      0.00, -0.01, 0.00, -0.01,
  };
  const std::vector<engine_common::TimestampNs> timestamps{10, 20, 30, 40, 50, 60};
  const auto prior = portfolio_math::build_prior_scenario_artifact(
      {values.data(), 6, 4, 4}, timestamps, 60, 70);
  portfolio_math::ViewSpecV1 view;
  view.view_id = "comparison-mean-view";
  view.available_at = 70;
  view.loading = {1.0, 0.0, 0.0, 0.0};
  view.target = 0.02;
  view.confidence = 0.75;
  view.observation_variance = 0.1;
  view.confidence_mapping_hash = 71;
  view.source_artifact_hash = 72;
  return portfolio_math::apply_ffv_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&view, 1));
}

portfolio_math::NcoPolicyComparisonOptions options() {
  portfolio_math::NcoPolicyComparisonOptions result;
  result.reconciler.reconciler_spec_hash = 101;
  result.reconciler.costs_available = true;
  result.reconciler.max_single_weight = 0.8;
  return result;
}

bool test_fixed_downstream_comparison() {
  const auto posterior = make_posterior();
  const std::vector<std::uint32_t> clusters{0, 0, 1, 1};
  const std::vector<double> current{0.25, 0.25, 0.25, 0.25};
  const std::vector<double> penalty(4, 1.0);
  const std::vector<double> linear(4, 0.001);
  const std::vector<double> quadratic(4, 0.1);
  const auto comparison = portfolio_math::compare_nco_policy_family(
      posterior, clusters, 2, current, penalty, linear, quadratic, options());
  bool ok = check(comparison.status == portfolio_math::OptimizationStatus::OK,
                  "fixed downstream comparison status");
  ok &= check(comparison.posterior_direct.anchor_status ==
                      portfolio_math::OptimizationStatus::OK &&
                  comparison.nco_ffv.anchor_status ==
                      portfolio_math::OptimizationStatus::OK &&
                  comparison.nco_risk_only.anchor_status ==
                      portfolio_math::OptimizationStatus::OK,
              "all comparison anchors status");
  ok &= check(comparison.posterior_direct.reconciler_status ==
                      portfolio_math::OptimizationStatus::OK &&
                  comparison.nco_ffv.reconciler_status ==
                      portfolio_math::OptimizationStatus::OK &&
                  comparison.nco_risk_only.reconciler_status ==
                      portfolio_math::OptimizationStatus::OK,
              "all comparison reconciler status");
  ok &= check(comparison.posterior_direct.target_weights.size() == 4 &&
                  comparison.nco_ffv.target_weights.size() == 4 &&
                  comparison.nco_risk_only.target_weights.size() == 4 &&
                  !comparison.winner_selected &&
                  !comparison.eligible_for_official_risk,
              "fixed downstream research boundary");
  ok &= check(comparison.artifact_hash ==
                  portfolio_math::nco_policy_comparison_artifact_hash(comparison) &&
                  comparison.artifact_hash != 0,
              "comparison artifact hash");
  const auto repeated = portfolio_math::compare_nco_policy_family(
      posterior, clusters, 2, current, penalty, linear, quadratic, options());
  ok &= check(repeated.artifact_hash == comparison.artifact_hash &&
                  repeated.nco_ffv.target_weights == comparison.nco_ffv.target_weights,
              "fixed downstream deterministic replay");
  const auto serialized = portfolio_math::serialize_nco_policy_comparison(comparison);
  ok &= check(serialized.find("\"posterior_direct\":") != std::string::npos &&
                  serialized.find("\"nco_ffv\":") != std::string::npos &&
                  serialized.find("\"nco_risk_only\":") != std::string::npos &&
                  serialized.find("\"winner_selected\":false") != std::string::npos,
              "comparison serialization");
  return ok;
}

bool test_fail_closed() {
  const auto posterior = make_posterior();
  const std::vector<std::uint32_t> clusters{0, 0, 1, 1};
  const std::vector<double> current{0.25, 0.25, 0.25, 0.25};
  const std::vector<double> penalty(4, 1.0);
  const std::vector<double> zero(4, 0.0);
  auto invalid = options();
  invalid.reconciler.costs_available = false;
  const auto result = portfolio_math::compare_nco_policy_family(
      posterior, clusters, 2, current, penalty, zero, zero, invalid);
  return check(result.status == portfolio_math::OptimizationStatus::INVALID_INPUT &&
                   result.posterior_direct.target_weights.empty() &&
                   result.nco_ffv.target_weights.empty() &&
                   result.nco_risk_only.target_weights.empty() &&
                   !result.winner_selected,
               "comparison missing costs fails closed");
}

}  // namespace

int main() {
  if (!(test_fixed_downstream_comparison() && test_fail_closed())) return 1;
  std::printf("test_nco_policy_comparison: all checks passed\n");
  return 0;
}

#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include "portfolio_math/posterior.h"
#include "portfolio_math/posterior_direct.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

portfolio_math::PriorScenarioArtifactV1 make_prior() {
  const std::vector<double> values{
      0.02, 0.00,
      0.01, 0.01,
      0.00, 0.02,
      0.03, 0.01,
      -0.01, 0.00,
      0.00, -0.01,
  };
  const std::vector<engine_common::TimestampNs> timestamps{10, 20, 30, 40, 50, 60};
  return portfolio_math::build_prior_scenario_artifact(
      {values.data(), 6, 2, 2}, timestamps, 60, 70);
}

bool test_reference() {
  const auto prior = make_prior();
  const auto posterior = portfolio_math::apply_ffv_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>{});
  const auto result = portfolio_math::solve_posterior_direct(posterior);
  bool ok = check(result.diagnostics.status == portfolio_math::OptimizationStatus::OK,
                  "posterior direct status");
  ok &= check(result.weights.size() == 2 &&
                  std::abs(result.diagnostics.weight_sum - 1.0) < 1e-10,
              "posterior direct simplex");
  ok &= check(result.weights[0] >= 0.0 && result.weights[1] >= 0.0 &&
                  result.diagnostics.variance >= 0.0 &&
                  result.diagnostics.posterior_artifact_hash == posterior.artifact_hash,
              "posterior direct diagnostics");
  ok &= check(result.weights[0] > result.weights[1],
              "posterior direct favors higher posterior mean");
  return ok;
}

bool test_bounds_and_fail_closed() {
  const auto prior = make_prior();
  const auto posterior = portfolio_math::apply_gaussian_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>{});
  auto bounded = portfolio_math::PosteriorDirectOptions{};
  bounded.max_single_weight = 0.6;
  const auto bounded_result = portfolio_math::solve_posterior_direct(
      posterior, bounded);
  bool ok = check(bounded_result.diagnostics.status == portfolio_math::OptimizationStatus::OK &&
                      bounded_result.weights[0] <= 0.6 + 1e-10 &&
                      bounded_result.weights[1] <= 0.6 + 1e-10,
                  "posterior direct cap");
  auto impossible = bounded;
  impossible.max_single_weight = 0.4;
  const auto impossible_result = portfolio_math::solve_posterior_direct(
      posterior, impossible);
  ok &= check(impossible_result.diagnostics.status ==
                  portfolio_math::OptimizationStatus::INVALID_INPUT &&
                  impossible_result.weights.empty(),
              "posterior direct infeasible cap");
  auto invalid = bounded;
  invalid.risk_aversion = 0.0;
  const auto invalid_result = portfolio_math::solve_posterior_direct(
      posterior, invalid);
  ok &= check(invalid_result.diagnostics.status ==
                  portfolio_math::OptimizationStatus::INVALID_INPUT,
              "posterior direct invalid options");
  return ok;
}

portfolio_math::ViewSpecV1 make_mean_view() {
  portfolio_math::ViewSpecV1 view;
  view.view_id = "bl-ffv-comparison-mean";
  view.available_at = 70;
  view.loading = {1.0, 0.0};
  view.target = 0.025;
  view.confidence = 1.0;
  view.observation_variance = 0.25;
  view.confidence_mapping_hash = 88;
  view.source_artifact_hash = 77;
  return view;
}

bool test_bl_ffv_policy_comparison() {
  const auto prior = make_prior();
  const auto no_view_bl = portfolio_math::apply_gaussian_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>{});
  const auto no_view_ffv = portfolio_math::apply_ffv_views(
      prior, std::span<const portfolio_math::ViewSpecV1>{});
  const auto parity = portfolio_math::compare_posterior_direct_policies(
      no_view_bl, no_view_ffv);
  bool ok = check(
      parity.status == portfolio_math::OptimizationStatus::OK &&
          parity.weight_l1_distance < 1e-10 &&
          std::abs(parity.objective_delta) < 1e-12 &&
          !parity.winner_selected &&
          parity.gaussian_artifact_hash == no_view_bl.artifact_hash &&
          parity.fully_flexible_artifact_hash == no_view_ffv.artifact_hash,
      "BL/FFV no-view policy parity");
  const auto view = make_mean_view();
  const auto viewed_bl = portfolio_math::apply_gaussian_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&view, 1));
  const auto viewed_ffv = portfolio_math::apply_ffv_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&view, 1));
  const auto comparison = portfolio_math::compare_posterior_direct_policies(
      viewed_bl, viewed_ffv);
  ok &= check(comparison.status == portfolio_math::OptimizationStatus::OK &&
                  comparison.weight_l1_distance >= 0.0 &&
                  std::isfinite(comparison.expected_return_delta) &&
                  std::isfinite(comparison.variance_delta) &&
                  std::isfinite(comparison.objective_delta) &&
                  !comparison.winner_selected,
              "BL/FFV fixed-downstream comparison diagnostics");
  const auto repeated = portfolio_math::compare_posterior_direct_policies(
      viewed_bl, viewed_ffv);
  ok &= check(repeated.weight_l1_distance == comparison.weight_l1_distance &&
                  repeated.objective_delta == comparison.objective_delta,
              "BL/FFV comparison determinism");
  const auto comparison_hash =
      portfolio_math::posterior_direct_policy_comparison_hash(comparison);
  ok &= check(comparison_hash ==
                  portfolio_math::posterior_direct_policy_comparison_hash(repeated) &&
                  comparison_hash != 0,
              "BL/FFV comparison hash replay");
  const auto serialized =
      portfolio_math::serialize_posterior_direct_policy_comparison(comparison);
  ok &= check(serialized.find("\"schema_version\":1") != std::string::npos &&
                  serialized.find("\"winner_selected\":false") != std::string::npos &&
                  serialized.find("\"comparison_hash\":") != std::string::npos,
              "BL/FFV comparison serialization");
  return ok;
}

}  // namespace

int main() {
  if (!(test_reference() && test_bounds_and_fail_closed() &&
        test_bl_ffv_policy_comparison())) return 1;
  std::printf("test_posterior_direct: all checks passed\n");
  return 0;
}

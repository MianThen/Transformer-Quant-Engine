#include <cstdio>
#include <cmath>
#include <span>
#include <string>
#include <vector>

#include "portfolio_math/posterior.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

portfolio_math::PriorScenarioArtifactV1 make_prior() {
  const std::vector<double> values{
      0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0,
      1.0, 1.0,
  };
  const std::vector<engine_common::TimestampNs> timestamps{10, 20, 30, 40};
  return portfolio_math::build_prior_scenario_artifact(
      {values.data(), 4, 2, 2}, timestamps, 40, 50);
}

portfolio_math::ViewSpecV1 make_view(double confidence) {
  portfolio_math::ViewSpecV1 view;
  view.view_id = "mean-asset-1";
  view.available_at = 50;
  view.loading = {1.0, 0.0};
  view.target = 0.8;
  view.confidence = confidence;
  view.observation_variance = 0.25;
  view.confidence_mapping_hash = 88;
  view.source_artifact_hash = 77;
  return view;
}

bool test_prior_and_no_view() {
  auto prior = make_prior();
  bool ok = check(portfolio_math::valid_prior_scenario_artifact(prior), "prior contract");
  ok &= check(prior.scenario_timestamps == std::vector<engine_common::TimestampNs>{10, 20, 30, 40},
              "prior timestamp provenance");
  const auto posterior = portfolio_math::apply_gaussian_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>{});
  ok &= check(portfolio_math::valid_posterior_scenario_artifact(posterior), "no-view posterior");
  ok &= check(posterior.view_count == 0 && posterior.posterior_mean == posterior.prior_mean,
              "no-view mean parity");
  ok &= check((posterior.posterior_covariance - posterior.prior_covariance).norm() == 0.0,
              "no-view covariance parity");
  ok &= check(portfolio_math::serialize_posterior_scenario_artifact(posterior).find(
                  "\"status\":\"ok\"") != std::string::npos,
              "posterior serialization");
  ok &= check(portfolio_math::serialize_posterior_scenario_artifact(posterior).find(
                  "\"scenario_timestamps\"") != std::string::npos &&
                  portfolio_math::serialize_posterior_scenario_artifact(posterior).find(
                      "\"support_min\"") != std::string::npos,
              "posterior provenance serialization");
  ok &= check(posterior.scenario_timestamps == prior.scenario_timestamps &&
                  posterior.support_min == prior.support_min &&
                  posterior.support_max == prior.support_max,
              "posterior support provenance");
  return ok;
}

bool test_confidence_limits_and_future_guard() {
  const auto prior = make_prior();
  bool ok = true;
  auto zero_view = make_view(0.0);
  const auto zero = portfolio_math::apply_gaussian_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&zero_view, 1));
  ok &= check(zero.posterior_mean == zero.prior_mean, "zero-confidence is no-view");
  auto full_view = make_view(1.0);
  const auto full = portfolio_math::apply_gaussian_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&full_view, 1));
  ok &= check(full.status == portfolio_math::PosteriorStatus::OK &&
                  std::abs(full.posterior_mean[0] - full_view.target) < 1e-10,
              "full-confidence mean oracle");
  auto future_view = make_view(0.5);
  future_view.available_at = 60;
  const auto future = portfolio_math::apply_gaussian_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&future_view, 1));
  ok &= check(future.status == portfolio_math::PosteriorStatus::FUTURE_DATA,
              "future view fail closed");
  const std::vector<engine_common::TimestampNs> future_timestamps{10, 20, 60, 70};
  const std::vector<double> values(8, 0.0);
  const auto bad_prior = portfolio_math::build_prior_scenario_artifact(
      {values.data(), 4, 2, 2}, future_timestamps, 40, 50);
  ok &= check(bad_prior.status == portfolio_math::PosteriorStatus::FUTURE_DATA,
              "future prior fail closed");
  return ok;
}

bool test_ffv_probability_projection_and_statistics() {
  const auto prior = make_prior();
  bool ok = true;
  const auto no_view = portfolio_math::apply_ffv_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>{});
  ok &= check(portfolio_math::valid_posterior_scenario_artifact(no_view),
              "FFV no-view artifact contract");
  ok &= check(no_view.posterior_probabilities == prior.prior_probabilities &&
                  no_view.posterior_mean == prior.prior_mean &&
                  no_view.kl_divergence == 0.0,
              "FFV no-view prior parity");
  auto zero_view = make_view(0.0);
  const auto zero = portfolio_math::apply_ffv_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&zero_view, 1));
  ok &= check(portfolio_math::valid_posterior_scenario_artifact(zero) &&
                  zero.posterior_probabilities == prior.prior_probabilities &&
                  zero.active_constraint_count == 0,
              "FFV zero-confidence prior parity");
  auto full_view = make_view(1.0);
  const auto full = portfolio_math::apply_ffv_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&full_view, 1));
  ok &= check(portfolio_math::valid_posterior_scenario_artifact(full) &&
                  std::abs(full.posterior_mean[0] - full_view.target) < 1e-10 &&
                  full.maximum_view_residual < 1e-10 &&
                  full.kl_divergence > 0.0 && full.maximum_scenario_weight < 0.5,
              "FFV full-confidence mean view");
  const auto statistics = portfolio_math::recompute_posterior_statistics(
      prior.scenario_values, prior.scenario_count, prior.asset_count,
      full.posterior_probabilities, full.posterior_quantile_levels);
  ok &= check(statistics.status == portfolio_math::PosteriorStatus::OK &&
                  statistics.mean == full.posterior_mean &&
                  (statistics.covariance - full.posterior_covariance).norm() < 1e-12 &&
                  statistics.quantiles == full.posterior_quantiles &&
                  statistics.expected_shortfall == full.posterior_expected_shortfall,
              "FFV posterior moment quantile ES recomputation parity");
  auto unsupported = make_view(1.0);
  unsupported.target = 2.0;
  const auto outside = portfolio_math::apply_ffv_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&unsupported, 1));
  ok &= check(outside.status == portfolio_math::PosteriorStatus::INFEASIBLE &&
                  !outside.support_guard_passed,
              "FFV out-of-support view fails closed");
  auto boundary = make_view(1.0);
  boundary.target = 1.0;
  const auto boundary_result = portfolio_math::apply_ffv_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&boundary, 1));
  ok &= check(boundary_result.status == portfolio_math::PosteriorStatus::INFEASIBLE &&
                  !boundary_result.support_guard_passed,
              "FFV positive-probability boundary fails closed");
  return ok;
}

bool test_relative_mean_view_and_duplicate_guard() {
  const auto prior = make_prior();
  auto relative = make_view(1.0);
  relative.view_id = "relative-mean";
  relative.loading = {1.0, -1.0};
  relative.target = 0.4;
  const auto gaussian = portfolio_math::apply_gaussian_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&relative, 1));
  const auto ffv = portfolio_math::apply_ffv_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&relative, 1));
  bool ok = check(gaussian.status == portfolio_math::PosteriorStatus::OK &&
                      std::abs(gaussian.posterior_mean[0] -
                               gaussian.posterior_mean[1] - relative.target) < 1e-10,
                  "Gaussian relative mean view");
  ok &= check(ffv.status == portfolio_math::PosteriorStatus::OK &&
                  std::abs(ffv.posterior_mean[0] - ffv.posterior_mean[1] -
                           relative.target) < 1e-10,
              "FFV relative mean view");
  const portfolio_math::ViewSpecV1 duplicates[]{relative, relative};
  const auto duplicate = portfolio_math::apply_ffv_mean_views(prior, duplicates);
  ok &= check(duplicate.status == portfolio_math::PosteriorStatus::NUMERICAL_FAILURE,
              "duplicate correlated FFV views fail closed");
  auto near_duplicate = relative;
  near_duplicate.view_id = "near-duplicate";
  near_duplicate.loading = {1.0 + 1e-12, -1.0 + 1e-12};
  const portfolio_math::ViewSpecV1 near_duplicates[]{relative, near_duplicate};
  const auto near_duplicate_result = portfolio_math::apply_ffv_mean_views(
      prior, near_duplicates);
  ok &= check(near_duplicate_result.status == portfolio_math::PosteriorStatus::NUMERICAL_FAILURE,
              "near-collinear FFV views fail closed");
  auto mismatched_mapping = relative;
  mismatched_mapping.view_id = "mapping-mismatch";
  mismatched_mapping.confidence_mapping_hash = 99;
  const portfolio_math::ViewSpecV1 mixed[]{relative, mismatched_mapping};
  const auto mixed_result = portfolio_math::apply_ffv_mean_views(prior, mixed);
  ok &= check(mixed_result.status == portfolio_math::PosteriorStatus::INVALID_INPUT,
              "mixed confidence mappings fail closed");
  return ok;
}

bool test_ffv_mean_inequality_views() {
  const auto prior = make_prior();
  bool ok = true;
  auto lower = make_view(1.0);
  lower.view_id = "lower-mean";
  lower.kind = portfolio_math::PosteriorViewKind::MEAN_LOWER_BOUND;
  lower.target = 0.75;
  const auto lower_result = portfolio_math::apply_ffv_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&lower, 1));
  ok &= check(portfolio_math::valid_posterior_scenario_artifact(lower_result) &&
                  lower_result.active_constraint_count == 1 &&
                  lower_result.posterior_mean[0] >= lower.target - 1e-10 &&
                  lower_result.maximum_view_residual < 1e-10,
              "FFV lower mean bound");
  auto upper = make_view(1.0);
  upper.view_id = "upper-mean";
  upper.kind = portfolio_math::PosteriorViewKind::MEAN_UPPER_BOUND;
  upper.target = 0.25;
  const auto upper_result = portfolio_math::apply_ffv_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&upper, 1));
  ok &= check(portfolio_math::valid_posterior_scenario_artifact(upper_result) &&
                  upper_result.active_constraint_count == 1 &&
                  upper_result.posterior_mean[0] <= upper.target + 1e-10 &&
                  upper_result.maximum_view_residual < 1e-10,
              "FFV upper mean bound");
  auto inactive = lower;
  inactive.view_id = "inactive-lower-mean";
  inactive.target = 0.25;
  const auto inactive_result = portfolio_math::apply_ffv_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&inactive, 1));
  ok &= check(portfolio_math::valid_posterior_scenario_artifact(inactive_result) &&
                  inactive_result.active_constraint_count == 0 &&
                  inactive_result.posterior_probabilities == prior.prior_probabilities,
              "FFV inactive lower mean bound parity");
  const auto gaussian_result = portfolio_math::apply_gaussian_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&lower, 1));
  ok &= check(gaussian_result.status == portfolio_math::PosteriorStatus::INVALID_INPUT,
              "Gaussian rejects inequality mean view");
  auto second_asset_upper = upper;
  second_asset_upper.view_id = "upper-mean-second-asset";
  second_asset_upper.loading = {0.0, 1.0};
  const portfolio_math::ViewSpecV1 coupled[]{lower, second_asset_upper};
  const auto coupled_result = portfolio_math::apply_ffv_mean_views(prior, coupled);
  ok &= check(portfolio_math::valid_posterior_scenario_artifact(coupled_result) &&
                  coupled_result.active_constraint_count == 2 &&
                  coupled_result.posterior_mean[0] >= lower.target - 1e-10 &&
                  coupled_result.posterior_mean[1] <= upper.target + 1e-10 &&
                  coupled_result.maximum_view_residual < 1e-10,
              "FFV coupled active-set bounds");
  return ok;
}

bool test_rich_view_family_calibration_gate() {
  const auto prior = make_prior();
  auto rich = make_view(0.8);
  rich.view_id = "direction-head-view";
  rich.family = portfolio_math::PosteriorViewFamily::DIRECTION;
  bool ok = check(
      !portfolio_math::valid_view_spec(rich, prior.asset_count, prior.decision_at),
      "uncalibrated rich view fails contract");
  rich.calibration_artifact_hash = 99;
  ok &= check(
      portfolio_math::valid_view_spec(rich, prior.asset_count, prior.decision_at),
      "calibrated rich view passes contract layer");
  const auto gaussian = portfolio_math::apply_gaussian_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&rich, 1));
  const auto ffv = portfolio_math::apply_ffv_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&rich, 1));
  ok &= check(gaussian.status == portfolio_math::PosteriorStatus::INVALID_INPUT &&
                  ffv.status == portfolio_math::PosteriorStatus::INVALID_INPUT,
              "rich view cannot enter mean-only solvers");
  const auto serialized = portfolio_math::serialize_view_spec(rich);
  ok &= check(serialized.find("\"family\":\"direction\"") != std::string::npos &&
                  serialized.find("\"calibration_artifact_hash\":99") !=
                      std::string::npos,
              "rich view provenance serialization");
  return ok;
}

bool test_direction_ffv_probability_oracle() {
  const auto prior = make_prior();
  auto direction = make_view(1.0);
  direction.view_id = "direction-probability";
  direction.family = portfolio_math::PosteriorViewFamily::DIRECTION;
  direction.calibration_artifact_hash = 123;
  direction.statistic_threshold = 0.5;
  direction.target = 0.75;
  const auto gaussian = portfolio_math::apply_gaussian_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&direction, 1));
  const auto ffv = portfolio_math::apply_ffv_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&direction, 1));
  bool ok = check(gaussian.status == portfolio_math::PosteriorStatus::INVALID_INPUT,
                  "Gaussian rejects direction view");
  ok &= check(ffv.status == portfolio_math::PosteriorStatus::OK &&
                  ffv.maximum_view_residual < 1e-10 &&
                  ffv.posterior_mean[0] > 0.6,
              "FFV direction probability equality oracle");
  const auto legacy = portfolio_math::apply_ffv_mean_views(
      prior, std::span<const portfolio_math::ViewSpecV1>(&direction, 1));
  ok &= check(legacy.status == portfolio_math::PosteriorStatus::INVALID_INPUT,
              "mean-only FFV wrapper rejects direction view");
  return ok;
}

}  // namespace

int main() {
  if (!(test_prior_and_no_view() && test_confidence_limits_and_future_guard() &&
        test_ffv_probability_projection_and_statistics() &&
        test_relative_mean_view_and_duplicate_guard() &&
        test_ffv_mean_inequality_views() &&
        test_rich_view_family_calibration_gate() &&
        test_direction_ffv_probability_oracle())) return 1;
  std::printf("test_posterior: all checks passed\n");
  return 0;
}

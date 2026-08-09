#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include "portfolio_math/nco_ffv.h"

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
  const std::vector<engine_common::TimestampNs> timestamps{
      10, 20, 30, 40, 50, 60,
  };
  const auto prior = portfolio_math::build_prior_scenario_artifact(
      {values.data(), 6, 4, 4}, timestamps, 60, 70);
  return portfolio_math::apply_ffv_views(
      prior, std::span<const portfolio_math::ViewSpecV1>{});
}

bool test_nco_ffv_reference_and_provenance() {
  const auto posterior = make_posterior();
  const std::vector<std::uint32_t> clusters{0, 0, 1, 1};
  const auto result = portfolio_math::solve_nco_ffv_minvar(
      posterior, clusters, 2);
  bool ok = check(result.status == portfolio_math::OptimizationStatus::OK &&
                      result.nco.weights.size() == 4 &&
                      std::abs(result.nco.diagnostics.weight_sum - 1.0) < 1e-10,
                  "NCO-FFV minvar reference");
  ok &= check(result.posterior_artifact_hash == posterior.artifact_hash &&
                  result.cluster_spec_hash != 0 &&
                  result.artifact_hash ==
                      portfolio_math::nco_ffv_policy_artifact_hash(result) &&
                  !result.eligible_for_official_risk,
              "NCO-FFV provenance and research boundary");
  const auto serialized =
      portfolio_math::serialize_nco_ffv_policy_result(result);
  ok &= check(serialized.find("\"posterior_artifact_hash\":") != std::string::npos &&
                  serialized.find("\"eligible_for_official_risk\":false") !=
                      std::string::npos &&
                  serialized.find("\"objective_ablation\":\"full\"") !=
                      std::string::npos &&
                  serialized.find("\"intra_objective_enabled\":true") !=
                      std::string::npos &&
                  serialized.find("\"inter_objective_enabled\":true") !=
                      std::string::npos &&
                  serialized.find("\"artifact_hash\":") != std::string::npos,
              "NCO-FFV serialization");
  const auto repeated = portfolio_math::solve_nco_ffv_minvar(
      posterior, clusters, 2);
  ok &= check(repeated.artifact_hash == result.artifact_hash &&
                  repeated.nco.weights == result.nco.weights,
              "NCO-FFV deterministic replay");
  return ok;
}

bool test_nco_ffv_fail_closed() {
  const auto posterior = make_posterior();
  const std::vector<std::uint32_t> missing_clusters{0, 0, 1};
  const auto missing = portfolio_math::solve_nco_ffv_minvar(
      posterior, missing_clusters, 2);
  bool ok = check(missing.status == portfolio_math::OptimizationStatus::INVALID_INPUT,
                  "NCO-FFV cluster shape guard");
  auto invalid_posterior = posterior;
  invalid_posterior.status = portfolio_math::PosteriorStatus::FUTURE_DATA;
  const std::vector<std::uint32_t> clusters{0, 0, 1, 1};
  const auto future = portfolio_math::solve_nco_ffv_minvar(
      invalid_posterior, clusters, 2);
  ok &= check(future.status == portfolio_math::OptimizationStatus::INVALID_INPUT,
              "NCO-FFV posterior future guard");

  portfolio_math::NcoFfvPolicyOptions invalid_options;
  invalid_options.nco.objective_ablation =
      static_cast<portfolio_math::NcoObjectiveAblation>(255);
  const auto invalid_objective = portfolio_math::solve_nco_ffv_minvar(
      posterior, clusters, 2, invalid_options);
  ok &= check(invalid_objective.status ==
                  portfolio_math::OptimizationStatus::INVALID_INPUT &&
                  invalid_objective.nco.weights.empty() &&
                  !invalid_objective.eligible_for_official_risk,
              "NCO-FFV invalid objective fails closed");
  return ok;
}

bool test_nco_ffv_objective_hashes() {
  const auto posterior = make_posterior();
  const std::vector<std::uint32_t> clusters{0, 0, 1, 1};
  portfolio_math::NcoFfvPolicyOptions intra_options;
  intra_options.nco.objective_ablation =
      portfolio_math::NcoObjectiveAblation::INTRA_ONLY;
  portfolio_math::NcoFfvPolicyOptions inter_options;
  inter_options.nco.objective_ablation =
      portfolio_math::NcoObjectiveAblation::INTER_ONLY;
  const auto full = portfolio_math::solve_nco_ffv_minvar(
      posterior, clusters, 2);
  const auto intra = portfolio_math::solve_nco_ffv_minvar(
      posterior, clusters, 2, intra_options);
  const auto inter = portfolio_math::solve_nco_ffv_minvar(
      posterior, clusters, 2, inter_options);
  bool ok = check(full.status == portfolio_math::OptimizationStatus::OK &&
                      intra.status == portfolio_math::OptimizationStatus::OK &&
                      inter.status == portfolio_math::OptimizationStatus::OK,
                  "NCO-FFV objective modes status");
  ok &= check(full.artifact_hash != intra.artifact_hash &&
                  full.artifact_hash != inter.artifact_hash &&
                  intra.artifact_hash != inter.artifact_hash,
              "NCO-FFV objective modes hash separation");
  const auto intra_serialized =
      portfolio_math::serialize_nco_ffv_policy_result(intra);
  const auto inter_serialized =
      portfolio_math::serialize_nco_ffv_policy_result(inter);
  ok &= check(intra_serialized.find("\"objective_ablation\":\"intra_only\"") !=
                      std::string::npos &&
                  inter_serialized.find("\"objective_ablation\":\"inter_only\"") !=
                      std::string::npos,
              "NCO-FFV objective mode serialization");
  return ok;
}

}  // namespace

int main() {
  if (!(test_nco_ffv_reference_and_provenance() &&
        test_nco_ffv_fail_closed() && test_nco_ffv_objective_hashes())) {
    return 1;
  }
  std::printf("test_nco_ffv: all checks passed\n");
  return 0;
}

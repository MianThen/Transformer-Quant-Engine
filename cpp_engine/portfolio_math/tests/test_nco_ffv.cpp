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
  return ok;
}

}  // namespace

int main() {
  if (!(test_nco_ffv_reference_and_provenance() &&
        test_nco_ffv_fail_closed())) {
    return 1;
  }
  std::printf("test_nco_ffv: all checks passed\n");
  return 0;
}

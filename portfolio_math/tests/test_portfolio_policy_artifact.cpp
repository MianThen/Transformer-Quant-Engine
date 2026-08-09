#include <cstdio>
#include <string>

#include "portfolio_math/portfolio_policy_artifact.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

std::string digest(char character) { return std::string(64, character); }

portfolio_math::PortfolioPolicyArtifact make_artifact() {
  portfolio_math::PortfolioPolicyArtifact artifact;
  artifact.policy = portfolio_math::PortfolioPolicyKind::NCO_MIN_VARIANCE;
  artifact.policy_id = "NCO-MINVAR-RESEARCH-V1";
  artifact.policy_config_hash = 11;
  artifact.official_risk_model_sha256 = digest('a');
  artifact.cluster_model_sha256 = digest('b');
  artifact.intra_cluster_objective_hash = 12;
  artifact.inter_cluster_objective_hash = 13;
  artifact.reconciler_spec_hash = 14;
  artifact.anchor_weights_sha256 = digest('c');
  artifact.target_weights_sha256 = digest('d');
  artifact.anchor_status = portfolio_math::OptimizationStatus::OK;
  artifact.reconciler_status = portfolio_math::OptimizationStatus::OK;
  artifact.anchor_weights = {0.4, 0.35, 0.25};
  artifact.target_weights = {0.35, 0.35, 0.25};
  artifact.anchor_distance = 0.05;
  artifact.max_constraint_violation = 0.0;
  artifact.predicted_cost = 0.001;
  artifact.predicted_linear_cost = 0.0006;
  artifact.predicted_quadratic_cost = 0.0004;
  artifact.turnover = 0.05;
  artifact.kkt_residual = 1e-10;
  artifact.active_constraint_count = 1;
  return artifact;
}

bool test_contract_and_serialization() {
  auto artifact = make_artifact();
  bool ok = check(portfolio_math::finalize_portfolio_policy_artifact(artifact) &&
                      portfolio_math::valid_portfolio_policy_artifact(artifact),
                  "policy artifact contract");
  const auto json = portfolio_math::serialize_portfolio_policy_artifact(artifact);
  ok &= check(json.find("\"policy\":\"nco_min_variance\"") != std::string::npos &&
                  json.find("\"anchor_weights\":[") != std::string::npos &&
                  json.find("0.4") != std::string::npos &&
                  json.find("\"target_weights\"") != std::string::npos &&
                  json.find("\"reconciler_status\":\"ok\"") != std::string::npos &&
                  json.find("\"predicted_linear_cost\":") != std::string::npos &&
                  json.find("\"predicted_quadratic_cost\":") != std::string::npos &&
                  json.find("\"artifact_hash\":") != std::string::npos,
              "policy artifact serialization");
  return ok;
}

bool test_tamper_and_failure_record() {
  auto artifact = make_artifact();
  bool ok = check(portfolio_math::finalize_portfolio_policy_artifact(artifact),
                  "artifact finalization");
  artifact.target_weights[0] = 0.2;
  ok &= check(!portfolio_math::valid_portfolio_policy_artifact(artifact),
              "policy artifact tamper detection");
  auto failure = make_artifact();
  failure.anchor_status = portfolio_math::OptimizationStatus::MAX_ITERATIONS;
  failure.reconciler_status = portfolio_math::OptimizationStatus::INFEASIBLE;
  failure.anchor_weights.clear();
  failure.target_weights.clear();
  failure.anchor_weights_sha256.clear();
  failure.target_weights_sha256.clear();
  ok &= check(portfolio_math::finalize_portfolio_policy_artifact(failure) &&
                  portfolio_math::valid_portfolio_policy_artifact(failure),
              "failure artifact record");
  return ok;
}

}  // namespace

int main() {
  if (!(test_contract_and_serialization() && test_tamper_and_failure_record())) {
    return 1;
  }
  std::printf("test_portfolio_policy_artifact: all checks passed\n");
  return 0;
}

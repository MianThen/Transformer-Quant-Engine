#include <cstdio>
#include <string>

#include "portfolio_math/cluster_model_artifact.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

std::string digest(char character) { return std::string(64, character); }

portfolio_math::ClusterModelArtifact make_hrp_artifact() {
  portfolio_math::ClusterModelArtifact artifact;
  artifact.kind = portfolio_math::ClusterModelKind::HRP_HIERARCHICAL_LINKAGE;
  artifact.correlation_source =
      portfolio_math::ClusterCorrelationSource::DENOISED_DETONED;
  artifact.official_risk_model_sha256 = digest('a');
  artifact.denoised_risk_sha256 = digest('b');
  artifact.detone_components = 1;
  artifact.merge_tree_sha256 = digest('c');
  artifact.symbols = {10, 11, 12, 13};
  artifact.quasi_diagonal_order = {0, 1, 2, 3};
  artifact.quasi_diagonal_order_sha256 = digest('d');
  artifact.fit_start = 100;
  artifact.fit_end = 200;
  artifact.available_at = 300;
  return artifact;
}

portfolio_math::ClusterModelArtifact make_onc_artifact() {
  portfolio_math::ClusterModelArtifact artifact;
  artifact.kind = portfolio_math::ClusterModelKind::ONC_PARTITION;
  artifact.correlation_source =
      portfolio_math::ClusterCorrelationSource::DENOISED;
  artifact.official_risk_model_sha256 = digest('e');
  artifact.denoised_risk_sha256 = digest('f');
  artifact.has_onc_spec = true;
  artifact.onc_spec.min_clusters = 2;
  artifact.onc_spec.max_clusters = 3;
  artifact.onc_spec.min_cluster_size = 2;
  artifact.onc_spec.repeats = 2;
  artifact.onc_spec.seeds = {17, 29};
  artifact.symbols = {10, 11, 12, 13};
  artifact.cluster_id_by_symbol = {0, 0, 1, 1};
  artifact.quasi_diagonal_order = {0, 1, 2, 3};
  artifact.cluster_count = 2;
  artifact.quality = 0.8;
  artifact.silhouette = {0.7, 0.8, 0.8, 0.9};
  artifact.cluster_id_by_symbol_sha256 = digest('1');
  artifact.quasi_diagonal_order_sha256 = digest('2');
  artifact.fit_start = 100;
  artifact.fit_end = 200;
  artifact.available_at = 300;
  return artifact;
}

bool test_variants_and_serialization() {
  auto hrp = make_hrp_artifact();
  auto onc = make_onc_artifact();
  bool ok = true;
  ok &= check(portfolio_math::finalize_cluster_model_artifact(hrp) &&
                  portfolio_math::valid_cluster_model_artifact(hrp),
              "HRP artifact contract");
  ok &= check(portfolio_math::finalize_cluster_model_artifact(onc) &&
                  portfolio_math::valid_cluster_model_artifact(onc),
              "ONC artifact contract");
  const auto json = portfolio_math::serialize_cluster_model_artifact(onc);
  ok &= check(json.find("\"kind\":\"onc_partition\"") != std::string::npos &&
                  json.find("\"symbols\":[10,11,12,13]") != std::string::npos &&
                  json.find("\"onc_spec\":{") != std::string::npos &&
                  json.find("\"silhouette\":[") != std::string::npos &&
                  json.find("\"artifact_hash\":") != std::string::npos &&
                  json.find("\"cluster_id_by_symbol_sha256\"") !=
                      std::string::npos,
              "artifact serialization fields");
  return ok;
}

bool test_provenance_failures() {
  auto artifact = make_onc_artifact();
  bool ok = true;
  artifact.correlation_source =
      portfolio_math::ClusterCorrelationSource::DENOISED_DETONED;
  ok &= check(!portfolio_math::finalize_cluster_model_artifact(artifact),
              "detoned artifact requires detone component");

  artifact = make_hrp_artifact();
  artifact.quasi_diagonal_order = {0, 1, 1, 3};
  ok &= check(!portfolio_math::finalize_cluster_model_artifact(artifact),
              "duplicate quasi-order fails closed");

  artifact = make_hrp_artifact();
  ok &= check(portfolio_math::finalize_cluster_model_artifact(artifact),
              "artifact finalization before tamper");
  artifact.quality = 0.1;
  ok &= check(!portfolio_math::valid_cluster_model_artifact(artifact),
              "artifact hash detects tamper");
  return ok;
}

}  // namespace

int main() {
  if (!(test_variants_and_serialization() && test_provenance_failures())) {
    return 1;
  }
  std::printf("test_cluster_model_artifact: all checks passed\n");
  return 0;
}

#include "portfolio_math/nco_ffv.h"

#include <bit>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace portfolio_math {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_byte(std::uint64_t& hash, std::uint8_t value) {
  hash ^= value;
  hash *= kFnvPrime;
}

void hash_value(std::uint64_t& hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash_byte(hash, static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffU));
  }
}

void hash_double(std::uint64_t& hash, double value) {
  hash_value(hash, std::bit_cast<std::uint64_t>(value));
}

std::uint64_t cluster_hash(std::span<const std::uint32_t> clusters,
                           std::uint32_t cluster_count) {
  std::uint64_t hash = kFnvOffset;
  hash_value(hash, cluster_count);
  hash_value(hash, clusters.size());
  for (const auto cluster : clusters) hash_value(hash, cluster);
  return hash;
}

const char* status_name(OptimizationStatus status) {
  switch (status) {
    case OptimizationStatus::OK: return "ok";
    case OptimizationStatus::INVALID_INPUT: return "invalid_input";
    case OptimizationStatus::NON_PSD_RISK_MODEL: return "non_psd_risk_model";
    case OptimizationStatus::INFEASIBLE: return "infeasible";
    case OptimizationStatus::MAX_ITERATIONS: return "max_iterations";
    case OptimizationStatus::NUMERICAL_FAILURE: return "numerical_failure";
  }
  return "invalid";
}

void json_weights(std::ostringstream& output, std::span<const double> values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << std::setprecision(17) << values[index];
  }
  output << ']';
}

}  // namespace

NcoFfvPolicyResult solve_nco_ffv_minvar(
    const PosteriorScenarioArtifactV1& posterior,
    std::span<const std::uint32_t> cluster_id_by_symbol,
    std::uint32_t cluster_count,
    NcoFfvPolicyOptions options) {
  NcoFfvPolicyResult result;
  result.posterior_artifact_hash = posterior.artifact_hash;
  result.cluster_spec_hash = cluster_hash(cluster_id_by_symbol, cluster_count);
  if (options.require_valid_posterior &&
      !valid_posterior_scenario_artifact(posterior)) {
    result.artifact_hash = nco_ffv_policy_artifact_hash(result);
    return result;
  }
  if (!valid_nco_policy_options(options.nco) ||
      posterior.asset_count == 0 ||
      cluster_id_by_symbol.size() != posterior.asset_count ||
      cluster_count == 0) {
    result.artifact_hash = nco_ffv_policy_artifact_hash(result);
    return result;
  }
  result.nco = solve_nco_minvar(
      quant_math::view(posterior.posterior_covariance),
      cluster_id_by_symbol, cluster_count, options.nco);
  result.status = result.nco.diagnostics.status;
  result.eligible_for_official_risk = false;
  result.artifact_hash = nco_ffv_policy_artifact_hash(result);
  return result;
}

std::uint64_t nco_ffv_policy_artifact_hash(
    const NcoFfvPolicyResult& result) noexcept {
  std::uint64_t hash = kFnvOffset;
  hash_byte(hash, static_cast<std::uint8_t>(result.status));
  hash_value(hash, result.posterior_artifact_hash);
  hash_value(hash, result.cluster_spec_hash);
  hash_value(hash, result.nco.diagnostics.cluster_count);
  hash_value(hash, result.nco.diagnostics.intra_cluster_iterations);
  hash_value(hash, result.nco.diagnostics.inter_cluster_iterations);
  hash_double(hash, result.nco.diagnostics.predicted_risk);
  hash_double(hash, result.nco.diagnostics.weight_sum);
  for (double value : result.nco.weights) hash_double(hash, value);
  hash_byte(hash, result.eligible_for_official_risk ? 1 : 0);
  return hash;
}

std::string serialize_nco_ffv_policy_result(
    const NcoFfvPolicyResult& result) {
  std::ostringstream output;
  output << "{\"schema_version\":1"
         << ",\"status\":\"" << status_name(result.status)
         << "\",\"posterior_artifact_hash\":" << result.posterior_artifact_hash
         << ",\"cluster_spec_hash\":" << result.cluster_spec_hash
         << ",\"weights\":";
  json_weights(output, result.nco.weights);
  output << ",\"cluster_count\":"
         << result.nco.diagnostics.cluster_count
         << ",\"predicted_risk\":" << std::setprecision(17)
         << result.nco.diagnostics.predicted_risk
         << ",\"eligible_for_official_risk\":"
         << (result.eligible_for_official_risk ? "true" : "false")
         << ",\"artifact_hash\":" << nco_ffv_policy_artifact_hash(result)
         << '}';
  return output.str();
}

}  // namespace portfolio_math

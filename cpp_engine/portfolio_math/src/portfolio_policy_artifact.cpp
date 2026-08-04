#include "portfolio_math/portfolio_policy_artifact.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <span>

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

void hash_string(std::uint64_t& hash, const std::string& value) {
  hash_value(hash, value.size());
  for (const unsigned char character : value) hash_byte(hash, character);
}

bool digest_like(const std::string& value, bool allow_empty = false) {
  if (value.empty()) return allow_empty;
  if (value.size() != 64) return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f') ||
           (character >= 'A' && character <= 'F');
  });
}

const char* policy_name(PortfolioPolicyKind policy) {
  switch (policy) {
    case PortfolioPolicyKind::TOPK_EQUAL_WEIGHT: return "topk_equal_weight";
    case PortfolioPolicyKind::HRP: return "hrp";
    case PortfolioPolicyKind::RISK_BUDGET: return "risk_budget";
    case PortfolioPolicyKind::POSTERIOR_DIRECT: return "posterior_direct";
    case PortfolioPolicyKind::NCO_MIN_VARIANCE: return "nco_min_variance";
    case PortfolioPolicyKind::NCO_RISK_BUDGET: return "nco_risk_budget";
    case PortfolioPolicyKind::NCO_FFV: return "nco_ffv";
  }
  return "invalid";
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

bool valid_policy(PortfolioPolicyKind policy) {
  switch (policy) {
    case PortfolioPolicyKind::TOPK_EQUAL_WEIGHT:
    case PortfolioPolicyKind::HRP:
    case PortfolioPolicyKind::RISK_BUDGET:
    case PortfolioPolicyKind::POSTERIOR_DIRECT:
    case PortfolioPolicyKind::NCO_MIN_VARIANCE:
    case PortfolioPolicyKind::NCO_RISK_BUDGET:
    case PortfolioPolicyKind::NCO_FFV:
      return true;
  }
  return false;
}

bool valid_status(OptimizationStatus status) {
  switch (status) {
    case OptimizationStatus::OK:
    case OptimizationStatus::INVALID_INPUT:
    case OptimizationStatus::NON_PSD_RISK_MODEL:
    case OptimizationStatus::INFEASIBLE:
    case OptimizationStatus::MAX_ITERATIONS:
    case OptimizationStatus::NUMERICAL_FAILURE:
      return true;
  }
  return false;
}

bool finite_nonnegative(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
}

bool valid_base(const PortfolioPolicyArtifact& artifact) {
  if (artifact.schema_version != 1 || !valid_policy(artifact.policy) ||
      !valid_status(artifact.anchor_status) ||
      !valid_status(artifact.reconciler_status) || artifact.policy_id.empty() ||
      artifact.policy_config_hash == 0 ||
      !digest_like(artifact.official_risk_model_sha256) ||
      !digest_like(artifact.cluster_model_sha256, true) ||
      !digest_like(artifact.posterior_scenario_sha256, true) ||
      artifact.intra_cluster_objective_hash == 0 ||
      artifact.inter_cluster_objective_hash == 0 ||
      artifact.reconciler_spec_hash == 0 ||
      !digest_like(artifact.anchor_weights_sha256, true) ||
      !digest_like(artifact.target_weights_sha256, true) ||
      !std::isfinite(artifact.anchor_distance) || artifact.anchor_distance < 0.0 ||
      !std::isfinite(artifact.max_constraint_violation) ||
      artifact.max_constraint_violation < 0.0 ||
      !std::isfinite(artifact.predicted_cost) || artifact.predicted_cost < 0.0 ||
      !std::isfinite(artifact.predicted_linear_cost) ||
      artifact.predicted_linear_cost < 0.0 ||
      !std::isfinite(artifact.predicted_quadratic_cost) ||
      artifact.predicted_quadratic_cost < 0.0 ||
      std::abs(artifact.predicted_cost -
               artifact.predicted_linear_cost -
               artifact.predicted_quadratic_cost) >
          1e-12 * std::max(1.0, std::abs(artifact.predicted_cost)) ||
      !std::isfinite(artifact.turnover) || artifact.turnover < 0.0 ||
      !std::isfinite(artifact.kkt_residual) || artifact.kkt_residual < 0.0 ||
      artifact.eligible_for_official_risk ||
      !finite_nonnegative(artifact.anchor_weights) ||
      !finite_nonnegative(artifact.target_weights)) {
    return false;
  }
  if (artifact.anchor_status == OptimizationStatus::OK) {
    if (artifact.anchor_weights.empty() || artifact.anchor_weights_sha256.empty()) return false;
  } else if (!artifact.anchor_weights.empty() || !artifact.anchor_weights_sha256.empty()) {
    return false;
  }
  if (artifact.reconciler_status == OptimizationStatus::OK) {
    if (artifact.target_weights.empty() || artifact.target_weights_sha256.empty()) return false;
  } else if (!artifact.target_weights.empty() || !artifact.target_weights_sha256.empty()) {
    return false;
  }
  if (!artifact.anchor_weights.empty() && !artifact.target_weights.empty() &&
      artifact.anchor_weights.size() != artifact.target_weights.size()) {
    return false;
  }
  if (artifact.anchor_weights.empty() && artifact.target_weights.empty()) return true;
  const auto& weights = !artifact.target_weights.empty() ? artifact.target_weights
                                                         : artifact.anchor_weights;
  const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  return sum >= 0.0 && sum <= 1.0 + 1e-10;
}

void hash_base(std::uint64_t& hash, const PortfolioPolicyArtifact& artifact) {
  hash_value(hash, artifact.schema_version);
  hash_value(hash, static_cast<std::uint8_t>(artifact.policy));
  hash_string(hash, artifact.policy_id);
  hash_value(hash, artifact.policy_config_hash);
  hash_string(hash, artifact.official_risk_model_sha256);
  hash_string(hash, artifact.cluster_model_sha256);
  hash_string(hash, artifact.posterior_scenario_sha256);
  hash_value(hash, artifact.intra_cluster_objective_hash);
  hash_value(hash, artifact.inter_cluster_objective_hash);
  hash_value(hash, artifact.reconciler_spec_hash);
  hash_string(hash, artifact.anchor_weights_sha256);
  hash_string(hash, artifact.target_weights_sha256);
  hash_value(hash, static_cast<std::uint8_t>(artifact.anchor_status));
  hash_value(hash, static_cast<std::uint8_t>(artifact.reconciler_status));
  for (const double value : artifact.anchor_weights) hash_double(hash, value);
  for (const double value : artifact.target_weights) hash_double(hash, value);
  hash_double(hash, artifact.anchor_distance);
  hash_double(hash, artifact.max_constraint_violation);
  hash_double(hash, artifact.predicted_cost);
  hash_double(hash, artifact.predicted_linear_cost);
  hash_double(hash, artifact.predicted_quadratic_cost);
  hash_double(hash, artifact.turnover);
  hash_double(hash, artifact.kkt_residual);
  hash_value(hash, artifact.active_constraint_count);
  hash_value(hash, artifact.eligible_for_official_risk ? 1U : 0U);
}

std::string json_escape(const std::string& value) {
  std::string escaped;
  for (const char character : value) {
    if (character == '\\' || character == '"') escaped.push_back('\\');
    escaped.push_back(character);
  }
  return escaped;
}

void append_double_array(std::ostringstream& output,
                         const std::vector<double>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
}

std::string unsigned_json(const PortfolioPolicyArtifact& artifact) {
  std::ostringstream output;
  output << std::setprecision(17);
  output << "{\"schema_version\":" << artifact.schema_version
         << ",\"policy\":\"" << policy_name(artifact.policy)
         << "\",\"policy_id\":\"" << json_escape(artifact.policy_id)
         << "\",\"policy_config_hash\":" << artifact.policy_config_hash
         << ",\"official_risk_model_sha256\":\""
         << json_escape(artifact.official_risk_model_sha256)
         << "\",\"cluster_model_sha256\":\""
         << json_escape(artifact.cluster_model_sha256)
         << "\",\"posterior_scenario_sha256\":\""
         << json_escape(artifact.posterior_scenario_sha256)
         << "\",\"intra_cluster_objective_hash\":"
         << artifact.intra_cluster_objective_hash
         << ",\"inter_cluster_objective_hash\":"
         << artifact.inter_cluster_objective_hash
         << ",\"reconciler_spec_hash\":" << artifact.reconciler_spec_hash
         << ",\"anchor_weights_sha256\":\""
         << json_escape(artifact.anchor_weights_sha256)
         << "\",\"target_weights_sha256\":\""
         << json_escape(artifact.target_weights_sha256)
         << "\",\"anchor_status\":\"" << status_name(artifact.anchor_status)
         << "\",\"reconciler_status\":\""
         << status_name(artifact.reconciler_status) << "\",\"anchor_weights\":";
  append_double_array(output, artifact.anchor_weights);
  output << ",\"target_weights\":";
  append_double_array(output, artifact.target_weights);
  output << ",\"anchor_distance\":" << artifact.anchor_distance
         << ",\"max_constraint_violation\":" << artifact.max_constraint_violation
         << ",\"predicted_cost\":" << artifact.predicted_cost
         << ",\"predicted_linear_cost\":" << artifact.predicted_linear_cost
         << ",\"predicted_quadratic_cost\":" << artifact.predicted_quadratic_cost
         << ",\"turnover\":" << artifact.turnover
         << ",\"kkt_residual\":" << artifact.kkt_residual
         << ",\"active_constraint_count\":" << artifact.active_constraint_count
         << ",\"eligible_for_official_risk\":false}";
  return output.str();
}

}  // namespace

bool valid_portfolio_policy_artifact(
    const PortfolioPolicyArtifact& artifact) noexcept {
  return valid_base(artifact) && artifact.artifact_hash != 0 &&
         artifact.artifact_hash == portfolio_policy_artifact_hash(artifact);
}

bool finalize_portfolio_policy_artifact(
    PortfolioPolicyArtifact& artifact) noexcept {
  if (!valid_base(artifact)) return false;
  artifact.artifact_hash = portfolio_policy_artifact_hash(artifact);
  return artifact.artifact_hash != 0;
}

std::uint64_t portfolio_policy_artifact_hash(
    const PortfolioPolicyArtifact& artifact) noexcept {
  if (!valid_base(artifact)) return 0;
  std::uint64_t hash = kFnvOffset;
  hash_base(hash, artifact);
  return hash;
}

std::string serialize_portfolio_policy_artifact(
    const PortfolioPolicyArtifact& artifact) {
  if (!valid_portfolio_policy_artifact(artifact)) return {};
  std::ostringstream output;
  output << unsigned_json(artifact);
  output.seekp(-1, std::ios_base::end);
  output << ",\"artifact_hash\":" << artifact.artifact_hash << '}';
  return output.str();
}

}  // namespace portfolio_math

#include "portfolio_math/nco_policy_comparison.h"

#include <bit>
#include <cmath>
#include <iomanip>
#include <numeric>
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

void hash_weights(std::uint64_t& hash, std::span<const double> weights) {
  hash_value(hash, weights.size());
  for (const double value : weights) hash_double(hash, value);
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

bool finite_nonnegative(std::span<const double> values) {
  for (const double value : values) {
    if (!std::isfinite(value) || value < 0.0) return false;
  }
  return true;
}

double evaluated_expected_return(const PosteriorScenarioArtifactV1& posterior,
                                 std::span<const double> weights) {
  if (posterior.posterior_mean.size() != weights.size()) return 0.0;
  const double value = std::inner_product(posterior.posterior_mean.begin(),
                                           posterior.posterior_mean.end(),
                                           weights.begin(), 0.0);
  return std::isfinite(value) ? value : 0.0;
}

double evaluated_variance(const PosteriorScenarioArtifactV1& posterior,
                          std::span<const double> weights) {
  if (posterior.posterior_covariance.rows() !=
          static_cast<Eigen::Index>(weights.size()) ||
      posterior.posterior_covariance.cols() !=
          static_cast<Eigen::Index>(weights.size())) return 0.0;
  const Eigen::Map<const Eigen::VectorXd> vector(
      weights.data(), static_cast<Eigen::Index>(weights.size()));
  const double variance =
      (vector.transpose() * posterior.posterior_covariance * vector)(0, 0);
  return std::isfinite(variance) && variance >= 0.0 ? variance : 0.0;
}

std::uint64_t anchor_hash(std::uint64_t provenance_hash,
                          OptimizationStatus status,
                          std::span<const double> weights,
                          double expected_return,
                          double variance) noexcept {
  std::uint64_t hash = kFnvOffset;
  hash_value(hash, provenance_hash);
  hash_byte(hash, static_cast<std::uint8_t>(status));
  hash_weights(hash, weights);
  hash_double(hash, expected_return);
  hash_double(hash, variance);
  return hash;
}

void hash_snapshot(std::uint64_t& hash,
                   const NcoPolicyComparisonSnapshot& snapshot) {
  hash_byte(hash, static_cast<std::uint8_t>(snapshot.anchor_status));
  hash_byte(hash, static_cast<std::uint8_t>(snapshot.reconciler_status));
  hash_value(hash, snapshot.anchor_artifact_hash);
  hash_weights(hash, snapshot.anchor_weights);
  hash_weights(hash, snapshot.target_weights);
  hash_double(hash, snapshot.evaluated_expected_return);
  hash_double(hash, snapshot.evaluated_variance);
  hash_byte(hash, static_cast<std::uint8_t>(snapshot.reconciler.status));
  hash_value(hash, snapshot.reconciler.iterations);
  hash_value(hash, snapshot.reconciler.active_constraint_count);
  hash_double(hash, snapshot.reconciler.anchor_distance);
  hash_double(hash, snapshot.reconciler.max_constraint_violation);
  hash_double(hash, snapshot.reconciler.predicted_cost);
  hash_double(hash, snapshot.reconciler.predicted_linear_cost);
  hash_double(hash, snapshot.reconciler.predicted_quadratic_cost);
  hash_double(hash, snapshot.reconciler.turnover);
  hash_double(hash, snapshot.reconciler.kkt_residual);
  hash_byte(hash, snapshot.reconciler.eligible_for_official_risk ? 1 : 0);
}

void serialize_snapshot(std::ostringstream& output,
                        const NcoPolicyComparisonSnapshot& snapshot) {
  output << "{\"anchor_status\":\"" << status_name(snapshot.anchor_status)
         << "\",\"reconciler_status\":\""
         << status_name(snapshot.reconciler_status)
         << "\",\"anchor_artifact_hash\":" << snapshot.anchor_artifact_hash
         << ",\"anchor_weights\":[";
  for (std::size_t index = 0; index < snapshot.anchor_weights.size(); ++index) {
    if (index != 0) output << ',';
    output << std::setprecision(17) << snapshot.anchor_weights[index];
  }
  output << "],\"target_weights\":[";
  for (std::size_t index = 0; index < snapshot.target_weights.size(); ++index) {
    if (index != 0) output << ',';
    output << std::setprecision(17) << snapshot.target_weights[index];
  }
  output << "],\"evaluated_expected_return\":"
         << std::setprecision(17) << snapshot.evaluated_expected_return
         << ",\"evaluated_variance\":" << snapshot.evaluated_variance
         << ",\"predicted_cost\":" << snapshot.reconciler.predicted_cost
         << ",\"predicted_linear_cost\":"
         << snapshot.reconciler.predicted_linear_cost
         << ",\"predicted_quadratic_cost\":"
         << snapshot.reconciler.predicted_quadratic_cost
         << ",\"turnover\":" << snapshot.reconciler.turnover
         << ",\"max_constraint_violation\":"
         << snapshot.reconciler.max_constraint_violation << '}';
}

bool same_target_investment(const NcoPolicyComparisonOptions& options) {
  return std::abs(options.posterior_direct.target_investment -
                  options.nco_ffv.nco.target_investment) <= 1e-12 &&
      std::abs(options.posterior_direct.target_investment -
               options.reconciler.target_investment) <= 1e-12;
}

void reconcile_snapshot(
    NcoPolicyComparisonSnapshot& snapshot,
    std::span<const double> current_weights,
    std::span<const double> anchor_penalty,
    std::span<const double> linear_cost,
    std::span<const double> quadratic_impact,
    const SinglePeriodReconcilerOptions& options) {
  if (snapshot.anchor_status != OptimizationStatus::OK) return;
  const auto reconciled = reconcile_single_period(
      snapshot.anchor_weights, current_weights, anchor_penalty, linear_cost,
      quadratic_impact, options);
  snapshot.reconciler_status = reconciled.diagnostics.status;
  snapshot.reconciler = reconciled.diagnostics;
  if (snapshot.reconciler_status == OptimizationStatus::OK) {
    snapshot.target_weights = reconciled.target_weights;
  }
}

}  // namespace

NcoPolicyComparisonArtifactV1 compare_nco_policy_family(
    const PosteriorScenarioArtifactV1& posterior,
    std::span<const std::uint32_t> cluster_id_by_symbol,
    std::uint32_t cluster_count,
    std::span<const double> current_weights,
    std::span<const double> anchor_penalty,
    std::span<const double> linear_cost,
    std::span<const double> quadratic_impact,
    NcoPolicyComparisonOptions options) {
  NcoPolicyComparisonArtifactV1 comparison;
  comparison.posterior_artifact_hash = posterior.artifact_hash;
  if (cluster_count > 0 && cluster_id_by_symbol.size() == posterior.asset_count) {
    comparison.cluster_spec_hash =
        solve_nco_ffv_minvar(posterior, cluster_id_by_symbol, cluster_count,
                             options.nco_ffv)
            .cluster_spec_hash;
  }
  const auto size = posterior.asset_count;
  if (!valid_posterior_scenario_artifact(posterior) || size == 0 ||
      cluster_count == 0 || cluster_id_by_symbol.size() != size ||
      current_weights.size() != size || anchor_penalty.size() != size ||
      linear_cost.size() != size || quadratic_impact.size() != size ||
      !finite_nonnegative(current_weights) || !finite_nonnegative(anchor_penalty) ||
      !finite_nonnegative(linear_cost) || !finite_nonnegative(quadratic_impact) ||
      !valid_posterior_direct_options(options.posterior_direct) ||
      !valid_nco_policy_options(options.nco_ffv.nco) ||
      !valid_single_period_reconciler_options(options.reconciler) ||
      !options.reconciler.costs_available || !same_target_investment(options)) {
    comparison.artifact_hash = nco_policy_comparison_artifact_hash(comparison);
    return comparison;
  }

  const auto direct = solve_posterior_direct(posterior, options.posterior_direct);
  comparison.posterior_direct.anchor_status = direct.diagnostics.status;
  comparison.posterior_direct.anchor_weights = direct.weights;
  comparison.posterior_direct.anchor_artifact_hash = anchor_hash(
      direct.diagnostics.posterior_artifact_hash, direct.diagnostics.status,
      direct.weights, direct.diagnostics.expected_return,
      direct.diagnostics.variance);
  comparison.posterior_direct.evaluated_expected_return =
      evaluated_expected_return(posterior, direct.weights);
  comparison.posterior_direct.evaluated_variance =
      evaluated_variance(posterior, direct.weights);

  const auto ffv = solve_nco_ffv_minvar(
      posterior, cluster_id_by_symbol, cluster_count, options.nco_ffv);
  comparison.nco_ffv.anchor_status = ffv.status;
  comparison.nco_ffv.anchor_weights = ffv.nco.weights;
  comparison.nco_ffv.anchor_artifact_hash = ffv.artifact_hash;
  comparison.nco_ffv.evaluated_expected_return =
      evaluated_expected_return(posterior, ffv.nco.weights);
  comparison.nco_ffv.evaluated_variance =
      evaluated_variance(posterior, ffv.nco.weights);

  const auto risk_only = solve_nco_minvar(
      quant_math::view(posterior.prior_covariance), cluster_id_by_symbol,
      cluster_count, options.nco_ffv.nco);
  comparison.nco_risk_only.anchor_status = risk_only.diagnostics.status;
  comparison.nco_risk_only.anchor_weights = risk_only.weights;
  comparison.nco_risk_only.anchor_artifact_hash = anchor_hash(
      posterior.prior_scenario_hash, risk_only.diagnostics.status,
      risk_only.weights, 0.0, risk_only.diagnostics.predicted_risk *
          risk_only.diagnostics.predicted_risk);
  comparison.nco_risk_only.evaluated_expected_return =
      evaluated_expected_return(posterior, risk_only.weights);
  comparison.nco_risk_only.evaluated_variance =
      evaluated_variance(posterior, risk_only.weights);

  reconcile_snapshot(comparison.posterior_direct, current_weights,
                     anchor_penalty, linear_cost, quadratic_impact,
                     options.reconciler);
  reconcile_snapshot(comparison.nco_ffv, current_weights, anchor_penalty,
                     linear_cost, quadratic_impact, options.reconciler);
  reconcile_snapshot(comparison.nco_risk_only, current_weights, anchor_penalty,
                     linear_cost, quadratic_impact, options.reconciler);

  comparison.status = comparison.posterior_direct.anchor_status;
  if (comparison.status == OptimizationStatus::OK)
    comparison.status = comparison.nco_ffv.anchor_status;
  if (comparison.status == OptimizationStatus::OK)
    comparison.status = comparison.nco_risk_only.anchor_status;
  if (comparison.status == OptimizationStatus::OK)
    comparison.status = comparison.posterior_direct.reconciler_status;
  if (comparison.status == OptimizationStatus::OK)
    comparison.status = comparison.nco_ffv.reconciler_status;
  if (comparison.status == OptimizationStatus::OK)
    comparison.status = comparison.nco_risk_only.reconciler_status;
  comparison.winner_selected = false;
  comparison.eligible_for_official_risk = false;
  comparison.artifact_hash = nco_policy_comparison_artifact_hash(comparison);
  return comparison;
}

std::uint64_t nco_policy_comparison_artifact_hash(
    const NcoPolicyComparisonArtifactV1& comparison) noexcept {
  std::uint64_t hash = kFnvOffset;
  hash_value(hash, comparison.schema_version);
  hash_byte(hash, static_cast<std::uint8_t>(comparison.status));
  hash_value(hash, comparison.posterior_artifact_hash);
  hash_value(hash, comparison.cluster_spec_hash);
  hash_snapshot(hash, comparison.posterior_direct);
  hash_snapshot(hash, comparison.nco_ffv);
  hash_snapshot(hash, comparison.nco_risk_only);
  hash_byte(hash, comparison.winner_selected ? 1 : 0);
  hash_byte(hash, comparison.eligible_for_official_risk ? 1 : 0);
  return hash;
}

std::string serialize_nco_policy_comparison(
    const NcoPolicyComparisonArtifactV1& comparison) {
  std::ostringstream output;
  output << "{\"schema_version\":" << comparison.schema_version
         << ",\"status\":\"" << status_name(comparison.status)
         << "\",\"posterior_artifact_hash\":"
         << comparison.posterior_artifact_hash
         << ",\"cluster_spec_hash\":" << comparison.cluster_spec_hash
         << ",\"posterior_direct\":";
  serialize_snapshot(output, comparison.posterior_direct);
  output << ",\"nco_ffv\":";
  serialize_snapshot(output, comparison.nco_ffv);
  output << ",\"nco_risk_only\":";
  serialize_snapshot(output, comparison.nco_risk_only);
  output << ",\"winner_selected\":false"
         << ",\"eligible_for_official_risk\":false"
         << ",\"artifact_hash\":"
         << nco_policy_comparison_artifact_hash(comparison) << '}';
  return output.str();
}

}  // namespace portfolio_math

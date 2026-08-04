#include "performance_analytics/factor_attribution_v2.h"

#include <bit>
#include <cmath>
#include <numeric>

namespace performance_analytics {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_value(std::uint64_t& hash, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    hash ^= (value >> shift) & 0xffU;
    hash *= kFnvPrime;
  }
}

void hash_double(std::uint64_t& hash, double value) {
  hash_value(hash, std::bit_cast<std::uint64_t>(value == 0.0 ? 0.0 : value));
}

bool finite_span(std::span<const double> values) {
  for (const double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

}  // namespace

FactorAttributionResultV2 compute_factor_attribution_v2(
    const FactorAttributionProblemV2& problem) {
  FactorAttributionResultV2 result;
  result.period_id = problem.period_id;
  result.accounting_portfolio_return = problem.accounting_portfolio_return;
  if (problem.period_id == 0 || problem.exposures.data == nullptr ||
      problem.exposures.rows == 0 || problem.exposures.cols == 0 ||
      problem.exposures.row_stride < problem.exposures.cols ||
      problem.asset_returns.size() != problem.exposures.rows ||
      problem.portfolio_weights.size() != problem.exposures.rows ||
      problem.factor_returns.size() != problem.exposures.cols ||
      !std::isfinite(problem.accounting_portfolio_return) ||
      !finite_span(problem.asset_returns) || !finite_span(problem.factor_returns) ||
      !finite_span(problem.portfolio_weights) ||
      !std::isfinite(problem.spec.reconciliation_tolerance) ||
      problem.spec.reconciliation_tolerance <= 0.0 ||
      problem.spec.factor_schema_hash == 0 || problem.spec.pit_exposure_hash == 0 ||
      problem.spec.config_hash == 0) {
    return result;
  }
  if (problem.available_at == 0 || problem.decision_at == 0 ||
      problem.available_at > problem.decision_at) {
    result.status = FactorAttributionStatus::FUTURE_DATA;
    return result;
  }
  result.portfolio_factor_exposure.assign(problem.exposures.cols, 0.0);
  result.factor_contributions.assign(problem.exposures.cols, 0.0);
  for (std::size_t asset = 0; asset < problem.exposures.rows; ++asset) {
    if (problem.portfolio_weights[asset] < 0.0) return result;
    const double weight = problem.portfolio_weights[asset];
    const double asset_return = problem.asset_returns[asset];
    double fitted_return = 0.0;
    for (std::size_t factor = 0; factor < problem.exposures.cols; ++factor) {
      const double exposure = problem.exposures(asset, factor);
      if (!std::isfinite(exposure)) return result;
      result.portfolio_factor_exposure[factor] += weight * exposure;
      fitted_return += exposure * problem.factor_returns[factor];
    }
    result.specific_contribution += weight * (asset_return - fitted_return);
  }
  for (std::size_t factor = 0; factor < problem.exposures.cols; ++factor) {
    result.factor_contributions[factor] =
        result.portfolio_factor_exposure[factor] * problem.factor_returns[factor];
    result.factor_contribution_total += result.factor_contributions[factor];
  }
  result.reconstructed_portfolio_return = result.factor_contribution_total +
      result.specific_contribution;
  result.reconciliation_residual = result.reconstructed_portfolio_return -
      problem.accounting_portfolio_return;
  const double tolerance = problem.spec.reconciliation_tolerance;
  if (std::abs(result.reconciliation_residual) > tolerance * std::max(
          1.0, std::abs(problem.accounting_portfolio_return))) {
    result.status = FactorAttributionStatus::RECONCILIATION_FAILURE;
    return result;
  }
  result.artifact_hash = kFnvOffset;
  hash_value(result.artifact_hash, problem.spec.factor_schema_hash);
  hash_value(result.artifact_hash, problem.spec.pit_exposure_hash);
  hash_value(result.artifact_hash, problem.spec.config_hash);
  hash_value(result.artifact_hash, problem.period_id);
  hash_value(result.artifact_hash, problem.available_at);
  hash_value(result.artifact_hash, problem.decision_at);
  for (const double value : problem.portfolio_weights) hash_double(result.artifact_hash, value);
  for (const double value : problem.asset_returns) hash_double(result.artifact_hash, value);
  for (const double value : problem.factor_returns) hash_double(result.artifact_hash, value);
  for (const double value : result.portfolio_factor_exposure) hash_double(result.artifact_hash, value);
  result.status = FactorAttributionStatus::OK;
  return result;
}

}  // namespace performance_analytics

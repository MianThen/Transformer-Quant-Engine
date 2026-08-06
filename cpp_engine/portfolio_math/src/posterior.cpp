#include "portfolio_math/posterior.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

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

const char* status_name(PosteriorStatus status) {
  switch (status) {
    case PosteriorStatus::OK: return "ok";
    case PosteriorStatus::INVALID_INPUT: return "invalid_input";
    case PosteriorStatus::FUTURE_DATA: return "future_data";
    case PosteriorStatus::NON_PSD_PRIOR: return "non_psd_prior";
    case PosteriorStatus::NUMERICAL_FAILURE: return "numerical_failure";
    case PosteriorStatus::INFEASIBLE: return "infeasible";
  }
  return "invalid";
}

const char* view_kind_name(PosteriorViewKind kind) {
  switch (kind) {
    case PosteriorViewKind::MEAN: return "mean";
    case PosteriorViewKind::MEAN_LOWER_BOUND: return "mean_lower_bound";
    case PosteriorViewKind::MEAN_UPPER_BOUND: return "mean_upper_bound";
  }
  return "invalid";
}

const char* view_family_name(PosteriorViewFamily family) {
  switch (family) {
    case PosteriorViewFamily::MEAN: return "mean";
    case PosteriorViewFamily::DIRECTION: return "direction";
    case PosteriorViewFamily::VOLATILITY: return "volatility";
    case PosteriorViewFamily::RANKING: return "ranking";
    case PosteriorViewFamily::QUANTILE: return "quantile";
  }
  return "invalid";
}

const char* engine_name(PosteriorEngineKind engine) {
  switch (engine) {
    case PosteriorEngineKind::GAUSSIAN_BL: return "gaussian_bl";
    case PosteriorEngineKind::FULLY_FLEXIBLE_VIEWS:
      return "fully_flexible_views";
  }
  return "invalid";
}

bool finite_vector(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

bool valid_probability_vector(std::span<const double> probabilities,
                              std::size_t expected_size) {
  if (probabilities.size() != expected_size || probabilities.empty()) return false;
  double sum = 0.0;
  for (double value : probabilities) {
    if (!std::isfinite(value) || value < 0.0) return false;
    sum += value;
  }
  return std::isfinite(sum) && std::abs(sum - 1.0) <= 1e-12;
}

std::uint64_t probability_hash(std::span<const double> probabilities) {
  std::uint64_t hash = kFnvOffset;
  hash_value(hash, probabilities.size());
  for (double value : probabilities) hash_double(hash, value);
  return hash;
}

bool finite_matrix(const quant_math::DenseMatrix& matrix) {
  return quant_math::validate_finite(quant_math::view(matrix)).ok;
}

bool psd(const quant_math::DenseMatrix& matrix) {
  return quant_math::is_positive_semidefinite(matrix, 1e-10);
}

quant_math::DenseMatrix copy_matrix(quant_math::MatrixView view) {
  quant_math::DenseMatrix result(
      static_cast<Eigen::Index>(view.rows), static_cast<Eigen::Index>(view.cols));
  for (std::size_t row = 0; row < view.rows; ++row) {
    for (std::size_t col = 0; col < view.cols; ++col) {
      result(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          view(row, col);
    }
  }
  return result;
}

void hash_matrix(std::uint64_t& hash, const quant_math::DenseMatrix& matrix) {
  hash_value(hash, static_cast<std::uint64_t>(matrix.rows()));
  hash_value(hash, static_cast<std::uint64_t>(matrix.cols()));
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      hash_double(hash, matrix(row, col));
    }
  }
}

std::uint64_t prior_hash(const PriorScenarioArtifactV1& artifact) {
  std::uint64_t hash = kFnvOffset;
  hash_value(hash, artifact.schema_version);
  hash_value(hash, static_cast<std::uint8_t>(artifact.status));
  hash_value(hash, static_cast<std::uint64_t>(artifact.fit_start));
  hash_value(hash, static_cast<std::uint64_t>(artifact.fit_end));
  hash_value(hash, static_cast<std::uint64_t>(artifact.available_at));
  hash_value(hash, static_cast<std::uint64_t>(artifact.decision_at));
  hash_value(hash, artifact.scenario_count);
  hash_value(hash, artifact.asset_count);
  hash_double(hash, artifact.effective_sample_size);
  for (auto timestamp : artifact.scenario_timestamps) {
    hash_value(hash, static_cast<std::uint64_t>(timestamp));
  }
  for (double value : artifact.scenario_values) hash_double(hash, value);
  for (double value : artifact.prior_probabilities) hash_double(hash, value);
  for (double value : artifact.support_min) hash_double(hash, value);
  for (double value : artifact.support_max) hash_double(hash, value);
  for (double value : artifact.prior_mean) hash_double(hash, value);
  hash_matrix(hash, artifact.prior_covariance);
  hash_value(hash, artifact.scenario_hash);
  hash_byte(hash, artifact.eligible_for_official_risk ? 1 : 0);
  return hash;
}

std::uint64_t view_set_hash(std::span<const ViewSpecV1> views) {
  std::uint64_t hash = kFnvOffset;
  hash_value(hash, views.size());
  for (const auto& view : views) hash_value(hash, view_spec_hash(view));
  return hash;
}

std::uint64_t posterior_hash_without_self(
    const PosteriorScenarioArtifactV1& artifact) {
  std::uint64_t hash = kFnvOffset;
  hash_value(hash, artifact.schema_version);
  hash_value(hash, static_cast<std::uint8_t>(artifact.status));
  hash_value(hash, static_cast<std::uint8_t>(artifact.engine));
  hash_value(hash, static_cast<std::uint64_t>(artifact.fit_start));
  hash_value(hash, static_cast<std::uint64_t>(artifact.fit_end));
  hash_value(hash, static_cast<std::uint64_t>(artifact.available_at));
  hash_value(hash, static_cast<std::uint64_t>(artifact.decision_at));
  hash_value(hash, artifact.scenario_count);
  hash_value(hash, artifact.asset_count);
  hash_double(hash, artifact.effective_sample_size);
  for (auto timestamp : artifact.scenario_timestamps) {
    hash_value(hash, static_cast<std::uint64_t>(timestamp));
  }
  for (double value : artifact.prior_mean) hash_double(hash, value);
  for (double value : artifact.posterior_mean) hash_double(hash, value);
  hash_matrix(hash, artifact.prior_covariance);
  hash_matrix(hash, artifact.posterior_covariance);
  for (double value : artifact.support_min) hash_double(hash, value);
  for (double value : artifact.support_max) hash_double(hash, value);
  hash_value(hash, artifact.view_count);
  hash_value(hash, artifact.active_constraint_count);
  hash_value(hash, artifact.iterations);
  hash_double(hash, artifact.maximum_view_residual);
  hash_double(hash, artifact.kl_divergence);
  hash_byte(hash, artifact.kl_divergence_available ? 1 : 0);
  hash_double(hash, artifact.maximum_scenario_weight);
  hash_value(hash, artifact.posterior_probability_hash);
  for (double value : artifact.posterior_probabilities) hash_double(hash, value);
  for (double value : artifact.posterior_quantile_levels) hash_double(hash, value);
  for (double value : artifact.posterior_quantiles) hash_double(hash, value);
  for (double value : artifact.posterior_expected_shortfall) hash_double(hash, value);
  hash_byte(hash, artifact.support_guard_passed ? 1 : 0);
  hash_value(hash, artifact.prior_scenario_hash);
  hash_value(hash, artifact.view_spec_hash);
  hash_value(hash, artifact.confidence_mapping_hash);
  hash_byte(hash, artifact.eligible_for_official_risk ? 1 : 0);
  return hash;
}

void json_vector(std::ostringstream& output, const std::vector<double>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << std::setprecision(17) << values[index];
  }
  output << ']';
}

void json_matrix(std::ostringstream& output, const quant_math::DenseMatrix& matrix) {
  output << '[';
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    if (row != 0) output << ',';
    output << '[';
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      if (col != 0) output << ',';
      output << std::setprecision(17) << matrix(row, col);
    }
    output << ']';
  }
  output << ']';
}

}  // namespace

bool valid_view_spec(const ViewSpecV1& view, std::size_t asset_count,
                     engine_common::TimestampNs decision_at) noexcept {
  return view.schema_version == 1 && !view.view_id.empty() &&
         (view.kind == PosteriorViewKind::MEAN ||
          view.kind == PosteriorViewKind::MEAN_LOWER_BOUND ||
          view.kind == PosteriorViewKind::MEAN_UPPER_BOUND) &&
         asset_count > 0 &&
         view.available_at > 0 && decision_at > 0 &&
         view.available_at <= decision_at && view.loading.size() == asset_count &&
         finite_vector(view.loading) && std::isfinite(view.target) &&
         std::isfinite(view.statistic_threshold) &&
         std::isfinite(view.confidence) && view.confidence >= 0.0 &&
         view.confidence <= 1.0 && std::isfinite(view.observation_variance) &&
         view.observation_variance > 0.0 && view.confidence_mapping_hash != 0 &&
         (view.family == PosteriorViewFamily::MEAN ||
          view.family == PosteriorViewFamily::DIRECTION ||
          view.family == PosteriorViewFamily::VOLATILITY ||
          view.family == PosteriorViewFamily::RANKING ||
          view.family == PosteriorViewFamily::QUANTILE) &&
         (view.family == PosteriorViewFamily::MEAN ||
          view.calibration_artifact_hash != 0) &&
         (view.family != PosteriorViewFamily::DIRECTION ||
          (view.kind == PosteriorViewKind::MEAN &&
           view.target >= 0.0 && view.target <= 1.0)) &&
         (view.family != PosteriorViewFamily::VOLATILITY ||
          (view.kind == PosteriorViewKind::MEAN && view.target > 0.0)) &&
         view.source_artifact_hash != 0;
}

std::uint64_t view_spec_hash(const ViewSpecV1& view) noexcept {
  std::uint64_t hash = kFnvOffset;
  hash_value(hash, view.schema_version);
  hash_string(hash, view.view_id);
  hash_value(hash, static_cast<std::uint8_t>(view.kind));
  hash_value(hash, static_cast<std::uint8_t>(view.family));
  hash_value(hash, static_cast<std::uint64_t>(view.available_at));
  for (double value : view.loading) hash_double(hash, value);
  hash_double(hash, view.target);
  hash_double(hash, view.statistic_threshold);
  hash_double(hash, view.confidence);
  hash_double(hash, view.observation_variance);
  hash_value(hash, view.confidence_mapping_hash);
  hash_value(hash, view.calibration_artifact_hash);
  hash_value(hash, view.source_artifact_hash);
  return hash;
}

bool valid_prior_scenario_artifact(
    const PriorScenarioArtifactV1& artifact) noexcept {
  return artifact.schema_version == 1 && artifact.status == PosteriorStatus::OK &&
         artifact.fit_start > 0 && artifact.fit_start <= artifact.fit_end &&
         artifact.available_at > 0 && artifact.decision_at > 0 &&
         artifact.available_at <= artifact.decision_at && artifact.scenario_count >= 2 &&
         artifact.asset_count > 0 &&
         std::isfinite(artifact.effective_sample_size) &&
         artifact.effective_sample_size > 0.0 &&
         artifact.scenario_timestamps.size() == artifact.scenario_count &&
         artifact.scenario_values.size() == artifact.scenario_count * artifact.asset_count &&
         valid_probability_vector(artifact.prior_probabilities, artifact.scenario_count) &&
         artifact.support_min.size() == artifact.asset_count &&
         artifact.support_max.size() == artifact.asset_count &&
         finite_vector(artifact.scenario_values) && finite_vector(artifact.support_min) &&
         finite_vector(artifact.support_max) &&
         std::all_of(artifact.scenario_timestamps.begin(),
                     artifact.scenario_timestamps.end(), [](auto timestamp) {
                       return timestamp > 0;
                     }) &&
         std::all_of(artifact.scenario_timestamps.begin(),
                     artifact.scenario_timestamps.end(), [&artifact](auto timestamp) {
                       return timestamp <= artifact.available_at &&
                              timestamp <= artifact.decision_at;
                     }) &&
         std::adjacent_find(artifact.scenario_timestamps.begin(),
                            artifact.scenario_timestamps.end(),
                            [](auto left, auto right) { return left >= right; }) ==
             artifact.scenario_timestamps.end() &&
         std::equal(artifact.support_min.begin(), artifact.support_min.end(),
                    artifact.support_max.begin(), [](double lower, double upper) {
                      return lower <= upper;
                    }) &&
         finite_vector(artifact.prior_mean) &&
         artifact.prior_mean.size() == artifact.asset_count &&
         artifact.prior_covariance.rows() ==
             static_cast<Eigen::Index>(artifact.prior_mean.size()) &&
         artifact.prior_covariance.cols() == artifact.prior_covariance.rows() &&
         finite_matrix(artifact.prior_covariance) && psd(artifact.prior_covariance) &&
         artifact.scenario_hash != 0 && !artifact.eligible_for_official_risk;
}

PriorScenarioArtifactV1 build_prior_scenario_artifact(
    quant_math::MatrixView scenario_returns,
    std::span<const engine_common::TimestampNs> scenario_timestamps,
    engine_common::TimestampNs available_at,
    engine_common::TimestampNs decision_at) {
  PriorScenarioArtifactV1 artifact;
  artifact.available_at = available_at;
  artifact.decision_at = decision_at;
  if (scenario_returns.rows < 2 || scenario_returns.cols == 0 ||
      scenario_timestamps.size() != scenario_returns.rows || available_at <= 0 ||
      decision_at <= 0 || available_at > decision_at ||
      !quant_math::validate_finite(scenario_returns).ok) {
    artifact.status = PosteriorStatus::INVALID_INPUT;
    artifact.artifact_hash = prior_hash(artifact);
    return artifact;
  }
  for (std::size_t index = 0; index < scenario_timestamps.size(); ++index) {
    const auto timestamp = scenario_timestamps[index];
    if (timestamp > decision_at || timestamp > available_at) {
      artifact.status = PosteriorStatus::FUTURE_DATA;
      artifact.artifact_hash = prior_hash(artifact);
      return artifact;
    }
    if (index > 0 && timestamp <= scenario_timestamps[index - 1]) {
      artifact.status = PosteriorStatus::INVALID_INPUT;
      artifact.artifact_hash = prior_hash(artifact);
      return artifact;
    }
  }
  const auto values = copy_matrix(scenario_returns);
  artifact.fit_start = scenario_timestamps.front();
  artifact.fit_end = scenario_timestamps.back();
  artifact.scenario_count = scenario_returns.rows;
  artifact.asset_count = scenario_returns.cols;
  artifact.effective_sample_size = static_cast<double>(scenario_returns.rows);
  artifact.scenario_timestamps.assign(scenario_timestamps.begin(),
                                      scenario_timestamps.end());
  artifact.scenario_values.assign(
      scenario_returns.data,
      scenario_returns.data + scenario_returns.rows * scenario_returns.row_stride);
  if (scenario_returns.row_stride != scenario_returns.cols) {
    artifact.scenario_values.clear();
    artifact.scenario_values.reserve(scenario_returns.rows * scenario_returns.cols);
    for (std::size_t row = 0; row < scenario_returns.rows; ++row) {
      for (std::size_t col = 0; col < scenario_returns.cols; ++col) {
        artifact.scenario_values.push_back(scenario_returns(row, col));
      }
    }
  }
  artifact.prior_probabilities.assign(
      scenario_returns.rows, 1.0 / static_cast<double>(scenario_returns.rows));
  artifact.support_min.assign(scenario_returns.cols,
                              std::numeric_limits<double>::infinity());
  artifact.support_max.assign(scenario_returns.cols,
                              -std::numeric_limits<double>::infinity());
  for (std::size_t row = 0; row < scenario_returns.rows; ++row) {
    for (std::size_t col = 0; col < scenario_returns.cols; ++col) {
      artifact.support_min[col] = std::min(artifact.support_min[col], scenario_returns(row, col));
      artifact.support_max[col] = std::max(artifact.support_max[col], scenario_returns(row, col));
    }
  }
  const Eigen::VectorXd mean = values.colwise().mean();
  artifact.prior_mean.assign(mean.data(), mean.data() + mean.size());
  const auto centered = values.rowwise() - mean.transpose();
  const auto covariance = centered.transpose() * centered /
                          static_cast<double>(scenario_returns.rows - 1);
  artifact.prior_covariance = 0.5 * (covariance + covariance.transpose());
  if (!psd(artifact.prior_covariance)) {
    artifact.status = PosteriorStatus::NON_PSD_PRIOR;
    artifact.artifact_hash = prior_hash(artifact);
    return artifact;
  }
  std::uint64_t scenario_digest = kFnvOffset;
  hash_value(scenario_digest, scenario_returns.rows);
  hash_value(scenario_digest, scenario_returns.cols);
  for (auto timestamp : scenario_timestamps) {
    hash_value(scenario_digest, static_cast<std::uint64_t>(timestamp));
  }
  for (std::size_t row = 0; row < scenario_returns.rows; ++row) {
    for (std::size_t col = 0; col < scenario_returns.cols; ++col) {
      hash_double(scenario_digest, scenario_returns(row, col));
    }
  }
  artifact.scenario_hash = scenario_digest;
  artifact.status = PosteriorStatus::OK;
  artifact.artifact_hash = prior_hash(artifact);
  return artifact;
}

bool valid_posterior_scenario_artifact(
    const PosteriorScenarioArtifactV1& artifact) noexcept {
  const bool base = artifact.schema_version == 1 && artifact.status == PosteriorStatus::OK &&
         artifact.fit_start > 0 && artifact.fit_start <= artifact.fit_end &&
         artifact.available_at > 0 && artifact.decision_at > 0 &&
         artifact.available_at <= artifact.decision_at && artifact.scenario_count >= 2 &&
         artifact.asset_count > 0 &&
         std::isfinite(artifact.effective_sample_size) && artifact.effective_sample_size > 0.0 &&
         artifact.scenario_timestamps.size() == artifact.scenario_count &&
         std::all_of(artifact.scenario_timestamps.begin(),
                     artifact.scenario_timestamps.end(), [](auto timestamp) {
                       return timestamp > 0;
                     }) &&
         std::all_of(artifact.scenario_timestamps.begin(),
                     artifact.scenario_timestamps.end(), [&artifact](auto timestamp) {
                       return timestamp <= artifact.available_at &&
                              timestamp <= artifact.decision_at;
                     }) &&
         std::adjacent_find(artifact.scenario_timestamps.begin(),
                            artifact.scenario_timestamps.end(),
                            [](auto left, auto right) { return left >= right; }) ==
             artifact.scenario_timestamps.end() &&
         finite_vector(artifact.prior_mean) && finite_vector(artifact.posterior_mean) &&
         artifact.prior_mean.size() == artifact.asset_count &&
         artifact.posterior_mean.size() == artifact.asset_count &&
         artifact.prior_covariance.rows() ==
             static_cast<Eigen::Index>(artifact.prior_mean.size()) &&
         artifact.posterior_covariance.rows() == artifact.prior_covariance.rows() &&
         artifact.prior_covariance.cols() == artifact.prior_covariance.rows() &&
         artifact.posterior_covariance.cols() == artifact.posterior_covariance.rows() &&
         artifact.support_min.size() == artifact.asset_count &&
         artifact.support_max.size() == artifact.asset_count &&
         finite_vector(artifact.support_min) && finite_vector(artifact.support_max) &&
         std::equal(artifact.support_min.begin(), artifact.support_min.end(),
                    artifact.support_max.begin(), [](double lower, double upper) {
                      return lower <= upper;
                    }) &&
         finite_matrix(artifact.prior_covariance) && finite_matrix(artifact.posterior_covariance) &&
         psd(artifact.prior_covariance) && psd(artifact.posterior_covariance) &&
         std::isfinite(artifact.maximum_view_residual) && artifact.maximum_view_residual >= 0.0 &&
         std::isfinite(artifact.kl_divergence) && artifact.kl_divergence >= 0.0 &&
         artifact.support_guard_passed && artifact.prior_scenario_hash != 0 &&
         artifact.view_spec_hash != 0 &&
         ((artifact.view_count == 0 && artifact.confidence_mapping_hash == 0) ||
          (artifact.view_count > 0 && artifact.confidence_mapping_hash != 0)) &&
         !artifact.eligible_for_official_risk;
  if (!base) return false;
  if (artifact.engine == PosteriorEngineKind::GAUSSIAN_BL) {
    return artifact.posterior_probabilities.empty() &&
           artifact.posterior_probability_hash == 0 &&
           artifact.posterior_quantile_levels.empty() &&
           artifact.posterior_quantiles.empty() &&
           artifact.posterior_expected_shortfall.empty() &&
           artifact.maximum_scenario_weight == 0.0;
  }
  if (artifact.engine != PosteriorEngineKind::FULLY_FLEXIBLE_VIEWS) return false;
  const std::size_t stat_count =
      artifact.posterior_quantile_levels.size() * artifact.asset_count;
  return valid_probability_vector(artifact.posterior_probabilities,
                                  artifact.scenario_count) &&
         artifact.posterior_probability_hash ==
             probability_hash(artifact.posterior_probabilities) &&
         finite_vector(artifact.posterior_quantile_levels) &&
         std::all_of(artifact.posterior_quantile_levels.begin(),
                     artifact.posterior_quantile_levels.end(), [](double level) {
                       return level > 0.0 && level < 1.0;
                     }) &&
         finite_vector(artifact.posterior_quantiles) &&
         finite_vector(artifact.posterior_expected_shortfall) &&
         artifact.posterior_quantiles.size() == stat_count &&
         artifact.posterior_expected_shortfall.size() == stat_count &&
         artifact.kl_divergence_available &&
         std::isfinite(artifact.maximum_scenario_weight) &&
         artifact.maximum_scenario_weight > 0.0 &&
         artifact.maximum_scenario_weight <= 1.0;
}

PosteriorScenarioArtifactV1 apply_gaussian_mean_views(
    const PriorScenarioArtifactV1& prior,
    std::span<const ViewSpecV1> views) {
  PosteriorScenarioArtifactV1 artifact;
  artifact.fit_start = prior.fit_start;
  artifact.fit_end = prior.fit_end;
  artifact.available_at = prior.available_at;
  artifact.decision_at = prior.decision_at;
  artifact.scenario_count = prior.scenario_count;
  artifact.asset_count = prior.asset_count;
  artifact.effective_sample_size = prior.effective_sample_size;
  artifact.scenario_timestamps = prior.scenario_timestamps;
  artifact.prior_mean = prior.prior_mean;
  artifact.posterior_mean = prior.prior_mean;
  artifact.prior_covariance = prior.prior_covariance;
  artifact.posterior_covariance = prior.prior_covariance;
  artifact.support_min = prior.support_min;
  artifact.support_max = prior.support_max;
  artifact.prior_scenario_hash = prior.scenario_hash;
  artifact.view_spec_hash = view_set_hash(views);
  artifact.view_count = views.size();
  if (!views.empty()) artifact.confidence_mapping_hash = views.front().confidence_mapping_hash;
  if (!valid_prior_scenario_artifact(prior)) {
    artifact.status = prior.status == PosteriorStatus::FUTURE_DATA
                          ? PosteriorStatus::FUTURE_DATA
                          : PosteriorStatus::INVALID_INPUT;
    artifact.artifact_hash = posterior_hash_without_self(artifact);
    return artifact;
  }
  for (const auto& view : views) {
    if (view.family != PosteriorViewFamily::MEAN ||
        view.kind != PosteriorViewKind::MEAN ||
        !valid_view_spec(view, prior.prior_mean.size(), prior.decision_at)) {
      artifact.status = view.available_at > prior.decision_at
                            ? PosteriorStatus::FUTURE_DATA
                            : PosteriorStatus::INVALID_INPUT;
      artifact.artifact_hash = posterior_hash_without_self(artifact);
      return artifact;
    }
    if (view.confidence_mapping_hash != artifact.confidence_mapping_hash) {
      artifact.status = PosteriorStatus::INVALID_INPUT;
      artifact.artifact_hash = posterior_hash_without_self(artifact);
      return artifact;
    }
  }
  std::vector<const ViewSpecV1*> active;
  for (const auto& view : views) {
    if (view.confidence > 0.0) active.push_back(&view);
  }
  if (!active.empty()) {
    const Eigen::Index dimensions = static_cast<Eigen::Index>(prior.prior_mean.size());
    const Eigen::Index count = static_cast<Eigen::Index>(active.size());
    Eigen::MatrixXd loading(count, dimensions);
    Eigen::VectorXd target(count);
    Eigen::VectorXd noise(count);
    for (Eigen::Index row = 0; row < count; ++row) {
      const auto& view = *active[static_cast<std::size_t>(row)];
      for (Eigen::Index col = 0; col < dimensions; ++col) {
        loading(row, col) = view.loading[static_cast<std::size_t>(col)];
      }
      target(row) = view.target;
      noise(row) = view.observation_variance * (1.0 - view.confidence);
    }
    const Eigen::MatrixXd covariance = prior.prior_covariance;
    Eigen::MatrixXd view_covariance = loading * covariance * loading.transpose();
    view_covariance.diagonal() += noise;
    Eigen::LDLT<Eigen::MatrixXd> solver(view_covariance);
    if (solver.info() != Eigen::Success ||
        (solver.vectorD().array() <= 1e-14).any()) {
      artifact.status = PosteriorStatus::NUMERICAL_FAILURE;
      artifact.artifact_hash = posterior_hash_without_self(artifact);
      return artifact;
    }
    const Eigen::VectorXd mean = Eigen::Map<const Eigen::VectorXd>(
        prior.prior_mean.data(), dimensions);
    const Eigen::VectorXd innovation = target - loading * mean;
    const Eigen::MatrixXd gain = covariance * loading.transpose() * solver.solve(
        Eigen::MatrixXd::Identity(count, count));
    const Eigen::VectorXd posterior_mean = mean + gain * innovation;
    Eigen::MatrixXd posterior_covariance = covariance - gain * loading * covariance;
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.transpose());
    if (!posterior_mean.allFinite() || !posterior_covariance.allFinite() ||
        !psd(posterior_covariance)) {
      artifact.status = PosteriorStatus::NUMERICAL_FAILURE;
      artifact.artifact_hash = posterior_hash_without_self(artifact);
      return artifact;
    }
    artifact.posterior_mean.assign(posterior_mean.data(), posterior_mean.data() + dimensions);
    artifact.posterior_covariance = std::move(posterior_covariance);
    const Eigen::VectorXd residual = loading * posterior_mean - target;
    artifact.maximum_view_residual = residual.cwiseAbs().maxCoeff();
    if ((noise.array() > 1e-14).all()) {
      const Eigen::MatrixXd inverse_prior = covariance.completeOrthogonalDecomposition().pseudoInverse();
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> prior_eigen(
          covariance, Eigen::EigenvaluesOnly);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> posterior_eigen(
          artifact.posterior_covariance, Eigen::EigenvaluesOnly);
      const bool positive_definite =
          prior_eigen.info() == Eigen::Success &&
          posterior_eigen.info() == Eigen::Success &&
          prior_eigen.eigenvalues().minCoeff() > 1e-14 &&
          posterior_eigen.eigenvalues().minCoeff() > 1e-14;
      if (positive_definite) {
        const double dimension = static_cast<double>(dimensions);
        const double trace_term = (inverse_prior * artifact.posterior_covariance).trace();
        const double quadratic =
            (posterior_mean - mean).transpose() * inverse_prior * (posterior_mean - mean);
        const double logdet_prior = prior_eigen.eigenvalues().array().log().sum();
        const double logdet_post = posterior_eigen.eigenvalues().array().log().sum();
        artifact.kl_divergence =
            0.5 * (trace_term + quadratic - dimension + logdet_prior - logdet_post);
        artifact.kl_divergence_available =
            std::isfinite(artifact.kl_divergence) && artifact.kl_divergence >= 0.0;
        if (!artifact.kl_divergence_available) artifact.kl_divergence = 0.0;
      }
    }
  }
  artifact.status = PosteriorStatus::OK;
  artifact.artifact_hash = posterior_hash_without_self(artifact);
  return artifact;
}

std::uint64_t posterior_scenario_artifact_hash(
    const PosteriorScenarioArtifactV1& artifact) noexcept {
  return posterior_hash_without_self(artifact);
}

PosteriorScenarioStatisticsV1 recompute_posterior_statistics(
    std::span<const double> scenario_values,
    std::size_t scenario_count,
    std::size_t asset_count,
    std::span<const double> probabilities,
    std::span<const double> quantile_levels) {
  PosteriorScenarioStatisticsV1 statistics;
  statistics.scenario_count = scenario_count;
  statistics.asset_count = asset_count;
  statistics.quantile_levels.assign(quantile_levels.begin(), quantile_levels.end());
  if (scenario_count < 2 || asset_count == 0 ||
      scenario_values.size() != scenario_count * asset_count ||
      !valid_probability_vector(probabilities, scenario_count) ||
      !finite_vector(statistics.quantile_levels) ||
      !std::all_of(statistics.quantile_levels.begin(), statistics.quantile_levels.end(),
                   [](double level) { return level > 0.0 && level < 1.0; }) ||
      !std::is_sorted(statistics.quantile_levels.begin(), statistics.quantile_levels.end())) {
    statistics.status = PosteriorStatus::INVALID_INPUT;
    return statistics;
  }
  const Eigen::Index scenarios = static_cast<Eigen::Index>(scenario_count);
  const Eigen::Index assets = static_cast<Eigen::Index>(asset_count);
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> values(
      scenario_values.data(), scenarios, assets);
  const Eigen::Map<const Eigen::VectorXd> weights(probabilities.data(), scenarios);
  if (!values.allFinite()) {
    statistics.status = PosteriorStatus::INVALID_INPUT;
    return statistics;
  }
  const Eigen::VectorXd mean = values.transpose() * weights;
  Eigen::MatrixXd centered = values.rowwise() - mean.transpose();
  Eigen::MatrixXd covariance =
      centered.transpose() * weights.asDiagonal() * centered;
  covariance = 0.5 * (covariance + covariance.transpose());
  statistics.mean.assign(mean.data(), mean.data() + mean.size());
  statistics.covariance = std::move(covariance);
  statistics.effective_sample_size =
      1.0 / weights.array().square().sum();
  statistics.maximum_scenario_weight = weights.maxCoeff();
  statistics.kl_divergence = 0.0;
  const double uniform_probability = 1.0 / static_cast<double>(scenario_count);
  for (double weight : probabilities) {
    if (weight > 0.0) {
      statistics.kl_divergence +=
          weight * std::log(weight / uniform_probability);
    }
  }
  statistics.quantiles.assign(
      statistics.quantile_levels.size() * asset_count, 0.0);
  statistics.expected_shortfall.assign(
      statistics.quantile_levels.size() * asset_count, 0.0);
  for (std::size_t asset = 0; asset < asset_count; ++asset) {
    std::vector<std::pair<double, double>> sorted;
    sorted.reserve(scenario_count);
    for (std::size_t scenario = 0; scenario < scenario_count; ++scenario) {
      sorted.emplace_back(values(static_cast<Eigen::Index>(scenario),
                                 static_cast<Eigen::Index>(asset)),
                          probabilities[scenario]);
    }
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const auto& left, const auto& right) {
                       return left.first < right.first;
                     });
    for (std::size_t level_index = 0;
         level_index < statistics.quantile_levels.size(); ++level_index) {
      const double level = statistics.quantile_levels[level_index];
      double cumulative = 0.0;
      double quantile = sorted.back().first;
      double tail_mass = 0.0;
      double tail_value = 0.0;
      for (const auto& [value, weight] : sorted) {
        const double included = std::min(weight, std::max(0.0, level - cumulative));
        tail_mass += included;
        tail_value += included * value;
        cumulative += weight;
        if (cumulative + 1e-15 >= level) {
          quantile = value;
          break;
        }
      }
      const std::size_t output_index = level_index * asset_count + asset;
      statistics.quantiles[output_index] = quantile;
      statistics.expected_shortfall[output_index] =
          tail_mass > 0.0 ? tail_value / tail_mass : quantile;
    }
  }
  if (!std::isfinite(statistics.effective_sample_size) ||
      !std::isfinite(statistics.maximum_scenario_weight) ||
      !std::isfinite(statistics.kl_divergence) ||
      !finite_matrix(statistics.covariance)) {
    statistics.status = PosteriorStatus::NUMERICAL_FAILURE;
    return statistics;
  }
  statistics.status = PosteriorStatus::OK;
  return statistics;
}

PosteriorScenarioArtifactV1 apply_ffv_views(
    const PriorScenarioArtifactV1& prior,
    std::span<const ViewSpecV1> views,
    std::span<const double> quantile_levels,
    FFVOptions options) {
  PosteriorScenarioArtifactV1 artifact;
  artifact.engine = PosteriorEngineKind::FULLY_FLEXIBLE_VIEWS;
  artifact.fit_start = prior.fit_start;
  artifact.fit_end = prior.fit_end;
  artifact.available_at = prior.available_at;
  artifact.decision_at = prior.decision_at;
  artifact.scenario_count = prior.scenario_count;
  artifact.asset_count = prior.asset_count;
  artifact.effective_sample_size = prior.effective_sample_size;
  artifact.scenario_timestamps = prior.scenario_timestamps;
  artifact.prior_mean = prior.prior_mean;
  artifact.posterior_mean = prior.prior_mean;
  artifact.prior_covariance = prior.prior_covariance;
  artifact.posterior_covariance = prior.prior_covariance;
  artifact.support_min = prior.support_min;
  artifact.support_max = prior.support_max;
  artifact.prior_scenario_hash = prior.scenario_hash;
  artifact.view_spec_hash = view_set_hash(views);
  artifact.view_count = views.size();
  if (!views.empty()) artifact.confidence_mapping_hash = views.front().confidence_mapping_hash;
  if (quantile_levels.empty()) {
    artifact.posterior_quantile_levels = {0.10, 0.90};
  } else {
    artifact.posterior_quantile_levels.assign(quantile_levels.begin(), quantile_levels.end());
  }
  auto finish = [&](PosteriorStatus status) {
    artifact.status = status;
    artifact.artifact_hash = posterior_hash_without_self(artifact);
    return artifact;
  };
  if (!valid_prior_scenario_artifact(prior) || options.max_iterations == 0 ||
      !std::isfinite(options.tolerance) || options.tolerance <= 0.0 ||
      !std::isfinite(options.min_probability) || options.min_probability <= 0.0 ||
      options.min_probability >= 1.0) {
    return finish(PosteriorStatus::INVALID_INPUT);
  }
  for (const auto& view : views) {
    if (!valid_view_spec(view, prior.asset_count, prior.decision_at) ||
        (view.family != PosteriorViewFamily::MEAN &&
         view.family != PosteriorViewFamily::DIRECTION &&
         view.family != PosteriorViewFamily::VOLATILITY)) {
      return finish(view.available_at > prior.decision_at
                        ? PosteriorStatus::FUTURE_DATA
                        : PosteriorStatus::INVALID_INPUT);
    }
    if (view.confidence_mapping_hash != artifact.confidence_mapping_hash) {
      return finish(PosteriorStatus::INVALID_INPUT);
    }
  }
  const auto stats_prior = recompute_posterior_statistics(
      prior.scenario_values, prior.scenario_count, prior.asset_count,
      prior.prior_probabilities, artifact.posterior_quantile_levels);
  if (stats_prior.status != PosteriorStatus::OK) {
    return finish(stats_prior.status);
  }
  struct ActiveView {
    const ViewSpecV1* spec{nullptr};
    double sign{1.0};
  };
  std::vector<ActiveView> active;
  for (const auto& view : views) {
    if (view.confidence <= 0.0) continue;
    if (view.kind == PosteriorViewKind::MEAN) {
      active.push_back({&view, 1.0});
      continue;
    }
    double prior_view = 0.0;
    for (std::size_t asset = 0; asset < prior.asset_count; ++asset) {
      prior_view += view.loading[asset] * prior.prior_mean[asset];
    }
    const double target = prior_view + view.confidence * (view.target - prior_view);
    const bool lower = view.kind == PosteriorViewKind::MEAN_LOWER_BOUND;
    const bool violated = lower
                              ? prior_view < target - options.tolerance
                              : prior_view > target + options.tolerance;
    if (violated) active.push_back({&view, lower ? 1.0 : -1.0});
  }
  artifact.active_constraint_count = active.size();
  if (active.empty()) {
    artifact.posterior_probabilities = prior.prior_probabilities;
    artifact.posterior_probability_hash = probability_hash(
        artifact.posterior_probabilities);
    artifact.effective_sample_size = stats_prior.effective_sample_size;
    artifact.maximum_scenario_weight = stats_prior.maximum_scenario_weight;
    artifact.kl_divergence = 0.0;
    artifact.kl_divergence_available = true;
    artifact.posterior_quantiles = stats_prior.quantiles;
    artifact.posterior_expected_shortfall = stats_prior.expected_shortfall;
    return finish(PosteriorStatus::OK);
  }
  const Eigen::Index scenarios = static_cast<Eigen::Index>(prior.scenario_count);
  const Eigen::Index assets = static_cast<Eigen::Index>(prior.asset_count);
  const Eigen::Index constraints = static_cast<Eigen::Index>(active.size());
  Eigen::MatrixXd functions(constraints, scenarios);
  Eigen::VectorXd targets(constraints);
  for (Eigen::Index row = 0; row < constraints; ++row) {
    const auto& view = *active[static_cast<std::size_t>(row)].spec;
    const double sign = active[static_cast<std::size_t>(row)].sign;
    double prior_view = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (Eigen::Index scenario = 0; scenario < scenarios; ++scenario) {
      double value = 0.0;
      for (Eigen::Index asset = 0; asset < assets; ++asset) {
        value += sign * view.loading[static_cast<std::size_t>(asset)] *
                 prior.scenario_values[static_cast<std::size_t>(scenario * assets + asset)];
      }
      if (view.family == PosteriorViewFamily::DIRECTION) {
        value = value >= view.statistic_threshold ? 1.0 : 0.0;
      } else if (view.family == PosteriorViewFamily::VOLATILITY) {
        const double centered = value - view.statistic_threshold;
        value = centered * centered;
      }
      functions(row, scenario) = value;
      prior_view += prior.prior_probabilities[static_cast<std::size_t>(scenario)] * value;
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
    }
    if (!functions.row(row).allFinite() || !std::isfinite(prior_view) ||
        !std::isfinite(minimum) || !std::isfinite(maximum)) {
      return finish(PosteriorStatus::NUMERICAL_FAILURE);
    }
    const double signed_target =
        view.family == PosteriorViewFamily::VOLATILITY
            ? view.target * view.target
            : sign * view.target;
    targets(row) = prior_view + view.confidence * (signed_target - prior_view);
    const double floor_mass = options.min_probability *
                              static_cast<double>(prior.scenario_count);
    if (!std::isfinite(floor_mass) || floor_mass >= 1.0) {
      return finish(PosteriorStatus::INVALID_INPUT);
    }
    double function_sum = 0.0;
    for (Eigen::Index scenario = 0; scenario < scenarios; ++scenario) {
      function_sum += functions(row, scenario);
    }
    const double remaining_mass = 1.0 - floor_mass;
    const double floor_minimum = options.min_probability * function_sum +
                                 remaining_mass * minimum;
    const double floor_maximum = options.min_probability * function_sum +
                                 remaining_mass * maximum;
    if (targets(row) < floor_minimum || targets(row) > floor_maximum) {
      artifact.support_guard_passed = false;
      return finish(PosteriorStatus::INFEASIBLE);
    }
  }
  for (Eigen::Index row = 0; row < constraints; ++row) {
    double mean = 0.0;
    for (Eigen::Index scenario = 0; scenario < scenarios; ++scenario) {
      mean += prior.prior_probabilities[static_cast<std::size_t>(scenario)] *
              functions(row, scenario);
    }
    double variance = 0.0;
    for (Eigen::Index scenario = 0; scenario < scenarios; ++scenario) {
      const double delta = functions(row, scenario) - mean;
      variance += prior.prior_probabilities[static_cast<std::size_t>(scenario)] *
                  delta * delta;
    }
    const double scale = std::max(1.0, std::abs(mean));
    if (!std::isfinite(variance) || variance <= 1e-14 * scale * scale) {
      return finish(PosteriorStatus::NUMERICAL_FAILURE);
    }
    for (Eigen::Index other = row + 1; other < constraints; ++other) {
      double other_mean = 0.0;
      for (Eigen::Index scenario = 0; scenario < scenarios; ++scenario) {
        other_mean += prior.prior_probabilities[static_cast<std::size_t>(scenario)] *
                      functions(other, scenario);
      }
      double other_variance = 0.0;
      double covariance = 0.0;
      for (Eigen::Index scenario = 0; scenario < scenarios; ++scenario) {
        const double left = functions(row, scenario) - mean;
        const double right = functions(other, scenario) - other_mean;
        const double probability =
            prior.prior_probabilities[static_cast<std::size_t>(scenario)];
        other_variance += probability * right * right;
        covariance += probability * left * right;
      }
      const double denominator = std::sqrt(variance * other_variance);
      if (!std::isfinite(denominator) || denominator <= 0.0 ||
          std::abs(covariance) >= (1.0 - 1e-10) * denominator) {
        return finish(PosteriorStatus::NUMERICAL_FAILURE);
      }
    }
  }
  Eigen::VectorXd lambda = Eigen::VectorXd::Zero(constraints);
  const Eigen::VectorXd prior_probability = Eigen::Map<const Eigen::VectorXd>(
      prior.prior_probabilities.data(), scenarios);
  Eigen::VectorXd probabilities = prior_probability;
  Eigen::VectorXd residual = Eigen::VectorXd::Zero(constraints);
  bool converged = false;
  auto calculate = [&](const Eigen::VectorXd& candidate,
                       Eigen::VectorXd& output_probabilities,
                       Eigen::VectorXd& output_residual) {
    const Eigen::VectorXd log_weights = functions.transpose() * candidate;
    const double maximum_log_weight = log_weights.maxCoeff();
    output_probabilities = (log_weights.array() - maximum_log_weight).exp() *
                           prior_probability.array();
    const double normalizer = output_probabilities.sum();
    if (!std::isfinite(normalizer) || normalizer <= 0.0) return false;
    output_probabilities /= normalizer;
    output_residual = functions * output_probabilities - targets;
    return output_probabilities.allFinite() && output_residual.allFinite();
  };
  for (std::uint32_t iteration = 0; iteration < options.max_iterations; ++iteration) {
    if (!calculate(lambda, probabilities, residual)) return finish(PosteriorStatus::NUMERICAL_FAILURE);
    artifact.iterations = iteration + 1;
    if (residual.cwiseAbs().maxCoeff() <= options.tolerance) {
      converged = true;
      break;
    }
    const Eigen::VectorXd moment = functions * probabilities;
    const Eigen::MatrixXd centered = functions.colwise() - moment;
    const Eigen::MatrixXd covariance =
        centered * probabilities.asDiagonal() * centered.transpose();
    Eigen::LDLT<Eigen::MatrixXd> solver(covariance);
    if (solver.info() != Eigen::Success ||
        (solver.vectorD().array().abs() <= 1e-14).any()) {
      return finish(PosteriorStatus::NUMERICAL_FAILURE);
    }
    const Eigen::VectorXd step = solver.solve(residual);
    if (!step.allFinite()) return finish(PosteriorStatus::NUMERICAL_FAILURE);
    const double base_loss = residual.squaredNorm();
    bool accepted = false;
    for (double scale = 1.0; scale >= 1.0 / 1024.0; scale *= 0.5) {
      Eigen::VectorXd candidate_probabilities;
      Eigen::VectorXd candidate_residual;
      if (!calculate(lambda - scale * step, candidate_probabilities, candidate_residual)) continue;
      if (candidate_residual.squaredNorm() < base_loss) {
        lambda -= scale * step;
        accepted = true;
        break;
      }
    }
    if (!accepted) return finish(PosteriorStatus::NUMERICAL_FAILURE);
  }
  if (!converged) {
    if (!calculate(lambda, probabilities, residual)) return finish(PosteriorStatus::NUMERICAL_FAILURE);
    artifact.maximum_view_residual = residual.cwiseAbs().maxCoeff();
    return finish(PosteriorStatus::INFEASIBLE);
  }
  artifact.maximum_view_residual = residual.cwiseAbs().maxCoeff();
  for (Eigen::Index row = 0; row < constraints; ++row) {
    const auto kind = active[static_cast<std::size_t>(row)].spec->kind;
    if (kind != PosteriorViewKind::MEAN &&
        lambda(row) < -std::max(1e-10, options.tolerance)) {
      return finish(PosteriorStatus::NUMERICAL_FAILURE);
    }
  }
  if ((probabilities.array() < 0.0).any() ||
      probabilities.minCoeff() < options.min_probability) {
    return finish(PosteriorStatus::INFEASIBLE);
  }
  std::vector<double> posterior_probabilities(
      probabilities.data(), probabilities.data() + probabilities.size());
  const auto statistics = recompute_posterior_statistics(
      prior.scenario_values, prior.scenario_count, prior.asset_count,
      posterior_probabilities, artifact.posterior_quantile_levels);
  if (statistics.status != PosteriorStatus::OK) return finish(statistics.status);
  artifact.posterior_probabilities = std::move(posterior_probabilities);
  artifact.posterior_probability_hash = probability_hash(artifact.posterior_probabilities);
  artifact.posterior_mean = statistics.mean;
  artifact.posterior_covariance = statistics.covariance;
  artifact.effective_sample_size = statistics.effective_sample_size;
  artifact.maximum_scenario_weight = statistics.maximum_scenario_weight;
  artifact.kl_divergence = statistics.kl_divergence;
  artifact.kl_divergence_available = true;
  artifact.posterior_quantiles = statistics.quantiles;
  artifact.posterior_expected_shortfall = statistics.expected_shortfall;
  return finish(PosteriorStatus::OK);
}

PosteriorScenarioArtifactV1 apply_ffv_mean_views(
    const PriorScenarioArtifactV1& prior,
    std::span<const ViewSpecV1> views,
    std::span<const double> quantile_levels,
    FFVOptions options) {
  for (const auto& view : views) {
    if (view.family != PosteriorViewFamily::MEAN) {
      PosteriorScenarioArtifactV1 artifact;
      artifact.engine = PosteriorEngineKind::FULLY_FLEXIBLE_VIEWS;
      artifact.status = PosteriorStatus::INVALID_INPUT;
      artifact.fit_start = prior.fit_start;
      artifact.fit_end = prior.fit_end;
      artifact.available_at = prior.available_at;
      artifact.decision_at = prior.decision_at;
      artifact.scenario_count = prior.scenario_count;
      artifact.asset_count = prior.asset_count;
      artifact.prior_scenario_hash = prior.scenario_hash;
      artifact.view_spec_hash = view_set_hash(views);
      artifact.view_count = views.size();
      artifact.artifact_hash = posterior_hash_without_self(artifact);
      return artifact;
    }
  }
  return apply_ffv_views(prior, views, quantile_levels, options);
}

std::string serialize_view_spec(const ViewSpecV1& view) {
  std::ostringstream output;
  output << "{\"schema_version\":" << view.schema_version
         << ",\"view_id\":\"" << view.view_id << "\",\"kind\":\""
         << view_kind_name(view.kind) << "\",\"family\":\""
         << view_family_name(view.family) << "\",\"available_at\":"
         << view.available_at << ",\"loading\":";
  json_vector(output, view.loading);
  output << ",\"target\":" << std::setprecision(17) << view.target
         << ",\"statistic_threshold\":" << view.statistic_threshold
         << ",\"confidence\":" << view.confidence
         << ",\"observation_variance\":" << view.observation_variance
         << ",\"confidence_mapping_hash\":" << view.confidence_mapping_hash
         << ",\"calibration_artifact_hash\":" << view.calibration_artifact_hash
         << ",\"source_artifact_hash\":" << view.source_artifact_hash
         << ",\"view_spec_hash\":" << view_spec_hash(view) << '}';
  return output.str();
}

std::string serialize_posterior_scenario_artifact(
    const PosteriorScenarioArtifactV1& artifact) {
  std::ostringstream output;
  output << "{\"schema_version\":" << artifact.schema_version
         << ",\"status\":\"" << status_name(artifact.status)
         << "\",\"engine\":\"" << engine_name(artifact.engine)
         << "\",\"fit_start\":" << artifact.fit_start
         << ",\"fit_end\":" << artifact.fit_end
         << ",\"available_at\":" << artifact.available_at
         << ",\"decision_at\":" << artifact.decision_at
         << ",\"scenario_count\":" << artifact.scenario_count
         << ",\"asset_count\":" << artifact.asset_count
         << ",\"effective_sample_size\":" << std::setprecision(17)
         << artifact.effective_sample_size << ",\"scenario_timestamps\":[";
  for (std::size_t index = 0; index < artifact.scenario_timestamps.size(); ++index) {
    if (index != 0) output << ',';
    output << artifact.scenario_timestamps[index];
  }
  output << "],\"prior_mean\":";
  json_vector(output, artifact.prior_mean);
  output << ",\"posterior_mean\":";
  json_vector(output, artifact.posterior_mean);
  output << ",\"prior_covariance\":";
  json_matrix(output, artifact.prior_covariance);
  output << ",\"posterior_covariance\":";
  json_matrix(output, artifact.posterior_covariance);
  output << ",\"support_min\":";
  json_vector(output, artifact.support_min);
  output << ",\"support_max\":";
  json_vector(output, artifact.support_max);
  output << ",\"view_count\":" << artifact.view_count
         << ",\"active_constraint_count\":" << artifact.active_constraint_count
         << ",\"iterations\":" << artifact.iterations
         << ",\"maximum_view_residual\":" << artifact.maximum_view_residual
         << ",\"kl_divergence\":" << artifact.kl_divergence
         << ",\"kl_divergence_available\":"
         << (artifact.kl_divergence_available ? "true" : "false")
         << ",\"support_guard_passed\":"
         << (artifact.support_guard_passed ? "true" : "false")
         << ",\"prior_scenario_hash\":" << artifact.prior_scenario_hash
         << ",\"view_spec_hash\":" << artifact.view_spec_hash
         << ",\"confidence_mapping_hash\":"
         << artifact.confidence_mapping_hash
         << ",\"posterior_probability_hash\":"
         << artifact.posterior_probability_hash
         << ",\"maximum_scenario_weight\":"
         << artifact.maximum_scenario_weight
         << ",\"posterior_probabilities\":";
  json_vector(output, artifact.posterior_probabilities);
  output << ",\"posterior_quantile_levels\":";
  json_vector(output, artifact.posterior_quantile_levels);
  output << ",\"posterior_quantiles\":";
  json_vector(output, artifact.posterior_quantiles);
  output << ",\"posterior_expected_shortfall\":";
  json_vector(output, artifact.posterior_expected_shortfall);
  output
         << ",\"eligible_for_official_risk\":false"
         << ",\"artifact_hash\":" << artifact.artifact_hash << '}';
  return output.str();
}

}  // namespace portfolio_math

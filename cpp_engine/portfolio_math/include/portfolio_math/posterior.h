#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "engine_common/types.h"
#include "portfolio_math/types.h"
#include "quant_math/matrix.h"

namespace portfolio_math {

enum class PosteriorStatus : std::uint8_t {
  OK,
  INVALID_INPUT,
  FUTURE_DATA,
  NON_PSD_PRIOR,
  NUMERICAL_FAILURE,
  INFEASIBLE,
};

enum class PosteriorViewKind : std::uint8_t {
  MEAN,
  MEAN_LOWER_BOUND,
  MEAN_UPPER_BOUND,
};

enum class PosteriorViewFamily : std::uint8_t {
  MEAN,
  DIRECTION,
  VOLATILITY,
  RANKING,
  QUANTILE,
};

enum class PosteriorEngineKind : std::uint8_t {
  GAUSSIAN_BL,
  FULLY_FLEXIBLE_VIEWS,
};

struct ViewSpecV1 {
  std::uint32_t schema_version{1};
  std::string view_id;
  PosteriorViewKind kind{PosteriorViewKind::MEAN};
  PosteriorViewFamily family{PosteriorViewFamily::MEAN};
  engine_common::TimestampNs available_at{0};
  std::vector<double> loading;
  double target{0.0};
  double confidence{0.0};
  double observation_variance{0.0};
  std::uint64_t confidence_mapping_hash{0};
  std::uint64_t calibration_artifact_hash{0};
  std::uint64_t source_artifact_hash{0};
};

[[nodiscard]] bool valid_view_spec(
    const ViewSpecV1& view, std::size_t asset_count,
    engine_common::TimestampNs decision_at) noexcept;

[[nodiscard]] std::uint64_t view_spec_hash(const ViewSpecV1& view) noexcept;

struct PriorScenarioArtifactV1 {
  std::uint32_t schema_version{1};
  PosteriorStatus status{PosteriorStatus::INVALID_INPUT};
  engine_common::TimestampNs fit_start{0};
  engine_common::TimestampNs fit_end{0};
  engine_common::TimestampNs available_at{0};
  engine_common::TimestampNs decision_at{0};
  std::size_t scenario_count{0};
  std::size_t asset_count{0};
  double effective_sample_size{0.0};
  std::vector<engine_common::TimestampNs> scenario_timestamps;
  std::vector<double> scenario_values;
  std::vector<double> prior_probabilities;
  std::vector<double> support_min;
  std::vector<double> support_max;
  std::vector<double> prior_mean;
  quant_math::DenseMatrix prior_covariance;
  std::uint64_t scenario_hash{0};
  bool eligible_for_official_risk{false};
  std::uint64_t artifact_hash{0};
};

[[nodiscard]] bool valid_prior_scenario_artifact(
    const PriorScenarioArtifactV1& artifact) noexcept;

[[nodiscard]] PriorScenarioArtifactV1 build_prior_scenario_artifact(
    quant_math::MatrixView scenario_returns,
    std::span<const engine_common::TimestampNs> scenario_timestamps,
    engine_common::TimestampNs available_at,
    engine_common::TimestampNs decision_at);

struct PosteriorScenarioArtifactV1 {
  std::uint32_t schema_version{1};
  PosteriorStatus status{PosteriorStatus::INVALID_INPUT};
  PosteriorEngineKind engine{PosteriorEngineKind::GAUSSIAN_BL};
  engine_common::TimestampNs fit_start{0};
  engine_common::TimestampNs fit_end{0};
  engine_common::TimestampNs available_at{0};
  engine_common::TimestampNs decision_at{0};
  std::size_t scenario_count{0};
  std::size_t asset_count{0};
  double effective_sample_size{0.0};
  std::vector<engine_common::TimestampNs> scenario_timestamps;
  std::vector<double> prior_mean;
  std::vector<double> posterior_mean;
  quant_math::DenseMatrix prior_covariance;
  quant_math::DenseMatrix posterior_covariance;
  std::vector<double> support_min;
  std::vector<double> support_max;
  std::size_t view_count{0};
  std::size_t active_constraint_count{0};
  std::uint32_t iterations{0};
  double maximum_view_residual{0.0};
  double kl_divergence{0.0};
  bool kl_divergence_available{false};
  double maximum_scenario_weight{0.0};
  std::uint64_t posterior_probability_hash{0};
  std::vector<double> posterior_probabilities;
  std::vector<double> posterior_quantile_levels;
  std::vector<double> posterior_quantiles;
  std::vector<double> posterior_expected_shortfall;
  bool support_guard_passed{true};
  std::uint64_t prior_scenario_hash{0};
  std::uint64_t view_spec_hash{0};
  std::uint64_t confidence_mapping_hash{0};
  bool eligible_for_official_risk{false};
  std::uint64_t artifact_hash{0};
};

[[nodiscard]] bool valid_posterior_scenario_artifact(
    const PosteriorScenarioArtifactV1& artifact) noexcept;

[[nodiscard]] PosteriorScenarioArtifactV1 apply_gaussian_mean_views(
    const PriorScenarioArtifactV1& prior,
    std::span<const ViewSpecV1> views);

struct FFVOptions {
  std::uint32_t max_iterations{200};
  double tolerance{1e-10};
  double min_probability{1e-14};
};

struct PosteriorScenarioStatisticsV1 {
  PosteriorStatus status{PosteriorStatus::INVALID_INPUT};
  std::size_t scenario_count{0};
  std::size_t asset_count{0};
  double effective_sample_size{0.0};
  double maximum_scenario_weight{0.0};
  double kl_divergence{0.0};
  std::vector<double> mean;
  quant_math::DenseMatrix covariance;
  std::vector<double> quantile_levels;
  std::vector<double> quantiles;
  std::vector<double> expected_shortfall;
};

[[nodiscard]] PosteriorScenarioStatisticsV1 recompute_posterior_statistics(
    std::span<const double> scenario_values,
    std::size_t scenario_count,
    std::size_t asset_count,
    std::span<const double> probabilities,
    std::span<const double> quantile_levels = {});

[[nodiscard]] PosteriorScenarioArtifactV1 apply_ffv_mean_views(
    const PriorScenarioArtifactV1& prior,
    std::span<const ViewSpecV1> views,
    std::span<const double> quantile_levels = {},
    FFVOptions options = {});

[[nodiscard]] std::uint64_t posterior_scenario_artifact_hash(
    const PosteriorScenarioArtifactV1& artifact) noexcept;

[[nodiscard]] std::string serialize_view_spec(const ViewSpecV1& view);
[[nodiscard]] std::string serialize_posterior_scenario_artifact(
    const PosteriorScenarioArtifactV1& artifact);

}  // namespace portfolio_math

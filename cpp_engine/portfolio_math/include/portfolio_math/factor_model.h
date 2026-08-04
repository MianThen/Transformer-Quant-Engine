#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "engine_common/types.h"
#include "quant_math/matrix.h"

namespace portfolio_math {

enum class FactorModelStatus : std::uint8_t {
  OK,
  INVALID_INPUT,
  INSUFFICIENT_OBSERVATIONS,
  FUTURE_DATA,
  NUMERICAL_FAILURE,
  NON_PSD,
};

enum class FactorEstimator : std::uint8_t {
  FACTOR_PIT_EWMA,
};

enum class FactorExposureSourceStatus : std::uint8_t {
  KNOWN,
  UNAVAILABLE,
  INVALID,
};

struct FactorExposureSourceManifest {
  FactorExposureSourceStatus status{FactorExposureSourceStatus::INVALID};
  std::uint64_t source_schema_hash{0};
  std::uint64_t snapshot_hash{0};
  std::uint64_t provenance_hash{0};
  engine_common::TimestampNs coverage_start{0};
  engine_common::TimestampNs coverage_end{0};
  engine_common::TimestampNs available_at{0};
  bool point_in_time{false};
};

[[nodiscard]] bool valid_factor_exposure_source_manifest(
    const FactorExposureSourceManifest& manifest,
    engine_common::TimestampNs fit_start,
    engine_common::TimestampNs fit_end,
    engine_common::TimestampNs available_at,
    engine_common::TimestampNs decision_at) noexcept;

struct FactorModelSpec {
  FactorEstimator estimator{FactorEstimator::FACTOR_PIT_EWMA};
  std::uint32_t minimum_observations{20};
  double ewma_decay{0.94};
  double factor_covariance_shrinkage{0.10};
  double specific_variance_shrinkage{0.10};
  double specific_variance_floor{1e-8};
  double psd_eigenvalue_floor{1e-12};
  double tolerance{1e-10};
  double annualization_factor{1.0};
  std::uint64_t factor_schema_hash{0};
  std::uint64_t wls_spec_hash{0};
  std::uint64_t config_hash{0};
};

[[nodiscard]] bool valid_factor_model_spec(
    const FactorModelSpec& spec) noexcept;

struct FactorModelInput {
  quant_math::MatrixView asset_returns;
  quant_math::MatrixView pit_exposures;
  quant_math::MatrixView pit_exposure_history;
  std::span<const double> asset_weights;
  std::span<const engine_common::TimestampNs> observation_timestamps;
  engine_common::TimestampNs fit_start{0};
  engine_common::TimestampNs fit_end{0};
  engine_common::TimestampNs available_at{0};
  engine_common::TimestampNs decision_at{0};
  std::uint64_t pit_exposure_hash{0};
  FactorExposureSourceManifest exposure_source;
};

struct FactorModelDiagnostics {
  double maximum_wls_orthogonality_error{0.0};
  double minimum_factor_covariance_eigenvalue{0.0};
  double minimum_specific_variance{0.0};
  double factor_covariance_trace{0.0};
  double specific_variance_floor_hit_rate{0.0};
  double factor_return_mean_norm{0.0};
  std::uint32_t floor_hit_count{0};
};

struct FactorRiskModelArtifact {
  FactorModelStatus status{FactorModelStatus::INVALID_INPUT};
  FactorEstimator estimator{FactorEstimator::FACTOR_PIT_EWMA};
  engine_common::TimestampNs fit_start{0};
  engine_common::TimestampNs fit_end{0};
  engine_common::TimestampNs available_at{0};
  std::uint64_t pit_exposure_hash{0};
  FactorExposureSourceManifest exposure_source;
  std::uint64_t factor_schema_hash{0};
  std::uint64_t wls_spec_hash{0};
  std::uint64_t config_hash{0};
  std::uint64_t artifact_hash{0};
  std::size_t effective_observations{0};
  quant_math::DenseMatrix exposures;
  quant_math::DenseMatrix exposure_history;
  quant_math::DenseMatrix factor_returns;
  quant_math::DenseMatrix specific_returns;
  quant_math::DenseMatrix factor_covariance;
  quant_math::DenseVector factor_mean;
  quant_math::DenseVector specific_variance;
  FactorModelDiagnostics diagnostics;
};

[[nodiscard]] FactorRiskModelArtifact fit_factor_pit_ewma(
    const FactorModelInput& input,
    const FactorModelSpec& spec = {});

[[nodiscard]] bool valid_factor_model_artifact(
    const FactorRiskModelArtifact& artifact,
    engine_common::TimestampNs decision_at,
    double tolerance = 1e-10) noexcept;

[[nodiscard]] std::uint64_t factor_model_artifact_hash(
    const FactorRiskModelArtifact& artifact) noexcept;

[[nodiscard]] std::string serialize_factor_model_artifact(
    const FactorRiskModelArtifact& artifact);

struct FactorRiskModelView {
  quant_math::MatrixView exposures;
  quant_math::MatrixView factor_covariance;
  std::span<const double> specific_variance;
};

[[nodiscard]] bool valid_factor_risk_model(
    const FactorRiskModelView& model,
    double tolerance = 1e-10) noexcept;

[[nodiscard]] double factor_form_variance(
    const FactorRiskModelView& model,
    std::span<const double> weights);

[[nodiscard]] std::vector<double> factor_form_gradient(
    const FactorRiskModelView& model,
    std::span<const double> weights);

[[nodiscard]] quant_math::DenseMatrix materialize_factor_covariance(
    const FactorRiskModelView& model);

enum class FactorOptimizerStatus : std::uint8_t {
  OK,
  INVALID_INPUT,
  INFEASIBLE,
  MAX_ITERATIONS,
  NUMERICAL_FAILURE,
};

struct FactorOptimizerOptions {
  std::uint32_t max_iterations{2'000};
  double tolerance{1e-10};
  double target_investment{1.0};
  double max_single_weight{1.0};
};

struct FactorOptimizerDiagnostics {
  FactorOptimizerStatus status{FactorOptimizerStatus::INVALID_INPUT};
  std::uint32_t iterations{0};
  double predicted_variance{0.0};
  double gradient_residual{0.0};
};

struct FactorOptimizerResult {
  std::vector<double> weights;
  FactorOptimizerDiagnostics diagnostics;
};

[[nodiscard]] FactorOptimizerResult solve_factor_form_min_variance(
    const FactorRiskModelView& model,
    FactorOptimizerOptions options = {});

}  // namespace portfolio_math

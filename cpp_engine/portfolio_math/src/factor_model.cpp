#include "portfolio_math/factor_model.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <iomanip>
#include <sstream>

#include <Eigen/Eigenvalues>

namespace portfolio_math {
namespace {

using quant_math::DenseMatrix;
using quant_math::DenseVector;

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

void hash_matrix(std::uint64_t& hash, const DenseMatrix& matrix) {
  hash_value(hash, static_cast<std::uint64_t>(matrix.rows()));
  hash_value(hash, static_cast<std::uint64_t>(matrix.cols()));
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      hash_double(hash, matrix(row, col));
    }
  }
}

void hash_exposure_source_manifest(
    std::uint64_t& hash, const FactorExposureSourceManifest& manifest) {
  hash_value(hash, static_cast<std::uint8_t>(manifest.status));
  hash_value(hash, manifest.source_schema_hash);
  hash_value(hash, manifest.snapshot_hash);
  hash_value(hash, manifest.provenance_hash);
  hash_value(hash, static_cast<std::uint64_t>(manifest.coverage_start));
  hash_value(hash, static_cast<std::uint64_t>(manifest.coverage_end));
  hash_value(hash, static_cast<std::uint64_t>(manifest.available_at));
  hash_value(hash, manifest.point_in_time ? 1U : 0U);
}

std::uint64_t hash_factor_artifact(const FactorRiskModelArtifact& artifact) {
  std::uint64_t hash = kFnvOffset;
  hash_value(hash, static_cast<std::uint8_t>(artifact.estimator));
  hash_value(hash, artifact.factor_schema_hash);
  hash_value(hash, artifact.wls_spec_hash);
  hash_value(hash, artifact.config_hash);
  hash_value(hash, artifact.pit_exposure_hash);
  hash_value(hash, artifact.fit_start);
  hash_value(hash, artifact.fit_end);
  hash_value(hash, artifact.available_at);
  hash_value(hash, artifact.effective_observations);
  hash_exposure_source_manifest(hash, artifact.exposure_source);
  hash_matrix(hash, artifact.exposures);
  hash_matrix(hash, artifact.exposure_history);
  hash_matrix(hash, artifact.factor_returns);
  hash_matrix(hash, artifact.specific_returns);
  hash_matrix(hash, artifact.factor_covariance);
  for (const double value : artifact.factor_mean) hash_double(hash, value);
  for (const double value : artifact.specific_variance) hash_double(hash, value);
  return hash;
}

void append_matrix_shape(std::ostringstream& output, const DenseMatrix& matrix) {
  output << "[" << matrix.rows() << "," << matrix.cols() << "]";
}

const char* exposure_source_status_name(FactorExposureSourceStatus status) {
  switch (status) {
    case FactorExposureSourceStatus::KNOWN:
      return "KNOWN";
    case FactorExposureSourceStatus::UNAVAILABLE:
      return "UNAVAILABLE";
    case FactorExposureSourceStatus::INVALID:
      return "INVALID";
  }
  return "INVALID";
}

bool valid_matrix(quant_math::MatrixView matrix) {
  return matrix.data != nullptr && matrix.rows > 0 && matrix.cols > 0 &&
      matrix.row_stride >= matrix.cols &&
      quant_math::validate_finite(matrix).ok;
}

DenseMatrix copy_matrix(quant_math::MatrixView matrix) {
  DenseMatrix result(static_cast<Eigen::Index>(matrix.rows),
                     static_cast<Eigen::Index>(matrix.cols));
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t col = 0; col < matrix.cols; ++col) {
      result(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          matrix(row, col);
    }
  }
  return result;
}

bool valid_positive_semidefinite(const DenseMatrix& matrix, double tolerance) {
  if (!quant_math::validate_symmetric(quant_math::view(matrix), tolerance).ok) {
    return false;
  }
  double minimum = 0.0;
  return quant_math::is_positive_semidefinite(matrix, tolerance, &minimum);
}

DenseMatrix repair_psd(const DenseMatrix& covariance, double floor,
                       bool* ok) {
  Eigen::SelfAdjointEigenSolver<DenseMatrix> solver(covariance);
  if (solver.info() != Eigen::Success) {
    *ok = false;
    return {};
  }
  DenseVector eigenvalues = solver.eigenvalues();
  for (Eigen::Index index = 0; index < eigenvalues.size(); ++index) {
    if (!std::isfinite(eigenvalues(index))) {
      *ok = false;
      return {};
    }
    eigenvalues(index) = std::max(floor, eigenvalues(index));
  }
  *ok = true;
  return solver.eigenvectors() * eigenvalues.asDiagonal() *
      solver.eigenvectors().transpose();
}

double sum_weights(std::span<const double> weights) {
  return std::accumulate(weights.begin(), weights.end(), 0.0);
}

std::vector<double> project_simplex_box(std::span<const double> values,
                                        double target, double upper) {
  std::vector<double> projected(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    projected[index] = std::clamp(values[index], 0.0, upper);
  }
  double sum = sum_weights(projected);
  if (std::abs(sum - target) <= 1e-14) return projected;
  double left = -upper;
  double right = upper;
  for (std::uint32_t iteration = 0; iteration < 100; ++iteration) {
    const double shift = 0.5 * (left + right);
    sum = 0.0;
    for (const double value : values) {
      sum += std::clamp(value - shift, 0.0, upper);
    }
    if (sum > target) {
      left = shift;
    } else {
      right = shift;
    }
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    projected[index] = std::clamp(values[index] - 0.5 * (left + right),
                                  0.0, upper);
  }
  return projected;
}

}  // namespace

bool valid_factor_exposure_source_manifest(
    const FactorExposureSourceManifest& manifest,
    engine_common::TimestampNs fit_start,
    engine_common::TimestampNs fit_end,
    engine_common::TimestampNs available_at,
    engine_common::TimestampNs decision_at) noexcept {
  return manifest.status == FactorExposureSourceStatus::KNOWN &&
      manifest.source_schema_hash != 0 && manifest.snapshot_hash != 0 &&
      manifest.provenance_hash != 0 && manifest.coverage_start != 0 &&
      manifest.coverage_start <= fit_start &&
      manifest.coverage_end >= fit_end &&
      manifest.available_at != 0 && manifest.available_at <= available_at &&
      available_at <= decision_at && manifest.point_in_time;
}

bool valid_factor_model_spec(const FactorModelSpec& spec) noexcept {
  return spec.estimator == FactorEstimator::FACTOR_PIT_EWMA &&
      spec.minimum_observations >= 2 && std::isfinite(spec.ewma_decay) &&
      spec.ewma_decay > 0.0 && spec.ewma_decay < 1.0 &&
      std::isfinite(spec.factor_covariance_shrinkage) &&
      spec.factor_covariance_shrinkage >= 0.0 &&
      spec.factor_covariance_shrinkage <= 1.0 &&
      std::isfinite(spec.specific_variance_shrinkage) &&
      spec.specific_variance_shrinkage >= 0.0 &&
      spec.specific_variance_shrinkage <= 1.0 &&
      std::isfinite(spec.specific_variance_floor) &&
      spec.specific_variance_floor > 0.0 &&
      std::isfinite(spec.psd_eigenvalue_floor) &&
      spec.psd_eigenvalue_floor > 0.0 && std::isfinite(spec.tolerance) &&
      spec.tolerance > 0.0 && std::isfinite(spec.annualization_factor) &&
      spec.annualization_factor > 0.0 && spec.factor_schema_hash != 0 &&
      spec.wls_spec_hash != 0 && spec.config_hash != 0;
}

FactorRiskModelArtifact fit_factor_pit_ewma(const FactorModelInput& input,
                                            const FactorModelSpec& spec) {
  FactorRiskModelArtifact result;
  result.estimator = spec.estimator;
  result.fit_start = input.fit_start;
  result.fit_end = input.fit_end;
  result.available_at = input.available_at;
  result.pit_exposure_hash = input.pit_exposure_hash;
  result.exposure_source = input.exposure_source;
  result.factor_schema_hash = spec.factor_schema_hash;
  result.wls_spec_hash = spec.wls_spec_hash;
  result.config_hash = spec.config_hash;
  const bool has_exposure_history =
      input.pit_exposure_history.data != nullptr ||
      input.pit_exposure_history.rows != 0 ||
      input.pit_exposure_history.cols != 0 ||
      input.pit_exposure_history.row_stride != 0;
  if (!valid_factor_model_spec(spec) || !valid_matrix(input.asset_returns) ||
      !valid_matrix(input.pit_exposures) || input.fit_start == 0 ||
      input.fit_end < input.fit_start || input.available_at < input.fit_end ||
      input.decision_at < input.available_at || input.pit_exposure_hash == 0 ||
      input.asset_returns.cols != input.pit_exposures.rows ||
      input.observation_timestamps.size() != input.asset_returns.rows ||
      (has_exposure_history &&
       (!valid_matrix(input.pit_exposure_history) ||
        input.pit_exposure_history.rows !=
            input.asset_returns.rows * input.asset_returns.cols ||
        input.pit_exposure_history.cols != input.pit_exposures.cols)) ||
      (input.asset_weights.size() != 0 &&
       input.asset_weights.size() != input.asset_returns.cols)) {
    return result;
  }
  if (!valid_factor_exposure_source_manifest(
          input.exposure_source, input.fit_start, input.fit_end,
          input.available_at, input.decision_at)) {
    const bool source_is_future =
        input.exposure_source.status == FactorExposureSourceStatus::KNOWN &&
        (input.exposure_source.available_at > input.available_at ||
         input.exposure_source.available_at > input.decision_at);
    result.status = source_is_future ? FactorModelStatus::FUTURE_DATA
                                     : FactorModelStatus::INVALID_INPUT;
    return result;
  }
  if (input.asset_returns.rows < spec.minimum_observations) {
    result.status = FactorModelStatus::INSUFFICIENT_OBSERVATIONS;
    return result;
  }
  if (input.pit_exposures.cols == 0 ||
      input.pit_exposures.cols >= input.asset_returns.rows) {
    return result;
  }
  engine_common::TimestampNs previous_timestamp = 0;
  for (const auto timestamp : input.observation_timestamps) {
    if (timestamp == 0 || timestamp < previous_timestamp) return result;
    if (timestamp < input.fit_start || timestamp > input.fit_end) {
      result.status = FactorModelStatus::FUTURE_DATA;
      return result;
    }
    previous_timestamp = timestamp;
  }

  const std::size_t asset_count = input.asset_returns.cols;
  const std::size_t factor_count = input.pit_exposures.cols;
  result.exposures = copy_matrix(input.pit_exposures);
  result.exposure_history = DenseMatrix(
      static_cast<Eigen::Index>(input.asset_returns.rows * asset_count),
      static_cast<Eigen::Index>(factor_count));
  if (has_exposure_history) {
    result.exposure_history = copy_matrix(input.pit_exposure_history);
  } else {
    for (std::size_t row = 0; row < input.asset_returns.rows; ++row) {
      result.exposure_history.block(
          static_cast<Eigen::Index>(row * asset_count), 0,
          static_cast<Eigen::Index>(asset_count),
          static_cast<Eigen::Index>(factor_count)) = result.exposures;
    }
  }
  result.factor_returns = DenseMatrix(
      static_cast<Eigen::Index>(input.asset_returns.rows),
      static_cast<Eigen::Index>(factor_count));
  result.specific_returns = DenseMatrix(
      static_cast<Eigen::Index>(input.asset_returns.rows),
      static_cast<Eigen::Index>(asset_count));
  result.factor_mean = DenseVector::Zero(static_cast<Eigen::Index>(factor_count));
  result.specific_variance = DenseVector::Zero(static_cast<Eigen::Index>(asset_count));
  DenseVector weights = DenseVector::Ones(static_cast<Eigen::Index>(asset_count));
  if (!input.asset_weights.empty()) {
    for (std::size_t index = 0; index < asset_count; ++index) {
      if (!std::isfinite(input.asset_weights[index]) ||
          input.asset_weights[index] < 0.0) return result;
      weights(static_cast<Eigen::Index>(index)) = input.asset_weights[index];
    }
    if (!(weights.sum() > 0.0)) return result;
  }
  double maximum_orthogonality_error = 0.0;
  for (std::size_t row = 0; row < input.asset_returns.rows; ++row) {
    const DenseMatrix exposure_t = result.exposure_history.block(
        static_cast<Eigen::Index>(row * asset_count), 0,
        static_cast<Eigen::Index>(asset_count),
        static_cast<Eigen::Index>(factor_count));
    const DenseMatrix normal = exposure_t.transpose() * weights.asDiagonal() *
        exposure_t;
    Eigen::LDLT<DenseMatrix> normal_solver(normal);
    if (!normal_solver.isPositive()) {
      result.status = FactorModelStatus::NUMERICAL_FAILURE;
      return result;
    }
    const DenseVector asset_return = DenseVector::Map(
        input.asset_returns.data + row * input.asset_returns.row_stride,
        static_cast<Eigen::Index>(asset_count));
    const DenseVector factor_return = normal_solver.solve(
        exposure_t.transpose() * weights.asDiagonal() * asset_return);
    if (factor_return.size() != static_cast<Eigen::Index>(factor_count) ||
        !factor_return.allFinite()) {
      result.status = FactorModelStatus::NUMERICAL_FAILURE;
      return result;
    }
    result.factor_returns.row(static_cast<Eigen::Index>(row)) = factor_return.transpose();
    const DenseVector residual = asset_return - exposure_t * factor_return;
    result.specific_returns.row(static_cast<Eigen::Index>(row)) = residual.transpose();
    const DenseVector orthogonality = exposure_t.transpose() *
        weights.asDiagonal() * residual;
    maximum_orthogonality_error = std::max(
        maximum_orthogonality_error, orthogonality.cwiseAbs().maxCoeff());
  }
  result.diagnostics.maximum_wls_orthogonality_error = maximum_orthogonality_error;

  DenseMatrix factor_covariance = DenseMatrix::Zero(
      static_cast<Eigen::Index>(factor_count), static_cast<Eigen::Index>(factor_count));
  DenseVector factor_mean = result.factor_returns.row(0).transpose();
  for (std::size_t row = 1; row < input.asset_returns.rows; ++row) {
    const DenseVector current = result.factor_returns.row(
        static_cast<Eigen::Index>(row)).transpose();
    const DenseVector centered = current - factor_mean;
    factor_covariance = spec.ewma_decay * factor_covariance +
        (1.0 - spec.ewma_decay) * centered * centered.transpose();
    factor_mean = spec.ewma_decay * factor_mean +
        (1.0 - spec.ewma_decay) * current;
  }
  const DenseMatrix diagonal_target = factor_covariance.diagonal().asDiagonal();
  factor_covariance = (1.0 - spec.factor_covariance_shrinkage) * factor_covariance +
      spec.factor_covariance_shrinkage * diagonal_target;
  factor_covariance *= spec.annualization_factor;
  bool repaired = false;
  factor_covariance = repair_psd(factor_covariance, spec.psd_eigenvalue_floor,
                                 &repaired);
  if (!repaired) {
    result.status = FactorModelStatus::NUMERICAL_FAILURE;
    return result;
  }

  DenseVector specific_variance = result.specific_returns.row(0).transpose();
  specific_variance = specific_variance.array().square().matrix();
  for (std::size_t row = 1; row < input.asset_returns.rows; ++row) {
    const DenseVector current = result.specific_returns.row(
        static_cast<Eigen::Index>(row)).transpose();
    specific_variance = spec.ewma_decay * specific_variance +
        (1.0 - spec.ewma_decay) * current.array().square().matrix();
  }
  const double target_variance = specific_variance.mean();
  specific_variance = (1.0 - spec.specific_variance_shrinkage) * specific_variance +
      spec.specific_variance_shrinkage *
          DenseVector::Constant(static_cast<Eigen::Index>(asset_count), target_variance);
  specific_variance *= spec.annualization_factor;
  for (Eigen::Index index = 0; index < specific_variance.size(); ++index) {
    specific_variance(index) = std::max(spec.specific_variance_floor,
                                        specific_variance(index));
  }

  result.factor_covariance = std::move(factor_covariance);
  result.factor_mean = std::move(factor_mean);
  result.specific_variance = std::move(specific_variance);
  result.diagnostics.minimum_factor_covariance_eigenvalue =
      Eigen::SelfAdjointEigenSolver<DenseMatrix>(result.factor_covariance)
          .eigenvalues().minCoeff();
  result.diagnostics.minimum_specific_variance = result.specific_variance.minCoeff();
  result.diagnostics.factor_covariance_trace = result.factor_covariance.trace();
  result.diagnostics.floor_hit_count = static_cast<std::uint32_t>(std::count_if(
      result.specific_variance.begin(), result.specific_variance.end(),
      [&](double value) { return value <= spec.specific_variance_floor * (1.0 + spec.tolerance); }));
  result.diagnostics.specific_variance_floor_hit_rate =
      static_cast<double>(result.diagnostics.floor_hit_count) /
      static_cast<double>(asset_count);
  result.diagnostics.factor_return_mean_norm = result.factor_mean.norm();
  if (!valid_positive_semidefinite(result.factor_covariance, spec.tolerance) ||
      result.diagnostics.minimum_specific_variance < spec.specific_variance_floor ||
      result.diagnostics.maximum_wls_orthogonality_error >
          100.0 * spec.tolerance) {
    result.status = FactorModelStatus::NUMERICAL_FAILURE;
    return result;
  }
  result.effective_observations = input.asset_returns.rows;
  result.status = FactorModelStatus::OK;
  result.artifact_hash = factor_model_artifact_hash(result);
  return result;
}

bool valid_factor_model_artifact(const FactorRiskModelArtifact& artifact,
                                 engine_common::TimestampNs decision_at,
                                 double tolerance) noexcept {
  if (artifact.status != FactorModelStatus::OK || artifact.artifact_hash == 0 ||
      artifact.fit_start == 0 || artifact.fit_start > artifact.fit_end ||
      artifact.fit_end > artifact.available_at ||
      artifact.available_at > decision_at || artifact.pit_exposure_hash == 0 ||
      !valid_factor_exposure_source_manifest(
          artifact.exposure_source, artifact.fit_start, artifact.fit_end,
          artifact.available_at, decision_at) ||
      artifact.factor_schema_hash == 0 || artifact.wls_spec_hash == 0 ||
      artifact.config_hash == 0 || artifact.effective_observations == 0 ||
      artifact.factor_returns.rows() !=
          static_cast<Eigen::Index>(artifact.effective_observations) ||
      artifact.specific_returns.rows() != artifact.factor_returns.rows() ||
      artifact.specific_returns.cols() != artifact.exposures.rows() ||
      artifact.factor_returns.cols() != artifact.exposures.cols() ||
      artifact.exposure_history.rows() !=
          artifact.factor_returns.rows() * artifact.exposures.rows() ||
      artifact.exposure_history.cols() != artifact.exposures.cols() ||
      artifact.factor_covariance.rows() != artifact.exposures.cols() ||
      artifact.factor_covariance.cols() != artifact.exposures.cols() ||
      artifact.factor_mean.size() != artifact.exposures.cols() ||
      artifact.specific_variance.size() != artifact.exposures.rows()) {
    return false;
  }
  const FactorRiskModelView view{
      quant_math::view(artifact.exposures),
      quant_math::view(artifact.factor_covariance),
      std::span<const double>(artifact.specific_variance.data(),
                              static_cast<std::size_t>(artifact.specific_variance.size())),
  };
  return valid_factor_risk_model(view, tolerance) &&
      std::isfinite(artifact.diagnostics.maximum_wls_orthogonality_error) &&
      artifact.diagnostics.maximum_wls_orthogonality_error >= 0.0 &&
      artifact.artifact_hash == factor_model_artifact_hash(artifact);
}

std::uint64_t factor_model_artifact_hash(
    const FactorRiskModelArtifact& artifact) noexcept {
  if (artifact.status != FactorModelStatus::OK || artifact.exposures.rows() == 0 ||
      artifact.factor_covariance.rows() == 0) {
    return 0;
  }
  return hash_factor_artifact(artifact);
}

std::string serialize_factor_model_artifact(
    const FactorRiskModelArtifact& artifact) {
  if (!valid_factor_model_artifact(artifact, artifact.available_at)) return {};
  std::ostringstream output;
  output << std::setprecision(17)
         << "{\"schema_version\":1,\"estimator\":\"FACTOR-PIT-EWMA\""
         << ",\"fit_start\":" << artifact.fit_start
         << ",\"fit_end\":" << artifact.fit_end
         << ",\"available_at\":" << artifact.available_at
         << ",\"effective_observations\":" << artifact.effective_observations
         << ",\"factor_schema_hash\":" << artifact.factor_schema_hash
         << ",\"wls_spec_hash\":" << artifact.wls_spec_hash
         << ",\"config_hash\":" << artifact.config_hash
         << ",\"pit_exposure_hash\":" << artifact.pit_exposure_hash
         << ",\"exposure_source\":{\"status\":\""
         << exposure_source_status_name(artifact.exposure_source.status)
         << "\",\"source_schema_hash\":"
         << artifact.exposure_source.source_schema_hash
         << ",\"snapshot_hash\":" << artifact.exposure_source.snapshot_hash
         << ",\"provenance_hash\":"
         << artifact.exposure_source.provenance_hash
         << ",\"coverage_start\":"
         << artifact.exposure_source.coverage_start
         << ",\"coverage_end\":"
         << artifact.exposure_source.coverage_end
         << ",\"available_at\":"
         << artifact.exposure_source.available_at
         << ",\"point_in_time\":"
         << (artifact.exposure_source.point_in_time ? "true" : "false")
         << "}"
         << ",\"exposures_shape\":";
  append_matrix_shape(output, artifact.exposures);
  output << ",\"exposure_history_shape\":";
  append_matrix_shape(output, artifact.exposure_history);
  output << ",\"factor_returns_shape\":";
  append_matrix_shape(output, artifact.factor_returns);
  output << ",\"specific_returns_shape\":";
  append_matrix_shape(output, artifact.specific_returns);
  output << ",\"factor_covariance_shape\":";
  append_matrix_shape(output, artifact.factor_covariance);
  output << ",\"specific_variance_floor_hit_rate\":"
         << artifact.diagnostics.specific_variance_floor_hit_rate
         << ",\"artifact_hash\":" << artifact.artifact_hash << '}';
  return output.str();
}

bool valid_factor_risk_model(const FactorRiskModelView& model,
                             double tolerance) noexcept {
  if (model.exposures.data == nullptr || model.factor_covariance.data == nullptr ||
      model.exposures.rows == 0 || model.exposures.cols == 0 ||
      model.factor_covariance.rows != model.exposures.cols ||
      model.factor_covariance.cols != model.exposures.cols ||
      model.specific_variance.size() != model.exposures.rows ||
      !quant_math::validate_finite(model.exposures).ok ||
      !quant_math::validate_finite(model.factor_covariance).ok ||
      !quant_math::validate_symmetric(model.factor_covariance, tolerance).ok) {
    return false;
  }
  if (!std::all_of(model.specific_variance.begin(), model.specific_variance.end(),
                  [](double value) { return std::isfinite(value) && value > 0.0; })) {
    return false;
  }
  const DenseMatrix factor_covariance = copy_matrix(model.factor_covariance);
  return quant_math::is_positive_semidefinite(factor_covariance, tolerance);
}

double factor_form_variance(const FactorRiskModelView& model,
                            std::span<const double> weights) {
  if (!valid_factor_risk_model(model) || weights.size() != model.exposures.rows) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const DenseMatrix exposures = copy_matrix(model.exposures);
  const DenseMatrix factor_covariance = copy_matrix(model.factor_covariance);
  const DenseVector portfolio_factor = exposures.transpose() *
      DenseVector::Map(weights.data(), static_cast<Eigen::Index>(weights.size()));
  const DenseVector weight_vector = DenseVector::Map(
      weights.data(), static_cast<Eigen::Index>(weights.size()));
  const double factor_part = (portfolio_factor.transpose() * factor_covariance *
                              portfolio_factor)(0, 0);
  const double specific_part = (weight_vector.array().square() *
                                Eigen::Map<const DenseVector>(
                                    model.specific_variance.data(),
                                    static_cast<Eigen::Index>(weights.size())).array()).sum();
  return factor_part + specific_part;
}

std::vector<double> factor_form_gradient(const FactorRiskModelView& model,
                                         std::span<const double> weights) {
  if (!valid_factor_risk_model(model) || weights.size() != model.exposures.rows) {
    return {};
  }
  const DenseMatrix exposures = copy_matrix(model.exposures);
  const DenseMatrix factor_covariance = copy_matrix(model.factor_covariance);
  const DenseVector weight_vector = DenseVector::Map(
      weights.data(), static_cast<Eigen::Index>(weights.size()));
  const DenseVector portfolio_factor = exposures.transpose() * weight_vector;
  const DenseVector marginal = exposures * factor_covariance * portfolio_factor +
      Eigen::Map<const DenseVector>(model.specific_variance.data(),
                                    static_cast<Eigen::Index>(weights.size())).cwiseProduct(weight_vector);
  std::vector<double> result(static_cast<std::size_t>(marginal.size()));
  for (Eigen::Index index = 0; index < marginal.size(); ++index) {
    result[static_cast<std::size_t>(index)] = 2.0 * marginal(index);
  }
  return result;
}

quant_math::DenseMatrix materialize_factor_covariance(
    const FactorRiskModelView& model) {
  if (model.exposures.data == nullptr || model.factor_covariance.data == nullptr ||
      model.exposures.cols != model.factor_covariance.rows ||
      model.factor_covariance.rows != model.factor_covariance.cols ||
      model.specific_variance.size() != model.exposures.rows) {
    return {};
  }
  const DenseMatrix exposures = copy_matrix(model.exposures);
  const DenseMatrix factor_covariance = copy_matrix(model.factor_covariance);
  DenseMatrix covariance = exposures * factor_covariance * exposures.transpose();
  for (Eigen::Index index = 0; index < covariance.rows(); ++index) {
    covariance(index, index) += model.specific_variance[static_cast<std::size_t>(index)];
  }
  return 0.5 * (covariance + covariance.transpose()).eval();
}

FactorOptimizerResult solve_factor_form_min_variance(
    const FactorRiskModelView& model, FactorOptimizerOptions options) {
  FactorOptimizerResult result;
  if (!valid_factor_risk_model(model) || options.max_iterations == 0 ||
      !std::isfinite(options.tolerance) || options.tolerance <= 0.0 ||
      !std::isfinite(options.target_investment) || options.target_investment <= 0.0 ||
      !std::isfinite(options.max_single_weight) || options.max_single_weight <= 0.0 ||
      model.exposures.rows * options.max_single_weight + options.tolerance <
          options.target_investment) {
    return result;
  }
  result.weights.assign(model.exposures.rows,
                        options.target_investment /
                            static_cast<double>(model.exposures.rows));
  result.weights = project_simplex_box(result.weights, options.target_investment,
                                       options.max_single_weight);
  double lipschitz = 1e-12;
  std::vector<double> probe(result.weights.size(), 1.0 /
      std::sqrt(static_cast<double>(result.weights.size())));
  for (int iteration = 0; iteration < 20; ++iteration) {
    const auto gradient = factor_form_gradient(model, probe);
    if (gradient.empty()) return result;
    double norm = 0.0;
    for (const double value : gradient) norm += value * value;
    norm = std::sqrt(norm);
    if (!(norm > 0.0) || !std::isfinite(norm)) break;
    for (std::size_t index = 0; index < probe.size(); ++index) {
      probe[index] = gradient[index] / norm;
    }
    const auto next = factor_form_gradient(model, probe);
    double rayleigh = 0.0;
    for (std::size_t index = 0; index < probe.size(); ++index) {
      rayleigh += probe[index] * next[index];
    }
    lipschitz = std::max(lipschitz, rayleigh);
  }
  const double step = 1.0 / lipschitz;
  for (std::uint32_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
    const auto gradient = factor_form_gradient(model, result.weights);
    if (gradient.empty()) return result;
    std::vector<double> candidate(result.weights.size());
    for (std::size_t index = 0; index < result.weights.size(); ++index) {
      candidate[index] = result.weights[index] - step * gradient[index];
    }
    candidate = project_simplex_box(candidate, options.target_investment,
                                    options.max_single_weight);
    double change = 0.0;
    for (std::size_t index = 0; index < candidate.size(); ++index) {
      change = std::max(change, std::abs(candidate[index] - result.weights[index]));
    }
    result.weights = std::move(candidate);
    result.diagnostics.iterations = iteration;
    if (change <= options.tolerance) {
      result.diagnostics.status = FactorOptimizerStatus::OK;
      break;
    }
    if (iteration == options.max_iterations) {
      result.diagnostics.status = FactorOptimizerStatus::MAX_ITERATIONS;
    }
  }
  result.diagnostics.predicted_variance = factor_form_variance(model, result.weights);
  const auto gradient = factor_form_gradient(model, result.weights);
  result.diagnostics.gradient_residual = gradient.empty()
      ? std::numeric_limits<double>::infinity()
      : *std::max_element(gradient.begin(), gradient.end(),
                          [](double left, double right) {
                            return std::abs(left) < std::abs(right);
                          });
  if (!std::isfinite(result.diagnostics.predicted_variance) ||
      result.diagnostics.predicted_variance < 0.0) {
    result.diagnostics.status = FactorOptimizerStatus::NUMERICAL_FAILURE;
  }
  return result;
}

}  // namespace portfolio_math

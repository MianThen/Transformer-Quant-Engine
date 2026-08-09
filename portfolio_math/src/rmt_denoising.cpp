#include "portfolio_math/rmt_denoising.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Eigenvalues>

namespace portfolio_math {
namespace {

using quant_math::DenseMatrix;
using quant_math::DenseVector;

RmtDenoisingResult invalid_result(CovarianceStatus status,
                                  CovarianceEstimator estimator) {
  RmtDenoisingResult result;
  result.covariance.status = status;
  result.covariance.estimator = estimator;
  result.covariance.loss_profile = CovarianceLossProfile::NOT_APPLICABLE;
  return result;
}

DenseMatrix copy_matrix(quant_math::MatrixView source) {
  DenseMatrix result(static_cast<Eigen::Index>(source.rows),
                     static_cast<Eigen::Index>(source.cols));
  for (std::size_t row = 0; row < source.rows; ++row) {
    for (std::size_t col = 0; col < source.cols; ++col) {
      result(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          source(row, col);
    }
  }
  return result;
}

}  // namespace

bool valid_rmt_denoising_spec(const RmtDenoisingSpec& spec) noexcept {
  return std::isfinite(spec.edge_tolerance) && spec.edge_tolerance >= 0.0 &&
         std::isfinite(spec.eigenvalue_floor) &&
         spec.eigenvalue_floor > 0.0 && std::isfinite(spec.psd_tolerance) &&
         spec.psd_tolerance > 0.0 &&
         std::isfinite(spec.targeted_shrinkage_intensity) &&
         spec.targeted_shrinkage_intensity >= 0.0 &&
         spec.targeted_shrinkage_intensity <= 1.0;
}

RmtDenoisingResult rmt_spectral_denoising(
    quant_math::MatrixView returns, RmtDenoisingSpec spec,
    CovarianceEstimator estimator, bool targeted_shrinkage) {
  if (!valid_rmt_denoising_spec(spec)) {
    return invalid_result(CovarianceStatus::INVALID_INPUT, estimator);
  }
  if (returns.rows < 2) {
    return invalid_result(CovarianceStatus::INSUFFICIENT_OBSERVATIONS,
                          estimator);
  }
  if (returns.cols == 0 || !quant_math::validate_finite(returns).ok) {
    return invalid_result(CovarianceStatus::INVALID_INPUT, estimator);
  }

  const Eigen::Index observations = static_cast<Eigen::Index>(returns.rows);
  const Eigen::Index assets = static_cast<Eigen::Index>(returns.cols);
  DenseMatrix values = copy_matrix(returns);
  values.rowwise() -= values.colwise().mean();
  DenseMatrix sample = values.transpose() * values /
                       static_cast<double>(observations);
  sample = 0.5 * (sample + sample.transpose()).eval();
  const DenseVector variances = sample.diagonal();
  if ((variances.array() <= spec.eigenvalue_floor).any()) {
    return invalid_result(CovarianceStatus::INVALID_INPUT, estimator);
  }

  const DenseVector standard_deviations = variances.cwiseSqrt();
  DenseMatrix correlation = sample;
  for (Eigen::Index row = 0; row < assets; ++row) {
    for (Eigen::Index col = 0; col < assets; ++col) {
      correlation(row, col) /=
          standard_deviations(row) * standard_deviations(col);
    }
  }
  correlation = 0.5 * (correlation + correlation.transpose()).eval();
  Eigen::SelfAdjointEigenSolver<DenseMatrix> eigensolver(correlation);
  if (eigensolver.info() != Eigen::Success) {
    return invalid_result(CovarianceStatus::NUMERICAL_FAILURE, estimator);
  }

  RmtDenoisingResult result;
  result.covariance.estimator = estimator;
  result.covariance.loss_profile = CovarianceLossProfile::NOT_APPLICABLE;
  result.covariance.effective_observations = returns.rows;
  result.covariance.concentration_ratio =
      static_cast<double>(returns.cols) / static_cast<double>(returns.rows);
  auto& diagnostics = result.diagnostics;
  diagnostics.aspect_ratio = result.covariance.concentration_ratio;
  diagnostics.targeted_shrinkage_intensity =
      targeted_shrinkage ? spec.targeted_shrinkage_intensity : 0.0;
  diagnostics.input_correlation_trace = correlation.trace();
  diagnostics.raw_eigenvalues.resize(static_cast<std::size_t>(assets));
  for (Eigen::Index index = 0; index < assets; ++index) {
    const double value = eigensolver.eigenvalues()(index);
    if (!std::isfinite(value) || value < -spec.psd_tolerance) {
      return invalid_result(CovarianceStatus::NUMERICAL_FAILURE, estimator);
    }
    diagnostics.raw_eigenvalues[static_cast<std::size_t>(index)] =
        std::max(0.0, value);
  }

  diagnostics.fitted_noise_boundary =
      std::pow(1.0 + std::sqrt(diagnostics.aspect_ratio), 2.0);
  const double boundary = diagnostics.fitted_noise_boundary +
                          spec.edge_tolerance *
                              std::max(1.0, diagnostics.fitted_noise_boundary);
  std::vector<double> cleaned = diagnostics.raw_eigenvalues;
  double noise_sum = 0.0;
  for (std::size_t index = 0; index < cleaned.size(); ++index) {
    if (cleaned[index] <= boundary) {
      noise_sum += cleaned[index];
      ++diagnostics.noise_eigenvalue_count;
    } else {
      ++diagnostics.retained_signal_rank;
    }
  }
  if (diagnostics.noise_eigenvalue_count > 0) {
    const double raw_residual =
        noise_sum / static_cast<double>(diagnostics.noise_eigenvalue_count);
    if (!std::isfinite(raw_residual) ||
        raw_residual < spec.eigenvalue_floor) {
      return invalid_result(CovarianceStatus::NUMERICAL_FAILURE, estimator);
    }
    diagnostics.constant_residual_eigenvalue = raw_residual;
    for (std::size_t index = 0; index < cleaned.size(); ++index) {
      if (diagnostics.raw_eigenvalues[index] <= boundary) {
        cleaned[index] = targeted_shrinkage
                             ? diagnostics.raw_eigenvalues[index] +
                                   spec.targeted_shrinkage_intensity *
                                       (raw_residual -
                                        diagnostics.raw_eigenvalues[index])
                             : diagnostics.constant_residual_eigenvalue;
      }
    }
  }
  diagnostics.cleaned_eigenvalues = cleaned;
  if (!std::all_of(cleaned.begin(), cleaned.end(), [&](double value) {
        return std::isfinite(value) && value >= spec.eigenvalue_floor;
      })) {
    return invalid_result(CovarianceStatus::NUMERICAL_FAILURE, estimator);
  }

  Eigen::Map<const DenseVector> cleaned_vector(
      cleaned.data(), static_cast<Eigen::Index>(cleaned.size()));
  DenseMatrix cleaned_correlation =
      eigensolver.eigenvectors() * cleaned_vector.asDiagonal() *
      eigensolver.eigenvectors().transpose();
  cleaned_correlation =
      0.5 * (cleaned_correlation + cleaned_correlation.transpose()).eval();
  const DenseVector cleaned_diagonal = cleaned_correlation.diagonal();
  if ((cleaned_diagonal.array() <= spec.eigenvalue_floor).any()) {
    return invalid_result(CovarianceStatus::NUMERICAL_FAILURE, estimator);
  }
  for (Eigen::Index row = 0; row < assets; ++row) {
    for (Eigen::Index col = 0; col < assets; ++col) {
      cleaned_correlation(row, col) /=
          std::sqrt(cleaned_diagonal(row) * cleaned_diagonal(col));
    }
  }
  cleaned_correlation =
      0.5 * (cleaned_correlation + cleaned_correlation.transpose()).eval();
  for (Eigen::Index index = 0; index < assets; ++index) {
    cleaned_correlation(index, index) = 1.0;
  }
  if (!quant_math::validate_finite(quant_math::view(cleaned_correlation)).ok ||
      !quant_math::is_positive_semidefinite(cleaned_correlation,
                                            spec.psd_tolerance)) {
    return invalid_result(CovarianceStatus::NUMERICAL_FAILURE, estimator);
  }

  DenseMatrix covariance =
      standard_deviations.asDiagonal() * cleaned_correlation *
      standard_deviations.asDiagonal();
  covariance = 0.5 * (covariance + covariance.transpose()).eval();
  if (!quant_math::validate_finite(quant_math::view(covariance)).ok ||
      !quant_math::is_positive_semidefinite(covariance, spec.psd_tolerance)) {
    return invalid_result(CovarianceStatus::NUMERICAL_FAILURE, estimator);
  }

  diagnostics.output_correlation_trace = cleaned_correlation.trace();
  diagnostics.trace_drift =
      std::abs(diagnostics.output_correlation_trace -
               diagnostics.input_correlation_trace);
  diagnostics.maximum_relative_diagonal_drift = 0.0;
  for (Eigen::Index index = 0; index < assets; ++index) {
    diagnostics.maximum_relative_diagonal_drift = std::max(
        diagnostics.maximum_relative_diagonal_drift,
        std::abs(covariance(index, index) - sample(index, index)) /
            std::max(spec.eigenvalue_floor, std::abs(sample(index, index))));
  }
  const auto [minimum, maximum] =
      std::minmax_element(cleaned.begin(), cleaned.end());
  diagnostics.condition_number = *maximum / *minimum;
  diagnostics.psd_repair_amount = 0.0;
  result.covariance.covariance = std::move(covariance);
  result.covariance.status = CovarianceStatus::OK;
  return result;
}

RmtDenoisingResult rmt_constant_residual_denoising(
    quant_math::MatrixView returns, RmtDenoisingSpec spec) {
  return rmt_spectral_denoising(
      returns, spec, CovarianceEstimator::RMT_CONSTANT_RESIDUAL, false);
}

RmtDenoisingResult rmt_targeted_shrinkage_denoising(
    quant_math::MatrixView returns, RmtDenoisingSpec spec) {
  return rmt_spectral_denoising(
      returns, spec, CovarianceEstimator::RMT_TARGETED_SHRINKAGE, true);
}

}  // namespace portfolio_math

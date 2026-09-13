#include "portfolio_math/factor_risk_diagnostics.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace portfolio_math {
namespace {

using quant_math::DenseMatrix;
using quant_math::DenseVector;

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

bool valid_covariance(quant_math::MatrixView matrix, double tolerance) {
  if (!valid_matrix(matrix) || matrix.rows != matrix.cols ||
      !quant_math::validate_symmetric(matrix, tolerance).ok) {
    return false;
  }
  const DenseMatrix covariance = copy_matrix(matrix);
  double minimum_eigenvalue = 0.0;
  return quant_math::is_positive_semidefinite(covariance, tolerance,
                                              &minimum_eigenvalue) &&
      minimum_eigenvalue > tolerance;
}

bool valid_factor_model_components(const FactorRiskModelView& model,
                                   double tolerance) {
  if (!valid_matrix(model.exposures) ||
      !valid_matrix(model.factor_covariance) ||
      model.factor_covariance.rows != model.factor_covariance.cols ||
      model.exposures.cols != model.factor_covariance.rows ||
      model.specific_variance.size() != model.exposures.rows ||
      !quant_math::validate_symmetric(model.factor_covariance, tolerance).ok) {
    return false;
  }
  return std::all_of(
      model.specific_variance.begin(), model.specific_variance.end(),
      [](double value) { return std::isfinite(value) && value > 0.0; });
}

bool valid_timestamps(std::span<const engine_common::TimestampNs> timestamps,
                      std::size_t expected_size,
                      engine_common::TimestampNs decision_at,
                      FactorRiskDiagnosticStatus* status) {
  if (timestamps.size() != expected_size) return false;
  engine_common::TimestampNs previous = 0;
  for (const auto timestamp : timestamps) {
    if (timestamp == 0 || timestamp < previous) return false;
    if (timestamp > decision_at) {
      *status = FactorRiskDiagnosticStatus::FUTURE_DATA;
      return false;
    }
    previous = timestamp;
  }
  return true;
}

DenseMatrix centered_covariance(quant_math::MatrixView returns) {
  DenseMatrix values = copy_matrix(returns);
  values.rowwise() -= values.colwise().mean();
  DenseMatrix covariance = values.transpose() * values /
      static_cast<double>(returns.rows);
  return 0.5 * (covariance + covariance.transpose()).eval();
}

bool positive_definite(const DenseMatrix& covariance, double tolerance) {
  Eigen::LLT<DenseMatrix> llt(covariance);
  if (llt.info() != Eigen::Success) return false;
  const DenseVector diagonal = llt.matrixL().toDenseMatrix().diagonal();
  return diagonal.allFinite() &&
      (diagonal.array() > std::sqrt(tolerance)).all();
}

double log_determinant(const DenseMatrix& covariance) {
  const Eigen::LLT<DenseMatrix> llt(covariance);
  if (llt.info() != Eigen::Success) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return 2.0 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
}

bool factor_form_risk_contributions(
    const FactorRiskModelView& model,
    std::span<const double> weights,
    double tolerance,
    double* factor_variance,
    double* specific_variance,
    double* portfolio_variance,
    double* reconciliation_error,
    std::vector<double>* contributions) {
  const DenseMatrix exposures = copy_matrix(model.exposures);
  const DenseMatrix factor_covariance = copy_matrix(model.factor_covariance);
  const DenseVector weight_vector = DenseVector::Map(
      weights.data(), static_cast<Eigen::Index>(weights.size()));
  const DenseVector specific = Eigen::Map<const DenseVector>(
      model.specific_variance.data(),
      static_cast<Eigen::Index>(model.specific_variance.size()));
  const DenseVector portfolio_factor = exposures.transpose() * weight_vector;
  const DenseVector marginal = exposures * factor_covariance * portfolio_factor +
      specific.cwiseProduct(weight_vector);
  *factor_variance =
      (portfolio_factor.transpose() * factor_covariance * portfolio_factor)(0, 0);
  *specific_variance =
      (weight_vector.array().square() * specific.array()).sum();
  *portfolio_variance = *factor_variance + *specific_variance;
  const double marginal_variance = weight_vector.dot(marginal);
  *reconciliation_error = std::abs(marginal_variance - *portfolio_variance);
  if (!std::isfinite(*factor_variance) || !std::isfinite(*specific_variance) ||
      !std::isfinite(*portfolio_variance) ||
      !std::isfinite(*reconciliation_error) || !marginal.allFinite() ||
      *portfolio_variance <= tolerance) {
    return false;
  }
  contributions->resize(weights.size());
  for (std::size_t index = 0; index < weights.size(); ++index) {
    (*contributions)[index] =
        weights[index] * marginal(static_cast<Eigen::Index>(index)) /
        *portfolio_variance;
  }
  return std::all_of(contributions->begin(), contributions->end(),
                     [](double value) { return std::isfinite(value); });
}

bool streaming_realized_risk_contributions(
    quant_math::MatrixView returns,
    std::span<const double> weights,
    double tolerance,
    double* portfolio_variance,
    std::vector<double>* contributions) {
  const std::size_t observation_count = returns.rows;
  const std::size_t asset_count = returns.cols;
  std::vector<double> asset_means(asset_count, 0.0);
  std::vector<double> portfolio_returns(observation_count, 0.0);
  double portfolio_mean = 0.0;
  for (std::size_t row = 0; row < observation_count; ++row) {
    double portfolio_return = 0.0;
    for (std::size_t asset = 0; asset < asset_count; ++asset) {
      const double asset_return = returns(row, asset);
      asset_means[asset] += asset_return;
      portfolio_return += weights[asset] * asset_return;
    }
    portfolio_returns[row] = portfolio_return;
    portfolio_mean += portfolio_return;
  }
  const double inverse_observations =
      1.0 / static_cast<double>(observation_count);
  portfolio_mean *= inverse_observations;
  for (double& asset_mean : asset_means) {
    asset_mean *= inverse_observations;
  }

  std::vector<double> marginal_covariance(asset_count, 0.0);
  double variance_sum = 0.0;
  for (std::size_t row = 0; row < observation_count; ++row) {
    const double centered_portfolio = portfolio_returns[row] - portfolio_mean;
    variance_sum += centered_portfolio * centered_portfolio;
    for (std::size_t asset = 0; asset < asset_count; ++asset) {
      marginal_covariance[asset] +=
          (returns(row, asset) - asset_means[asset]) * centered_portfolio;
    }
  }
  *portfolio_variance = variance_sum * inverse_observations;
  if (!std::isfinite(*portfolio_variance) ||
      *portfolio_variance <= tolerance) {
    return false;
  }
  contributions->resize(asset_count);
  for (std::size_t asset = 0; asset < asset_count; ++asset) {
    (*contributions)[asset] = weights[asset] * marginal_covariance[asset] *
        inverse_observations / *portfolio_variance;
  }
  return std::all_of(contributions->begin(), contributions->end(),
                     [](double value) { return std::isfinite(value); });
}

}  // namespace

FactorRiskDiagnosticResult evaluate_factor_risk_diagnostics(
    const FactorRiskDiagnosticInput& input) {
  FactorRiskDiagnosticResult result;
  if (!std::isfinite(input.tolerance) || input.tolerance <= 0.0 ||
      input.available_at == 0 || input.decision_at == 0 ||
      input.predicted_risk_artifact_hash == 0) {
    return result;
  }
  result.predicted_risk_artifact_hash =
      input.predicted_risk_artifact_hash;
  if (input.available_at > input.decision_at) {
    result.status = FactorRiskDiagnosticStatus::FUTURE_DATA;
    return result;
  }
  FactorRiskDiagnosticStatus timestamp_status =
      FactorRiskDiagnosticStatus::INVALID_INPUT;
  if (!valid_timestamps(input.observation_timestamps,
                        input.realized_factor_returns.rows,
                        input.decision_at, &timestamp_status)) {
    result.status = timestamp_status;
    return result;
  }
  if (!valid_matrix(input.realized_factor_returns) ||
      !valid_matrix(input.realized_asset_returns) ||
      !valid_factor_model_components(input.predicted_risk_model,
                                     input.tolerance) ||
      input.realized_factor_returns.rows < 2 ||
      input.realized_asset_returns.rows != input.realized_factor_returns.rows ||
      input.predicted_risk_model.factor_covariance.rows !=
          input.realized_factor_returns.cols ||
      input.predicted_risk_model.exposures.rows !=
          input.realized_asset_returns.cols ||
      input.portfolio_weights.size() != input.realized_asset_returns.cols ||
      input.portfolio_weights.empty() ||
      !std::all_of(input.portfolio_weights.begin(), input.portfolio_weights.end(),
                   [](double value) { return std::isfinite(value); })) {
    return result;
  }
  if (!valid_covariance(input.predicted_risk_model.factor_covariance,
                        input.tolerance)) {
    result.status = FactorRiskDiagnosticStatus::NON_PSD;
    return result;
  }

  const DenseMatrix predicted_factor_covariance =
      copy_matrix(input.predicted_risk_model.factor_covariance);
  const DenseMatrix realized_factor_covariance =
      centered_covariance(input.realized_factor_returns);
  if (!positive_definite(predicted_factor_covariance, input.tolerance) ||
      !quant_math::validate_finite(
          quant_math::view(realized_factor_covariance)).ok) {
    result.status = FactorRiskDiagnosticStatus::NUMERICAL_FAILURE;
    return result;
  }

  const Eigen::LLT<DenseMatrix> factor_llt(predicted_factor_covariance);
  const DenseMatrix scaled_realized_covariance =
      factor_llt.solve(realized_factor_covariance);
  const double trace_term = scaled_realized_covariance.trace();
  const double qlike = log_determinant(predicted_factor_covariance) + trace_term;
  if (!scaled_realized_covariance.allFinite() || !std::isfinite(qlike)) {
    result.status = FactorRiskDiagnosticStatus::NUMERICAL_FAILURE;
    return result;
  }

  double predicted_factor_variance = 0.0;
  double predicted_specific_variance = 0.0;
  double predicted_variance = 0.0;
  double predicted_reconciliation_error = 0.0;
  double realized_variance = 0.0;
  std::vector<double> predicted_contributions;
  std::vector<double> realized_contributions;
  if (!factor_form_risk_contributions(
          input.predicted_risk_model, input.portfolio_weights,
          input.tolerance, &predicted_factor_variance,
          &predicted_specific_variance, &predicted_variance,
          &predicted_reconciliation_error, &predicted_contributions) ||
      !streaming_realized_risk_contributions(
          input.realized_asset_returns, input.portfolio_weights,
          input.tolerance, &realized_variance, &realized_contributions)) {
    result.status = FactorRiskDiagnosticStatus::NUMERICAL_FAILURE;
    return result;
  }
  result.status = FactorRiskDiagnosticStatus::OK;
  result.effective_observations = input.realized_factor_returns.rows;
  result.factor_covariance_qlike = qlike;
  result.predicted_factor_variance = predicted_factor_variance;
  result.predicted_specific_variance = predicted_specific_variance;
  result.predicted_variance_reconciliation_error =
      predicted_reconciliation_error;
  result.predicted_portfolio_variance = predicted_variance;
  result.realized_portfolio_variance = realized_variance;
  result.realized_predicted_variance_ratio = realized_variance / predicted_variance;
  result.predicted_risk_contributions = std::move(predicted_contributions);
  result.realized_risk_contributions = std::move(realized_contributions);
  double absolute_error_sum = 0.0;
  for (std::size_t index = 0; index < result.predicted_risk_contributions.size();
       ++index) {
    const double error = std::abs(result.predicted_risk_contributions[index] -
                                  result.realized_risk_contributions[index]);
    absolute_error_sum += error;
    result.maximum_absolute_risk_contribution_error = std::max(
        result.maximum_absolute_risk_contribution_error, error);
  }
  result.mean_absolute_risk_contribution_error =
      absolute_error_sum /
      static_cast<double>(result.predicted_risk_contributions.size());
  return result;
}

}  // namespace portfolio_math

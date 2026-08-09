#include "portfolio_math/factor_risk_diagnostics.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include <Eigen/Core>

namespace {

bool close(double left, double right, double tolerance = 1e-8) {
  return std::abs(left - right) <= tolerance *
      std::max({1.0, std::abs(left), std::abs(right)});
}

double sum(std::span<const double> values) {
  return std::accumulate(values.begin(), values.end(), 0.0);
}

}  // namespace

int main() {
  using quant_math::DenseMatrix;
  using quant_math::DenseVector;
  using portfolio_math::FactorRiskDiagnosticInput;
  using portfolio_math::FactorRiskDiagnosticStatus;
  using portfolio_math::FactorRiskModelView;

  DenseMatrix predicted_factor_covariance(2, 2);
  predicted_factor_covariance << 2.0e-4, 2.5e-5,
                                 2.5e-5, 1.5e-4;
  DenseMatrix exposures(3, 2);
  exposures << 1.0, 0.2,
               0.1, 1.0,
               0.5, 0.4;
  std::vector<double> specific_variance{5.0e-5, 4.0e-5, 3.0e-5};
  const FactorRiskModelView risk_model{
      quant_math::view(exposures),
      quant_math::view(predicted_factor_covariance),
      std::span<const double>(specific_variance),
  };

  DenseMatrix factor_returns(40, 2);
  DenseMatrix asset_returns(40, 3);
  std::vector<engine_common::TimestampNs> timestamps;
  for (int row = 0; row < factor_returns.rows(); ++row) {
    const double first = 0.01 * std::sin(0.13 * static_cast<double>(row));
    const double second = 0.008 * std::cos(0.17 * static_cast<double>(row));
    factor_returns(row, 0) = first;
    factor_returns(row, 1) = second;
    asset_returns(row, 0) = first + 0.2 * second +
        0.002 * std::sin(0.41 * static_cast<double>(row));
    asset_returns(row, 1) = 0.1 * first + second +
        0.001 * std::cos(0.23 * static_cast<double>(row));
    asset_returns(row, 2) = 0.5 * first + 0.4 * second +
        0.0015 * std::sin(0.31 * static_cast<double>(row));
    timestamps.push_back(static_cast<engine_common::TimestampNs>(row + 1));
  }
  std::vector<double> weights{0.4, 0.35, 0.25};
  FactorRiskDiagnosticInput input{
      risk_model,
      101,
      quant_math::view(factor_returns),
      quant_math::view(asset_returns),
      weights,
      timestamps,
      40,
      40,
      1e-12,
  };
  const auto result = portfolio_math::evaluate_factor_risk_diagnostics(input);
  bool ok = result.status == FactorRiskDiagnosticStatus::OK;
  ok = ok && result.predicted_risk_artifact_hash == 101;
  ok = ok && result.effective_observations == 40;
  ok = ok && std::isfinite(result.factor_covariance_qlike);
  ok = ok && result.predicted_risk_contributions.size() == 3;
  ok = ok && result.realized_risk_contributions.size() == 3;
  ok = ok && close(result.realized_predicted_variance_ratio,
                   result.realized_portfolio_variance /
                       result.predicted_portfolio_variance);
  ok = ok && close(sum(result.predicted_risk_contributions), 1.0, 1e-10);
  ok = ok && close(sum(result.realized_risk_contributions), 1.0, 1e-10);
  ok = ok && close(result.predicted_portfolio_variance,
                   result.predicted_factor_variance +
                       result.predicted_specific_variance,
                   1e-12);
  ok = ok && result.predicted_variance_reconciliation_error < 1e-15;

  const DenseMatrix dense_covariance =
      portfolio_math::materialize_factor_covariance(risk_model);
  const DenseVector weight_vector = DenseVector::Map(
      weights.data(), static_cast<Eigen::Index>(weights.size()));
  const DenseVector predicted_marginal = dense_covariance * weight_vector;
  const double dense_predicted_variance = weight_vector.dot(predicted_marginal);
  ok = ok && close(result.predicted_portfolio_variance,
                   dense_predicted_variance, 1e-12);
  for (std::size_t asset = 0; asset < weights.size(); ++asset) {
    const double expected_contribution =
        weights[asset] * predicted_marginal(static_cast<Eigen::Index>(asset)) /
        dense_predicted_variance;
    ok = ok && close(result.predicted_risk_contributions[asset],
                     expected_contribution, 1e-12);
  }

  DenseMatrix centered_returns = asset_returns;
  centered_returns.rowwise() -= centered_returns.colwise().mean();
  const DenseMatrix realized_covariance = centered_returns.transpose() *
      centered_returns / static_cast<double>(asset_returns.rows());
  const DenseVector realized_marginal = realized_covariance * weight_vector;
  const double dense_realized_variance = weight_vector.dot(realized_marginal);
  ok = ok && close(result.realized_portfolio_variance,
                   dense_realized_variance, 1e-12);
  for (std::size_t asset = 0; asset < weights.size(); ++asset) {
    const double expected_contribution =
        weights[asset] * realized_marginal(static_cast<Eigen::Index>(asset)) /
        dense_realized_variance;
    ok = ok && close(result.realized_risk_contributions[asset],
                     expected_contribution, 1e-12);
  }

  auto future_timestamps = timestamps;
  future_timestamps.back() = 41;
  input.observation_timestamps = future_timestamps;
  const auto future_observation =
      portfolio_math::evaluate_factor_risk_diagnostics(input);
  ok = ok && future_observation.status == FactorRiskDiagnosticStatus::FUTURE_DATA;

  input.observation_timestamps = timestamps;
  input.available_at = 41;
  const auto future_availability =
      portfolio_math::evaluate_factor_risk_diagnostics(input);
  ok = ok && future_availability.status == FactorRiskDiagnosticStatus::FUTURE_DATA;
  input.available_at = 40;

  input.predicted_risk_artifact_hash = 0;
  const auto missing_artifact =
      portfolio_math::evaluate_factor_risk_diagnostics(input);
  ok = ok && missing_artifact.status == FactorRiskDiagnosticStatus::INVALID_INPUT;
  input.predicted_risk_artifact_hash = 101;

  DenseMatrix non_psd = predicted_factor_covariance;
  non_psd(0, 0) = -1.0;
  input.predicted_risk_model.factor_covariance = quant_math::view(non_psd);
  const auto invalid_covariance =
      portfolio_math::evaluate_factor_risk_diagnostics(input);
  ok = ok && invalid_covariance.status == FactorRiskDiagnosticStatus::NON_PSD;
  input.predicted_risk_model.factor_covariance =
      quant_math::view(predicted_factor_covariance);

  auto invalid_specific_variance = specific_variance;
  invalid_specific_variance[0] = 0.0;
  input.predicted_risk_model.specific_variance = invalid_specific_variance;
  const auto invalid_specific =
      portfolio_math::evaluate_factor_risk_diagnostics(input);
  ok = ok && invalid_specific.status == FactorRiskDiagnosticStatus::INVALID_INPUT;
  input.predicted_risk_model.specific_variance = specific_variance;

  factor_returns(0, 0) = std::numeric_limits<double>::quiet_NaN();
  const auto non_finite =
      portfolio_math::evaluate_factor_risk_diagnostics(input);
  ok = ok && non_finite.status == FactorRiskDiagnosticStatus::INVALID_INPUT;
  factor_returns(0, 0) = 0.0;

  constexpr std::size_t large_asset_count = 5'000;
  constexpr std::size_t large_factor_count = 4;
  constexpr std::size_t large_observation_count = 32;
  DenseMatrix large_exposures(large_asset_count, large_factor_count);
  for (std::size_t asset = 0; asset < large_asset_count; ++asset) {
    large_exposures(static_cast<Eigen::Index>(asset), 0) = 1.0;
    large_exposures(static_cast<Eigen::Index>(asset), 1) =
        (static_cast<double>(asset % 17) - 8.0) / 8.0;
    large_exposures(static_cast<Eigen::Index>(asset), 2) =
        (static_cast<double>(asset % 11) - 5.0) / 5.0;
    large_exposures(static_cast<Eigen::Index>(asset), 3) =
        (static_cast<double>(asset % 7) - 3.0) / 3.0;
  }
  DenseMatrix large_factor_covariance = DenseMatrix::Zero(
      large_factor_count, large_factor_count);
  large_factor_covariance.diagonal() << 2.0e-4, 1.5e-4, 1.2e-4, 1.0e-4;
  large_factor_covariance(0, 1) = 1.0e-5;
  large_factor_covariance(1, 0) = 1.0e-5;
  std::vector<double> large_specific_variance(large_asset_count);
  std::vector<double> large_weights(
      large_asset_count, 1.0 / static_cast<double>(large_asset_count));
  for (std::size_t asset = 0; asset < large_asset_count; ++asset) {
    large_specific_variance[asset] =
        2.0e-5 + 1.0e-6 * static_cast<double>(asset % 13);
  }
  DenseMatrix large_factor_returns(large_observation_count,
                                   large_factor_count);
  DenseMatrix large_asset_returns(large_observation_count,
                                  large_asset_count);
  std::vector<engine_common::TimestampNs> large_timestamps;
  for (std::size_t row = 0; row < large_observation_count; ++row) {
    for (std::size_t factor = 0; factor < large_factor_count; ++factor) {
      large_factor_returns(static_cast<Eigen::Index>(row),
                           static_cast<Eigen::Index>(factor)) =
          0.01 * std::sin(0.11 * static_cast<double>((row + 1) * (factor + 1)));
    }
    for (std::size_t asset = 0; asset < large_asset_count; ++asset) {
      double asset_return = 0.0;
      for (std::size_t factor = 0; factor < large_factor_count; ++factor) {
        asset_return +=
            large_exposures(static_cast<Eigen::Index>(asset),
                            static_cast<Eigen::Index>(factor)) *
            large_factor_returns(static_cast<Eigen::Index>(row),
                                 static_cast<Eigen::Index>(factor));
      }
      asset_return += 0.001 * std::sin(
          0.013 * static_cast<double>((row + 1) * ((asset % 97) + 1)));
      large_asset_returns(static_cast<Eigen::Index>(row),
                          static_cast<Eigen::Index>(asset)) = asset_return;
    }
    large_timestamps.push_back(
        static_cast<engine_common::TimestampNs>(row + 1));
  }
  const FactorRiskModelView large_risk_model{
      quant_math::view(large_exposures),
      quant_math::view(large_factor_covariance),
      std::span<const double>(large_specific_variance),
  };
  const FactorRiskDiagnosticInput large_input{
      large_risk_model,
      5'000,
      quant_math::view(large_factor_returns),
      quant_math::view(large_asset_returns),
      large_weights,
      large_timestamps,
      large_observation_count,
      large_observation_count,
      1e-12,
  };
  const auto large_result =
      portfolio_math::evaluate_factor_risk_diagnostics(large_input);
  const auto repeated_large_result =
      portfolio_math::evaluate_factor_risk_diagnostics(large_input);
  ok = ok && large_result.status == FactorRiskDiagnosticStatus::OK;
  ok = ok && large_result.predicted_risk_contributions.size() ==
      large_asset_count;
  ok = ok && large_result.realized_risk_contributions.size() ==
      large_asset_count;
  ok = ok && close(sum(large_result.predicted_risk_contributions), 1.0, 1e-9);
  ok = ok && close(sum(large_result.realized_risk_contributions), 1.0, 1e-9);
  ok = ok && large_result.predicted_variance_reconciliation_error < 1e-14;
  ok = ok && repeated_large_result.status == FactorRiskDiagnosticStatus::OK;
  ok = ok && close(large_result.factor_covariance_qlike,
                   repeated_large_result.factor_covariance_qlike, 0.0);
  ok = ok && close(large_result.realized_portfolio_variance,
                   repeated_large_result.realized_portfolio_variance, 0.0);
  ok = ok && large_result.predicted_risk_contributions ==
      repeated_large_result.predicted_risk_contributions;
  ok = ok && large_result.realized_risk_contributions ==
      repeated_large_result.realized_risk_contributions;

  std::cout << (ok ? "test_factor_risk_diagnostics: all checks passed\n"
                   : "test_factor_risk_diagnostics: failed\n");
  return ok ? 0 : 1;
}

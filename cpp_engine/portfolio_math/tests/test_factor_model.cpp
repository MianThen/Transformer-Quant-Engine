#include "portfolio_math/factor_model.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace {

bool close(double left, double right, double tolerance = 1e-8) {
  return std::abs(left - right) <= tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

}  // namespace

int main() {
  using quant_math::DenseMatrix;
  using portfolio_math::FactorModelInput;
  using portfolio_math::FactorModelSpec;
  using portfolio_math::FactorModelStatus;

  DenseMatrix exposures(3, 2);
  exposures << 1.0, 0.0,
               0.0, 1.0,
               1.0, 1.0;
  DenseMatrix returns(30, 3);
  for (int row = 0; row < returns.rows(); ++row) {
    const double market = 0.001 * static_cast<double>(row + 1);
    const double style = 0.002 * std::sin(static_cast<double>(row));
    returns(row, 0) = market + 0.0001;
    returns(row, 1) = style - 0.0002;
    returns(row, 2) = market + style + 0.0003;
  }
  std::vector<engine_common::TimestampNs> timestamps;
  for (int row = 0; row < returns.rows(); ++row) {
    timestamps.push_back(static_cast<engine_common::TimestampNs>(row + 1));
  }
  FactorModelSpec spec;
  spec.minimum_observations = 20;
  spec.factor_schema_hash = 11;
  spec.wls_spec_hash = 13;
  spec.config_hash = 17;
  FactorModelInput input{
      quant_math::view(returns), quant_math::view(exposures), {}, {}, timestamps,
      1, 30, 30, 30, 19, {},
  };
  input.exposure_source.status =
      portfolio_math::FactorExposureSourceStatus::KNOWN;
  input.exposure_source.source_schema_hash = 23;
  input.exposure_source.snapshot_hash = 29;
  input.exposure_source.provenance_hash = 31;
  input.exposure_source.coverage_start = 1;
  input.exposure_source.coverage_end = 30;
  input.exposure_source.available_at = 30;
  input.exposure_source.point_in_time = true;
  const auto artifact = portfolio_math::fit_factor_pit_ewma(input, spec);
  bool ok = artifact.status == FactorModelStatus::OK;
  ok = ok && artifact.artifact_hash != 0;
  ok = ok && portfolio_math::valid_factor_model_artifact(artifact, 30);
  const auto serialized = portfolio_math::serialize_factor_model_artifact(artifact);
  ok = ok && portfolio_math::factor_model_artifact_hash(artifact) == artifact.artifact_hash &&
      serialized.find("\"exposure_source\"") != std::string::npos &&
      serialized.find("\"status\":\"KNOWN\"") != std::string::npos;
  ok = ok && artifact.factor_covariance.rows() == 2;
  ok = ok && artifact.specific_variance.size() == 3;
  ok = ok && artifact.diagnostics.maximum_wls_orthogonality_error < 1e-10;
  ok = ok && quant_math::is_positive_semidefinite(artifact.factor_covariance, 1e-10);
  const auto artifact_hash = artifact.artifact_hash;
  exposures(0, 0) = 99.0;
  ok = ok && artifact.artifact_hash == artifact_hash &&
      close(artifact.exposures(0, 0), 1.0);
  const auto source_hash = artifact.artifact_hash;
  input.exposure_source.provenance_hash = 37;
  const auto changed_source = portfolio_math::fit_factor_pit_ewma(input, spec);
  ok = ok && changed_source.status == FactorModelStatus::OK &&
      changed_source.artifact_hash != source_hash;
  input.exposure_source.provenance_hash = 31;

  DenseMatrix exposure_history(returns.rows() * exposures.rows(), exposures.cols());
  for (int row = 0; row < returns.rows(); ++row) {
    exposure_history.block(row * exposures.rows(), 0, exposures.rows(), exposures.cols()) = exposures;
  }
  input.pit_exposure_history = quant_math::view(exposure_history);
  const auto history_artifact = portfolio_math::fit_factor_pit_ewma(input, spec);
  ok = ok && history_artifact.status == FactorModelStatus::OK &&
      history_artifact.exposure_history.rows() == returns.rows() * exposures.rows();

  portfolio_math::FactorRiskModelView view{
      quant_math::view(artifact.exposures), quant_math::view(artifact.factor_covariance),
      std::span<const double>(artifact.specific_variance.data(),
                              static_cast<std::size_t>(artifact.specific_variance.size())),
  };
  std::vector<double> weights{0.2, 0.3, 0.5};
  const double factor_variance = portfolio_math::factor_form_variance(view, weights);
  const DenseMatrix dense = portfolio_math::materialize_factor_covariance(view);
  Eigen::VectorXd weight_vector = Eigen::Map<Eigen::VectorXd>(weights.data(), 3);
  const double dense_variance = (weight_vector.transpose() * dense * weight_vector)(0, 0);
  ok = ok && close(factor_variance, dense_variance, 1e-10);
  const auto gradient = portfolio_math::factor_form_gradient(view, weights);
  const double bump = 1e-7;
  auto bumped = weights;
  bumped[0] += bump;
  const double finite_difference =
      (portfolio_math::factor_form_variance(view, bumped) - factor_variance) / bump;
  ok = ok && close(gradient[0], finite_difference, 1e-5);
  const auto optimizer = portfolio_math::solve_factor_form_min_variance(view);
  ok = ok && optimizer.diagnostics.status == portfolio_math::FactorOptimizerStatus::OK;
  ok = ok && close(std::accumulate(optimizer.weights.begin(), optimizer.weights.end(), 0.0), 1.0, 1e-8);

  auto future_timestamps = timestamps;
  future_timestamps.back() = 31;
  input.observation_timestamps = future_timestamps;
  const auto future = portfolio_math::fit_factor_pit_ewma(input, spec);
  ok = ok && future.status == FactorModelStatus::FUTURE_DATA;
  input.observation_timestamps = timestamps;
  input.exposure_source.available_at = 31;
  const auto future_source = portfolio_math::fit_factor_pit_ewma(input, spec);
  ok = ok && future_source.status == FactorModelStatus::FUTURE_DATA;
  input.exposure_source.available_at = 30;
  input.exposure_source.status =
      portfolio_math::FactorExposureSourceStatus::UNAVAILABLE;
  const auto unavailable_source = portfolio_math::fit_factor_pit_ewma(input, spec);
  ok = ok && unavailable_source.status == FactorModelStatus::INVALID_INPUT;
  std::cout << (ok ? "test_factor_model: all checks passed\n"
                   : "test_factor_model: failed\n");
  return ok ? 0 : 1;
}

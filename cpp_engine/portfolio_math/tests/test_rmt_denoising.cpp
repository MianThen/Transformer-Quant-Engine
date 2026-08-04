#include <cmath>
#include <cstdio>

#include "portfolio_math/covariance.h"
#include "portfolio_math/rmt_denoising.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAILED: %s\n", message);
  }
  return condition;
}

bool near(double left, double right, double tolerance = 1e-10) {
  return std::abs(left - right) <= tolerance;
}

quant_math::DenseMatrix make_block_returns() {
  quant_math::DenseMatrix returns(30, 3);
  for (Eigen::Index row = 0; row < returns.rows(); ++row) {
    const int position = static_cast<int>(row % 3);
    const double factor = static_cast<double>(position - 1);
    const double orthogonal = position == 1 ? -2.0 : 1.0;
    returns(row, 0) = factor;
    returns(row, 1) = factor;
    returns(row, 2) = orthogonal;
  }
  return returns;
}

bool test_block_spectrum() {
  const auto returns = make_block_returns();
  const auto result = portfolio_math::rmt_constant_residual_denoising(
      quant_math::view(returns));
  bool ok = true;
  ok &= check(result.covariance.status == portfolio_math::CovarianceStatus::OK,
              "RMT constant residual status");
  ok &= check(result.covariance.estimator ==
                  portfolio_math::CovarianceEstimator::RMT_CONSTANT_RESIDUAL,
              "RMT estimator identity");
  ok &= check(result.diagnostics.retained_signal_rank == 1 &&
                  result.diagnostics.noise_eigenvalue_count == 2,
              "MP boundary separates one signal eigenvalue");
  ok &= check(near(result.diagnostics.aspect_ratio, 0.1) &&
                  near(result.diagnostics.fitted_noise_boundary,
                       std::pow(1.0 + std::sqrt(0.1), 2.0)) &&
                  near(result.diagnostics.constant_residual_eigenvalue, 0.5),
              "MP boundary and constant residual oracle");
  ok &= check(near(result.covariance.covariance(0, 0), 2.0 / 3.0) &&
                  near(result.covariance.covariance(1, 1), 2.0 / 3.0) &&
                  near(result.covariance.covariance(2, 2), 2.0),
              "volatility mapping preserves covariance diagonal");
  ok &= check(near(result.covariance.covariance(0, 1), 0.4),
              "constant residual reduces perfect block correlation");
  ok &= check(result.diagnostics.trace_drift < 1e-12 &&
                  result.diagnostics.maximum_relative_diagonal_drift < 1e-12 &&
                  result.diagnostics.psd_repair_amount == 0.0 &&
                  !result.diagnostics.eligible_for_official_risk &&
                  result.diagnostics.condition_number >= 1.0,
              "RMT preservation and diagnostics");
  ok &= check(quant_math::is_positive_semidefinite(
                  result.covariance.covariance, 1e-10),
              "RMT covariance is PSD");

  const auto selection = portfolio_math::estimate_research_covariance(
      quant_math::view(returns),
      portfolio_math::CovarianceEstimator::RMT_CONSTANT_RESIDUAL);
  ok &= check(selection.has_rmt_diagnostics &&
                  selection.covariance.status == portfolio_math::CovarianceStatus::OK,
              "research selection exposes RMT diagnostics");
  return ok;
}

bool test_permutation_and_failure() {
  const auto source = make_block_returns();
  quant_math::DenseMatrix permuted(source.rows(), source.cols());
  permuted.col(0) = source.col(2);
  permuted.col(1) = source.col(0);
  permuted.col(2) = source.col(1);
  const auto first = portfolio_math::rmt_constant_residual_denoising(
      quant_math::view(source));
  const auto second = portfolio_math::rmt_constant_residual_denoising(
      quant_math::view(permuted));
  bool ok = first.covariance.status == portfolio_math::CovarianceStatus::OK &&
            second.covariance.status == portfolio_math::CovarianceStatus::OK;
  ok &= check(ok, "permutation fixture status");
  if (ok) {
    const std::size_t mapping[3]{2, 0, 1};
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t col = 0; col < 3; ++col) {
        ok &= check(near(second.covariance.covariance(
                              static_cast<Eigen::Index>(row),
                              static_cast<Eigen::Index>(col)),
                          first.covariance.covariance(
                              static_cast<Eigen::Index>(mapping[row]),
                              static_cast<Eigen::Index>(mapping[col])),
                          1e-10),
                    "RMT column permutation equivariance");
      }
    }
  }

  quant_math::DenseMatrix constant(4, 2);
  constant << 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0;
  const auto failed = portfolio_math::rmt_constant_residual_denoising(
      quant_math::view(constant));
  ok &= check(failed.covariance.status ==
                  portfolio_math::CovarianceStatus::INVALID_INPUT,
              "zero-variance column fails closed");

  auto invalid_spec = portfolio_math::RmtDenoisingSpec{};
  invalid_spec.eigenvalue_floor = 0.0;
  const auto invalid = portfolio_math::rmt_constant_residual_denoising(
      quant_math::view(source), invalid_spec);
  ok &= check(invalid.covariance.status ==
                  portfolio_math::CovarianceStatus::INVALID_INPUT,
              "invalid RMT spec fails closed");
  return ok;
}

bool test_targeted_shrinkage() {
  const auto returns = make_block_returns();
  auto spec = portfolio_math::RmtDenoisingSpec{};
  spec.targeted_shrinkage_intensity = 0.5;
  const auto result = portfolio_math::rmt_targeted_shrinkage_denoising(
      quant_math::view(returns), spec);
  bool ok = true;
  ok &= check(result.covariance.status == portfolio_math::CovarianceStatus::OK,
              "RMT targeted shrinkage status");
  ok &= check(result.covariance.estimator ==
                  portfolio_math::CovarianceEstimator::RMT_TARGETED_SHRINKAGE,
              "RMT targeted estimator identity");
  ok &= check(result.diagnostics.retained_signal_rank == 1 &&
                  result.diagnostics.noise_eigenvalue_count == 2,
              "targeted MP boundary separates signal and noise");
  ok &= check(near(result.diagnostics.cleaned_eigenvalues[0], 0.25) &&
                  near(result.diagnostics.cleaned_eigenvalues[1], 0.75) &&
                  near(result.diagnostics.cleaned_eigenvalues[2], 2.0),
              "targeted shrinkage preserves signal eigenvalue");
  ok &= check(near(result.diagnostics.targeted_shrinkage_intensity, 0.5),
              "targeted shrinkage intensity is auditable");
  ok &= check(near(result.covariance.covariance(0, 1), 14.0 / 27.0) &&
                  result.diagnostics.trace_drift < 1e-12 &&
                  !result.diagnostics.eligible_for_official_risk,
              "targeted covariance and research-only diagnostics");

  const auto selection = portfolio_math::estimate_research_covariance(
      quant_math::view(returns),
      portfolio_math::CovarianceEstimator::RMT_TARGETED_SHRINKAGE);
  ok &= check(selection.has_rmt_diagnostics &&
                  selection.covariance.status == portfolio_math::CovarianceStatus::OK &&
                  selection.covariance.estimator ==
                      portfolio_math::CovarianceEstimator::RMT_TARGETED_SHRINKAGE,
              "research selection exposes targeted diagnostics");

  spec.targeted_shrinkage_intensity = 1.1;
  const auto invalid = portfolio_math::rmt_targeted_shrinkage_denoising(
      quant_math::view(returns), spec);
  ok &= check(invalid.covariance.status ==
                  portfolio_math::CovarianceStatus::INVALID_INPUT,
              "targeted shrinkage intensity is bounded");
  return ok;
}

bool test_singular_p_gt_t_fails_without_trace_drift() {
  quant_math::DenseMatrix returns(4, 6);
  for (Eigen::Index row = 0; row < returns.rows(); ++row) {
    const double factor = static_cast<double>(row - 1);
    for (Eigen::Index col = 0; col < returns.cols(); ++col) {
      returns(row, col) = factor * (1.0 + 0.1 * static_cast<double>(col));
    }
  }
  const auto result = portfolio_math::rmt_constant_residual_denoising(
      quant_math::view(returns));
  return check(result.covariance.status ==
                   portfolio_math::CovarianceStatus::NUMERICAL_FAILURE,
               "RMT rejects floor-induced trace drift for exact low rank p>T");
}

}  // namespace

int main() {
  if (!(test_block_spectrum() && test_permutation_and_failure() &&
        test_targeted_shrinkage() &&
        test_singular_p_gt_t_fails_without_trace_drift())) {
    return 1;
  }
  std::printf("test_rmt_denoising: all checks passed\n");
  return 0;
}

#include "portfolio_math/covariance.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace portfolio_math {
namespace {

using quant_math::DenseMatrix;
using quant_math::DenseVector;

struct CenteredReturns {
  bool ok{false};
  DenseMatrix values;
};

CenteredReturns center(quant_math::MatrixView returns) {
  if (returns.rows < 2 || returns.cols == 0 ||
      !quant_math::validate_finite(returns).ok) {
    return {};
  }
  DenseMatrix values(static_cast<Eigen::Index>(returns.rows),
                     static_cast<Eigen::Index>(returns.cols));
  for (std::size_t row = 0; row < returns.rows; ++row) {
    for (std::size_t col = 0; col < returns.cols; ++col) {
      values(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          returns(row, col);
    }
  }
  values.rowwise() -= values.colwise().mean();
  return {true, std::move(values)};
}

std::uint64_t fingerprint(quant_math::MatrixView matrix) {
  std::uint64_t hash = 1469598103934665603ULL;
  const auto append = [&](std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
      hash ^= (value >> (byte * 8)) & 0xffU;
      hash *= 1099511628211ULL;
    }
  };
  append(matrix.rows);
  append(matrix.cols);
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t col = 0; col < matrix.cols; ++col) {
      append(std::bit_cast<std::uint64_t>(matrix(row, col)));
    }
  }
  return hash;
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

double minimum_variance_loss(const DenseMatrix &estimate,
                             const DenseMatrix &population) {
  const DenseMatrix inverse = estimate.ldlt().solve(
      DenseMatrix::Identity(estimate.rows(), estimate.cols()));
  const double denominator = inverse.trace();
  return (inverse * population * inverse).trace() / (denominator * denominator);
}

} // namespace

CovarianceResult sample_covariance(quant_math::MatrixView returns) {
  auto centered = center(returns);
  if (!centered.ok) {
    CovarianceResult result;
    result.status = returns.rows < 2
                        ? CovarianceStatus::INSUFFICIENT_OBSERVATIONS
                        : CovarianceStatus::INVALID_INPUT;
    return result;
  }
  const double observations = static_cast<double>(returns.rows);
  DenseMatrix covariance =
      centered.values.transpose() * centered.values / observations;
  covariance = 0.5 * (covariance + covariance.transpose()).eval();
  return {CovarianceStatus::OK,
          CovarianceEstimator::SAMPLE,
          CovarianceLossProfile::NOT_APPLICABLE,
          std::move(covariance),
          0.0,
          0.0,
          returns.rows,
          static_cast<double>(returns.cols) /
              static_cast<double>(returns.rows)};
}

CovarianceResult
ledoit_wolf_linear_constant_correlation(quant_math::MatrixView returns) {
  auto centered = center(returns);
  if (!centered.ok) {
    CovarianceResult result;
    result.status = returns.rows < 2
                        ? CovarianceStatus::INSUFFICIENT_OBSERVATIONS
                        : CovarianceStatus::INVALID_INPUT;
    result.estimator =
        CovarianceEstimator::LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION;
    result.loss_profile = CovarianceLossProfile::FROBENIUS;
    return result;
  }
  const Eigen::Index asset_count = centered.values.cols();
  const double observations = static_cast<double>(centered.values.rows());
  DenseMatrix sample =
      centered.values.transpose() * centered.values / observations;
  sample = 0.5 * (sample + sample.transpose()).eval();
  if (asset_count == 1) {
    return {CovarianceStatus::OK,
            CovarianceEstimator::LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION,
            CovarianceLossProfile::FROBENIUS,
            std::move(sample),
            0.0,
            0.0,
            returns.rows,
            1.0 / static_cast<double>(returns.rows)};
  }

  const DenseVector variances = sample.diagonal();
  if ((variances.array() < 0.0).any()) {
    CovarianceResult result;
    result.status = CovarianceStatus::NUMERICAL_FAILURE;
    result.estimator =
        CovarianceEstimator::LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION;
    result.loss_profile = CovarianceLossProfile::FROBENIUS;
    return result;
  }
  const DenseVector standard_deviations = variances.cwiseSqrt();
  double correlation_sum = 0.0;
  std::size_t valid_correlation_count = 0;
  for (Eigen::Index row = 0; row < asset_count; ++row) {
    for (Eigen::Index col = row + 1; col < asset_count; ++col) {
      const double denominator =
          standard_deviations(row) * standard_deviations(col);
      if (denominator > 0.0) {
        correlation_sum += sample(row, col) / denominator;
        ++valid_correlation_count;
      }
    }
  }
  const double average_correlation =
      valid_correlation_count == 0
          ? 0.0
          : correlation_sum / static_cast<double>(valid_correlation_count);

  DenseMatrix target = DenseMatrix::Zero(asset_count, asset_count);
  target.diagonal() = variances;
  for (Eigen::Index row = 0; row < asset_count; ++row) {
    for (Eigen::Index col = row + 1; col < asset_count; ++col) {
      const double value = average_correlation * standard_deviations(row) *
                           standard_deviations(col);
      target(row, col) = value;
      target(col, row) = value;
    }
  }

  const DenseMatrix squared = centered.values.array().square().matrix();
  const DenseMatrix phi_matrix = squared.transpose() * squared / observations -
                                 sample.array().square().matrix();
  const double phi = phi_matrix.sum();
  const DenseMatrix cubed = centered.values.array().cube().matrix();
  const DenseMatrix theta = cubed.transpose() * centered.values / observations -
                            variances.asDiagonal() * sample;
  double rho = phi_matrix.diagonal().sum();
  for (Eigen::Index row = 0; row < asset_count; ++row) {
    for (Eigen::Index col = 0; col < asset_count; ++col) {
      if (row == col || standard_deviations(row) == 0.0)
        continue;
      rho += average_correlation * standard_deviations(col) /
             standard_deviations(row) * theta(row, col);
    }
  }
  const double gamma = (sample - target).squaredNorm();
  double shrinkage = 1.0;
  if (gamma > std::numeric_limits<double>::epsilon()) {
    shrinkage = std::clamp((phi - rho) / (gamma * observations), 0.0, 1.0);
  }
  DenseMatrix covariance = shrinkage * target + (1.0 - shrinkage) * sample;
  covariance = 0.5 * (covariance + covariance.transpose()).eval();
  if (!quant_math::validate_finite(quant_math::view(covariance)).ok) {
    CovarianceResult result;
    result.status = CovarianceStatus::NUMERICAL_FAILURE;
    result.estimator =
        CovarianceEstimator::LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION;
    result.loss_profile = CovarianceLossProfile::FROBENIUS;
    result.shrinkage_intensity = shrinkage;
    result.average_correlation = average_correlation;
    return result;
  }
  return {CovarianceStatus::OK,
          CovarianceEstimator::LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION,
          CovarianceLossProfile::FROBENIUS,
          std::move(covariance),
          shrinkage,
          average_correlation,
          returns.rows,
          static_cast<double>(returns.cols) /
              static_cast<double>(returns.rows)};
}

NonlinearCovarianceResult
ledoit_wolf_nonlinear_minimum_variance_quest(quant_math::MatrixView returns,
                                             QuestSpec spec) {
  NonlinearCovarianceResult output;
  output.covariance.estimator =
      CovarianceEstimator::LEDOIT_WOLF_NONLINEAR_QUEST;
  output.covariance.loss_profile = CovarianceLossProfile::MINIMUM_VARIANCE;
  output.diagnostics.raw_observations = returns.rows;

  auto centered = center(returns);
  if (!centered.ok) {
    output.covariance.status = returns.rows < 2
                                   ? CovarianceStatus::INSUFFICIENT_OBSERVATIONS
                                   : CovarianceStatus::INVALID_INPUT;
    return output;
  }
  const std::size_t dimension = returns.cols;
  // Estimating the column means consumes one degree of freedom. This freezes
  // the V1 centered-panel convention and preserves the paper's rank branches.
  const std::size_t observations = returns.rows - 1;
  output.diagnostics.quest_effective_observations = observations;
  output.covariance.effective_observations = observations;
  const double c =
      static_cast<double>(dimension) / static_cast<double>(observations);
  output.covariance.concentration_ratio = c;
  if (std::abs(c - 1.0) <= spec.concentration_ratio_guard) {
    output.diagnostics.quest_status =
        QuestStatus::CONCENTRATION_RATIO_TOO_CLOSE_TO_ONE;
    output.diagnostics.dimensional_branch =
        NonlinearDimensionalBranch::GUARDED_P_EQ_N;
    output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
    return output;
  }
  output.diagnostics.dimensional_branch =
      dimension < observations ? NonlinearDimensionalBranch::REGULAR_P_LT_N
                               : NonlinearDimensionalBranch::SINGULAR_P_GT_N;

  DenseMatrix sample = centered.values.transpose() * centered.values /
                       static_cast<double>(observations);
  sample = 0.5 * (sample + sample.transpose()).eval();
  Eigen::SelfAdjointEigenSolver<DenseMatrix> eigensolver(sample);
  if (eigensolver.info() != Eigen::Success) {
    output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
    return output;
  }
  output.diagnostics.sample_eigenvalues.resize(dimension);
  for (std::size_t index = 0; index < dimension; ++index) {
    const double value =
        eigensolver.eigenvalues()(static_cast<Eigen::Index>(index));
    const double scale = std::max(1.0, sample.diagonal().cwiseAbs().maxCoeff());
    if (value < -1e-10 * scale) {
      output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
      return output;
    }
    output.diagnostics.sample_eigenvalues[index] = std::max(0.0, value);
  }

  auto inverse =
      quest_inverse(output.diagnostics.sample_eigenvalues, observations, spec);
  output.diagnostics.quest_status = inverse.status;
  output.diagnostics.quest = inverse.diagnostics;
  output.diagnostics.population_eigenvalues = inverse.population_eigenvalues;
  if (inverse.status != QuestStatus::OK ||
      inverse.population_eigenvalues.size() != dimension) {
    output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
    return output;
  }

  std::size_t structural_zeros = 0;
  if (dimension > observations) {
    const double zero_tolerance =
        1e-10 * std::max(spec.eigenvalue_floor,
                         output.diagnostics.sample_eigenvalues.back());
    while (structural_zeros < dimension &&
           output.diagnostics.sample_eigenvalues[structural_zeros] <=
               zero_tolerance) {
      ++structural_zeros;
    }
    if (structural_zeros < dimension - observations) {
      output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
      return output;
    }
  }
  output.diagnostics.structural_zero_count = structural_zeros;
  output.diagnostics.shrunk_eigenvalues.assign(dimension, 0.0);
  output.diagnostics.minimum_angle_weight =
      std::numeric_limits<double>::infinity();

  if (structural_zeros > 0) {
    const auto equation = [&](double m0) {
      double right = 0.0;
      for (double tau : inverse.population_eigenvalues) {
        right += tau / (1.0 + tau * m0);
      }
      right /= static_cast<double>(observations);
      return 1.0 / m0 - right;
    };
    double lower = std::max(spec.eigenvalue_floor, 1e-15);
    double upper = 1.0;
    while (equation(upper) > 0.0 && upper < 1e16)
      upper *= 2.0;
    if (!(equation(lower) > 0.0) || !(equation(upper) < 0.0)) {
      output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
      return output;
    }
    for (std::uint32_t iteration = 0; iteration < spec.max_root_iterations;
         ++iteration) {
      const double middle = std::midpoint(lower, upper);
      if (equation(middle) > 0.0) {
        lower = middle;
      } else {
        upper = middle;
      }
      if (upper - lower <=
          spec.root_tolerance * std::max({1.0, lower, upper})) {
        break;
      }
    }
    const double m0 = std::midpoint(lower, upper);
    output.diagnostics.null_equation_residual = std::abs(equation(m0));
    double shrunk = 0.0;
    double row_mass = 0.0;
    for (double tau : inverse.population_eigenvalues) {
      const double theta = 1.0 / ((1.0 - 1.0 / c) * (1.0 + m0 * tau));
      if (!std::isfinite(theta) || theta < 0.0) {
        output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
        return output;
      }
      output.diagnostics.minimum_angle_weight =
          std::min(output.diagnostics.minimum_angle_weight, theta);
      row_mass += theta / static_cast<double>(dimension);
      shrunk += tau * theta / static_cast<double>(dimension);
    }
    output.diagnostics.maximum_angle_row_mass_error = std::abs(row_mass - 1.0);
    output.diagnostics.null_space_shrinkage = shrunk;
    std::fill_n(output.diagnostics.shrunk_eigenvalues.begin(), structural_zeros,
                shrunk);
  }

  for (std::size_t row = structural_zeros; row < dimension; ++row) {
    const double lambda = output.diagnostics.sample_eigenvalues[row];
    if (!(lambda > 0.0)) {
      output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
      return output;
    }
    const auto stieltjes = quest_boundary_stieltjes(
        inverse.population_eigenvalues, observations, lambda, spec);
    if (stieltjes.status != QuestStatus::OK) {
      output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
      return output;
    }
    output.diagnostics.maximum_stieltjes_residual = std::max(
        output.diagnostics.maximum_stieltjes_residual, stieltjes.residual);
    double shrunk = 0.0;
    double row_mass = 0.0;
    for (double tau : inverse.population_eigenvalues) {
      const std::complex<double> denominator =
          tau * (1.0 - c - c * lambda * stieltjes.value) - lambda;
      const double squared_norm = std::norm(denominator);
      const double theta = c * lambda * tau / squared_norm;
      if (!std::isfinite(theta) || theta < 0.0 || !(squared_norm > 0.0)) {
        output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
        return output;
      }
      output.diagnostics.minimum_angle_weight =
          std::min(output.diagnostics.minimum_angle_weight, theta);
      row_mass += theta / static_cast<double>(dimension);
      shrunk += tau * theta / static_cast<double>(dimension);
    }
    output.diagnostics.maximum_angle_row_mass_error =
        std::max(output.diagnostics.maximum_angle_row_mass_error,
                 std::abs(row_mass - 1.0));
    output.diagnostics.shrunk_eigenvalues[row] = shrunk;
  }

  if (output.diagnostics.maximum_angle_row_mass_error >
      spec.angle_row_mass_tolerance) {
    output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
    return output;
  }

  if (!std::all_of(
          output.diagnostics.shrunk_eigenvalues.begin(),
          output.diagnostics.shrunk_eigenvalues.end(), [&](double value) {
            return std::isfinite(value) && value >= spec.eigenvalue_floor;
          })) {
    output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
    return output;
  }
  Eigen::Map<const DenseVector> shrunk(
      output.diagnostics.shrunk_eigenvalues.data(),
      static_cast<Eigen::Index>(dimension));
  DenseMatrix covariance = eigensolver.eigenvectors() * shrunk.asDiagonal() *
                           eigensolver.eigenvectors().transpose();
  covariance = 0.5 * (covariance + covariance.transpose()).eval();
  if (!quant_math::validate_finite(quant_math::view(covariance)).ok ||
      !quant_math::is_positive_semidefinite(covariance, 1e-10)) {
    output.covariance.status = CovarianceStatus::NUMERICAL_FAILURE;
    return output;
  }

  const auto [minimum, maximum] =
      std::minmax_element(output.diagnostics.shrunk_eigenvalues.begin(),
                          output.diagnostics.shrunk_eigenvalues.end());
  output.diagnostics.condition_number = *maximum / *minimum;
  const double sample_trace = sample.trace();
  output.diagnostics.relative_trace_drift =
      std::abs(covariance.trace() - sample_trace) /
      std::max(spec.eigenvalue_floor, std::abs(sample_trace));
  for (std::size_t index = 0; index < dimension; ++index) {
    const auto eigen_index = static_cast<Eigen::Index>(index);
    output.diagnostics.maximum_relative_diagonal_drift =
        std::max(output.diagnostics.maximum_relative_diagonal_drift,
                 std::abs(covariance(eigen_index, eigen_index) -
                          sample(eigen_index, eigen_index)) /
                     std::max(spec.eigenvalue_floor,
                              std::abs(sample(eigen_index, eigen_index))));
  }
  output.diagnostics.psd_repair_amount = 0.0;
  output.covariance.covariance = std::move(covariance);
  output.covariance.status = CovarianceStatus::OK;
  return output;
}

ResearchCovarianceSelection
estimate_research_covariance(quant_math::MatrixView returns,
                             CovarianceEstimator frozen_estimator,
                             QuestSpec quest_spec) {
  ResearchCovarianceSelection output;
  switch (frozen_estimator) {
  case CovarianceEstimator::SAMPLE:
    output.covariance = sample_covariance(returns);
    return output;
  case CovarianceEstimator::LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION:
    output.covariance = ledoit_wolf_linear_constant_correlation(returns);
    return output;
  case CovarianceEstimator::LEDOIT_WOLF_NONLINEAR_QUEST: {
    auto nonlinear =
        ledoit_wolf_nonlinear_minimum_variance_quest(returns, quest_spec);
    output.covariance = std::move(nonlinear.covariance);
    output.nonlinear_diagnostics = std::move(nonlinear.diagnostics);
    output.has_nonlinear_diagnostics = true;
    return output;
  }
  default:
    output.covariance.estimator = frozen_estimator;
    return output;
  }
}

FixedInputCovarianceComparison
compare_covariance_estimators_fixed_input(quant_math::MatrixView returns,
                                          QuestSpec quest_spec) {
  FixedInputCovarianceComparison report;
  report.input_fingerprint = fingerprint(returns);
  report.sample = estimate_research_covariance(
      returns, CovarianceEstimator::SAMPLE, quest_spec);
  report.linear_constant_correlation = estimate_research_covariance(
      returns, CovarianceEstimator::LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION,
      quest_spec);
  report.nonlinear_minimum_variance_quest = estimate_research_covariance(
      returns, CovarianceEstimator::LEDOIT_WOLF_NONLINEAR_QUEST, quest_spec);
  report.all_estimators_succeeded =
      report.sample.covariance.status == CovarianceStatus::OK &&
      report.linear_constant_correlation.covariance.status ==
          CovarianceStatus::OK &&
      report.nonlinear_minimum_variance_quest.covariance.status ==
          CovarianceStatus::OK;
  if (report.all_estimators_succeeded) {
    report.linear_distance_from_sample =
        (report.linear_constant_correlation.covariance.covariance -
         report.sample.covariance.covariance)
            .norm();
    report.nonlinear_distance_from_sample =
        (report.nonlinear_minimum_variance_quest.covariance.covariance -
         report.sample.covariance.covariance)
            .norm();
  }
  return report;
}

CovarianceOracleEvaluation
evaluate_covariance_oracle(quant_math::MatrixView estimated_covariance,
                           quant_math::MatrixView sample_covariance_view,
                           quant_math::MatrixView population_covariance) {
  CovarianceOracleEvaluation result;
  if (estimated_covariance.rows == 0 ||
      estimated_covariance.rows != estimated_covariance.cols ||
      sample_covariance_view.rows != estimated_covariance.rows ||
      sample_covariance_view.cols != estimated_covariance.cols ||
      population_covariance.rows != estimated_covariance.rows ||
      population_covariance.cols != estimated_covariance.cols ||
      !quant_math::validate_symmetric(estimated_covariance, 1e-10).ok ||
      !quant_math::validate_symmetric(sample_covariance_view, 1e-10).ok ||
      !quant_math::validate_symmetric(population_covariance, 1e-10).ok) {
    return result;
  }
  const DenseMatrix estimate = copy_matrix(estimated_covariance);
  const DenseMatrix sample = copy_matrix(sample_covariance_view);
  const DenseMatrix population = copy_matrix(population_covariance);
  Eigen::SelfAdjointEigenSolver<DenseMatrix> estimate_eigenvalues(
      estimate, Eigen::EigenvaluesOnly);
  Eigen::SelfAdjointEigenSolver<DenseMatrix> sample_eigensolver(sample);
  Eigen::SelfAdjointEigenSolver<DenseMatrix> population_eigenvalues(
      population, Eigen::EigenvaluesOnly);
  if (estimate_eigenvalues.info() != Eigen::Success ||
      sample_eigensolver.info() != Eigen::Success ||
      population_eigenvalues.info() != Eigen::Success ||
      estimate_eigenvalues.eigenvalues().minCoeff() <= 0.0 ||
      sample_eigensolver.eigenvalues().minCoeff() <= 0.0 ||
      population_eigenvalues.eigenvalues().minCoeff() <= 0.0) {
    result.status = CovarianceStatus::NUMERICAL_FAILURE;
    return result;
  }

  const DenseMatrix rotated_population =
      sample_eigensolver.eigenvectors().transpose() * population *
      sample_eigensolver.eigenvectors();
  const DenseVector oracle_eigenvalues = rotated_population.diagonal();
  const DenseMatrix oracle = sample_eigensolver.eigenvectors() *
                             oracle_eigenvalues.asDiagonal() *
                             sample_eigensolver.eigenvectors().transpose();
  result.minimum_variance_loss = minimum_variance_loss(estimate, population);
  result.sample_minimum_variance_loss =
      minimum_variance_loss(sample, population);
  result.finite_sample_oracle_loss = minimum_variance_loss(oracle, population);
  const double improvement_denominator =
      result.sample_minimum_variance_loss - result.finite_sample_oracle_loss;
  if (std::abs(improvement_denominator) >
      std::numeric_limits<double>::epsilon()) {
    result.prial_percent =
        100.0 *
        (result.sample_minimum_variance_loss - result.minimum_variance_loss) /
        improvement_denominator;
  }
  result.distance_to_finite_sample_oracle = (estimate - oracle).norm();
  const DenseVector ones = DenseVector::Ones(estimate.rows());
  DenseVector weights = estimate.ldlt().solve(ones);
  const double weight_sum = weights.sum();
  if (!weights.allFinite() ||
      std::abs(weight_sum) <= std::numeric_limits<double>::epsilon()) {
    result.status = CovarianceStatus::NUMERICAL_FAILURE;
    return result;
  }
  weights /= weight_sum;
  result.predicted_portfolio_variance = weights.dot(estimate * weights);
  result.realized_population_variance = weights.dot(population * weights);
  if (!std::isfinite(result.minimum_variance_loss) ||
      !std::isfinite(result.prial_percent) ||
      !std::isfinite(result.distance_to_finite_sample_oracle) ||
      !std::isfinite(result.predicted_portfolio_variance) ||
      !std::isfinite(result.realized_population_variance)) {
    result.status = CovarianceStatus::NUMERICAL_FAILURE;
    return result;
  }
  result.status = CovarianceStatus::OK;
  return result;
}

} // namespace portfolio_math

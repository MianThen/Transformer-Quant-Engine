#pragma once

#include <cstdint>
#include <vector>

#include "portfolio_math/quest.h"
#include "quant_math/matrix.h"

namespace portfolio_math {

enum class CovarianceEstimator {
  SAMPLE,
  LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION,
  LEDOIT_WOLF_NONLINEAR_QUEST,
  RMT_CONSTANT_RESIDUAL,
  RMT_TARGETED_SHRINKAGE,
  FACTOR_MODEL_PIT_BASELINE,
  FACTOR_MODEL_DYNAMIC_CONDITIONAL,
};

enum class CovarianceLossProfile {
  NOT_APPLICABLE,
  FROBENIUS,
  MINIMUM_VARIANCE,
  STEIN,
  LOG_EUCLIDEAN,
};

enum class CovarianceStatus {
  OK,
  INVALID_INPUT,
  INSUFFICIENT_OBSERVATIONS,
  NUMERICAL_FAILURE,
};

enum class NonlinearDimensionalBranch : std::uint8_t {
  REGULAR_P_LT_N,
  SINGULAR_P_GT_N,
  GUARDED_P_EQ_N,
};

struct CovarianceResult {
  CovarianceStatus status{CovarianceStatus::INVALID_INPUT};
  CovarianceEstimator estimator{CovarianceEstimator::SAMPLE};
  CovarianceLossProfile loss_profile{CovarianceLossProfile::NOT_APPLICABLE};
  quant_math::DenseMatrix covariance;
  double shrinkage_intensity{0.0};
  double average_correlation{0.0};
  std::size_t effective_observations{0};
  double concentration_ratio{0.0};
};

struct NonlinearShrinkageDiagnostics {
  QuestDiagnostics quest;
  QuestStatus quest_status{QuestStatus::INVALID_INPUT};
  NonlinearDimensionalBranch dimensional_branch{
      NonlinearDimensionalBranch::GUARDED_P_EQ_N};
  std::vector<double> sample_eigenvalues;
  std::vector<double> population_eigenvalues;
  std::vector<double> shrunk_eigenvalues;
  std::size_t raw_observations{0};
  std::size_t quest_effective_observations{0};
  bool demeaned_returns{true};
  double maximum_angle_row_mass_error{0.0};
  double minimum_angle_weight{0.0};
  double null_space_shrinkage{0.0};
  std::size_t structural_zero_count{0};
  double null_equation_residual{0.0};
  double condition_number{0.0};
  double relative_trace_drift{0.0};
  double maximum_relative_diagonal_drift{0.0};
  double psd_repair_amount{0.0};
  double maximum_stieltjes_residual{0.0};
};

struct NonlinearCovarianceResult {
  CovarianceResult covariance;
  NonlinearShrinkageDiagnostics diagnostics;
};

struct RmtDenoisingDiagnostics {
  double aspect_ratio{0.0};
  double fitted_noise_boundary{0.0};
  double constant_residual_eigenvalue{0.0};
  double targeted_shrinkage_intensity{0.0};
  std::size_t retained_signal_rank{0};
  std::size_t noise_eigenvalue_count{0};
  std::vector<double> raw_eigenvalues;
  std::vector<double> cleaned_eigenvalues;
  double input_correlation_trace{0.0};
  double output_correlation_trace{0.0};
  double trace_drift{0.0};
  double maximum_relative_diagonal_drift{0.0};
  double condition_number{0.0};
  double psd_repair_amount{0.0};
  bool eligible_for_official_risk{false};
};

struct ResearchCovarianceSelection {
  CovarianceResult covariance;
  NonlinearShrinkageDiagnostics nonlinear_diagnostics;
  bool has_nonlinear_diagnostics{false};
  RmtDenoisingDiagnostics rmt_diagnostics;
  bool has_rmt_diagnostics{false};
};

struct FixedInputCovarianceComparison {
  std::uint64_t input_fingerprint{0};
  ResearchCovarianceSelection sample;
  ResearchCovarianceSelection linear_constant_correlation;
  ResearchCovarianceSelection nonlinear_minimum_variance_quest;
  double linear_distance_from_sample{0.0};
  double nonlinear_distance_from_sample{0.0};
  bool all_estimators_succeeded{false};
};

struct CovarianceOracleEvaluation {
  CovarianceStatus status{CovarianceStatus::INVALID_INPUT};
  double minimum_variance_loss{0.0};
  double sample_minimum_variance_loss{0.0};
  double finite_sample_oracle_loss{0.0};
  double prial_percent{0.0};
  double distance_to_finite_sample_oracle{0.0};
  double predicted_portfolio_variance{0.0};
  double realized_population_variance{0.0};
};

[[nodiscard]] CovarianceResult
sample_covariance(quant_math::MatrixView returns);
[[nodiscard]] CovarianceResult
ledoit_wolf_linear_constant_correlation(quant_math::MatrixView returns);
[[nodiscard]] NonlinearCovarianceResult
ledoit_wolf_nonlinear_minimum_variance_quest(quant_math::MatrixView returns,
                                             QuestSpec spec = {});
[[nodiscard]] ResearchCovarianceSelection
estimate_research_covariance(quant_math::MatrixView returns,
                             CovarianceEstimator frozen_estimator,
                             QuestSpec quest_spec = {});
[[nodiscard]] FixedInputCovarianceComparison
compare_covariance_estimators_fixed_input(quant_math::MatrixView returns,
                                          QuestSpec quest_spec = {});
[[nodiscard]] CovarianceOracleEvaluation
evaluate_covariance_oracle(quant_math::MatrixView estimated_covariance,
                           quant_math::MatrixView sample_covariance,
                           quant_math::MatrixView population_covariance);

} // namespace portfolio_math

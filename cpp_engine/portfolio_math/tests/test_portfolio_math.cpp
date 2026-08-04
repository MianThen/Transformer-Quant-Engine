#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "portfolio_math/covariance.h"
#include "portfolio_math/risk_budget.h"
#include "portfolio_math/risk_model.h"
#include "portfolio_math/tail_risk.h"

namespace {

bool check(bool condition, const char *message) {
  if (!condition)
    std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

bool near(double actual, double expected, double tolerance = 1e-9) {
  return std::abs(actual - expected) <= tolerance;
}

bool test_covariance() {
  bool ok = true;
  portfolio_math::RiskPreprocessorSpec linear_spec;
  linear_spec.official_estimator = portfolio_math::CovarianceEstimator::
      LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION;
  linear_spec.covariance_loss =
      portfolio_math::CovarianceLossProfile::FROBENIUS;
  linear_spec.lookback_observations = 252;
  linear_spec.balanced_panel_policy_hash = 11;
  linear_spec.config_hash = 12;
  ok &= check(portfolio_math::valid_risk_preprocessor_spec(linear_spec),
              "frozen LW-LIN-CC spec");
  auto rmt_spec = linear_spec;
  rmt_spec.official_estimator =
      portfolio_math::CovarianceEstimator::RMT_CONSTANT_RESIDUAL;
  rmt_spec.covariance_loss = portfolio_math::CovarianceLossProfile::NOT_APPLICABLE;
  rmt_spec.concentration_ratio_guard = 0.02;
  rmt_spec.eigenvalue_floor = 1e-12;
  rmt_spec.rmt_spec_hash = 14;
  ok &= check(portfolio_math::valid_risk_preprocessor_spec(rmt_spec),
              "RMT spec contract");
  rmt_spec.rmt_spec_hash = 0;
  ok &= check(!portfolio_math::valid_risk_preprocessor_spec(rmt_spec),
              "RMT requires frozen spec hash");
  rmt_spec.rmt_spec_hash = 14;
  rmt_spec.official_estimator =
      portfolio_math::CovarianceEstimator::RMT_TARGETED_SHRINKAGE;
  ok &= check(portfolio_math::valid_risk_preprocessor_spec(rmt_spec),
              "targeted RMT spec contract");
  auto quest_spec = linear_spec;
  quest_spec.official_estimator =
      portfolio_math::CovarianceEstimator::LEDOIT_WOLF_NONLINEAR_QUEST;
  quest_spec.covariance_loss =
      portfolio_math::CovarianceLossProfile::MINIMUM_VARIANCE;
  quest_spec.concentration_ratio_guard = 0.02;
  quest_spec.quest_solver_spec_hash = 13;
  ok &= check(portfolio_math::valid_risk_preprocessor_spec(quest_spec),
              "frozen LW-NLS-MV-QUEST spec contract");
  quest_spec.uniform_observation_weights = false;
  ok &= check(!portfolio_math::valid_risk_preprocessor_spec(quest_spec),
              "QuEST rejects non-uniform observation weights");

  quant_math::DenseMatrix returns(5, 3);
  returns << 0.01, 0.02, -0.01, 0.02, 0.01, 0.00, -0.01, 0.00, 0.02, 0.00, 0.01,
      0.01, 0.03, 0.02, -0.02;
  const auto sample =
      portfolio_math::sample_covariance(quant_math::view(returns));
  ok &= check(sample.status == portfolio_math::CovarianceStatus::OK,
              "sample covariance status");
  ok &= check(near(sample.covariance(0, 0), 0.0002, 1e-14),
              "sample covariance uses documented 1/T convention");

  const auto shrunk = portfolio_math::ledoit_wolf_linear_constant_correlation(
      quant_math::view(returns));
  ok &= check(shrunk.status == portfolio_math::CovarianceStatus::OK,
              "Ledoit-Wolf status");
  ok &= check(shrunk.estimator == portfolio_math::CovarianceEstimator::
                                      LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION &&
                  shrunk.loss_profile ==
                      portfolio_math::CovarianceLossProfile::FROBENIUS &&
                  near(shrunk.concentration_ratio, 3.0 / 5.0),
              "LW-LIN-CC identity and balanced-panel diagnostics");
  ok &= check(shrunk.shrinkage_intensity >= 0.0 &&
                  shrunk.shrinkage_intensity <= 1.0,
              "shrinkage clipping");
  ok &= check(near(shrunk.shrinkage_intensity, 0.28163416396134855, 1e-12),
              "Ledoit-Wolf independent oracle shrinkage");
  const double oracle_covariance[3][3] = {
      {0.0002, 4.665009828752064e-05, -0.000149752157270073},
      {4.665009828752064e-05, 0.000056, -8.265575219943662e-05},
      {-0.000149752157270073, -8.265575219943662e-05, 0.0002},
  };
  for (Eigen::Index row = 0; row < 3; ++row) {
    for (Eigen::Index col = 0; col < 3; ++col) {
      ok &= check(
          near(shrunk.covariance(row, col), oracle_covariance[row][col], 1e-14),
          "Ledoit-Wolf independent oracle covariance");
    }
  }
  ok &= check(
      quant_math::validate_symmetric(quant_math::view(shrunk.covariance), 1e-12)
          .ok,
      "Ledoit-Wolf symmetry");
  ok &= check(quant_math::is_positive_semidefinite(shrunk.covariance, 1e-12),
              "Ledoit-Wolf PSD");

  portfolio_math::QuestSpec quest_options;
  quest_options.max_inverse_iterations = 100;
  const auto nonlinear =
      portfolio_math::ledoit_wolf_nonlinear_minimum_variance_quest(
          quant_math::view(returns), quest_options);
  ok &= check(nonlinear.covariance.status ==
                      portfolio_math::CovarianceStatus::OK &&
                  nonlinear.covariance.estimator ==
                      portfolio_math::CovarianceEstimator::
                          LEDOIT_WOLF_NONLINEAR_QUEST &&
                  nonlinear.covariance.loss_profile ==
                      portfolio_math::CovarianceLossProfile::MINIMUM_VARIANCE,
              "LW-NLS-MV-QUEST regular branch status and identity");
  ok &= check(
      !nonlinear.diagnostics.population_eigenvalues.empty() &&
          std::is_sorted(nonlinear.diagnostics.population_eigenvalues.begin(),
                         nonlinear.diagnostics.population_eigenvalues.end()) &&
          nonlinear.diagnostics.minimum_angle_weight >= 0.0 &&
          nonlinear.diagnostics.maximum_angle_row_mass_error < 1e-6 &&
          nonlinear.diagnostics.maximum_stieltjes_residual < 1e-8,
      "LW-NLS-MV-QUEST population spectrum and angle diagnostics");
  ok &= check(nonlinear.diagnostics.raw_observations ==
                      static_cast<std::size_t>(returns.rows()) &&
                  nonlinear.diagnostics.quest_effective_observations + 1 ==
                      static_cast<std::size_t>(returns.rows()) &&
                  nonlinear.diagnostics.demeaned_returns,
              "LW-NLS-MV-QUEST centered-panel degrees of freedom");
  ok &= check(quant_math::is_positive_semidefinite(
                  nonlinear.covariance.covariance, 1e-10) &&
                  nonlinear.diagnostics.condition_number >= 1.0 &&
                  nonlinear.diagnostics.psd_repair_amount == 0.0,
              "LW-NLS-MV-QUEST PSD and condition diagnostics");

  quant_math::DenseMatrix singular_returns(4, 6);
  singular_returns << 1.0, 0.0, 2.0, -1.0, 0.5, 1.5, 0.0, 1.0, 1.0, 0.0, 1.5,
      0.5, -1.0, 0.5, 0.0, 1.0, 2.0, 1.0, 2.0, 1.5, 3.0, -2.0, 0.0, 2.5;
  const auto singular_nonlinear =
      portfolio_math::ledoit_wolf_nonlinear_minimum_variance_quest(
          quant_math::view(singular_returns), quest_options);
  ok &= check(
      singular_nonlinear.covariance.status ==
              portfolio_math::CovarianceStatus::OK &&
          singular_nonlinear.diagnostics.dimensional_branch ==
              portfolio_math::NonlinearDimensionalBranch::SINGULAR_P_GT_N &&
          singular_nonlinear.diagnostics.null_space_shrinkage > 0.0 &&
          singular_nonlinear.diagnostics.structural_zero_count == 3 &&
          singular_nonlinear.diagnostics.null_equation_residual < 1e-8 &&
          singular_nonlinear.diagnostics.maximum_angle_row_mass_error < 1e-6,
      "LW-NLS-MV-QUEST singular branch null-space shrinkage");
  ok &= check(quant_math::is_positive_semidefinite(
                  singular_nonlinear.covariance.covariance, 1e-10),
              "LW-NLS-MV-QUEST singular covariance PSD");

  quant_math::DenseMatrix transformed(returns.rows(), returns.cols());
  transformed.col(0) = -returns.col(2);
  transformed.col(1) = returns.col(0);
  transformed.col(2) = returns.col(1);
  const auto equivariant =
      portfolio_math::ledoit_wolf_nonlinear_minimum_variance_quest(
          quant_math::view(transformed), quest_options);
  const std::size_t source[3]{2, 0, 1};
  const double sign[3]{-1.0, 1.0, 1.0};
  bool covariance_equivariant =
      equivariant.covariance.status == portfolio_math::CovarianceStatus::OK;
  for (std::size_t row = 0; row < 3 && covariance_equivariant; ++row) {
    for (std::size_t col = 0; col < 3; ++col) {
      covariance_equivariant &= near(
          equivariant.covariance.covariance(static_cast<Eigen::Index>(row),
                                            static_cast<Eigen::Index>(col)),
          sign[row] * sign[col] *
              nonlinear.covariance.covariance(
                  static_cast<Eigen::Index>(source[row]),
                  static_cast<Eigen::Index>(source[col])),
          2e-8);
    }
  }
  ok &= check(covariance_equivariant,
              "LW-NLS-MV-QUEST sign and permutation equivariance");

  const auto comparison =
      portfolio_math::compare_covariance_estimators_fixed_input(
          quant_math::view(returns), quest_options);
  ok &= check(comparison.all_estimators_succeeded &&
                  comparison.input_fingerprint != 0 &&
                  comparison.linear_distance_from_sample > 0.0 &&
                  comparison.nonlinear_distance_from_sample > 0.0,
              "fixed-input sample/LW-LIN-CC/LW-NLS-MV paired report");

  portfolio_math::QuestSpec failing_options = quest_options;
  failing_options.max_inverse_iterations = 1;
  failing_options.objective_tolerance = 0.0;
  const auto failed_selection = portfolio_math::estimate_research_covariance(
      quant_math::view(returns),
      portfolio_math::CovarianceEstimator::LEDOIT_WOLF_NONLINEAR_QUEST,
      failing_options);
  ok &= check(failed_selection.covariance.status ==
                      portfolio_math::CovarianceStatus::NUMERICAL_FAILURE &&
                  failed_selection.covariance.estimator ==
                      portfolio_math::CovarianceEstimator::
                          LEDOIT_WOLF_NONLINEAR_QUEST &&
                  failed_selection.covariance.covariance.size() == 0 &&
                  failed_selection.has_nonlinear_diagnostics,
              "frozen QuEST selection fails closed without linear fallback");

  quant_math::DenseMatrix guarded_returns(4, 3);
  guarded_returns << 1.0, 0.0, 2.0, 0.0, 1.0, 1.0, -1.0, 0.5, 0.0, 2.0, 1.5,
      3.0;
  const auto guarded =
      portfolio_math::ledoit_wolf_nonlinear_minimum_variance_quest(
          quant_math::view(guarded_returns), quest_options);
  ok &= check(
      guarded.covariance.status ==
              portfolio_math::CovarianceStatus::NUMERICAL_FAILURE &&
          guarded.diagnostics.quest_status ==
              portfolio_math::QuestStatus::CONCENTRATION_RATIO_TOO_CLOSE_TO_ONE,
      "LW-NLS-MV-QUEST p/n guard fails closed");

  quant_math::DenseMatrix one_asset(3, 1);
  one_asset << 1.0, 2.0, 3.0;
  const auto one = portfolio_math::ledoit_wolf_linear_constant_correlation(
      quant_math::view(one_asset));
  ok &= check(one.status == portfolio_math::CovarianceStatus::OK &&
                  near(one.covariance(0, 0), 2.0 / 3.0),
              "single-asset covariance");
  const auto nonlinear_one =
      portfolio_math::ledoit_wolf_nonlinear_minimum_variance_quest(
          quant_math::view(one_asset), quest_options);
  ok &= check(nonlinear_one.covariance.status ==
                      portfolio_math::CovarianceStatus::OK &&
                  nonlinear_one.covariance.covariance(0, 0) > 0.0,
              "LW-NLS-MV-QUEST single-asset fixture");

  returns(0, 0) = std::numeric_limits<double>::quiet_NaN();
  ok &=
      check(portfolio_math::ledoit_wolf_linear_constant_correlation(
                quant_math::view(returns))
                    .status == portfolio_math::CovarianceStatus::INVALID_INPUT,
            "non-finite return rejection");
  return ok;
}

bool test_nonlinear_monte_carlo_oracle() {
  bool ok = true;
  constexpr std::size_t observations = 40;
  constexpr std::size_t dimension = 4;
  const double variances[dimension]{0.5, 1.0, 2.0, 4.0};
  quant_math::DenseMatrix population =
      quant_math::DenseMatrix::Zero(static_cast<Eigen::Index>(dimension),
                                    static_cast<Eigen::Index>(dimension));
  for (std::size_t index = 0; index < dimension; ++index) {
    population(static_cast<Eigen::Index>(index),
               static_cast<Eigen::Index>(index)) = variances[index];
  }
  std::mt19937_64 generator(20260730);
  std::normal_distribution<double> normal;
  quant_math::DenseMatrix returns(static_cast<Eigen::Index>(observations),
                                  static_cast<Eigen::Index>(dimension));
  for (std::size_t row = 0; row < observations; ++row) {
    for (std::size_t col = 0; col < dimension; ++col) {
      returns(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          std::sqrt(variances[col]) * normal(generator);
    }
  }

  portfolio_math::QuestSpec options;
  options.max_inverse_iterations = 100;
  const auto sample =
      portfolio_math::sample_covariance(quant_math::view(returns));
  const auto nonlinear =
      portfolio_math::ledoit_wolf_nonlinear_minimum_variance_quest(
          quant_math::view(returns), options);
  ok &= check(sample.status == portfolio_math::CovarianceStatus::OK &&
                  nonlinear.covariance.status ==
                      portfolio_math::CovarianceStatus::OK,
              "deterministic Monte Carlo covariance estimates");
  if (sample.status != portfolio_math::CovarianceStatus::OK ||
      nonlinear.covariance.status != portfolio_math::CovarianceStatus::OK) {
    return false;
  }
  const auto evaluation = portfolio_math::evaluate_covariance_oracle(
      quant_math::view(nonlinear.covariance.covariance),
      quant_math::view(sample.covariance), quant_math::view(population));
  ok &= check(
      evaluation.status == portfolio_math::CovarianceStatus::OK &&
          evaluation.minimum_variance_loss > 0.0 &&
          evaluation.sample_minimum_variance_loss > 0.0 &&
          evaluation.finite_sample_oracle_loss > 0.0 &&
          evaluation.finite_sample_oracle_loss <=
              evaluation.sample_minimum_variance_loss + 1e-12 &&
          std::isfinite(evaluation.prial_percent) &&
          evaluation.distance_to_finite_sample_oracle >= 0.0 &&
          evaluation.predicted_portfolio_variance > 0.0 &&
          evaluation.realized_population_variance > 0.0,
      "known-population MV loss, PRIAL, oracle distance and variance report");
  return ok;
}

bool test_risk_budget() {
  bool ok = true;
  quant_math::DenseMatrix covariance(2, 2);
  covariance << 1.0, 0.0, 0.0, 4.0;
  const std::vector<double> budgets{0.5, 0.5};
  const std::vector<double> current{0.5, 0.5};
  const auto solved = portfolio_math::solve_long_only_risk_budget(
      quant_math::view(covariance), budgets, current);
  ok &=
      check(solved.diagnostics.status == portfolio_math::OptimizationStatus::OK,
            "risk-budget status");
  ok &= check(solved.weights.size() == 2 &&
                  near(solved.weights[0], 2.0 / 3.0, 1e-8) &&
                  near(solved.weights[1], 1.0 / 3.0, 1e-8),
              "diagonal analytic risk-budget weights");
  ok &= check(near(solved.diagnostics.turnover, 1.0 / 6.0, 1e-8),
              "one-way turnover diagnostic");

  const auto contributions = portfolio_math::risk_contributions(
      quant_math::view(covariance), solved.weights);
  ok &= check(contributions.status == portfolio_math::OptimizationStatus::OK &&
                  near(contributions.contribution_shares[0], 0.5, 1e-8) &&
                  near(contributions.contribution_shares[1], 0.5, 1e-8),
              "risk contribution shares");

  covariance *= 7.0;
  const auto scaled = portfolio_math::solve_long_only_risk_budget(
      quant_math::view(covariance), budgets);
  ok &= check(scaled.diagnostics.status ==
                      portfolio_math::OptimizationStatus::OK &&
                  near(scaled.weights[0], solved.weights[0], 1e-8),
              "covariance scale invariance");

  covariance << 1.0, 0.0, 0.0, 4.0;
  const std::vector<double> lower_bounds{0.0, 0.0};
  const std::vector<double> upper_bounds{0.60, 0.80};
  const auto bounded = portfolio_math::solve_bounded_long_only_risk_budget(
      quant_math::view(covariance), budgets, lower_bounds, upper_bounds,
      current);
  ok &= check(
      bounded.diagnostics.status == portfolio_math::OptimizationStatus::OK &&
          bounded.weights.size() == 2 && near(bounded.weights[0], 0.60, 1e-6) &&
          near(bounded.weights[1], 0.40, 1e-6),
      "bounded risk-budget solution");
  ok &= check(bounded.diagnostics.active_bound_count == 1 &&
                  bounded.diagnostics.max_risk_budget_error > 0.0,
              "active bound and risk-budget distortion diagnostics");

  const std::vector<double> infeasible_lower{0.6, 0.6};
  const auto infeasible = portfolio_math::solve_bounded_long_only_risk_budget(
      quant_math::view(covariance), budgets, infeasible_lower, upper_bounds);
  ok &= check(infeasible.diagnostics.status ==
                      portfolio_math::OptimizationStatus::INFEASIBLE &&
                  infeasible.weights.empty(),
              "infeasible bounds fail closed");

  covariance << 1.0, 2.0, 2.0, 1.0;
  const auto non_psd = portfolio_math::solve_long_only_risk_budget(
      quant_math::view(covariance), budgets);
  ok &= check(non_psd.diagnostics.status ==
                      portfolio_math::OptimizationStatus::NON_PSD_RISK_MODEL &&
                  non_psd.weights.empty(),
              "non-PSD failure closes output");
  return ok;
}

bool test_cvar() {
  bool ok = true;
  const std::vector<double> returns{-3.0, -1.0, 1.0, 2.0};
  const std::vector<engine_common::SymbolId> symbols{1};
  const std::vector<double> fixed_weights{1.0};
  const std::vector<engine_common::TimestampNs> timestamps{1, 2, 3, 4};
  portfolio_math::TailRiskProblemView problem;
  problem.decision_at = 5;
  problem.symbols = symbols;
  problem.history_timestamps = timestamps;
  problem.fixed_portfolio_weights = fixed_weights;
  problem.portfolio_return_history = returns;
  problem.spec.confidence_level = 0.75;
  problem.spec.config_hash = 501;
  const auto result = portfolio_math::estimate_tail_risk(problem);
  ok &= check(result.status == portfolio_math::TailRiskStatus::OK &&
                  result.estimator == portfolio_math::TailRiskEstimatorKind::
                                          EMPIRICAL_ROCKAFELLAR_URYASEV,
              "TAIL-EMPIRICAL-ES status and identity");
  ok &= check(near(*result.value_at_risk_loss, 1.0) &&
                  near(*result.expected_shortfall_loss, 3.0) &&
                  near(*result.return_cvar, -3.0),
              "ES sign and discrete tail");
  ok &= check(result.effective_observations == returns.size() &&
                  result.input_hash != 0 && result.artifact_hash != 0,
              "TAIL-EMPIRICAL-ES provenance");
  portfolio_math::TailRiskArtifactSpec artifact_spec;
  artifact_spec.source_dataset_fingerprint = std::string(64, '1');
  artifact_spec.reference_price_quality = "PROXY";
  artifact_spec.promotion_eligible = true;
  artifact_spec.limitations.push_back("REFERENCE_PRICE_PROXY");
  const auto artifact = portfolio_math::serialize_tail_risk_artifact(
      result, problem.spec, artifact_spec);
  ok &= check(artifact.find("\"return_cvar\":-3") != std::string::npos &&
                  artifact.find("\"promotion_eligible\":false") !=
                      std::string::npos,
              "TAIL-EMPIRICAL-ES artifact closes proxy promotion");

  const std::vector<double> mass_returns{0.0, -10.0};
  const std::vector<double> probabilities{0.95, 0.05};
  const std::vector<engine_common::TimestampNs> mass_timestamps{1, 2};
  problem.history_timestamps = mass_timestamps;
  problem.portfolio_return_history = mass_returns;
  problem.scenario_probabilities = probabilities;
  problem.spec.confidence_level = 0.95;
  const auto mass = portfolio_math::estimate_tail_risk(problem);
  ok &= check(near(*mass.value_at_risk_loss, 0.0) &&
                  near(*mass.expected_shortfall_loss, 10.0),
              "probability mass at VaR is allocated exactly");
  const std::vector<double> unnormalized{0.5, 0.4};
  problem.scenario_probabilities = unnormalized;
  ok &= check(portfolio_math::estimate_tail_risk(problem).status ==
                  portfolio_math::TailRiskStatus::INVALID_INPUT,
              "unnormalized scenario probabilities are rejected");

  const std::vector<double> shifted{-2.0, 0.0, 2.0, 3.0};
  problem.history_timestamps = timestamps;
  problem.portfolio_return_history = shifted;
  problem.scenario_probabilities = {};
  problem.spec.confidence_level = 0.75;
  const auto translated = portfolio_math::estimate_tail_risk(problem);
  ok &= check(near(*translated.return_cvar, *result.return_cvar + 1.0),
              "return-CVaR translation");
  problem.portfolio_return_history = returns;
  problem.spec.confidence_level = 0.49;
  ok &= check(portfolio_math::estimate_tail_risk(problem).status ==
                  portfolio_math::TailRiskStatus::INVALID_INPUT,
              "tail confidence below the frozen domain is rejected");
  problem.spec.confidence_level = 0.75;
  problem.decision_at = 3;
  ok &= check(portfolio_math::estimate_tail_risk(problem).status ==
                  portfolio_math::TailRiskStatus::INVALID_INPUT,
              "future tail observation is rejected");
  return ok;
}

} // namespace

int main() {
  if (!(test_covariance() && test_nonlinear_monte_carlo_oracle() &&
        test_risk_budget() && test_cvar()))
    return 1;
  std::printf("test_portfolio_math: all checks passed\n");
  return 0;
}

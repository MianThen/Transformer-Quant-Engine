#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
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

bool test_tail_risk_backtest() {
  bool ok = true;
  const std::vector<engine_common::TimestampNs> timestamps{1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<double> returns{0.1, -0.2, 0.0, -0.6, 0.3, -0.8, 0.1, -0.1};
  const std::vector<double> var_loss(timestamps.size(), 0.4);
  const std::vector<double> es_loss(timestamps.size(), 0.7);
  portfolio_math::TailRiskBacktestProblemView problem{
      timestamps, returns, var_loss, es_loss, 0.75, 8, 503,
  };
  const auto result = portfolio_math::backtest_tail_risk(problem);
  ok &= check(result.status == portfolio_math::TailRiskBacktestStatus::OK &&
                  result.effective_observations == 8 &&
                  result.exception_count == 2 && result.es_violation_count == 1,
              "VaR/ES joint backtest counts exceptions");
  ok &= check(near(result.exception_rate, 0.25) &&
                  near(result.es_violation_rate, 0.125) &&
                  near(result.mean_exceedance_loss, 0.3) &&
                  near(result.mean_es_excess_loss, 0.1) &&
                  near(result.mean_fz0_score, std::log(0.7)),
              "VaR/ES joint backtest tail diagnostics");
  ok &= check(result.transition_00 == 3 && result.transition_01 == 2 &&
                  result.transition_10 == 2 && result.transition_11 == 0 &&
                  result.input_hash != 0 && result.artifact_hash != 0 &&
                  std::isfinite(result.kupiec_p_value) &&
                  std::isfinite(result.christoffersen_p_value),
              "VaR/ES joint backtest independence diagnostics");
  portfolio_math::TailRiskArtifactSpec artifact_spec;
  artifact_spec.reference_price_quality = "PROXY";
  artifact_spec.promotion_eligible = true;
  const auto serialized = portfolio_math::serialize_tail_risk_backtest_artifact(
      result, artifact_spec);
  ok &= check(serialized.find("\"role\":\"tail_risk_backtest\"") !=
                  std::string::npos &&
                  serialized.find("\"transition_01\":2") !=
                      std::string::npos &&
                  serialized.find("\"mean_fz0_score\"") !=
                      std::string::npos &&
                  serialized.find("\"promotion_eligible\":false") !=
                      std::string::npos,
              "VaR/ES joint backtest artifact closes proxy promotion");
  auto future_timestamps = timestamps;
  future_timestamps.back() = 9;
  problem.realization_timestamps = future_timestamps;
  ok &= check(portfolio_math::backtest_tail_risk(problem).status ==
                  portfolio_math::TailRiskBacktestStatus::INVALID_INPUT,
              "future realized return closes VaR/ES backtest");
  problem.realization_timestamps = timestamps;
  auto invalid_es = es_loss;
  invalid_es.front() = 0.3;
  problem.expected_shortfall_loss = invalid_es;
  ok &= check(portfolio_math::backtest_tail_risk(problem).status ==
                  portfolio_math::TailRiskBacktestStatus::INVALID_INPUT,
              "ES below VaR closes VaR/ES backtest");
  problem.expected_shortfall_loss = es_loss;
  const std::vector<double> negative_var_loss(timestamps.size(), -0.2);
  const std::vector<double> zero_es_loss(timestamps.size(), 0.0);
  problem.value_at_risk_loss = negative_var_loss;
  problem.expected_shortfall_loss = zero_es_loss;
  ok &= check(portfolio_math::backtest_tail_risk(problem).status ==
                  portfolio_math::TailRiskBacktestStatus::FZ0_DOMAIN_FAILURE,
              "nonpositive loss-ES closes FZ0 scoring domain");
  problem.value_at_risk_loss = var_loss;
  problem.expected_shortfall_loss = es_loss;
  problem.realized_returns = std::span<const double>(returns.data(), 2);
  problem.realization_timestamps = std::span<const engine_common::TimestampNs>(
      timestamps.data(), 2);
  problem.value_at_risk_loss = std::span<const double>(var_loss.data(), 2);
  problem.expected_shortfall_loss = std::span<const double>(es_loss.data(), 2);
  ok &= check(portfolio_math::backtest_tail_risk(problem).status ==
                  portfolio_math::TailRiskBacktestStatus::INSUFFICIENT_OBSERVATIONS,
              "short VaR/ES backtest closes with insufficient observations");
  return ok;
}

bool test_garch_fhs() {
  bool ok = true;
  constexpr std::size_t observation_count = 256;
  std::vector<double> returns;
  std::vector<engine_common::TimestampNs> timestamps;
  returns.reserve(observation_count);
  timestamps.reserve(observation_count);
  std::uint64_t state = 88172645463325252ULL;
  double variance = 0.0001;
  constexpr double mean = 0.0002;
  constexpr double omega = 0.000002;
  constexpr double alpha = 0.08;
  constexpr double beta = 0.88;
  for (std::size_t index = 0; index < observation_count; ++index) {
    double normal_approximation = 0.0;
    for (int draw = 0; draw < 3; ++draw) {
      state = state * 2862933555777941757ULL + 3037000493ULL;
      const double uniform = static_cast<double>(state >> 11) /
          static_cast<double>(1ULL << 53);
      normal_approximation += uniform;
    }
    const double standardized = 2.0 * normal_approximation - 3.0;
    const double value = mean + std::sqrt(variance) * standardized;
    returns.push_back(value);
    timestamps.push_back(static_cast<engine_common::TimestampNs>(index + 1));
    const double epsilon = value - mean;
    variance = omega + alpha * epsilon * epsilon + beta * variance;
  }
  const std::vector<engine_common::SymbolId> symbols{17};
  const std::vector<double> weights{1.0};
  portfolio_math::TailRiskProblemView problem;
  problem.decision_at = static_cast<engine_common::TimestampNs>(observation_count);
  problem.symbols = symbols;
  problem.history_timestamps = timestamps;
  problem.fixed_portfolio_weights = weights;
  problem.portfolio_return_history = returns;
  problem.spec.estimator = portfolio_math::TailRiskEstimatorKind::
      GARCH_FILTERED_HISTORICAL_SIMULATION;
  problem.spec.confidence_level = 0.95;
  problem.spec.mean_model_spec_hash = 701;
  problem.spec.volatility_model_spec_hash = 702;
  problem.spec.config_hash = 703;
  const auto result = portfolio_math::estimate_garch_fhs_tail_risk(problem);
  ok &= check(result.status == portfolio_math::TailRiskStatus::OK &&
                  result.estimator == portfolio_math::TailRiskEstimatorKind::
                      GARCH_FILTERED_HISTORICAL_SIMULATION &&
                  result.scenario_model == portfolio_math::TailScenarioModelKind::
                      PORTFOLIO_RETURN_SERIES,
              "GARCH-FHS status and identity");
  ok &= check(result.value_at_risk_loss && result.expected_shortfall_loss &&
                  result.return_cvar &&
                  result.garch_omega && result.garch_alpha && result.garch_beta &&
                  result.garch_stationarity_margin &&
                  *result.garch_omega > 0.0 && *result.garch_alpha >= 0.0 &&
                  *result.garch_beta >= 0.0 &&
                  *result.garch_stationarity_margin > 0.0 &&
                  *result.expected_shortfall_loss >= *result.value_at_risk_loss,
              "GARCH-FHS positivity and ES ordering");
  ok &= check(result.standardized_residual_mean &&
                  result.standardized_residual_variance &&
                  result.residual_ljung_box && result.squared_residual_ljung_box &&
                  result.arch_lm_statistic && result.maximum_standardized_residual &&
                  std::abs(*result.standardized_residual_mean) < 0.25 &&
                  *result.standardized_residual_variance > 0.25 &&
                  *result.standardized_residual_variance < 2.5 &&
                  std::isfinite(*result.residual_ljung_box) &&
                  std::isfinite(*result.squared_residual_ljung_box),
              "GARCH-FHS standardized residual diagnostics");
  const auto replay = portfolio_math::estimate_tail_risk(problem);
  ok &= check(replay.status == portfolio_math::TailRiskStatus::OK &&
                  replay.artifact_hash == result.artifact_hash &&
                  near(*replay.return_cvar, *result.return_cvar),
              "GARCH-FHS deterministic replay");
  portfolio_math::TailRiskArtifactSpec artifact_spec;
  artifact_spec.reference_price_quality = "PROXY";
  artifact_spec.promotion_eligible = true;
  const auto artifact = portfolio_math::serialize_tail_risk_artifact(
      result, problem.spec, artifact_spec);
  ok &= check(artifact.find("\"garch\"") != std::string::npos &&
                  artifact.find("\"stationarity_margin\"") != std::string::npos &&
                  artifact.find("\"promotion_eligible\":false") !=
                      std::string::npos,
              "GARCH-FHS diagnostics artifact and proxy gate");
  auto future_problem = problem;
  future_problem.decision_at = static_cast<engine_common::TimestampNs>(
      observation_count - 1);
  ok &= check(portfolio_math::estimate_garch_fhs_tail_risk(future_problem).status ==
                  portfolio_math::TailRiskStatus::INVALID_INPUT,
              "GARCH-FHS future observation closes replay");
  auto invalid_spec = problem;
  invalid_spec.spec.mean_model_spec_hash = 0;
  ok &= check(portfolio_math::estimate_garch_fhs_tail_risk(invalid_spec).status ==
                  portfolio_math::TailRiskStatus::INVALID_INPUT,
              "GARCH-FHS unfrozen mean model closes replay");
  quant_math::DenseMatrix synchronized_returns(observation_count, 2);
  for (std::size_t row = 0; row < observation_count; ++row) {
    synchronized_returns(static_cast<Eigen::Index>(row), 0) = returns[row];
    synchronized_returns(static_cast<Eigen::Index>(row), 1) = returns[row];
  }
  const std::vector<engine_common::SymbolId> vector_symbols{17, 18};
  const std::vector<double> vector_weights{0.6, 0.4};
  auto synchronized_vector = problem;
  synchronized_vector.symbols = vector_symbols;
  synchronized_vector.fixed_portfolio_weights = vector_weights;
  synchronized_vector.portfolio_return_history = {};
  synchronized_vector.asset_return_history =
      quant_math::view(synchronized_returns);
  synchronized_vector.spec.scenario_model =
      portfolio_math::TailScenarioModelKind::ASSET_VECTOR_SYNCHRONIZED;
  synchronized_vector.spec.synchronized_residual_rows = true;
  synchronized_vector.spec.scenario_seed = 704;
  synchronized_vector.spec.config_hash = 705;
  const auto vector_result =
      portfolio_math::estimate_garch_fhs_tail_risk(synchronized_vector);
  ok &= check(vector_result.status == portfolio_math::TailRiskStatus::OK &&
                  vector_result.scenario_model ==
                      portfolio_math::TailScenarioModelKind::
                          ASSET_VECTOR_SYNCHRONIZED &&
                  vector_result.asset_garch_diagnostics.size() == 2 &&
                  vector_result.value_at_risk_loss &&
                  vector_result.expected_shortfall_loss &&
                  near(*vector_result.value_at_risk_loss,
                       *result.value_at_risk_loss) &&
                  near(*vector_result.expected_shortfall_loss,
                       *result.expected_shortfall_loss),
              "asset-vector FHS preserves synchronized residual-row parity");
  const auto vector_replay =
      portfolio_math::estimate_tail_risk(synchronized_vector);
  ok &= check(vector_replay.status == portfolio_math::TailRiskStatus::OK &&
                  vector_replay.artifact_hash == vector_result.artifact_hash,
              "asset-vector FHS deterministic replay");
  const auto vector_artifact = portfolio_math::serialize_tail_risk_artifact(
      vector_result, synchronized_vector.spec, artifact_spec);
  ok &= check(vector_artifact.find("\"asset_garch\":[{") !=
                      std::string::npos &&
                  vector_artifact.find("\"symbol_id\":18") !=
                      std::string::npos,
              "asset-vector FHS records per-asset diagnostics");
  auto unsynchronized_vector = synchronized_vector;
  unsynchronized_vector.spec.synchronized_residual_rows = false;
  ok &= check(portfolio_math::estimate_garch_fhs_tail_risk(
                  unsynchronized_vector).status ==
                  portfolio_math::TailRiskStatus::INVALID_INPUT,
              "independent per-asset residual sampling is rejected");
  auto future_vector = synchronized_vector;
  future_vector.decision_at = static_cast<engine_common::TimestampNs>(
      observation_count - 1);
  ok &= check(portfolio_math::estimate_garch_fhs_tail_risk(future_vector).status ==
                  portfolio_math::TailRiskStatus::INVALID_INPUT,
              "future asset-vector observation closes replay");
  auto missing_problem = problem;
  std::vector<double> missing_returns = returns;
  missing_returns[observation_count / 2] = std::numeric_limits<double>::quiet_NaN();
  missing_problem.portfolio_return_history = missing_returns;
  ok &= check(portfolio_math::estimate_garch_fhs_tail_risk(missing_problem).status ==
                  portfolio_math::TailRiskStatus::INVALID_INPUT,
              "GARCH-FHS missing return closes replay");
  auto evt_problem = problem;
  evt_problem.spec.estimator = portfolio_math::TailRiskEstimatorKind::
      GARCH_FHS_POT_GPD;
  evt_problem.spec.evt_minimum_exceedances = 12;
  evt_problem.spec.evt_threshold_grid_points = 4;
  evt_problem.spec.evt_threshold_quantile_min = 0.75;
  evt_problem.spec.evt_threshold_quantile_max = 0.90;
  evt_problem.spec.evt_shape_upper_guard = 0.99;
  evt_problem.spec.evt_max_shape_spread = 0.35;
  evt_problem.spec.evt_max_relative_es_spread = 0.25;
  evt_problem.spec.evt_threshold_spec_hash = 704;
  evt_problem.spec.config_hash = 705;
  const auto evt = portfolio_math::estimate_garch_fhs_evt_tail_risk(evt_problem);
  if (evt.status != portfolio_math::TailRiskStatus::OK) {
    std::fprintf(stderr, "EVT status=%d diagnostics=%zu spread=%g/%g\n",
                 static_cast<int>(evt.status),
                 evt.evt_threshold_diagnostics.size(),
                 evt.evt_shape_spread.value_or(-1.0),
                 evt.evt_relative_es_spread.value_or(-1.0));
  }
  ok &= check(evt.status == portfolio_math::TailRiskStatus::OK &&
                  evt.estimator == portfolio_math::TailRiskEstimatorKind::
                      GARCH_FHS_POT_GPD && evt.evt_threshold && evt.gpd_shape &&
                  evt.gpd_scale && evt.evt_exceedance_count >= 12 &&
                  evt.evt_threshold_diagnostics.size() == 4 &&
                  evt.evt_selected_threshold_quantile &&
                  evt.value_at_risk_loss && evt.expected_shortfall_loss &&
                  *evt.expected_shortfall_loss >= *evt.value_at_risk_loss,
              "GARCH-FHS POT-GPD finite tail output");
  const auto evt_replay = portfolio_math::estimate_tail_risk(evt_problem);
  ok &= check(evt_replay.status == portfolio_math::TailRiskStatus::OK &&
                  evt_replay.artifact_hash == evt.artifact_hash,
              "GARCH-FHS POT-GPD deterministic replay");
  const auto evt_artifact = portfolio_math::serialize_tail_risk_artifact(
      evt, evt_problem.spec, artifact_spec);
  ok &= check(evt_artifact.find("\"evt\"") != std::string::npos &&
                  evt_artifact.find("WEIGHTED_MOMENTS_FOUR_POINT_GRID_V1") !=
                      std::string::npos &&
                  evt_artifact.find("\"threshold_diagnostics\":[{") !=
                      std::string::npos,
              "GARCH-FHS POT-GPD artifact diagnostics");
  auto synchronized_evt = synchronized_vector;
  synchronized_evt.spec.estimator = portfolio_math::TailRiskEstimatorKind::
      GARCH_FHS_POT_GPD;
  synchronized_evt.spec.evt_minimum_exceedances = 12;
  synchronized_evt.spec.evt_threshold_grid_points = 4;
  synchronized_evt.spec.evt_threshold_quantile_min = 0.75;
  synchronized_evt.spec.evt_threshold_quantile_max = 0.90;
  synchronized_evt.spec.evt_shape_upper_guard = 0.99;
  synchronized_evt.spec.evt_max_shape_spread = 0.35;
  synchronized_evt.spec.evt_max_relative_es_spread = 0.25;
  synchronized_evt.spec.evt_threshold_spec_hash = 706;
  synchronized_evt.spec.config_hash = 707;
  const auto synchronized_evt_result =
      portfolio_math::estimate_garch_fhs_evt_tail_risk(synchronized_evt);
  if (synchronized_evt_result.status != portfolio_math::TailRiskStatus::OK) {
    std::fprintf(stderr,
                 "vector EVT status=%d diagnostics=%zu spread=%g/%g\n",
                 static_cast<int>(synchronized_evt_result.status),
                 synchronized_evt_result.evt_threshold_diagnostics.size(),
                 synchronized_evt_result.evt_shape_spread.value_or(-1.0),
                 synchronized_evt_result.evt_relative_es_spread.value_or(-1.0));
  }
  ok &= check(
      synchronized_evt_result.status == portfolio_math::TailRiskStatus::OK &&
          synchronized_evt_result.estimator ==
              portfolio_math::TailRiskEstimatorKind::GARCH_FHS_POT_GPD &&
          synchronized_evt_result.scenario_model ==
              portfolio_math::TailScenarioModelKind::
                  ASSET_VECTOR_SYNCHRONIZED &&
          synchronized_evt_result.asset_garch_diagnostics.size() == 2 &&
          synchronized_evt_result.evt_threshold &&
          synchronized_evt_result.gpd_shape &&
          synchronized_evt_result.gpd_scale &&
          synchronized_evt_result.evt_threshold_diagnostics.size() == 4 &&
          synchronized_evt_result.value_at_risk_loss &&
          synchronized_evt_result.expected_shortfall_loss &&
          synchronized_evt_result.evt_exceedance_count >= 12 &&
          *synchronized_evt_result.gpd_scale > 0.0 &&
          *synchronized_evt_result.gpd_shape <
              synchronized_evt.spec.evt_shape_upper_guard &&
          *synchronized_evt_result.expected_shortfall_loss >=
              *synchronized_evt_result.value_at_risk_loss &&
          synchronized_evt_result.input_hash != 0 &&
          synchronized_evt_result.artifact_hash != 0,
      "asset-vector synchronized FHS POT-GPD finite tail output");
  if (synchronized_evt_result.status == portfolio_math::TailRiskStatus::OK) {
    const double support = 1.0 + *synchronized_evt_result.gpd_shape *
        (*synchronized_evt_result.value_at_risk_loss -
         *synchronized_evt_result.evt_threshold) /
        *synchronized_evt_result.gpd_scale;
    ok &= check(support > 0.0 && std::isfinite(support),
                "asset-vector POT-GPD VaR remains inside support");
    ok &= check(
        synchronized_evt_result.evt_exceedance_count ==
                evt.evt_exceedance_count &&
            near(*synchronized_evt_result.evt_threshold,
                 *evt.evt_threshold, 1e-12) &&
            near(*synchronized_evt_result.gpd_shape,
                 *evt.gpd_shape, 1e-12) &&
            near(*synchronized_evt_result.gpd_scale,
                 *evt.gpd_scale, 1e-12) &&
            near(*synchronized_evt_result.value_at_risk_loss,
                 *evt.value_at_risk_loss, 1e-12) &&
            near(*synchronized_evt_result.expected_shortfall_loss,
                 *evt.expected_shortfall_loss, 1e-12),
        "identical assets preserve synchronized portfolio-loss EVT parity");
  }
  const auto synchronized_evt_replay =
      portfolio_math::estimate_tail_risk(synchronized_evt);
  ok &= check(
      synchronized_evt_replay.status == portfolio_math::TailRiskStatus::OK &&
          synchronized_evt_replay.artifact_hash ==
              synchronized_evt_result.artifact_hash,
      "asset-vector synchronized POT-GPD deterministic replay");
  quant_math::DenseMatrix heterogeneous_returns(observation_count, 2);
  quant_math::DenseMatrix permuted_heterogeneous_returns(observation_count, 2);
  std::uint64_t second_state = 1099511628211ULL;
  double first_variance = 0.0001;
  double second_variance = 0.00016;
  constexpr double second_mean = -0.0001;
  constexpr double second_omega = 0.000003;
  constexpr double second_alpha = 0.06;
  constexpr double second_beta = 0.90;
  for (std::size_t row = 0; row < observation_count; ++row) {
    double independent_normal = 0.0;
    for (int draw = 0; draw < 3; ++draw) {
      second_state = second_state * 2862933555777941757ULL + 3037000493ULL;
      const double uniform = static_cast<double>(second_state >> 11) /
          static_cast<double>(1ULL << 53);
      independent_normal += uniform;
    }
    independent_normal = 2.0 * independent_normal - 3.0;
    const double first_epsilon = returns[row] - mean;
    const double first_standardized =
        first_epsilon / std::sqrt(first_variance);
    const double second_standardized =
        0.6 * first_standardized + 0.8 * independent_normal;
    const double second_asset =
        second_mean + std::sqrt(second_variance) * second_standardized;
    heterogeneous_returns(static_cast<Eigen::Index>(row), 0) = returns[row];
    heterogeneous_returns(static_cast<Eigen::Index>(row), 1) = second_asset;
    permuted_heterogeneous_returns(static_cast<Eigen::Index>(row), 0) =
        second_asset;
    permuted_heterogeneous_returns(static_cast<Eigen::Index>(row), 1) =
        returns[row];
    first_variance = omega + alpha * first_epsilon * first_epsilon +
        beta * first_variance;
    const double second_epsilon = second_asset - second_mean;
    second_variance = second_omega +
        second_alpha * second_epsilon * second_epsilon +
        second_beta * second_variance;
  }
  const std::vector<double> heterogeneous_weights{0.35, 0.65};
  const std::vector<double> permuted_heterogeneous_weights{0.65, 0.35};
  auto heterogeneous_evt = synchronized_evt;
  heterogeneous_evt.fixed_portfolio_weights = heterogeneous_weights;
  heterogeneous_evt.asset_return_history =
      quant_math::view(heterogeneous_returns);
  heterogeneous_evt.spec.config_hash = 1707;
  const auto heterogeneous_result =
      portfolio_math::estimate_garch_fhs_evt_tail_risk(heterogeneous_evt);
  auto permuted_heterogeneous_evt = heterogeneous_evt;
  permuted_heterogeneous_evt.fixed_portfolio_weights =
      permuted_heterogeneous_weights;
  permuted_heterogeneous_evt.asset_return_history =
      quant_math::view(permuted_heterogeneous_returns);
  permuted_heterogeneous_evt.spec.config_hash = 1707;
  const auto permuted_heterogeneous_result =
      portfolio_math::estimate_garch_fhs_evt_tail_risk(
          permuted_heterogeneous_evt);
  if (heterogeneous_result.status != portfolio_math::TailRiskStatus::OK ||
      permuted_heterogeneous_result.status !=
          portfolio_math::TailRiskStatus::OK) {
    std::fprintf(stderr,
                 "heterogeneous EVT status=%d/%d diagnostics=%zu/%zu\n",
                 static_cast<int>(heterogeneous_result.status),
                 static_cast<int>(permuted_heterogeneous_result.status),
                 heterogeneous_result.evt_threshold_diagnostics.size(),
                 permuted_heterogeneous_result.evt_threshold_diagnostics.size());
  }
  ok &= check(
      heterogeneous_result.status == portfolio_math::TailRiskStatus::OK &&
          permuted_heterogeneous_result.status ==
              portfolio_math::TailRiskStatus::OK &&
          heterogeneous_result.value_at_risk_loss &&
          permuted_heterogeneous_result.value_at_risk_loss &&
          heterogeneous_result.expected_shortfall_loss &&
          permuted_heterogeneous_result.expected_shortfall_loss &&
          near(*heterogeneous_result.value_at_risk_loss,
               *permuted_heterogeneous_result.value_at_risk_loss, 1e-12) &&
          near(*heterogeneous_result.expected_shortfall_loss,
               *permuted_heterogeneous_result.expected_shortfall_loss,
               1e-12),
      "non-identical correlated assets preserve column-order EVT distribution parity");
  auto changed_evt_hash = synchronized_evt;
  changed_evt_hash.spec.evt_threshold_spec_hash = 708;
  const auto changed_evt_hash_result =
      portfolio_math::estimate_garch_fhs_evt_tail_risk(changed_evt_hash);
  ok &= check(
      changed_evt_hash_result.status == portfolio_math::TailRiskStatus::OK &&
          changed_evt_hash_result.artifact_hash !=
              synchronized_evt_result.artifact_hash,
      "asset-vector POT-GPD threshold spec hash binds artifact");
  auto missing_evt_hash = synchronized_evt;
  missing_evt_hash.spec.evt_threshold_spec_hash = 0;
  ok &= check(
      portfolio_math::estimate_garch_fhs_evt_tail_risk(missing_evt_hash).status ==
          portfolio_math::TailRiskStatus::INVALID_INPUT,
      "asset-vector POT-GPD missing threshold hash closes replay");
  auto non_training_evt = synchronized_evt;
  non_training_evt.spec.training_only_tail_calibration = false;
  ok &= check(
      portfolio_math::estimate_garch_fhs_evt_tail_risk(non_training_evt).status ==
          portfolio_math::TailRiskStatus::INVALID_INPUT,
      "asset-vector POT-GPD rejects non-training tail calibration");
  auto unsynchronized_evt = synchronized_evt;
  unsynchronized_evt.spec.synchronized_residual_rows = false;
  ok &= check(
      portfolio_math::estimate_garch_fhs_evt_tail_risk(unsynchronized_evt).status ==
          portfolio_math::TailRiskStatus::INVALID_INPUT,
      "asset-vector POT-GPD rejects independent residual sampling");
  auto future_evt = synchronized_evt;
  future_evt.decision_at = static_cast<engine_common::TimestampNs>(
      observation_count - 1);
  ok &= check(
      portfolio_math::estimate_garch_fhs_evt_tail_risk(future_evt).status ==
          portfolio_math::TailRiskStatus::INVALID_INPUT,
      "future asset-vector POT-GPD observation closes replay");
  auto insufficient_vector_evt = synchronized_evt;
  insufficient_vector_evt.spec.evt_minimum_exceedances = 1000;
  ok &= check(
      portfolio_math::estimate_garch_fhs_evt_tail_risk(
          insufficient_vector_evt).status ==
          portfolio_math::TailRiskStatus::INSUFFICIENT_TAIL,
      "asset-vector POT-GPD insufficient tail closes replay");
  quant_math::DenseMatrix oversized_returns(observation_count, 201);
  std::vector<engine_common::SymbolId> oversized_symbols(201);
  std::vector<double> oversized_weights(201, 1.0 / 201.0);
  for (std::size_t col = 0; col < oversized_symbols.size(); ++col) {
    oversized_symbols[col] = static_cast<engine_common::SymbolId>(1000 + col);
    for (std::size_t row = 0; row < observation_count; ++row) {
      oversized_returns(static_cast<Eigen::Index>(row),
                        static_cast<Eigen::Index>(col)) = returns[row];
    }
  }
  auto oversized_evt = synchronized_evt;
  oversized_evt.symbols = oversized_symbols;
  oversized_evt.fixed_portfolio_weights = oversized_weights;
  oversized_evt.asset_return_history = quant_math::view(oversized_returns);
  ok &= check(
      portfolio_math::estimate_garch_fhs_evt_tail_risk(oversized_evt).status ==
          portfolio_math::TailRiskStatus::INVALID_INPUT,
      "asset-vector POT-GPD enforces N at most 200");
  auto insufficient_evt = evt_problem;
  insufficient_evt.spec.evt_minimum_exceedances = 1000;
  ok &= check(portfolio_math::estimate_garch_fhs_evt_tail_risk(insufficient_evt).status ==
                  portfolio_math::TailRiskStatus::INSUFFICIENT_TAIL,
              "GARCH-FHS POT-GPD insufficient tail closes replay");
  auto expectile_problem = problem;
  expectile_problem.spec.estimator = portfolio_math::TailRiskEstimatorKind::
      EXPECTILE_DIRECT;
  expectile_problem.spec.expectile_level = 0.95;
  expectile_problem.spec.expectile_feature_spec_hash = 706;
  expectile_problem.spec.training_only_tail_calibration = true;
  expectile_problem.spec.mean_model_spec_hash = 0;
  expectile_problem.spec.volatility_model_spec_hash = 0;
  expectile_problem.spec.evt_minimum_exceedances = 0;
  expectile_problem.spec.evt_threshold_quantile_min = 0.0;
  expectile_problem.spec.evt_threshold_quantile_max = 0.0;
  expectile_problem.spec.evt_shape_upper_guard = 0.0;
  expectile_problem.spec.evt_threshold_spec_hash = 0;
  expectile_problem.spec.config_hash = 707;
  const auto expectile = portfolio_math::estimate_expectile_tail_risk(
      expectile_problem);
  ok &= check(expectile.status == portfolio_math::TailRiskStatus::OK &&
                  expectile.expectile_loss && expectile.calibrated_expectile_level &&
                  std::isfinite(*expectile.expectile_loss) &&
                  near(*expectile.calibrated_expectile_level, 0.95),
              "direct expectile training calibration");
  const auto expectile_replay = portfolio_math::estimate_tail_risk(expectile_problem);
  ok &= check(expectile_replay.status == portfolio_math::TailRiskStatus::OK &&
                  expectile_replay.artifact_hash == expectile.artifact_hash,
              "direct expectile deterministic replay");
  const auto expectile_artifact = portfolio_math::serialize_tail_risk_artifact(
      expectile, expectile_problem.spec, artifact_spec);
  ok &= check(expectile_artifact.find("\"expectile_loss\"") != std::string::npos,
              "direct expectile artifact diagnostics");
  auto mapped_expectile = expectile_problem;
  mapped_expectile.spec.estimator = portfolio_math::TailRiskEstimatorKind::
      EXPECTILE_TAYLOR_MAPPED_ES;
  ok &= check(portfolio_math::estimate_tail_risk(mapped_expectile).status ==
                  portfolio_math::TailRiskStatus::INVALID_INPUT,
              "unfrozen Taylor expectile mapping closes replay");
  return ok;
}

} // namespace

int main() {
  if (!(test_covariance() && test_nonlinear_monte_carlo_oracle() &&
        test_risk_budget() && test_cvar() && test_tail_risk_backtest() &&
        test_garch_fhs()))
    return 1;
  std::printf("test_portfolio_math: all checks passed\n");
  return 0;
}

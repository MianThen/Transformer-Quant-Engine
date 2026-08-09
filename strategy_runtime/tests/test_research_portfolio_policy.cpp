#include <array>
#include <cmath>
#include <cstdio>

#include "strategy_runtime/research_portfolio_policy.h"

namespace {

bool check(bool condition, const char *message) {
  if (!condition)
    std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

qbt::strategy::ResearchPortfolioPolicyConfig
config(qbt::strategy::ResearchPortfolioPolicyKind kind) {
  qbt::strategy::ResearchPortfolioPolicyConfig value;
  value.kind = kind;
  value.selection.max_positions = 2;
  value.selection.max_position_weight = 0.70F;
  value.selection.minimum_expected_return = 0.0F;
  value.selection.minimum_confidence = 0.0F;
  value.target_investment = 1.0;
  value.policy_config_hash = 101;
  return value;
}

} // namespace

int main() {
  bool ok = true;
  std::array<engine_common::ModelPrediction, 2> predictions{};
  predictions[0] = {1,    100,  0.02F,
                    0.1F, 0.6F, -0.1F,
                    0.1F, 0.8F, engine_common::PREDICTION_VALID};
  predictions[1] = {2,    100,  0.01F,
                    0.2F, 0.6F, -0.1F,
                    0.1F, 0.8F, engine_common::PREDICTION_VALID};
  engine_common::PredictionBatch prediction_batch{100, 7, predictions,
                                                  predictions.size()};

  std::array<engine_common::MarketBar, 2> bars{};
  for (std::size_t index = 0; index < bars.size(); ++index) {
    bars[index].symbol_id = static_cast<engine_common::SymbolId>(index + 1);
    bars[index].timestamp = 100;
    bars[index].close = 10.0;
    bars[index].lot_size = 1;
    bars[index].flags =
        engine_common::MARKET_LISTED | engine_common::MARKET_DATA_TRUSTED;
  }
  const engine_common::MarketFrameBatchView market{100, bars, true};
  const engine_common::PortfolioView portfolio{
      {}, 12'000.0, 12'000.0, 0.0, 0.0};

  std::array<engine_common::SymbolId, 2> risk_symbols{1, 2};
  quant_math::DenseMatrix covariance(2, 2);
  covariance << 1.0, 0.0, 0.0, 4.0;
  portfolio_math::DenseRiskModelView risk_model;
  risk_model.estimator = portfolio_math::CovarianceEstimator::
      LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION;
  risk_model.loss_profile = portfolio_math::CovarianceLossProfile::FROBENIUS;
  risk_model.fit_start = 1;
  risk_model.fit_end = 98;
  risk_model.available_at = 99;
  risk_model.symbols = risk_symbols;
  risk_model.covariance = quant_math::view(covariance);
  risk_model.symbol_mapping_hash = 200;
  risk_model.balanced_panel_policy_hash = 201;
  risk_model.effective_observations = 20;
  risk_model.concentration_ratio = 0.1;
  risk_model.dimensional_branch =
      portfolio_math::DimensionalBranch::NOT_APPLICABLE;
  risk_model.artifact_hash = 202;

  qbt::strategy::ResearchPortfolioPolicy risk_budget(
      config(qbt::strategy::ResearchPortfolioPolicyKind::RISK_BUDGET));
  const auto targets =
      risk_budget.build(prediction_batch, market, portfolio, &risk_model);
  ok &= check(targets.size() == 2 && targets[0].target_quantity == 800 &&
                  targets[1].target_quantity == 400,
              "risk-budget target quantities");
  ok &= check(std::abs(targets[0].target_weight - 2.0F / 3.0F) < 1e-6F &&
                  risk_budget.diagnostics().status ==
                      portfolio_math::OptimizationStatus::OK &&
                  !risk_budget.diagnostics().hold_current_weights &&
                  risk_budget.diagnostics().risk_model_hash == 202,
              "risk-budget diagnostics and provenance");

  risk_model.estimator =
      portfolio_math::CovarianceEstimator::LEDOIT_WOLF_NONLINEAR_QUEST;
  risk_model.loss_profile =
      portfolio_math::CovarianceLossProfile::MINIMUM_VARIANCE;
  risk_model.dimensional_branch =
      portfolio_math::DimensionalBranch::REGULAR_P_LT_N;
  risk_model.artifact_hash = 203;
  ok &=
      check(risk_budget.build(prediction_batch, market, portfolio, &risk_model)
                        .size() == 2 &&
                risk_budget.diagnostics().risk_model_hash == 203 &&
                !risk_budget.diagnostics().hold_current_weights,
            "frozen Replay consumes LW-NLS-MV-QUEST risk artifact");

  risk_model.estimator = portfolio_math::CovarianceEstimator::SAMPLE;
  risk_model.loss_profile =
      portfolio_math::CovarianceLossProfile::NOT_APPLICABLE;
  risk_model.dimensional_branch =
      portfolio_math::DimensionalBranch::NOT_APPLICABLE;
  risk_model.artifact_hash = 204;
  ok &=
      check(risk_budget.build(prediction_batch, market, portfolio, &risk_model)
                        .size() == 2 &&
                risk_budget.diagnostics().risk_model_hash == 204,
            "frozen Replay consumes sample risk artifact");

  risk_model.estimator = portfolio_math::CovarianceEstimator::
      LEDOIT_WOLF_LINEAR_CONSTANT_CORRELATION;
  risk_model.loss_profile = portfolio_math::CovarianceLossProfile::FROBENIUS;
  risk_model.artifact_hash = 202;

  risk_model.available_at = 101;
  ok &=
      check(risk_budget.build(prediction_batch, market, portfolio, &risk_model)
                    .empty() &&
                risk_budget.diagnostics().hold_current_weights,
            "future risk artifact fails closed to HOLD");
  risk_model.available_at = 99;
  covariance << 1.0, 2.0, 2.0, 1.0;
  ok &=
      check(risk_budget.build(prediction_batch, market, portfolio, &risk_model)
                    .empty() &&
                risk_budget.diagnostics().status ==
                    portfolio_math::OptimizationStatus::NON_PSD_RISK_MODEL,
            "non-PSD risk artifact fails closed");
  ok &= check(risk_budget.diagnostics().policy ==
                      qbt::strategy::ResearchPortfolioPolicyKind::RISK_BUDGET &&
                  risk_budget.diagnostics().hold_current_weights,
              "risk-budget failure does not fall back to another policy");

  covariance << 1.0, 0.0, 0.0, 4.0;
  auto infeasible_config =
      config(qbt::strategy::ResearchPortfolioPolicyKind::RISK_BUDGET);
  infeasible_config.selection.max_position_weight = 0.40F;
  infeasible_config.policy_config_hash = 102;
  qbt::strategy::ResearchPortfolioPolicy infeasible_policy(infeasible_config);
  ok &= check(
      infeasible_policy.build(prediction_batch, market, portfolio, &risk_model)
              .empty() &&
          infeasible_policy.diagnostics().status ==
              portfolio_math::OptimizationStatus::INFEASIBLE,
      "infeasible exposure cap fails closed");

  std::array<engine_common::PortfolioItem, 1> uncovered_holding{};
  uncovered_holding[0].symbol_id = 3;
  uncovered_holding[0].position_quantity = 10;
  const engine_common::PortfolioView uncovered_portfolio{
      uncovered_holding, 11'900.0, 12'000.0, 100.0, 100.0};
  ok &= check(
      risk_budget
              .build(prediction_batch, market, uncovered_portfolio, &risk_model)
              .empty() &&
          risk_budget.diagnostics().hold_current_weights,
      "risk artifact must cover every current holding");

  qbt::strategy::ResearchPortfolioPolicy topk(
      config(qbt::strategy::ResearchPortfolioPolicyKind::TOPK_EQUAL_WEIGHT));
  const auto topk_targets =
      topk.build(prediction_batch, market, portfolio, nullptr);
  ok &= check(
      topk_targets.size() == 2 &&
          topk.diagnostics().status == portfolio_math::OptimizationStatus::OK,
      "frozen policy switch supports Top-K control without a risk artifact");

  auto raw_score_config =
      config(qbt::strategy::ResearchPortfolioPolicyKind::TOPK_EQUAL_WEIGHT);
  raw_score_config.selection.max_positions = 1;
  raw_score_config.policy_config_hash = 301;
  predictions[0].expected_return = 0.02F;
  predictions[0].expected_volatility = 0.20F;
  predictions[1].expected_return = 0.01F;
  predictions[1].expected_volatility = 0.01F;
  qbt::strategy::ResearchPortfolioPolicy raw_score_policy(raw_score_config);
  const auto raw_score_target =
      raw_score_policy.build(prediction_batch, market, portfolio, nullptr);
  ok &=
      check(raw_score_target.size() == 1 && raw_score_target[0].symbol_id == 1,
            "raw-return RankingScoreSpec selects the highest return");

  auto adjusted_score_config = raw_score_config;
  adjusted_score_config.selection.ranking_score_mode =
      engine_common::RankingScoreMode::RISK_ADJUSTED_RETURN;
  adjusted_score_config.selection.ranking_risk_floor = 0.01F;
  adjusted_score_config.policy_config_hash = 302;
  qbt::strategy::ResearchPortfolioPolicy adjusted_score_policy(
      adjusted_score_config);
  const auto adjusted_score_target =
      adjusted_score_policy.build(prediction_batch, market, portfolio, nullptr);
  ok &= check(adjusted_score_target.size() == 1 &&
                  adjusted_score_target[0].symbol_id == 2,
              "risk-adjusted RankingScoreSpec matches the frozen formula");

  if (!ok)
    return 1;
  std::printf("test_research_portfolio_policy: all checks passed\n");
  return 0;
}

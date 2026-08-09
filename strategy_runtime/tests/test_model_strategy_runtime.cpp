#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

#include "engine.h"
#include "ml_runtime/bar_v1_feature_pipeline.h"
#include "ml_runtime/mock_inference_backend.h"
#include "strategy_runtime/model_strategy_runtime.h"

namespace {

qbt::MarketSnapshot bar(int64_t timestamp, double close) {
  qbt::MarketSnapshot value;
  value.symbol = "TEST";
  value.timestamp = timestamp;
  value.open = close;
  value.high = close;
  value.low = close;
  value.close = close;
  value.volume = 10'000 + timestamp;
  value.lot_size = 1;
  return value;
}

} // namespace

int main() {
  qbt::ml::ModelArtifact artifact;
  artifact.descriptor.model_id = "mock-return";
  artifact.descriptor.model_version = "1";
  artifact.descriptor.feature_schema_hash = 42;
  artifact.descriptor.model_version_hash = 99;
  artifact.descriptor.lookback = 2;
  artifact.descriptor.feature_count = qbt::ml::kBarV1FeatureCount;
  artifact.descriptor.max_batch_size = 4;

  qbt::strategy::ModelStrategyConfig strategy_config;
  strategy_config.artifact = artifact;
  strategy_config.runtime_options.max_batch_size = 4;
  strategy_config.policy.max_positions = 1;
  strategy_config.policy.max_position_weight = 0.10F;
  strategy_config.policy.minimum_expected_return = 0.0F;
  strategy_config.policy.minimum_confidence = 0.01F;
  strategy_config.risk.max_order_quantity = 10'000;
  auto mismatched_config = strategy_config;
  mismatched_config.artifact.descriptor.ranking_cutoff = 1;
  mismatched_config.artifact.descriptor.ranking_score_spec_hash = 77;
  mismatched_config.artifact.descriptor.label_spec_hash = 78;
  mismatched_config.artifact.descriptor.ranking_score_mode =
      engine_common::RankingScoreMode::RISK_ADJUSTED_RETURN;
  bool ranking_mismatch_rejected = false;
  try {
    qbt::strategy::ModelStrategyRuntime rejected(
        std::make_unique<qbt::ml::MockInferenceBackend>(), mismatched_config);
  } catch (const std::invalid_argument &) {
    ranking_mismatch_rejected = true;
  }
  if (!ranking_mismatch_rejected)
    return 1;

  auto matched_config = mismatched_config;
  matched_config.policy.ranking_score_mode =
      engine_common::RankingScoreMode::RISK_ADJUSTED_RETURN;
  matched_config.policy.ranking_risk_floor =
      matched_config.artifact.descriptor.ranking_risk_floor;
  qbt::strategy::ModelStrategyRuntime matched(
      std::make_unique<qbt::ml::MockInferenceBackend>(), matched_config);
  (void)matched;
  auto runtime = std::make_shared<qbt::strategy::ModelStrategyRuntime>(
      std::make_unique<qbt::ml::MockInferenceBackend>(), strategy_config);

  qbt::ExecutionConfig execution;
  execution.enforce_t_plus_one = false;
  qbt::BacktestEngine engine(10'000.0, qbt::FillTiming::CLOSE, execution);
  engine_common::StrategySessionContext context;
  context.feature_schema_hash = 42;
  context.model_version_hash = 99;
  engine.set_strategy_runtime(runtime, context);

  for (int64_t timestamp = 1; timestamp <= 60; ++timestamp) {
    engine.process_market_data(bar(timestamp, 10.0 + timestamp * 0.01));
  }
  if (engine.get_order_count() != 0)
    return 1;
  engine.process_market_data(bar(61, 11.0));
  const auto bought = engine.get_position("TEST").quantity;
  const auto buy_decision = runtime->last_decision();
  if (bought <= 0 || engine.get_order_count() != 1 ||
      !buy_decision.valid() || buy_decision.decision_at != 61 ||
      buy_decision.targets.size() != 1 ||
      buy_decision.targets.front().target_quantity != bought)
    return 1;
  const auto buy_decision_id = buy_decision.decision_id;
  engine.process_market_data(bar(62, 9.0));
  const auto exit_decision = runtime->last_decision();
  if (engine.get_position("TEST").quantity != 0 ||
      engine.get_order_count() != 2 ||
      runtime->metrics().generated_intents != 2 ||
      runtime->metrics().insufficient_history == 0 ||
      !exit_decision.valid() || exit_decision.decision_id <= buy_decision_id ||
      exit_decision.decision_at != 62 || exit_decision.targets.size() != 1 ||
      exit_decision.targets.front().target_quantity != 0) {
    return 1;
  }

  qbt::strategy::LongOnlyTopKPolicy policy(strategy_config.policy);
  std::array<engine_common::ModelPrediction, 1> invalid_predictions{};
  invalid_predictions[0].symbol_id = 0;
  invalid_predictions[0].flags = engine_common::INSUFFICIENT_HISTORY;
  engine_common::PredictionBatch batch{62, 99, invalid_predictions, 1};
  std::array<engine_common::MarketBar, 1> market_bars{};
  market_bars[0].symbol_id = 0;
  market_bars[0].timestamp = 62;
  market_bars[0].close = 9.0;
  market_bars[0].lot_size = 1;
  market_bars[0].flags =
      engine_common::MARKET_LISTED | engine_common::MARKET_DATA_TRUSTED;
  std::array<engine_common::PortfolioItem, 1> positions{};
  positions[0].symbol_id = 0;
  positions[0].position_quantity = 100;
  const auto no_targets = policy.build(batch, {62, market_bars, true},
                                       {positions, 0.0, 10'000.0, 0.0, 0.0});
  if (!no_targets.empty())
    return 1;

  qbt::strategy::ModelStrategyConfig partial_config = strategy_config;
  partial_config.policy.minimum_confidence = 0.0F;
  auto partial_runtime = std::make_shared<qbt::strategy::ModelStrategyRuntime>(
      std::make_unique<qbt::ml::MockInferenceBackend>(), partial_config);
  qbt::ExecutionConfig partial_execution;
  partial_execution.enforce_t_plus_one = false;
  partial_execution.max_volume_participation = 0.10;
  qbt::BacktestEngine partial_engine(10'000.0, qbt::FillTiming::CLOSE,
                                     partial_execution);
  partial_engine.set_strategy_runtime(partial_runtime, context);
  for (int64_t timestamp = 1; timestamp <= 60; ++timestamp) {
    partial_engine.process_market_data(bar(timestamp, 10.0 + timestamp * 0.01));
  }
  auto thin_bar = bar(61, 11.0);
  thin_bar.volume = 10;
  partial_engine.process_market_data(thin_bar);
  if (partial_engine.get_order_count() != 1 ||
      partial_engine.get_position("TEST").quantity != 1) {
    return 1;
  }
  thin_bar = bar(62, 11.001);
  thin_bar.volume = 10;
  partial_engine.process_market_data(thin_bar);
  if (partial_engine.get_order_count() != 1 ||
      partial_engine.get_position("TEST").quantity != 2) {
    return 1;
  }
  std::printf("test_model_strategy_runtime: all checks passed\n");
  return 0;
}

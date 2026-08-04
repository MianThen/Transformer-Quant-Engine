#include "strategy_runtime/portfolio_policy.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace qbt::strategy {

double ranking_score(const engine_common::ModelPrediction &prediction,
                     const LongOnlyPolicyConfig &config) noexcept {
  if (!prediction.finite() || prediction.expected_volatility < 0.0F) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (config.ranking_score_mode ==
      engine_common::RankingScoreMode::RAW_RETURN) {
    return prediction.expected_return;
  }
  if (config.ranking_score_mode ==
      engine_common::RankingScoreMode::RISK_ADJUSTED_RETURN) {
    return static_cast<double>(prediction.expected_return) /
           std::max(static_cast<double>(prediction.expected_volatility),
                    static_cast<double>(config.ranking_risk_floor));
  }
  return std::numeric_limits<double>::quiet_NaN();
}

LongOnlyTopKPolicy::LongOnlyTopKPolicy(LongOnlyPolicyConfig config)
    : config_(config) {
  if (config_.max_positions == 0 ||
      !std::isfinite(config_.max_position_weight) ||
      config_.max_position_weight <= 0.0F ||
      config_.max_position_weight > 1.0F ||
      !std::isfinite(config_.minimum_expected_return) ||
      (!std::isnan(config_.minimum_ranking_score) &&
       !std::isfinite(config_.minimum_ranking_score)) ||
      !std::isfinite(config_.minimum_confidence) ||
      config_.minimum_confidence < 0.0F || config_.minimum_confidence > 1.0F ||
      !std::isfinite(config_.ranking_risk_floor) ||
      config_.ranking_risk_floor <= 0.0F ||
      (config_.ranking_score_mode !=
           engine_common::RankingScoreMode::RAW_RETURN &&
       config_.ranking_score_mode !=
           engine_common::RankingScoreMode::RISK_ADJUSTED_RETURN)) {
    throw std::invalid_argument("invalid long-only policy configuration");
  }
}

std::span<const engine_common::TargetPosition>
LongOnlyTopKPolicy::build(const engine_common::PredictionBatch &predictions,
                          const engine_common::MarketFrameBatchView &market,
                          const engine_common::PortfolioView &portfolio) {
  targets_.clear();
  std::vector<const engine_common::ModelPrediction *> ranked;
  std::unordered_set<engine_common::SymbolId> valid_symbols;
  ranked.reserve(predictions.size);
  valid_symbols.reserve(predictions.size);
  for (size_t index = 0; index < predictions.size; ++index) {
    const auto &value = predictions.values[index];
    if ((value.flags & engine_common::PREDICTION_VALID) == 0 || !value.finite())
      continue;
    valid_symbols.insert(value.symbol_id);
    const double score = ranking_score(value, config_);
    const double minimum_score = std::isnan(config_.minimum_ranking_score)
                                     ? config_.minimum_expected_return
                                     : config_.minimum_ranking_score;
    if (std::isfinite(score) && score >= minimum_score &&
        value.confidence >= config_.minimum_confidence)
      ranked.push_back(&value);
  }
  std::sort(ranked.begin(), ranked.end(),
            [&](const auto *lhs, const auto *rhs) {
              const double lhs_score = ranking_score(*lhs, config_);
              const double rhs_score = ranking_score(*rhs, config_);
              if (lhs_score != rhs_score) {
                return lhs_score > rhs_score;
              }
              return lhs->symbol_id < rhs->symbol_id;
            });
  if (ranked.size() > config_.max_positions)
    ranked.resize(config_.max_positions);
  std::unordered_set<engine_common::SymbolId> selected;
  selected.reserve(ranked.size());
  for (const auto *prediction : ranked)
    selected.insert(prediction->symbol_id);

  for (const auto &item : portfolio.items) {
    if (item.position_quantity != 0 &&
        valid_symbols.count(item.symbol_id) != 0 &&
        selected.count(item.symbol_id) == 0) {
      targets_.push_back({item.symbol_id, 0, 0.0F});
    }
  }
  for (const auto *prediction : ranked) {
    const auto found = std::find_if(
        market.bars.begin(), market.bars.end(), [&](const auto &bar) {
          return bar.symbol_id == prediction->symbol_id;
        });
    if (found == market.bars.end() || !std::isfinite(found->close) ||
        found->close <= 0.0 || found->lot_size <= 0 ||
        portfolio.equity <= 0.0) {
      continue;
    }
    const double notional = portfolio.equity * config_.max_position_weight;
    const auto raw_quantity = static_cast<engine_common::Quantity>(
        std::floor(notional / found->close));
    const auto quantity = raw_quantity - raw_quantity % found->lot_size;
    targets_.push_back({prediction->symbol_id,
                        std::max<engine_common::Quantity>(quantity, 0),
                        config_.max_position_weight});
  }
  std::sort(targets_.begin(), targets_.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.symbol_id < rhs.symbol_id;
            });
  return targets_;
}

} // namespace qbt::strategy

#include "strategy_runtime/model_strategy_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace qbt::strategy {

namespace {

bool same_targets(std::span<const engine_common::TargetPosition> left,
                  std::span<const engine_common::TargetPosition> right) {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin(),
                    [](const auto &lhs, const auto &rhs) {
                      return lhs.symbol_id == rhs.symbol_id &&
                             lhs.target_quantity == rhs.target_quantity &&
                             lhs.target_weight == rhs.target_weight;
                    });
}

} // namespace

ModelStrategyRuntime::ModelStrategyRuntime(
    std::unique_ptr<qbt::ml::IInferenceBackend> backend,
    ModelStrategyConfig config)
    : backend_(std::move(backend)), config_(std::move(config)),
      window_store_(config_.artifact.descriptor.lookback,
                    config_.artifact.descriptor.feature_count,
                    config_.artifact.descriptor.static_feature_count),
      policy_(config_.policy), risk_(config_.risk) {
  const auto feature_count = config_.artifact.descriptor.feature_count;
  const auto &descriptor = config_.artifact.descriptor;
  const bool valid_ranking_contract =
      descriptor.ranking_cutoff == 0 ||
      (descriptor.ranking_cutoff == config_.policy.max_positions &&
       descriptor.ranking_score_spec_hash != 0 &&
       descriptor.label_spec_hash != 0 &&
       descriptor.ranking_score_mode == config_.policy.ranking_score_mode &&
       std::abs(descriptor.ranking_risk_floor -
                config_.policy.ranking_risk_floor) <= 1e-12F);
  if (!backend_ ||
      (feature_count != 1 && feature_count != qbt::ml::kBarV1FeatureCount) ||
      config_.artifact.descriptor.static_feature_count != 0 ||
      config_.max_order_intents == 0 || !valid_ranking_contract) {
    throw std::invalid_argument("BAR runtime requires the BAR_V1 schema or the "
                                "one-feature mock schema");
  }
}

engine_common::StrategyStatus ModelStrategyRuntime::start(
    const engine_common::StrategySessionContext &context) {
  if (started_ || context.live ||
      context.feature_schema_hash !=
          config_.artifact.descriptor.feature_schema_hash ||
      context.model_version_hash !=
          config_.artifact.descriptor.model_version_hash) {
    return engine_common::StrategyStatus::INVALID_CONFIGURATION;
  }
  if (backend_->load(config_.artifact, config_.runtime_options) !=
          qbt::ml::InferenceStatus::OK ||
      backend_->warmup() != qbt::ml::InferenceStatus::OK) {
    return engine_common::StrategyStatus::MODEL_ERROR;
  }
  allow_orders_ = context.allow_orders && !context.shadow;
  last_targets_.clear();
  last_decision_id_ = 0;
  last_decision_at_ = 0;
  has_decision_ = false;
  started_ = true;
  return engine_common::StrategyStatus::OK;
}

engine_common::StrategyStatus ModelStrategyRuntime::on_market_batch(
    const engine_common::MarketFrameBatchView &market,
    const engine_common::PortfolioView &portfolio,
    engine_common::OrderIntentBuffer &output) noexcept {
  output.clear();
  if (!started_)
    return engine_common::StrategyStatus::NOT_READY;
  ++metrics_.market_batches;
  if (!market.data_trusted)
    return engine_common::StrategyStatus::DATA_UNTRUSTED;
  try {
    symbols_.clear();
    symbols_.reserve(market.bars.size());
    if (config_.artifact.descriptor.feature_count ==
        qbt::ml::kBarV1FeatureCount) {
      const auto rows = bar_v1_features_.update(market);
      for (size_t row = 0; row < rows.symbols.size(); ++row) {
        const size_t offset = row * qbt::ml::kBarV1FeatureCount;
        window_store_.update(
            rows.symbols[row], market.asof_timestamp,
            rows.values.subspan(offset, qbt::ml::kBarV1FeatureCount),
            rows.valid[row] != 0);
        symbols_.push_back(rows.symbols[row]);
      }
    } else {
      for (const auto &bar : market.bars) {
        const size_t required = static_cast<size_t>(bar.symbol_id) + 1;
        if (previous_signal_close_.size() < required) {
          previous_signal_close_.resize(required, 0.0);
          has_previous_close_.resize(required, 0);
        }
        const double close =
            bar.signal_close > 0.0 ? bar.signal_close : bar.close;
        const bool tradable =
            (bar.flags & engine_common::MARKET_LISTED) != 0 &&
            (bar.flags & engine_common::MARKET_SUSPENDED) == 0 &&
            std::isfinite(close) && close > 0.0;
        float value = 0.0F;
        bool valid = tradable && has_previous_close_[bar.symbol_id] != 0;
        if (valid) {
          value = static_cast<float>(
              std::log(close / previous_signal_close_[bar.symbol_id]));
          valid = std::isfinite(value);
        }
        window_store_.update(bar.symbol_id, market.asof_timestamp,
                             std::array<float, 1>{value}, valid);
        if (tradable) {
          previous_signal_close_[bar.symbol_id] = close;
          has_previous_close_[bar.symbol_id] = 1;
        }
        symbols_.push_back(bar.symbol_id);
      }
    }
    auto features = window_store_.batch(
        market.asof_timestamp, config_.artifact.descriptor.feature_schema_hash,
        symbols_);
    predictions_.resize(symbols_.size());
    engine_common::PredictionBatch predictions{0, 0, predictions_, 0};
    const auto status = backend_->infer(features, predictions);
    if (status != qbt::ml::InferenceStatus::OK) {
      ++metrics_.inference_errors;
      return engine_common::StrategyStatus::MODEL_ERROR;
    }
    for (size_t index = 0; index < predictions.size; ++index) {
      if ((predictions.values[index].flags &
           engine_common::INSUFFICIENT_HISTORY) != 0) {
        ++metrics_.insufficient_history;
      }
    }
    const auto targets = policy_.build(predictions, market, portfolio);
    const bool target_changed = has_decision_ &&
        !same_targets(last_targets_, targets);
    if ((!has_decision_ && !targets.empty()) || target_changed) {
      last_targets_.assign(targets.begin(), targets.end());
      ++last_decision_id_;
      last_decision_at_ = market.asof_timestamp;
      has_decision_ = true;
    }
    if (!allow_orders_)
      return engine_common::StrategyStatus::OK;
    for (const auto &target : targets) {
      const auto *current = portfolio_item(target.symbol_id, portfolio);
      const engine_common::Quantity current_quantity =
          current == nullptr ? 0 : current->position_quantity;
      const engine_common::Quantity active_buy_quantity =
          current == nullptr ? 0 : current->active_buy_quantity;
      const engine_common::Quantity active_sell_quantity =
          current == nullptr ? 0 : current->active_sell_quantity;
      const engine_common::Quantity projected_quantity =
          current_quantity + active_buy_quantity - active_sell_quantity;
      const engine_common::Quantity difference =
          target.target_quantity - projected_quantity;
      if (difference == 0)
        continue;
      engine_common::OrderIntent intent;
      intent.symbol_id = target.symbol_id;
      intent.side =
          difference > 0 ? engine_common::Side::BUY : engine_common::Side::SELL;
      intent.quantity = std::abs(difference);
      intent.timestamp = market.asof_timestamp;
      const auto decision = risk_.evaluate(intent, market);
      if (decision != RiskDecision::APPROVED) {
        ++metrics_.risk_rejections;
        continue;
      }
      if (output.size >= config_.max_order_intents || !output.push(intent)) {
        return engine_common::StrategyStatus::OUTPUT_OVERFLOW;
      }
      ++metrics_.generated_intents;
    }
    return engine_common::StrategyStatus::OK;
  } catch (...) {
    ++metrics_.inference_errors;
    return engine_common::StrategyStatus::MODEL_ERROR;
  }
}

void ModelStrategyRuntime::on_execution(
    const engine_common::ExecutionEvent &) noexcept {}

void ModelStrategyRuntime::on_reset(engine_common::ResetReason,
                                    engine_common::TimestampNs) noexcept {
  window_store_.reset();
  bar_v1_features_.reset();
  std::fill(has_previous_close_.begin(), has_previous_close_.end(), 0);
  last_targets_.clear();
  last_decision_at_ = 0;
  has_decision_ = false;
}

engine_common::StrategyDecisionView
ModelStrategyRuntime::last_decision() const noexcept {
  if (!has_decision_)
    return {};
  return {last_decision_id_, last_decision_at_, last_targets_};
}

void ModelStrategyRuntime::stop() noexcept {
  started_ = false;
  allow_orders_ = false;
  backend_->reset();
}

const engine_common::PortfolioItem *ModelStrategyRuntime::portfolio_item(
    engine_common::SymbolId symbol,
    const engine_common::PortfolioView &portfolio) const noexcept {
  const auto found =
      std::find_if(portfolio.items.begin(), portfolio.items.end(),
                   [&](const auto &item) { return item.symbol_id == symbol; });
  return found == portfolio.items.end() ? nullptr : &*found;
}

const engine_common::MarketBar *ModelStrategyRuntime::market_bar(
    engine_common::SymbolId symbol,
    const engine_common::MarketFrameBatchView &market) const noexcept {
  const auto found =
      std::find_if(market.bars.begin(), market.bars.end(),
                   [&](const auto &bar) { return bar.symbol_id == symbol; });
  return found == market.bars.end() ? nullptr : &*found;
}

} // namespace qbt::strategy

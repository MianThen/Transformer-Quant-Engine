#include "strategy_runtime/precomputed_prediction_runtime.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace qbt::strategy {

PrecomputedPredictionRuntime::PrecomputedPredictionRuntime(
    std::vector<PrecomputedPredictionFrame> frames,
    PrecomputedPredictionConfig config)
    : frames_(std::move(frames)), config_(config), policy_(config.policy),
      order_planner_(config.max_order_intents), risk_(config.risk) {
    if (frames_.empty() || config_.max_order_intents == 0 ||
        config_.model_version_hash == 0) {
        throw std::invalid_argument("precomputed prediction configuration is invalid");
    }
    engine_common::TimestampNs previous = 0;
    for (const auto& frame : frames_) {
        if (frame.timestamp <= previous || frame.predictions.empty()) {
            throw std::invalid_argument(
                "precomputed prediction frames must be non-empty and strictly ordered");
        }
        for (const auto& prediction : frame.predictions) {
            if (!prediction.finite()) {
                throw std::invalid_argument("precomputed prediction is non-finite");
            }
        }
        previous = frame.timestamp;
    }
}

engine_common::StrategyStatus PrecomputedPredictionRuntime::start(
    const engine_common::StrategySessionContext& context) {
    if (started_ || context.live) {
        return engine_common::StrategyStatus::INVALID_CONFIGURATION;
    }
    next_frame_ = 0;
    next_decision_id_ = 0;
    allow_orders_ = context.allow_orders && !context.shadow;
    started_ = true;
    return engine_common::StrategyStatus::OK;
}

engine_common::StrategyStatus PrecomputedPredictionRuntime::on_market_batch(
    const engine_common::MarketFrameBatchView& market,
    const engine_common::PortfolioView& portfolio,
    engine_common::OrderIntentBuffer& output) noexcept {
    output.clear();
    if (!started_) return engine_common::StrategyStatus::NOT_READY;
    if (!market.data_trusted) return engine_common::StrategyStatus::DATA_UNTRUSTED;
    while (next_frame_ < frames_.size() &&
           frames_[next_frame_].timestamp < market.asof_timestamp) {
        ++metrics_.missing_frames;
        ++next_frame_;
    }
    if (next_frame_ >= frames_.size() ||
        frames_[next_frame_].timestamp != market.asof_timestamp) {
        return engine_common::StrategyStatus::OK;
    }
    auto& frame = frames_[next_frame_++];
    if (frame.predictions.size() != market.bars.size() ||
        ++next_decision_id_ == 0) {
        ++metrics_.invalid_frames;
        return engine_common::StrategyStatus::MODEL_ERROR;
    }
    symbols_.resize(market.bars.size());
    for (size_t index = 0; index < market.bars.size(); ++index) {
        symbols_[index] = market.bars[index].symbol_id;
        frame.predictions[index].symbol_id = market.bars[index].symbol_id;
        frame.predictions[index].asof_timestamp = market.asof_timestamp;
    }
    engine_common::PredictionBatch batch{
        market.asof_timestamp, config_.model_version_hash,
        frame.predictions, frame.predictions.size()};
    if (validator_.validate(
            batch, market.asof_timestamp, config_.model_version_hash, symbols_) !=
        PredictionValidationStatus::OK) {
        ++metrics_.invalid_frames;
        return engine_common::StrategyStatus::MODEL_ERROR;
    }
    ++metrics_.matched_frames;
    const auto targets = policy_.build(batch, market, portfolio);
    if (!allow_orders_) return engine_common::StrategyStatus::OK;
    const auto intents = order_planner_.build(
        targets, portfolio, market.asof_timestamp, next_decision_id_);
    risk_.begin_batch(market, portfolio);
    for (const auto& intent : intents) {
        if (risk_.evaluate(intent, market, portfolio) != RiskDecision::APPROVED) {
            ++metrics_.risk_rejections;
            continue;
        }
        if (!output.push(intent)) {
            output.clear();
            return engine_common::StrategyStatus::OUTPUT_OVERFLOW;
        }
        ++metrics_.generated_intents;
    }
    return engine_common::StrategyStatus::OK;
}

void PrecomputedPredictionRuntime::stop() noexcept {
    started_ = false;
    allow_orders_ = false;
    next_frame_ = 0;
}

}  // namespace qbt::strategy

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "engine_common/model_types.h"
#include "engine_common/strategy.h"
#include "strategy_runtime/order_planner.h"
#include "strategy_runtime/portfolio_policy.h"
#include "strategy_runtime/prediction_validator.h"
#include "strategy_runtime/risk_manager.h"

namespace qbt::strategy {

struct PrecomputedPredictionFrame {
    engine_common::TimestampNs timestamp = 0;
    std::vector<engine_common::ModelPrediction> predictions;
};

struct PrecomputedPredictionConfig {
    LongOnlyPolicyConfig policy;
    RiskConfig risk;
    size_t max_order_intents = 1024;
    uint64_t model_version_hash = 1;
};

struct PrecomputedPredictionMetrics {
    uint64_t matched_frames = 0;
    uint64_t missing_frames = 0;
    uint64_t invalid_frames = 0;
    uint64_t generated_intents = 0;
    uint64_t risk_rejections = 0;
};

class PrecomputedPredictionRuntime final : public engine_common::IStrategyRuntime {
public:
    PrecomputedPredictionRuntime(
        std::vector<PrecomputedPredictionFrame> frames,
        PrecomputedPredictionConfig config);

    engine_common::StrategyStatus start(
        const engine_common::StrategySessionContext& context) override;
    engine_common::StrategyStatus on_market_batch(
        const engine_common::MarketFrameBatchView& market,
        const engine_common::PortfolioView& portfolio,
        engine_common::OrderIntentBuffer& output) noexcept override;
    void on_execution(const engine_common::ExecutionEvent&) noexcept override {}
    void on_reset(engine_common::ResetReason, engine_common::TimestampNs) noexcept override {}
    void stop() noexcept override;

    const PrecomputedPredictionMetrics& metrics() const noexcept { return metrics_; }

private:
    std::vector<PrecomputedPredictionFrame> frames_;
    PrecomputedPredictionConfig config_;
    LongOnlyTopKPolicy policy_;
    LongOnlyOrderPlanner order_planner_;
    BasicRiskManager risk_;
    PredictionValidator validator_;
    std::vector<engine_common::SymbolId> symbols_;
    size_t next_frame_ = 0;
    uint64_t next_decision_id_ = 0;
    bool started_ = false;
    bool allow_orders_ = false;
    PrecomputedPredictionMetrics metrics_;
};

}  // namespace qbt::strategy

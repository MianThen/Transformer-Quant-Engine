#pragma once

#include <memory>
#include <vector>

#include "engine_common/strategy.h"
#include "ml_runtime/bar_v1_feature_pipeline.h"
#include "ml_runtime/feature_window_store.h"
#include "ml_runtime/inference_backend.h"
#include "strategy_runtime/portfolio_policy.h"
#include "strategy_runtime/risk_manager.h"

namespace qbt::strategy {

struct ModelStrategyConfig {
    qbt::ml::ModelArtifact artifact;
    qbt::ml::RuntimeOptions runtime_options;
    LongOnlyPolicyConfig policy;
    RiskConfig risk;
    size_t max_order_intents = 1024;
};

struct ModelStrategyMetrics {
    uint64_t market_batches = 0;
    uint64_t inference_errors = 0;
    uint64_t insufficient_history = 0;
    uint64_t risk_rejections = 0;
    uint64_t generated_intents = 0;
};

class ModelStrategyRuntime final : public engine_common::IStrategyRuntime {
public:
    ModelStrategyRuntime(std::unique_ptr<qbt::ml::IInferenceBackend> backend,
                         ModelStrategyConfig config);

    engine_common::StrategyStatus start(
        const engine_common::StrategySessionContext& context) override;
    engine_common::StrategyStatus on_market_batch(
        const engine_common::MarketFrameBatchView& market,
        const engine_common::PortfolioView& portfolio,
        engine_common::OrderIntentBuffer& output) noexcept override;
    void on_execution(const engine_common::ExecutionEvent& execution) noexcept override;
    void on_reset(engine_common::ResetReason reason,
                  engine_common::TimestampNs timestamp) noexcept override;
    engine_common::StrategyDecisionView last_decision() const noexcept override;
    void stop() noexcept override;

    const ModelStrategyMetrics& metrics() const noexcept { return metrics_; }

private:
    const engine_common::PortfolioItem* portfolio_item(
        engine_common::SymbolId symbol,
        const engine_common::PortfolioView& portfolio) const noexcept;
    const engine_common::MarketBar* market_bar(
        engine_common::SymbolId symbol,
        const engine_common::MarketFrameBatchView& market) const noexcept;

    std::unique_ptr<qbt::ml::IInferenceBackend> backend_;
    ModelStrategyConfig config_;
    qbt::ml::BarV1FeaturePipeline bar_v1_features_;
    qbt::ml::FeatureWindowStore window_store_;
    LongOnlyTopKPolicy policy_;
    BasicRiskManager risk_;
    std::vector<double> previous_signal_close_;
    std::vector<uint8_t> has_previous_close_;
    std::vector<engine_common::SymbolId> symbols_;
    std::vector<engine_common::ModelPrediction> predictions_;
    std::vector<engine_common::TargetPosition> last_targets_;
    ModelStrategyMetrics metrics_;
    std::uint64_t last_decision_id_ = 0;
    engine_common::TimestampNs last_decision_at_ = 0;
    bool has_decision_ = false;
    bool started_ = false;
    bool allow_orders_ = false;
};

}  // namespace qbt::strategy

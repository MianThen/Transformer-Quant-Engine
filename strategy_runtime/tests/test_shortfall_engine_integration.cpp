#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>

#include "engine.h"
#include "ml_runtime/mock_inference_backend.h"
#include "performance_analytics/shortfall_replay_sink.h"
#include "strategy_runtime/model_strategy_runtime.h"

namespace {

qbt::MarketSnapshot bar(std::int64_t timestamp, double close) {
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

performance_analytics::PerformanceSpecV1 performance_spec() {
    performance_analytics::PerformanceSpecV1 value;
    value.frequency = performance_analytics::ReturnFrequency::DAILY;
    value.calendar_id = "XSHG_TRADING_DAY_V1";
    value.calendar_periods_per_year = 242.0;
    value.benchmark_id = "INTERNAL_FROZEN_CONTROL";
    value.config_hash = 45;
    return value;
}

}  // namespace

int main() {
    qbt::ml::ModelArtifact artifact;
    artifact.descriptor.model_id = "mock-return";
    artifact.descriptor.model_version = "1";
    artifact.descriptor.feature_schema_hash = 42;
    artifact.descriptor.model_version_hash = 99;
    artifact.descriptor.lookback = 2;
    artifact.descriptor.feature_count = 1;
    artifact.descriptor.max_batch_size = 4;

    qbt::strategy::ModelStrategyConfig strategy_config;
    strategy_config.artifact = artifact;
    strategy_config.runtime_options.max_batch_size = 4;
    strategy_config.policy.max_positions = 1;
    strategy_config.policy.max_position_weight = 0.10F;
    strategy_config.policy.minimum_expected_return = 0.0F;
    strategy_config.policy.minimum_confidence = 0.01F;
    strategy_config.risk.max_order_quantity = 10'000;

    auto runtime = std::make_shared<qbt::strategy::ModelStrategyRuntime>(
        std::make_unique<qbt::ml::MockInferenceBackend>(), strategy_config);
    auto analytics = std::make_shared<performance_analytics::ShortfallReplaySink>(
        performance_spec());

    qbt::ExecutionConfig execution;
    execution.enforce_t_plus_one = false;
    qbt::BacktestEngine engine(10'000.0, qbt::FillTiming::CLOSE, execution);
    engine_common::StrategySessionContext context;
    context.feature_schema_hash = 42;
    context.model_version_hash = 99;
    engine.set_strategy_runtime(runtime, context);
    engine.set_replay_analytics_sink(analytics);
    engine.set_commission_fn([](double notional, bool) {
        return notional * 0.001;
    });

    std::int64_t current_timestamp = 0;
    try {
        for (std::int64_t timestamp = 1; timestamp <= 60; ++timestamp) {
            current_timestamp = timestamp;
            engine.process_market_data(bar(timestamp, 10.0 + timestamp * 0.01));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "unexpected failure at %lld: %s, shortfall_status=%u\n",
                     static_cast<long long>(current_timestamp), error.what(),
                     static_cast<unsigned>(analytics->last_shortfall_status()));
        return 1;
    }
    if (!analytics->ledger().records().empty()) return 1;

    try {
        current_timestamp = 61;
        engine.process_market_data(bar(61, 11.0));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "unexpected failure at 61: %s, shortfall_status=%u\n",
                     error.what(),
                     static_cast<unsigned>(analytics->last_shortfall_status()));
        return 1;
    }
    const auto bought = engine.get_position("TEST").quantity;
    if (bought <= 0 || analytics->ledger().records().size() != 0) return 1;

    try {
        current_timestamp = 62;
        engine.process_market_data(bar(62, 9.0));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "unexpected failure at 62: %s, shortfall_status=%u\n",
                     error.what(),
                     static_cast<unsigned>(analytics->last_shortfall_status()));
        return 1;
    }
    if (engine.get_position("TEST").quantity != 0 ||
        analytics->ledger().records().size() != 1) {
        return 1;
    }
    engine.finalize(63);

    const auto records = analytics->ledger().records();
    if (analytics->failed() || records.size() != 2 ||
        records[0].decision_id != 1 || records[1].decision_id != 2 ||
        records[0].promotion_eligible || records[1].promotion_eligible ||
        records[0].explicit_fees <= 0.0 || records[1].explicit_fees <= 0.0 ||
        std::abs(records[0].identity_residual) > 1e-10 ||
        std::abs(records[1].identity_residual) > 1e-10) {
        return 1;
    }

    auto delayed_runtime = std::make_shared<qbt::strategy::ModelStrategyRuntime>(
        std::make_unique<qbt::ml::MockInferenceBackend>(), strategy_config);
    auto delayed_analytics =
        std::make_shared<performance_analytics::ShortfallReplaySink>(
            performance_spec());
    qbt::ExecutionConfig partial_execution = execution;
    partial_execution.max_volume_participation = 0.10;
    qbt::BacktestEngine delayed_engine(
        10'000.0, qbt::FillTiming::CLOSE, partial_execution);
    delayed_engine.set_strategy_runtime(delayed_runtime, context);
    delayed_engine.set_replay_analytics_sink(delayed_analytics);
    for (std::int64_t timestamp = 1; timestamp <= 60; ++timestamp) {
        delayed_engine.process_market_data(
            bar(timestamp, 10.0 + timestamp * 0.01));
    }
    auto thin = bar(61, 11.0);
    thin.volume = 10;
    delayed_engine.process_market_data(thin);
    thin = bar(62, 9.0);
    thin.volume = 10;
    delayed_engine.process_market_data(thin);
    bool delayed_fill_rejected = false;
    try {
        thin = bar(63, 9.1);
        thin.volume = 10;
        delayed_engine.process_market_data(thin);
    } catch (const std::runtime_error&) {
        delayed_fill_rejected = true;
    }
    if (!delayed_fill_rejected || !delayed_analytics->failed() ||
        delayed_analytics->ledger().records().size() != 1) {
        return 1;
    }

    std::printf("test_shortfall_engine_integration: all checks passed\n");
    return 0;
}

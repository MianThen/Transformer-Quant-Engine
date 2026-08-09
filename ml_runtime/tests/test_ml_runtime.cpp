#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ml_runtime/bar_v1_feature_pipeline.h"
#include "ml_runtime/feature_window_store.h"
#include "ml_runtime/mock_inference_backend.h"

namespace {

bool test_bar_v1_golden() {
    qbt::ml::BarV1FeaturePipeline pipeline;
    qbt::ml::BarV1FeatureRows rows;
    std::array<engine_common::MarketBar, 2> bars{};
    for (int64_t timestamp = 1; timestamp <= 61; ++timestamp) {
        for (size_t index = 0; index < bars.size(); ++index) {
            const double close = 10.0 + static_cast<double>(index) +
                0.02 * static_cast<double>(timestamp) +
                0.05 * std::sin(static_cast<double>(timestamp) / 3.0 +
                                static_cast<double>(index) * 0.2);
            auto& bar = bars[index];
            bar = {};
            bar.symbol_id = static_cast<engine_common::SymbolId>(index);
            bar.timestamp = timestamp;
            bar.open = close * (0.998 + 0.0002 * static_cast<double>(index));
            bar.high = close * 1.01;
            bar.low = close * 0.99;
            bar.close = close;
            bar.volume = 1000 + timestamp * static_cast<int64_t>(index + 1);
            bar.lot_size = 1;
            bar.flags = engine_common::MARKET_LISTED |
                        engine_common::MARKET_DATA_TRUSTED;
        }
        rows = pipeline.update({timestamp, bars, true});
    }
    constexpr std::array<float, 46> expected{
        0.00214639725F, 0.014200354F, 0.0267873369F, 0.0366207734F,
        0.0200006664F, 0.00200200267F, 0.000144394595F, 6.96790934F,
        1.6428231F, 0.000421126373F, 0.000521805487F, 0.00102415949F,
        0.00109028118F, 0.00509533752F, 0.0125271026F, 0.0212500133F,
        1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F,
        0.00170403766F, 0.0122037148F, 0.0247148089F, 0.033270251F,
        0.0200006664F, 0.00180162198F, -9.75842631e-05F, 7.02375889F,
        1.63859153F, 0.000467059581F, 0.000456647074F, 0.000936611032F,
        0.000987623818F, 0.0042250324F, 0.0109976772F, 0.0194899794F,
        1.0F, 0.0F, 0.5F, 0.0F, 1.0F, 0.0F, 1.0F,
    };
    if (rows.symbols.size() != 2 || rows.values.size() != expected.size() ||
        rows.valid.size() != 2 || rows.valid[0] != 1 || rows.valid[1] != 1) {
        return false;
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        if (std::abs(rows.values[index] - expected[index]) > 2.0e-6F) {
            return false;
        }
    }
    return true;
}

bool test_bar_v1_resets_history_after_missing_cross_section() {
    qbt::ml::BarV1FeaturePipeline pipeline;
    engine_common::MarketBar bar;
    bar.symbol_id = 0;
    bar.open = bar.high = bar.low = bar.close = 10.0;
    bar.volume = 100;
    bar.lot_size = 1;
    bar.flags = engine_common::MARKET_LISTED | engine_common::MARKET_DATA_TRUSTED;
    for (int64_t timestamp = 1; timestamp <= 61; ++timestamp) {
        bar.timestamp = timestamp;
        bar.close = bar.open = bar.high = bar.low = 10.0 + timestamp * 0.01;
        bar.volume = 100 + timestamp;
        pipeline.update({timestamp, std::span<const engine_common::MarketBar>(&bar, 1), true});
    }
    engine_common::MarketBar other = bar;
    other.symbol_id = 1;
    other.timestamp = 62;
    const auto missing = pipeline.update(
        {62, std::span<const engine_common::MarketBar>(&other, 1), true});
    if (missing.valid[0] != 0) return false;
    bar.timestamp = 63;
    const auto returned = pipeline.update(
        {63, std::span<const engine_common::MarketBar>(&bar, 1), true});
    return returned.valid[0] == 0;
}

}  // namespace

int main() {
    if (!test_bar_v1_golden()) return 1;
    if (!test_bar_v1_resets_history_after_missing_cross_section()) return 1;
    qbt::ml::FeatureWindowStore store(2, 2);
    store.update(3, 1, std::array<float, 2>{1.0F, 2.0F}, true);
    store.update(3, 2, std::array<float, 2>{3.0F, 4.0F}, true);
    store.update(3, 3, std::array<float, 2>{5.0F, 6.0F}, true);
    const std::array<engine_common::SymbolId, 1> symbols{3};
    auto input = store.batch(3, 42, symbols);
    if (!input.valid_shape() || input.values[0] != 3.0F || input.values[2] != 5.0F) {
        return 1;
    }

    qbt::ml::ModelArtifact artifact;
    artifact.descriptor.feature_schema_hash = 42;
    artifact.descriptor.model_version_hash = 99;
    artifact.descriptor.lookback = 2;
    artifact.descriptor.feature_count = 2;
    artifact.descriptor.max_batch_size = 4;
    qbt::ml::MockInferenceBackend backend;
    if (backend.load(artifact, {}) != qbt::ml::InferenceStatus::OK ||
        backend.warmup() != qbt::ml::InferenceStatus::OK) {
        return 1;
    }
    std::vector<engine_common::ModelPrediction> values(1);
    engine_common::PredictionBatch output{0, 0, values, 0};
    if (backend.infer(input, output) != qbt::ml::InferenceStatus::OK ||
        output.size != 1 || output.values[0].expected_return != 5.0F ||
        (output.values[0].flags & engine_common::PREDICTION_VALID) == 0) {
        return 1;
    }
    std::printf("test_ml_runtime: all checks passed\n");
    return 0;
}

#include "ml_runtime/mock_inference_backend.h"

#include <algorithm>
#include <cmath>

namespace qbt::ml {

InferenceStatus MockInferenceBackend::load(
    const ModelArtifact& artifact, const RuntimeOptions& options) {
    descriptor_ = artifact.descriptor;
    options_ = options;
    loaded_ = descriptor_.feature_schema_hash != 0 &&
        descriptor_.model_version_hash != 0 && descriptor_.lookback > 0 &&
        descriptor_.feature_count > 0;
    return loaded_ ? InferenceStatus::OK : InferenceStatus::INVALID_INPUT;
}

InferenceStatus MockInferenceBackend::warmup() {
    return loaded_ ? InferenceStatus::OK : InferenceStatus::NOT_LOADED;
}

InferenceStatus MockInferenceBackend::infer(
    const engine_common::FeatureBatchView& input,
    engine_common::PredictionBatch& output) noexcept {
    if (!loaded_) return InferenceStatus::NOT_LOADED;
    if (!input.valid_shape()) return InferenceStatus::INVALID_INPUT;
    if (input.feature_schema_hash != descriptor_.feature_schema_hash ||
        input.lookback != descriptor_.lookback ||
        input.feature_count != descriptor_.feature_count ||
        input.static_feature_count != descriptor_.static_feature_count) {
        return InferenceStatus::SCHEMA_MISMATCH;
    }
    const uint32_t maximum = options_.max_batch_size != 0
        ? options_.max_batch_size : descriptor_.max_batch_size;
    if ((maximum != 0 && input.batch_size > maximum) ||
        output.values.size() < input.batch_size) {
        return InferenceStatus::OUTPUT_OVERFLOW;
    }
    const size_t row_size = static_cast<size_t>(input.lookback) * input.feature_count;
    for (uint32_t row = 0; row < input.batch_size; ++row) {
        auto& prediction = output.values[row];
        prediction = {};
        prediction.symbol_id = input.symbols[row];
        prediction.asof_timestamp = input.asof_timestamp;
        const size_t mask_offset = static_cast<size_t>(row) * input.lookback;
        const size_t last = input.lookback - 1;
        const bool any_valid = std::any_of(
            input.valid_mask.begin() + static_cast<std::ptrdiff_t>(mask_offset),
            input.valid_mask.begin() + static_cast<std::ptrdiff_t>(mask_offset + input.lookback),
            [](uint8_t value) { return value != 0; });
        if (!any_valid || input.valid_mask[mask_offset + last] == 0) {
            prediction.flags = engine_common::INSUFFICIENT_HISTORY;
            continue;
        }
        const float score = input.values[static_cast<size_t>(row) * row_size +
            last * input.feature_count];
        if (!std::isfinite(score)) return InferenceStatus::NON_FINITE_OUTPUT;
        prediction.expected_return = score;
        prediction.expected_volatility = std::abs(score);
        prediction.direction_probability = 1.0F / (1.0F + std::exp(-score));
        prediction.lower_quantile = score - std::abs(score);
        prediction.upper_quantile = score + std::abs(score);
        prediction.confidence = std::clamp(std::abs(score), 0.0F, 1.0F);
        prediction.flags = engine_common::PREDICTION_VALID;
    }
    output.asof_timestamp = input.asof_timestamp;
    output.model_version_hash = descriptor_.model_version_hash;
    output.size = input.batch_size;
    return InferenceStatus::OK;
}

void MockInferenceBackend::reset() noexcept {
    loaded_ = false;
    descriptor_ = {};
    options_ = {};
}

}  // namespace qbt::ml

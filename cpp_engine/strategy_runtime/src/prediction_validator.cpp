#include "strategy_runtime/prediction_validator.h"

namespace qbt::strategy {

PredictionValidationStatus PredictionValidator::validate(
    const engine_common::PredictionBatch& predictions,
    engine_common::TimestampNs expected_timestamp,
    uint64_t expected_model_version_hash,
    std::span<const engine_common::SymbolId> expected_symbols) const noexcept {
    if (!predictions.valid_shape() || predictions.size != expected_symbols.size()) {
        return PredictionValidationStatus::INVALID_BATCH;
    }
    if (predictions.asof_timestamp != expected_timestamp) {
        return PredictionValidationStatus::TIMESTAMP_MISMATCH;
    }
    if (predictions.model_version_hash != expected_model_version_hash) {
        return PredictionValidationStatus::MODEL_VERSION_MISMATCH;
    }
    for (size_t index = 0; index < predictions.size; ++index) {
        const auto& value = predictions.values[index];
        if (value.symbol_id != expected_symbols[index]) {
            return PredictionValidationStatus::SYMBOL_MISMATCH;
        }
        if (value.asof_timestamp != expected_timestamp) {
            return PredictionValidationStatus::TIMESTAMP_MISMATCH;
        }
        if (!value.finite()) return PredictionValidationStatus::NON_FINITE;
        if ((value.flags & engine_common::INPUT_STALE) != 0) {
            return PredictionValidationStatus::STALE;
        }
        constexpr uint32_t allowed_flags = engine_common::PREDICTION_VALID |
            engine_common::INSUFFICIENT_HISTORY | engine_common::INPUT_STALE |
            engine_common::MODEL_FALLBACK;
        if ((value.flags & ~allowed_flags) != 0 ||
            ((value.flags & engine_common::PREDICTION_VALID) != 0 &&
             (value.flags & engine_common::INSUFFICIENT_HISTORY) != 0)) {
            return PredictionValidationStatus::INVALID_SEMANTICS;
        }
        if ((value.flags & engine_common::PREDICTION_VALID) == 0) continue;
        if (value.expected_volatility < 0.0F ||
            value.direction_probability < 0.0F ||
            value.direction_probability > 1.0F || value.confidence < 0.0F ||
            value.confidence > 1.0F || value.lower_quantile > value.expected_return ||
            value.expected_return > value.upper_quantile) {
            return PredictionValidationStatus::INVALID_SEMANTICS;
        }
    }
    return PredictionValidationStatus::OK;
}

}  // namespace qbt::strategy

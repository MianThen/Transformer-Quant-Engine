#pragma once

#include <cstdint>
#include <span>

#include "engine_common/model_types.h"

namespace qbt::strategy {

enum class PredictionValidationStatus : uint8_t {
    OK,
    INVALID_BATCH,
    TIMESTAMP_MISMATCH,
    MODEL_VERSION_MISMATCH,
    SYMBOL_MISMATCH,
    NON_FINITE,
    STALE,
    INVALID_SEMANTICS,
};

class PredictionValidator {
public:
    [[nodiscard]] PredictionValidationStatus validate(
        const engine_common::PredictionBatch& predictions,
        engine_common::TimestampNs expected_timestamp,
        uint64_t expected_model_version_hash,
        std::span<const engine_common::SymbolId> expected_symbols) const noexcept;
};

}  // namespace qbt::strategy

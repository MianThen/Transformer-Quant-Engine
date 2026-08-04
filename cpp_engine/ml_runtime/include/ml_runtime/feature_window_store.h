#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "engine_common/model_types.h"

namespace qbt::ml {

class FeatureWindowStore {
public:
    FeatureWindowStore(uint32_t lookback, uint32_t feature_count,
                       uint32_t static_feature_count = 0);

    void update(engine_common::SymbolId symbol,
                engine_common::TimestampNs timestamp,
                std::span<const float> features,
                bool valid,
                std::span<const float> static_features = {});
    engine_common::FeatureBatchView batch(
        engine_common::TimestampNs asof_timestamp,
        uint64_t feature_schema_hash,
        std::span<const engine_common::SymbolId> symbols);
    void reset() noexcept;
    void reset(engine_common::SymbolId symbol) noexcept;

private:
    struct Window {
        std::vector<float> values;
        std::vector<uint8_t> mask;
        std::vector<float> static_values;
        engine_common::TimestampNs last_timestamp = 0;
        uint32_t next = 0;
        uint32_t size = 0;
    };

    Window& window_for(engine_common::SymbolId symbol);

    uint32_t lookback_;
    uint32_t feature_count_;
    uint32_t static_feature_count_;
    std::vector<Window> windows_;
    std::vector<engine_common::SymbolId> batch_symbols_;
    std::vector<float> batch_values_;
    std::vector<uint8_t> batch_mask_;
    std::vector<float> batch_static_values_;
};

}  // namespace qbt::ml

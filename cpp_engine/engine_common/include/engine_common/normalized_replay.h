#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <span>
#include <string>
#include <stdexcept>

namespace engine_common {

constexpr uint16_t kNormalizedPayloadSchemaVersion = 1;

#pragma pack(push, 1)
struct NormalizedMarketRecord {
    std::array<char, 32> symbol{};
    int64_t timestamp = 0;
    int64_t open_ticks = 0;
    int64_t high_ticks = 0;
    int64_t low_ticks = 0;
    int64_t close_ticks = 0;
    int64_t volume = 0;
};

struct NormalizedOrderRecord {
    int64_t order_id = 0;
    std::array<char, 32> symbol{};
    uint8_t side = 0;
    uint8_t type = 0;
    uint16_t reserved = 0;
    int64_t quantity = 0;
    int64_t limit_price_ticks = 0;
    int64_t timestamp = 0;
};

struct NormalizedExecutionRecord {
    int64_t order_id = 0;
    int64_t execution_id = 0;
    uint8_t status = 0;
    uint8_t reserved[7]{};
    int64_t quantity = 0;
    int64_t cumulative_quantity = 0;
    int64_t price_ticks = 0;
    int64_t timestamp = 0;
};

struct NormalizedPayloadPrefix {
    uint16_t schema_version = kNormalizedPayloadSchemaVersion;
    uint16_t flags = 0;
    uint32_t payload_size = 0;
    uint64_t feature_schema_hash = 0;
    uint64_t model_version_hash = 0;
    int64_t asof_timestamp = 0;
    uint64_t decision_id = 0;
};

struct NormalizedPredictionRecord {
    NormalizedPayloadPrefix prefix;
    uint32_t symbol_id = 0;
    float expected_return = 0.0F;
    float expected_volatility = 0.0F;
    float direction_probability = 0.5F;
    float confidence = 0.0F;
    uint32_t prediction_flags = 0;
};

struct NormalizedRiskDecisionRecord {
    NormalizedPayloadPrefix prefix;
    int64_t client_order_id = 0;
    uint32_t symbol_id = 0;
    uint16_t reject_reason = 0;
    uint8_t approved = 0;
    uint8_t reserved = 0;
};
#pragma pack(pop)

inline std::string normalized_symbol(const std::array<char, 32>& value) {
    return std::string(value.data(),
                       std::find(value.begin(), value.end(), '\0'));
}

template <class Record>
inline std::span<const std::byte> normalized_bytes(const Record& record) {
    return {reinterpret_cast<const std::byte*>(&record), sizeof(record)};
}

template <class Record>
inline Record normalized_record(std::span<const std::byte> bytes) {
    if (bytes.size() != sizeof(Record)) throw std::invalid_argument("invalid replay record size");
    Record record{};
    std::memcpy(&record, bytes.data(), sizeof(record));
    return record;
}

}  // namespace engine_common

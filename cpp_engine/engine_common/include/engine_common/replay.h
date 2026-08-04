#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine_common {

enum class ReplayRecordType : uint16_t {
    MARKET_BYTES = 1,
    ORDER_INTENT = 2,
    EXECUTION = 3,
    METRIC = 4,
    NORMALIZED_MARKET_EVENT = 5,
    NORMALIZED_BAR = 6,
    FEATURE_BATCH = 7,
    PREDICTION_BATCH = 8,
    TARGET_POSITION_BATCH = 9,
    RISK_DECISION = 10,
    MODEL_LIFECYCLE = 11,
};

#pragma pack(push, 1)
struct ReplayRecordHeader {
    std::array<char, 4> magic{'Q', 'B', 'T', 'R'};
    uint16_t version = 1;
    ReplayRecordType type = ReplayRecordType::MARKET_BYTES;
    uint32_t payload_size = 0;
    uint32_t checksum = 0;
    int64_t timestamp_ns = 0;
};
#pragma pack(pop)

inline uint32_t replay_checksum(std::span<const std::byte> bytes) {
    uint32_t value = 2166136261U;
    for (std::byte byte : bytes) {
        value ^= static_cast<uint8_t>(byte);
        value *= 16777619U;
    }
    return value;
}

class ReplayWriter {
public:
    explicit ReplayWriter(const std::string& path)
        : output_(path, std::ios::binary | std::ios::app) {
        if (!output_) throw std::runtime_error("cannot open replay output");
    }

    void append(ReplayRecordType type, int64_t timestamp_ns,
                std::span<const std::byte> payload) {
        ReplayRecordHeader header;
        header.type = type;
        header.payload_size = static_cast<uint32_t>(payload.size());
        header.checksum = replay_checksum(payload);
        header.timestamp_ns = timestamp_ns;
        output_.write(reinterpret_cast<const char*>(&header), sizeof(header));
        output_.write(reinterpret_cast<const char*>(payload.data()),
                      static_cast<std::streamsize>(payload.size()));
        if (!output_) throw std::runtime_error("replay write failed");
    }

private:
    std::ofstream output_;
};

struct ReplayRecord {
    ReplayRecordHeader header;
    std::vector<std::byte> payload;
};

class ReplayReader {
public:
    explicit ReplayReader(const std::string& path) : input_(path, std::ios::binary) {
        if (!input_) throw std::runtime_error("cannot open replay input");
    }

    bool next(ReplayRecord& record) {
        ReplayRecordHeader header;
        input_.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (input_.eof()) return false;
        if (!input_ || header.magic != std::array<char, 4>{'Q', 'B', 'T', 'R'} ||
            header.version != 1 || header.payload_size > 16 * 1024 * 1024)
            throw std::runtime_error("invalid replay header");
        record.header = header;
        record.payload.resize(header.payload_size);
        input_.read(reinterpret_cast<char*>(record.payload.data()), header.payload_size);
        if (!input_ || replay_checksum(record.payload) != header.checksum)
            throw std::runtime_error("invalid replay payload");
        return true;
    }

private:
    std::ifstream input_;
};

}  // namespace engine_common

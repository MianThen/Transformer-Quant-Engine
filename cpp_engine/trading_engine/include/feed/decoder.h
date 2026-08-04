#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>

#include "feed/protocol.h"

namespace te {

// MockCodec 的解码中间结构；MockMarketDataAdapter 会将其转换为统一 MarketEvent。
struct MarketUpdate {
    char symbol[8];
    int64_t timestamp = 0;
    double bid = 0.0;
    double ask = 0.0;
    int32_t bid_size = 0;
    int32_t ask_size = 0;
    uint32_t seq_num = 0;
    int64_t enqueue_timestamp_ns = 0;
};

// Mock 行情解码器:字节流 → MarketUpdate,并做乱序检测/重排。
//
// 面试要点:
//   - 处理 TCP 粘包/半包:缓冲区可能只收到半条消息,需按 length 切分
//   - 乱序检测:比较 seq_num 与期望值,gap 说明丢包,回退说明重复
class Decoder {
public:
    using OnUpdate = std::function<void(const MarketUpdate&)>;

    void set_on_update(OnUpdate cb) { on_update_ = std::move(cb); }

    // 喂入一段新收到的字节(可能含 0..N 条完整消息 + 半条)。
    // 内部累积、按消息边界切分、解码,对每条完整消息回调 on_update_。
    // 返回本次消费的字节数。
    size_t feed(const uint8_t* data, size_t len);

    uint64_t gap_count() const { return gap_count_; }
    uint64_t duplicate_count() const { return duplicate_count_; }
    uint64_t malformed_count() const { return malformed_count_; }
    uint64_t recovered_gap_count() const { return recovered_gap_count_; }
    uint64_t permanent_gap_count() const { return permanent_gap_count_; }
    bool has_unresolved_gap() const { return gap_active_; }
    size_t buffered_bytes() const { return write_offset_ - read_offset_; }
    void reset_sequence();

private:
    struct SequencedMessage {
        MarketUpdate update{};
        uint32_t seq_num = 0;
        bool valid = false;
        bool emits_update = false;
    };

    struct ReorderSlot {
        SequencedMessage message{};
        uint32_t seq_num = 0;
        bool occupied = false;
    };

    SequencedMessage decode_one(const uint8_t* msg, size_t len) const;
    void process_message(const uint8_t* message, size_t len);
    void emit_message(const SequencedMessage& message);
    void drain_reorder_buffer();
    void clear_reorder_buffer();
    void compact_buffer();

    OnUpdate on_update_;
    uint32_t expected_seq_ = 0;
    bool sequence_initialized_ = false;
    bool gap_active_ = false;
    std::array<uint8_t, 256 * 1024> buffer_{};
    size_t read_offset_ = 0;
    size_t write_offset_ = 0;
    std::array<ReorderSlot, 64> reorder_buffer_{};
    size_t reorder_size_ = 0;
    uint64_t gap_count_ = 0;
    uint64_t recovered_gap_count_ = 0;
    uint64_t permanent_gap_count_ = 0;
    uint64_t duplicate_count_ = 0;
    uint64_t malformed_count_ = 0;

    static constexpr size_t kMaxMessageSize = 64 * 1024;
    static constexpr size_t kMaxBufferedBytes = 256 * 1024;
    static constexpr uint32_t kReorderWindow = 64;
    static_assert((kReorderWindow & (kReorderWindow - 1)) == 0,
                  "reorder window must be a power of two");
};

}  // namespace te

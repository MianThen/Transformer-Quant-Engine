#pragma once

#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <vector>

#include "feed/decoder.h"

namespace te {

class LegacyDecoder {
public:
    using OnUpdate = std::function<void(const MarketUpdate&)>;
    void set_on_update(OnUpdate callback) { on_update_ = std::move(callback); }

    size_t feed(const uint8_t* data, size_t len) {
        if (len == 0) return 0;
        if (data == nullptr || len > kMaxBufferedBytes ||
            buffer_.size() > kMaxBufferedBytes - len) {
            buffer_.clear();
            return 0;
        }
        buffer_.insert(buffer_.end(), data, data + len);
        size_t offset = 0;
        while (buffer_.size() - offset >= sizeof(MsgHeader)) {
            const uint8_t* bytes = buffer_.data() + offset;
            const uint16_t length = be16_to_host(
                read_unaligned<uint16_t>(bytes + offsetof(MsgHeader, length)));
            if (length < sizeof(MsgHeader) || length > kMaxMessageSize) {
                ++offset;
                continue;
            }
            if (buffer_.size() - offset < length) break;
            std::vector<uint8_t> message(bytes, bytes + length);
            process(std::move(message));
            offset += length;
        }
        if (offset != 0) buffer_.erase(buffer_.begin(), buffer_.begin() + offset);
        return len;
    }

    uint64_t recovered_gap_count() const { return recovered_gap_count_; }

private:
    std::optional<MarketUpdate> decode(const std::vector<uint8_t>& message) {
        const uint8_t* bytes = message.data();
        const size_t len = message.size();
        const uint8_t type = bytes[offsetof(MsgHeader, type)];
        if (type != static_cast<uint8_t>(MsgType::QUOTE) || len != sizeof(QuoteMsg))
            return std::nullopt;
        MarketUpdate update{};
        update.seq_num = be32_to_host(
            read_unaligned<uint32_t>(bytes + offsetof(MsgHeader, seq_num)));
        update.timestamp = static_cast<int64_t>(be64_to_host(
            read_unaligned<uint64_t>(bytes + offsetof(MsgHeader, timestamp))));
        std::memcpy(update.symbol, bytes + offsetof(QuoteMsg, symbol), sizeof(update.symbol));
        update.bid = to_price(be64_to_host(
            read_unaligned<int64_t>(bytes + offsetof(QuoteMsg, bid_price))));
        update.ask = to_price(be64_to_host(
            read_unaligned<int64_t>(bytes + offsetof(QuoteMsg, ask_price))));
        update.bid_size = be32_to_host(
            read_unaligned<int32_t>(bytes + offsetof(QuoteMsg, bid_size)));
        update.ask_size = be32_to_host(
            read_unaligned<int32_t>(bytes + offsetof(QuoteMsg, ask_size)));
        return update;
    }

    void emit(const std::vector<uint8_t>& message) {
        auto update = decode(message);
        if (update && on_update_) on_update_(*update);
    }

    void process(std::vector<uint8_t> message) {
        const uint32_t sequence = be32_to_host(read_unaligned<uint32_t>(
            message.data() + offsetof(MsgHeader, seq_num)));
        if (!initialized_) { expected_ = sequence; initialized_ = true; }
        if (sequence < expected_) return;
        if (sequence == expected_) {
            emit(message);
            ++expected_;
            drain();
            return;
        }
        if (reorder_.find(sequence) != reorder_.end()) return;
        if (sequence - expected_ <= kWindow && reorder_.size() < kWindow) {
            reorder_.emplace(sequence, std::move(message));
            return;
        }
        reorder_.clear();
        expected_ = sequence;
        emit(message);
        ++expected_;
    }

    void drain() {
        bool recovered = false;
        while (true) {
            auto found = reorder_.find(expected_);
            if (found == reorder_.end()) break;
            auto message = std::move(found->second);
            reorder_.erase(found);
            emit(message);
            ++expected_;
            recovered = true;
        }
        if (recovered && reorder_.empty()) ++recovered_gap_count_;
    }

    OnUpdate on_update_;
    std::vector<uint8_t> buffer_;
    std::map<uint32_t, std::vector<uint8_t>> reorder_;
    uint32_t expected_ = 0;
    bool initialized_ = false;
    uint64_t recovered_gap_count_ = 0;
    static constexpr size_t kMaxMessageSize = 64 * 1024;
    static constexpr size_t kMaxBufferedBytes = 256 * 1024;
    static constexpr uint32_t kWindow = 64;
};

}  // namespace te

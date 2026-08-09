#include "feed/decoder.h"

#include <cstring>

namespace te {

namespace {

int32_t sequence_distance(uint32_t sequence, uint32_t expected) {
    return static_cast<int32_t>(sequence - expected);
}

}  // namespace

size_t Decoder::feed(const uint8_t* data, size_t len) {
    if (len == 0) return 0;
    if (data == nullptr || len > kMaxBufferedBytes ||
        buffered_bytes() > kMaxBufferedBytes - len) {
        ++malformed_count_;
        read_offset_ = write_offset_ = 0;
        return 0;
    }

    if (buffer_.size() - write_offset_ < len) compact_buffer();
    std::memcpy(buffer_.data() + write_offset_, data, len);
    write_offset_ += len;

    while (write_offset_ - read_offset_ >= sizeof(MsgHeader)) {
        const uint8_t* message = buffer_.data() + read_offset_;
        const uint16_t message_length = be16_to_host(
            read_unaligned<uint16_t>(message + offsetof(MsgHeader, length)));
        if (message_length < sizeof(MsgHeader) || message_length > kMaxMessageSize) {
            ++malformed_count_;
            ++read_offset_;
            continue;
        }
        if (write_offset_ - read_offset_ < message_length) break;
        process_message(message, message_length);
        read_offset_ += message_length;
    }

    if (read_offset_ == write_offset_) {
        read_offset_ = write_offset_ = 0;
    } else if (read_offset_ >= buffer_.size() / 2) {
        compact_buffer();
    }
    return len;
}

Decoder::SequencedMessage Decoder::decode_one(const uint8_t* msg, size_t len) const {
    SequencedMessage message;
    if (msg == nullptr || len < sizeof(MsgHeader)) return message;
    message.seq_num = be32_to_host(
        read_unaligned<uint32_t>(msg + offsetof(MsgHeader, seq_num)));
    message.update.seq_num = message.seq_num;
    message.update.timestamp = static_cast<int64_t>(be64_to_host(
        read_unaligned<uint64_t>(msg + offsetof(MsgHeader, timestamp))));
    const uint16_t wire_length = be16_to_host(
        read_unaligned<uint16_t>(msg + offsetof(MsgHeader, length)));
    if (wire_length != len) return message;

    const uint8_t type = msg[offsetof(MsgHeader, type)];
    if (type == static_cast<uint8_t>(MsgType::HEARTBEAT)) {
        message.valid = len == sizeof(MsgHeader);
        return message;
    }
    if (type == static_cast<uint8_t>(MsgType::QUOTE)) {
        if (len != sizeof(QuoteMsg)) return message;
        std::memcpy(message.update.symbol, msg + offsetof(QuoteMsg, symbol),
                    sizeof(message.update.symbol));
        message.update.bid = to_price(be64_to_host(
            read_unaligned<int64_t>(msg + offsetof(QuoteMsg, bid_price))));
        message.update.ask = to_price(be64_to_host(
            read_unaligned<int64_t>(msg + offsetof(QuoteMsg, ask_price))));
        message.update.bid_size = be32_to_host(
            read_unaligned<int32_t>(msg + offsetof(QuoteMsg, bid_size)));
        message.update.ask_size = be32_to_host(
            read_unaligned<int32_t>(msg + offsetof(QuoteMsg, ask_size)));
        message.valid = message.emits_update = true;
        return message;
    }
    if (type == static_cast<uint8_t>(MsgType::TRADE)) {
        if (len != sizeof(TradeMsg)) return message;
        std::memcpy(message.update.symbol, msg + offsetof(TradeMsg, symbol),
                    sizeof(message.update.symbol));
        const double price = to_price(be64_to_host(
            read_unaligned<int64_t>(msg + offsetof(TradeMsg, price))));
        const int32_t size = be32_to_host(
            read_unaligned<int32_t>(msg + offsetof(TradeMsg, size)));
        message.update.bid = message.update.ask = price;
        const uint8_t side = msg[offsetof(TradeMsg, side)];
        if (side == 0) message.update.bid_size = size;
        else if (side == 1) message.update.ask_size = size;
        else return message;
        message.valid = message.emits_update = true;
    }
    return message;
}

void Decoder::process_message(const uint8_t* bytes, size_t len) {
    const SequencedMessage message = decode_one(bytes, len);
    const uint32_t sequence = message.seq_num;
    if (!sequence_initialized_) {
        expected_seq_ = sequence;
        sequence_initialized_ = true;
    }

    const int32_t distance = sequence_distance(sequence, expected_seq_);
    if (distance < 0) {
        ++duplicate_count_;
        return;
    }
    if (distance == 0) {
        emit_message(message);
        ++expected_seq_;
        drain_reorder_buffer();
        return;
    }

    if (!gap_active_) {
        ++gap_count_;
        gap_active_ = true;
    }
    if (static_cast<uint32_t>(distance) > kReorderWindow) {
        ++permanent_gap_count_;
        clear_reorder_buffer();
        gap_active_ = false;
        expected_seq_ = sequence;
        emit_message(message);
        ++expected_seq_;
        return;
    }

    ReorderSlot& slot = reorder_buffer_[sequence & (kReorderWindow - 1)];
    if (slot.occupied && slot.seq_num == sequence) {
        ++duplicate_count_;
        return;
    }
    if (slot.occupied) {
        ++permanent_gap_count_;
        clear_reorder_buffer();
        gap_active_ = false;
        expected_seq_ = sequence;
        emit_message(message);
        ++expected_seq_;
        return;
    }
    slot.message = message;
    slot.seq_num = sequence;
    slot.occupied = true;
    ++reorder_size_;
}

void Decoder::emit_message(const SequencedMessage& message) {
    if (!message.valid) {
        ++malformed_count_;
        return;
    }
    if (message.emits_update && on_update_) on_update_(message.update);
}

void Decoder::drain_reorder_buffer() {
    while (reorder_size_ != 0) {
        ReorderSlot& slot = reorder_buffer_[expected_seq_ & (kReorderWindow - 1)];
        if (!slot.occupied || slot.seq_num != expected_seq_) break;
        emit_message(slot.message);
        slot.occupied = false;
        --reorder_size_;
        ++expected_seq_;
    }
    if (gap_active_ && reorder_size_ == 0) {
        ++recovered_gap_count_;
        gap_active_ = false;
    }
}

void Decoder::clear_reorder_buffer() {
    for (ReorderSlot& slot : reorder_buffer_) slot.occupied = false;
    reorder_size_ = 0;
}

void Decoder::compact_buffer() {
    const size_t remaining = buffered_bytes();
    if (remaining != 0) {
        std::memmove(buffer_.data(), buffer_.data() + read_offset_, remaining);
    }
    read_offset_ = 0;
    write_offset_ = remaining;
}

void Decoder::reset_sequence() {
    read_offset_ = write_offset_ = 0;
    expected_seq_ = 0;
    sequence_initialized_ = false;
    gap_active_ = false;
    clear_reorder_buffer();
}

}  // namespace te

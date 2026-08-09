#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "feed/decoder.h"
#include "feed/feed_handler.h"
#include "feed/protocol.h"

using namespace te;

static int g_failures = 0;
#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s @ %d\n", #cond, __LINE__);       \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

QuoteMsg make_quote(uint32_t seq, double bid, double ask) {
    QuoteMsg message{};
    message.header.type = static_cast<uint8_t>(MsgType::QUOTE);
    message.header.length = host_to_be16(static_cast<uint16_t>(sizeof(message)));
    message.header.seq_num = host_to_be32(seq);
    message.header.timestamp = host_to_be64(static_cast<uint64_t>(seq) * 1000);
    std::memcpy(message.symbol, "AAPL", 4);
    message.bid_price = host_to_be64(from_price(bid));
    message.ask_price = host_to_be64(from_price(ask));
    message.bid_size = host_to_be32(100);
    message.ask_size = host_to_be32(200);
    return message;
}

void append(std::vector<uint8_t>& bytes, const QuoteMsg& message) {
    const auto* begin = reinterpret_cast<const uint8_t*>(&message);
    bytes.insert(bytes.end(), begin, begin + sizeof(message));
}

void test_sticky_and_partial_packets() {
    Decoder decoder;
    std::vector<MarketUpdate> updates;
    decoder.set_on_update([&](const MarketUpdate& update) { updates.push_back(update); });

    const QuoteMsg first = make_quote(1, 101.25, 101.50);
    const QuoteMsg second = make_quote(2, 102.25, 102.50);
    std::vector<uint8_t> bytes;
    append(bytes, first);
    append(bytes, second);

    const size_t split = sizeof(QuoteMsg) / 2;
    CHECK(decoder.feed(bytes.data(), split) == split);
    CHECK(updates.empty());
    CHECK(decoder.buffered_bytes() == split);
    CHECK(decoder.feed(bytes.data() + split, bytes.size() - split) == bytes.size() - split);
    CHECK(updates.size() == 2);
    CHECK(updates[0].seq_num == 1);
    CHECK(std::fabs(updates[0].bid - 101.25) < 1e-9);
    CHECK(updates[1].seq_num == 2);
}

void test_reorder_and_duplicate() {
    Decoder decoder;
    std::vector<uint32_t> sequences;
    decoder.set_on_update([&](const MarketUpdate& update) { sequences.push_back(update.seq_num); });

    const QuoteMsg one = make_quote(10, 10.0, 10.1);
    const QuoteMsg three = make_quote(12, 12.0, 12.1);
    const QuoteMsg two = make_quote(11, 11.0, 11.1);
    decoder.feed(reinterpret_cast<const uint8_t*>(&one), sizeof(one));
    decoder.feed(reinterpret_cast<const uint8_t*>(&three), sizeof(three));
    CHECK(sequences.size() == 1);
    CHECK(decoder.gap_count() == 1);
    decoder.feed(reinterpret_cast<const uint8_t*>(&two), sizeof(two));
    CHECK((sequences == std::vector<uint32_t>{10, 11, 12}));
    CHECK(decoder.recovered_gap_count() == 1);
    CHECK(!decoder.has_unresolved_gap());
    decoder.feed(reinterpret_cast<const uint8_t*>(&two), sizeof(two));
    CHECK(decoder.duplicate_count() == 1);
}

void test_permanent_gap_and_sequence_wrap() {
    Decoder decoder;
    std::vector<uint32_t> sequences;
    decoder.set_on_update([&](const MarketUpdate& update) { sequences.push_back(update.seq_num); });
    const QuoteMsg one = make_quote(1, 10.0, 10.1);
    const QuoteMsg far = make_quote(100, 11.0, 11.1);
    decoder.feed(reinterpret_cast<const uint8_t*>(&one), sizeof(one));
    decoder.feed(reinterpret_cast<const uint8_t*>(&far), sizeof(far));
    CHECK(decoder.gap_count() == 1);
    CHECK(decoder.permanent_gap_count() == 1);
    CHECK(!decoder.has_unresolved_gap());

    decoder.reset_sequence();
    const QuoteMsg maximum = make_quote(std::numeric_limits<uint32_t>::max(), 12.0, 12.1);
    const QuoteMsg zero = make_quote(0, 13.0, 13.1);
    decoder.feed(reinterpret_cast<const uint8_t*>(&maximum), sizeof(maximum));
    decoder.feed(reinterpret_cast<const uint8_t*>(&zero), sizeof(zero));
    CHECK(sequences[sequences.size() - 2] == std::numeric_limits<uint32_t>::max());
    CHECK(sequences.back() == 0);
}

void test_feed_backpressure_protection() {
    const QuoteMsg one = make_quote(1, 10.0, 10.1);
    const QuoteMsg two = make_quote(2, 11.0, 11.1);

    FeedHandler fail_fast(2, BackpressurePolicy::FAIL_FAST);
    fail_fast.ingest_bytes(reinterpret_cast<const uint8_t*>(&one), sizeof(one));
    fail_fast.ingest_bytes(reinterpret_cast<const uint8_t*>(&two), sizeof(two));
    CHECK(fail_fast.state() == FeedState::DEGRADED);
    CHECK(!fail_fast.market_data_trusted());
    CHECK(fail_fast.queue_drops() == 1);
    CHECK(fail_fast.queue_high_watermark() == 1);
    CHECK(fail_fast.max_consecutive_full_count() == 1);

    FeedHandler resync(2, BackpressurePolicy::DROP_AND_RESYNC);
    resync.ingest_bytes(reinterpret_cast<const uint8_t*>(&one), sizeof(one));
    resync.ingest_bytes(reinterpret_cast<const uint8_t*>(&two), sizeof(two));
    CHECK(resync.state() == FeedState::RESYNC_REQUIRED);
    resync.mark_resynchronized();
    CHECK(resync.state() == FeedState::RUNNING);
    CHECK(resync.market_data_trusted());
    CHECK(resync.queue_depth() == 0);
}

int main() {
    test_sticky_and_partial_packets();
    test_reorder_and_duplicate();
    test_permanent_gap_and_sequence_wrap();
    test_feed_backpressure_protection();
    if (g_failures == 0) {
        std::printf("test_decoder: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_decoder: %d failure(s)\n", g_failures);
    return 1;
}

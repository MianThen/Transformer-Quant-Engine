#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#include "adapter/mock_market_data_adapter.h"
#include "engine_common/replay.h"
#include "feed/feed_handler.h"
#include "feed/protocol.h"

using namespace te;

static QuoteMsg make_quote(uint32_t sequence, double bid, double ask) {
    QuoteMsg message{};
    message.header.type = static_cast<uint8_t>(MsgType::QUOTE);
    message.header.length = host_to_be16(static_cast<uint16_t>(sizeof(message)));
    message.header.seq_num = host_to_be32(sequence);
    message.header.timestamp = host_to_be64(static_cast<uint64_t>(sequence) * 1000);
    std::memcpy(message.symbol, "AAPL", 4);
    message.bid_price = host_to_be64(from_price(bid));
    message.ask_price = host_to_be64(from_price(ask));
    message.bid_size = host_to_be32(100);
    message.ask_size = host_to_be32(200);
    return message;
}

int main() {
    const auto path = std::filesystem::temp_directory_path() / "qbt-feed-capture.bin";
    std::filesystem::remove(path);
    const QuoteMsg first = make_quote(1, 10.0, 10.1);
    const QuoteMsg second = make_quote(2, 11.0, 11.1);
    std::vector<uint8_t> bytes(sizeof(first) + sizeof(second));
    std::memcpy(bytes.data(), &first, sizeof(first));
    std::memcpy(bytes.data() + sizeof(first), &second, sizeof(second));
    uint64_t live_checksum = 0;
    {
        FeedHandler feed(8);
        feed.enable_capture(path.string());
        if (feed.ingest_bytes(bytes.data(), bytes.size()) != bytes.size()) return 1;
        engine_common::MarketEvent update{};
        while (feed.try_pop(update)) live_checksum += update.sequence;
    }

    uint64_t replay_checksum = 0;
    uint64_t replay_count = 0;
    {
        engine_common::ReplayReader reader(path.string());
        MockMarketDataAdapter adapter;
        adapter.set_on_event([&](const engine_common::MarketEvent& update) {
            replay_checksum += update.sequence;
            ++replay_count;
        });
        engine_common::ReplayRecord record;
        while (reader.next(record)) {
            if (record.header.type == engine_common::ReplayRecordType::MARKET_BYTES) {
                adapter.feed(reinterpret_cast<const uint8_t*>(record.payload.data()),
                             record.payload.size());
            }
        }
    }
    std::filesystem::remove(path);
    if (live_checksum != 3 || replay_checksum != live_checksum || replay_count != 2) return 1;
    std::printf("test_replay_feed: all checks passed\n");
    return 0;
}

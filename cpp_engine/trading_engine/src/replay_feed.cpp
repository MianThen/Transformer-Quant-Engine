#include <cstdio>
#include <string>

#include "adapter/mock_market_data_adapter.h"
#include "engine_common/replay.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: replay_feed <capture.bin>\n");
        return 2;
    }
    try {
        engine_common::ReplayReader reader(argv[1]);
        te::MockMarketDataAdapter adapter;
        uint64_t updates = 0;
        uint64_t checksum = 0;
        adapter.set_on_event([&](const engine_common::MarketEvent& update) {
            ++updates;
            checksum += update.sequence;
        });
        engine_common::ReplayRecord record;
        while (reader.next(record)) {
            if (record.header.type != engine_common::ReplayRecordType::MARKET_BYTES) continue;
            adapter.feed(reinterpret_cast<const uint8_t*>(record.payload.data()),
                         record.payload.size());
        }
        std::printf("updates=%llu checksum=%llu gaps=%llu malformed=%llu\n",
                    static_cast<unsigned long long>(updates),
                    static_cast<unsigned long long>(checksum),
                    static_cast<unsigned long long>(adapter.gap_count()),
                    static_cast<unsigned long long>(adapter.malformed_count()));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}

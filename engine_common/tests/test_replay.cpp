#include <cstdio>
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "engine_common/replay.h"

int main() {
    const auto process_id =
#ifdef _WIN32
        _getpid();
#else
        getpid();
#endif
    const auto path = std::filesystem::temp_directory_path() /
        ("qbt-replay-test-" + std::to_string(process_id) + ".bin");
    std::filesystem::remove(path);
    const std::array<std::byte, 3> payload{std::byte{1}, std::byte{2}, std::byte{3}};
    {
        engine_common::ReplayWriter writer(path.string());
        writer.append(engine_common::ReplayRecordType::MARKET_BYTES, 42, payload);
    }
    bool valid = false;
    {
        engine_common::ReplayReader reader(path.string());
        engine_common::ReplayRecord record;
        valid = reader.next(record) && record.header.timestamp_ns == 42 &&
                record.payload.size() == payload.size() && !reader.next(record);
    }
    std::filesystem::remove(path);
    if (!valid) return 1;
    std::printf("test_replay: all checks passed\n");
    return 0;
}

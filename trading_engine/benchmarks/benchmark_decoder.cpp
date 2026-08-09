#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <new>
#include <string>
#include <vector>

#include "feed/decoder.h"
#ifdef TE_BENCHMARK_LEGACY
#include "legacy_decoder.h"
#endif

namespace {
std::atomic<bool> track_allocations{false};
std::atomic<std::uint64_t> allocation_count{0};
}

void* operator new(std::size_t size) {
    if (track_allocations.load(std::memory_order_relaxed))
        allocation_count.fetch_add(1, std::memory_order_relaxed);
    if (void* memory = std::malloc(size)) return memory;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

namespace {
using namespace te;
using Clock = std::chrono::steady_clock;
#ifdef TE_BENCHMARK_LEGACY
using BenchmarkDecoder = LegacyDecoder;
#else
using BenchmarkDecoder = Decoder;
#endif

struct Result {
    std::string name;
    std::uint64_t operations;
    std::uint64_t elapsed_ns;
    std::uint64_t checksum;
    std::uint64_t allocations;
};

QuoteMsg quote(uint32_t sequence) {
    QuoteMsg message{};
    message.header.type = static_cast<uint8_t>(MsgType::QUOTE);
    message.header.length = host_to_be16(sizeof(QuoteMsg));
    message.header.seq_num = host_to_be32(sequence);
    message.header.timestamp = host_to_be64(static_cast<uint64_t>(sequence) * 1000);
    std::memcpy(message.symbol, "AAPL", 4);
    message.bid_price = host_to_be64(from_price(100.0));
    message.ask_price = host_to_be64(from_price(100.01));
    message.bid_size = host_to_be32(100);
    message.ask_size = host_to_be32(200);
    return message;
}

template <class Function>
Result timed(std::string name, std::uint64_t operations, Function function) {
    allocation_count.store(0, std::memory_order_relaxed);
    track_allocations.store(true, std::memory_order_release);
    const auto start = Clock::now();
    const auto checksum = function();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - start).count();
    track_allocations.store(false, std::memory_order_release);
    return {std::move(name), operations, static_cast<std::uint64_t>(elapsed), checksum,
            allocation_count.load(std::memory_order_relaxed)};
}

Result contiguous(bool quick) {
    const std::size_t count = quick ? 100'000 : 1'000'000;
    std::vector<QuoteMsg> messages(count);
    for (std::size_t index = 0; index < count; ++index) messages[index] = quote(index + 1);
    BenchmarkDecoder decoder;
    std::uint64_t checksum = 0;
    decoder.set_on_update([&](const MarketUpdate& update) { checksum += update.seq_num; });
    return timed("decoder_contiguous", count, [&] {
        const auto* bytes = reinterpret_cast<const uint8_t*>(messages.data());
        const size_t total = messages.size() * sizeof(QuoteMsg);
        for (size_t offset = 0; offset < total; offset += 4096) {
            decoder.feed(bytes + offset, std::min<size_t>(4096, total - offset));
        }
        return checksum;
    });
}

Result partial(bool quick) {
    const std::size_t count = quick ? 20'000 : 250'000;
    BenchmarkDecoder decoder;
    std::uint64_t checksum = 0;
    decoder.set_on_update([&](const MarketUpdate& update) { checksum += update.seq_num; });
    return timed("decoder_partial_messages", count, [&] {
        for (std::size_t index = 0; index < count; ++index) {
            const QuoteMsg message = quote(static_cast<uint32_t>(index + 1));
            const auto* bytes = reinterpret_cast<const uint8_t*>(&message);
            decoder.feed(bytes, 7);
            decoder.feed(bytes + 7, sizeof(message) - 7);
        }
        return checksum;
    });
}

Result reordered(bool quick) {
    const std::size_t groups = quick ? 20'000 : 200'000;
    BenchmarkDecoder decoder;
    std::uint64_t checksum = 0;
    decoder.set_on_update([&](const MarketUpdate& update) { checksum += update.seq_num; });
    return timed("decoder_reorder_recovery", groups * 3, [&] {
        for (std::size_t group = 0; group < groups; ++group) {
            const uint32_t base = static_cast<uint32_t>(group * 3 + 1);
            const QuoteMsg first = quote(base);
            const QuoteMsg third = quote(base + 2);
            const QuoteMsg second = quote(base + 1);
            decoder.feed(reinterpret_cast<const uint8_t*>(&first), sizeof(first));
            decoder.feed(reinterpret_cast<const uint8_t*>(&third), sizeof(third));
            decoder.feed(reinterpret_cast<const uint8_t*>(&second), sizeof(second));
        }
        return checksum + decoder.recovered_gap_count();
    });
}

void output(const std::vector<Result>& results, bool quick) {
    std::cout << std::setprecision(17)
              << "{\"schema_version\":1,\"dataset\":{\"name\":\"te-decoder-v1\","
              << "\"version\":1,\"seed\":0,\"quick\":" << (quick ? "true" : "false") << "},"
              << "\"build\":{\"type\":\"" << QBT_BUILD_TYPE << "\",\"compiler_id\":\""
              << QBT_COMPILER_ID << "\",\"compiler_version\":\"" << QBT_COMPILER_VERSION
              << "\",\"lto\":" << (QBT_LTO_ENABLED ? "true" : "false") << "},\"results\":[";
    for (std::size_t index = 0; index < results.size(); ++index) {
        if (index) std::cout << ',';
        const Result& result = results[index];
        std::cout << "{\"name\":\"" << result.name << "\",\"category\":\"feed\",\"scale\":"
                  << result.operations << ",\"operations\":" << result.operations
                  << ",\"elapsed_ns\":" << result.elapsed_ns << ",\"ns_per_operation\":"
                  << static_cast<double>(result.elapsed_ns) / result.operations
                  << ",\"checksum\":" << result.checksum
                  << ",\"allocations\":" << result.allocations << '}';
    }
    std::cout << "]}\n";
}
}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
    output({contiguous(quick), partial(quick), reordered(quick)}, quick);
}

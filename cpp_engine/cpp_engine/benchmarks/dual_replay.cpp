#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <unordered_map>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "engine.h"
#include "engine_common/normalized_replay.h"
#include "engine_common/replay.h"

namespace {
using namespace qbt;
using namespace engine_common;

struct Capture {
    std::vector<NormalizedMarketRecord> markets;
    std::vector<NormalizedOrderRecord> orders;
    std::vector<NormalizedExecutionRecord> executions;
};

Capture read_capture(const std::string& path) {
    ReplayReader reader(path);
    ReplayRecord record;
    Capture capture;
    while (reader.next(record)) {
        switch (record.header.type) {
            case ReplayRecordType::MARKET_BYTES:
                capture.markets.push_back(
                    normalized_record<NormalizedMarketRecord>(record.payload));
                break;
            case ReplayRecordType::ORDER_INTENT:
                capture.orders.push_back(
                    normalized_record<NormalizedOrderRecord>(record.payload));
                break;
            case ReplayRecordType::EXECUTION:
                capture.executions.push_back(
                    normalized_record<NormalizedExecutionRecord>(record.payload));
                break;
            case ReplayRecordType::METRIC:
            case ReplayRecordType::NORMALIZED_MARKET_EVENT:
            case ReplayRecordType::NORMALIZED_BAR:
            case ReplayRecordType::FEATURE_BATCH:
            case ReplayRecordType::PREDICTION_BATCH:
            case ReplayRecordType::TARGET_POSITION_BATCH:
            case ReplayRecordType::RISK_DECISION:
            case ReplayRecordType::MODEL_LIFECYCLE:
                break;
        }
    }
    return capture;
}

bool verify(const Capture& capture, std::string& error) {
    if (capture.markets.empty()) {
        error = "capture has no market records";
        return false;
    }
    std::map<Timestamp, std::vector<Order>> intents;
    for (const auto& source : capture.orders) {
        Order order;
        order.id = source.order_id;
        order.symbol = normalized_symbol(source.symbol);
        order.side = static_cast<qbt::Side>(source.side);
        order.type = static_cast<qbt::OrderType>(source.type);
        order.quantity = source.quantity;
        order.limit_price = static_cast<Price>(source.limit_price_ticks) / 10'000.0;
        order.timestamp = source.timestamp;
        intents[source.timestamp].push_back(std::move(order));
    }
    BacktestEngine engine(1'000'000.0, FillTiming::CLOSE);
    Timestamp callback_timestamp = 0;
    engine.set_on_market_data([&](const MarketSnapshot& market) {
        callback_timestamp = market.timestamp;
        auto found = intents.find(callback_timestamp);
        return found == intents.end() ? std::vector<Order>{} : found->second;
    });
    for (const auto& source : capture.markets) {
        MarketSnapshot market;
        market.symbol = normalized_symbol(source.symbol);
        market.timestamp = source.timestamp;
        market.open = source.open_ticks / 10'000.0;
        market.high = source.high_ticks / 10'000.0;
        market.low = source.low_ticks / 10'000.0;
        market.close = source.close_ticks / 10'000.0;
        market.volume = source.volume;
        engine.process_market_data(market);
    }
    std::unordered_map<int64_t, NormalizedExecutionRecord> expected;
    for (const auto& execution : capture.executions) expected[execution.order_id] = execution;
    for (const auto& actual : engine.get_order_history()) {
        auto found = expected.find(actual.order.id);
        if (found == expected.end()) continue;
        if (actual.filled_quantity != found->second.cumulative_quantity) {
            error = "filled quantity mismatch for order " + std::to_string(actual.order.id);
            return false;
        }
        const auto expected_status = static_cast<OrderStatus>(found->second.status);
        if (actual.status != expected_status) {
            error = "status mismatch for order " + std::to_string(actual.order.id);
            return false;
        }
    }
    return true;
}

bool self_test(const std::string& path) {
    std::filesystem::remove(path);
    NormalizedMarketRecord market;
    std::memcpy(market.symbol.data(), "AAA", 3);
    market.timestamp = 1;
    market.open_ticks = market.high_ticks = market.low_ticks = market.close_ticks = 1'000'000;
    market.volume = 100;
    NormalizedOrderRecord order;
    order.order_id = 1;
    std::memcpy(order.symbol.data(), "AAA", 3);
    order.side = static_cast<uint8_t>(qbt::Side::BUY);
    order.type = static_cast<uint8_t>(qbt::OrderType::MARKET);
    order.quantity = 10;
    order.timestamp = 1;
    NormalizedExecutionRecord execution;
    execution.order_id = 1;
    execution.status = static_cast<uint8_t>(OrderStatus::FILLED);
    execution.cumulative_quantity = 10;
    {
        ReplayWriter writer(path);
        writer.append(ReplayRecordType::MARKET_BYTES, 1, normalized_bytes(market));
        writer.append(ReplayRecordType::ORDER_INTENT, 1, normalized_bytes(order));
        writer.append(ReplayRecordType::EXECUTION, 1, normalized_bytes(execution));
    }
    std::string error;
    const bool valid = verify(read_capture(path), error);
    std::filesystem::remove(path);
    if (!valid) std::fprintf(stderr, "%s\n", error.c_str());
    return valid;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--self-test") {
        const auto process_id =
#ifdef _WIN32
            _getpid();
#else
            getpid();
#endif
        const auto path = std::filesystem::temp_directory_path() /
            ("qbt-dual-replay-" + std::to_string(process_id) + ".bin");
        return self_test(path.string())
            ? 0 : 1;
    }
    if (argc != 2) {
        std::fprintf(stderr, "usage: qbt_dual_replay <capture>\n");
        return 2;
    }
    std::string error;
    const bool valid = verify(read_capture(argv[1]), error);
    std::cout << "{\"valid\":" << (valid ? "true" : "false")
              << ",\"error\":\"" << error << "\"}\n";
    return valid ? 0 : 1;
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "feed/decoder.h"
#include "oms/order_gateway.h"
#include "runtime/runtime.h"

namespace {
using namespace te;
using Clock = std::chrono::steady_clock;

class BenchmarkTransport final : public ITransport {
public:
    bool connect(const std::string&, uint16_t) override {
        ++connect_calls_;
        open_ = connect_calls_ > failed_connects_;
        status_ = open_ ? IoStatus::OK : IoStatus::FATAL_ERROR;
        return open_;
    }
    ssize_t receive(void*, size_t) override {
        status_ = IoStatus::WOULD_BLOCK;
        return 0;
    }
    ssize_t transmit(const void*, size_t size) override {
        ++transmit_calls_;
        if (!open_) return -1;
        if (block_every_ != 0 && transmit_calls_ % block_every_ == 0) {
            status_ = IoStatus::WOULD_BLOCK;
            return 0;
        }
        status_ = IoStatus::OK;
        const size_t sent = max_chunk_ == 0 ? size : std::min(size, max_chunk_);
        bytes_sent_ += sent;
        return static_cast<ssize_t>(sent);
    }
    bool wait_readable(int) override { return false; }
    bool wait_writable(int) override { return open_; }
    bool is_open() const override { return open_; }
    IoStatus last_status() const override { return status_; }
    int last_error() const override { return 0; }
    intptr_t native_handle() const override { return -1; }
    void close() override { open_ = false; }

    size_t max_chunk_ = 0;
    size_t block_every_ = 0;
    size_t failed_connects_ = 0;
    size_t bytes_sent_ = 0;
    size_t transmit_calls_ = 0;
    size_t connect_calls_ = 0;

private:
    bool open_ = false;
    IoStatus status_ = IoStatus::OK;
};

struct Result {
    std::string name;
    uint64_t events = 0;
    uint64_t elapsed_ns = 0;
    uint64_t p50_ns = 0;
    uint64_t p99_ns = 0;
    uint64_t p999_ns = 0;
    uint64_t max_ns = 0;
    uint64_t rss_bytes = 0;
    uint64_t queue_high_watermark = 0;
    uint64_t degraded = 0;
    uint64_t checksum = 0;
};

QuoteMsg quote(uint32_t sequence) {
    QuoteMsg message{};
    message.header.type = static_cast<uint8_t>(MsgType::QUOTE);
    message.header.length = host_to_be16(sizeof(message));
    message.header.seq_num = host_to_be32(sequence);
    message.header.timestamp = host_to_be64(static_cast<uint64_t>(sequence) * 1'000);
    std::memcpy(message.symbol, "AAPL", 4);
    message.bid_price = host_to_be64(from_price(100.0));
    message.ask_price = host_to_be64(from_price(100.01));
    message.bid_size = host_to_be32(100);
    message.ask_size = host_to_be32(100);
    return message;
}

uint64_t percentile(std::vector<uint64_t> values, double probability) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const size_t index = std::min(
        values.size() - 1,
        static_cast<size_t>(std::ceil(values.size() * probability)) - 1);
    return values[index];
}

Result pipeline_case(std::string name, size_t count, bool partial,
                     size_t max_chunk, size_t block_every) {
    auto transport = std::make_unique<BenchmarkTransport>();
    BenchmarkTransport* raw_transport = transport.get();
    raw_transport->max_chunk_ = max_chunk;
    raw_transport->block_every_ = block_every;
    GatewayConfig config;
    config.send_batch_bytes = 4096;
    config.flush_on_enqueue = false;
    OrderGateway gateway(config, std::move(transport));
    gateway.connect("benchmark", 1);
    Decoder decoder;
    std::vector<uint64_t> latencies;
    latencies.reserve(count);
    int64_t next_order_id = 1;
    uint64_t checksum = 0;
    decoder.set_on_update([&](const MarketUpdate& update) {
        const auto start = Clock::now();
        NewOrderRequest request;
        request.client_order_id = next_order_id++;
        request.symbol.assign(update.symbol,
                              std::find(update.symbol, update.symbol + 8, '\0'));
        request.quantity = 1;
        request.price = from_price((update.bid + update.ask) * 0.5);
        const int64_t order_id = gateway.send_order(request);
        if (order_id > 0) {
            gateway.process_report({order_id, order_id, OrderStatus::FILLED,
                                    1, 1, request.price,
                                    engine_common::RejectReason::NONE});
            checksum += static_cast<uint64_t>(order_id);
        }
        latencies.push_back(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - start).count()));
    });
    const auto started = Clock::now();
    for (size_t index = 0; index < count; ++index) {
        const QuoteMsg message = quote(static_cast<uint32_t>(index + 1));
        const auto* bytes = reinterpret_cast<const uint8_t*>(&message);
        if (partial) {
            decoder.feed(bytes, 7);
            decoder.feed(bytes + 7, sizeof(message) - 7);
        } else {
            decoder.feed(bytes, sizeof(message));
        }
    }
    gateway.flush();
    const uint64_t elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - started).count());
    const ProcessMetrics metrics = process_metrics();
    return {std::move(name), count, elapsed, percentile(latencies, 0.50),
            percentile(latencies, 0.99), percentile(latencies, 0.999),
            latencies.empty() ? 0 : *std::max_element(latencies.begin(), latencies.end()),
            metrics.resident_bytes, gateway.send_queue_high_watermark(),
            gateway.state() == GatewayState::DEGRADED ? 1U : 0U, checksum};
}

Result reordered_ack_case(size_t count) {
    auto transport = std::make_unique<BenchmarkTransport>();
    GatewayConfig config;
    config.send_batch_bytes = 4096;
    OrderGateway gateway(config, std::move(transport));
    gateway.connect("benchmark", 1);
    std::vector<int64_t> order_ids;
    order_ids.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        NewOrderRequest request;
        request.symbol = "AAPL";
        request.quantity = 1;
        request.price = from_price(100.0);
        order_ids.push_back(gateway.send_order(request));
    }
    const auto started = Clock::now();
    for (auto iterator = order_ids.rbegin(); iterator != order_ids.rend(); ++iterator) {
        gateway.process_report({*iterator, *iterator, OrderStatus::FILLED, 1, 1,
                                from_price(100.0),
                                engine_common::RejectReason::NONE});
    }
    gateway.flush();
    const uint64_t elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - started).count());
    const ProcessMetrics metrics = process_metrics();
    return {"ack_out_of_order", count, elapsed, 0, 0, 0, 0,
            metrics.resident_bytes, gateway.send_queue_high_watermark(), 0,
            gateway.archived_order_count()};
}

Result wal_recovery_case(size_t count) {
    const auto path = std::filesystem::temp_directory_path() / "qbt-e2e-benchmark.wal";
    std::filesystem::remove(path);
    GatewayConfig config;
    config.wal_path = path.string();
    config.wal_durability = WalDurability::GROUP_COMMIT;
    config.wal_group_size = 64;
    {
        auto transport = std::make_unique<BenchmarkTransport>();
        OrderGateway gateway(config, std::move(transport));
        gateway.connect("benchmark", 1);
        for (size_t index = 0; index < count; ++index) {
            NewOrderRequest request;
            request.symbol = "AAPL";
            request.quantity = 1;
            request.price = from_price(100.0);
            gateway.send_order(request);
        }
        gateway.flush();
    }
    const auto started = Clock::now();
    uint64_t checksum = 0;
    {
        OrderGateway recovered(config);
        checksum = recovered.wal_recovered_orders();
    }
    const uint64_t elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - started).count());
    const ProcessMetrics metrics = process_metrics();
    std::filesystem::remove(path);
    return {"wal_recovery", count, elapsed, 0, 0, 0, 0,
            metrics.resident_bytes, 0, 0, checksum};
}

Result order_burst_case(size_t count, bool wal_enabled) {
    const auto path = std::filesystem::temp_directory_path() / "qbt-order-burst.wal";
    std::filesystem::remove(path);
    GatewayConfig config;
    config.send_batch_bytes = 4096;
    if (wal_enabled) {
        config.wal_path = path.string();
        config.wal_durability = WalDurability::ASYNC;
    }
    Result result;
    {
        auto transport = std::make_unique<BenchmarkTransport>();
        OrderGateway gateway(config, std::move(transport));
        gateway.connect("benchmark", 1);
        const auto started = Clock::now();
        uint64_t checksum = 0;
        for (size_t index = 0; index < count; ++index) {
            NewOrderRequest request;
            request.symbol = "AAPL";
            request.quantity = 1;
            request.price = from_price(100.0);
            const int64_t id = gateway.send_order(request);
            if (id > 0) checksum += static_cast<uint64_t>(id);
        }
        gateway.poll_once();
        const uint64_t elapsed = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                Clock::now() - started).count());
        const ProcessMetrics metrics = process_metrics();
        result = {wal_enabled ? "order_burst_wal_async" : "order_burst_wal_off",
                  count, elapsed, 0, 0, 0, 0, metrics.resident_bytes,
                  gateway.send_queue_high_watermark(), 0, checksum};
        gateway.flush_wal();
    }
    std::error_code error;
    if (wal_enabled) std::filesystem::remove(path, error);
    return result;
}

Result reconnect_case() {
    auto transport = std::make_unique<BenchmarkTransport>();
    BenchmarkTransport* raw_transport = transport.get();
    raw_transport->failed_connects_ = 2;
    GatewayConfig config;
    config.max_retries = 4;
    config.sleep_for = [](std::chrono::milliseconds) {};
    OrderGateway gateway(config, std::move(transport));
    gateway.connect("benchmark", 1);
    const auto started = Clock::now();
    const bool recovered = gateway.reconnect();
    const uint64_t elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - started).count());
    return {"reconnect_storm", 3, elapsed, 0, 0, 0, 0,
            process_metrics().resident_bytes, 0, recovered ? 0U : 1U,
            gateway.reconnect_count()};
}

void print(const std::vector<Result>& results, bool quick) {
    std::cout << "{\"schema_version\":1,\"quick\":" << (quick ? "true" : "false")
              << ",\"results\":[";
    for (size_t index = 0; index < results.size(); ++index) {
        if (index != 0) std::cout << ',';
        const Result& result = results[index];
        const double throughput = result.elapsed_ns == 0 ? 0.0 :
            static_cast<double>(result.events) * 1e9 / result.elapsed_ns;
        std::cout << "{\"name\":\"" << result.name << "\",\"events\":"
                  << result.events << ",\"elapsed_ns\":" << result.elapsed_ns
                  << ",\"throughput_per_second\":" << throughput
                  << ",\"p50_ns\":" << result.p50_ns << ",\"p99_ns\":"
                  << result.p99_ns << ",\"p999_ns\":" << result.p999_ns
                  << ",\"max_pause_ns\":" << result.max_ns
                  << ",\"rss_bytes\":" << result.rss_bytes
                  << ",\"queue_high_watermark\":" << result.queue_high_watermark
                  << ",\"degraded_count\":" << result.degraded
                  << ",\"checksum\":" << result.checksum << '}';
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
    const size_t count = quick ? 2'000 : 100'000;
    std::vector<Result> results;
    results.push_back(pipeline_case("market_burst", count, false, 0, 0));
    results.push_back(pipeline_case("partial_packets", count, true, 7, 0));
    results.push_back(pipeline_case("send_backpressure", count, false, 17, 11));
    results.push_back(reordered_ack_case(quick ? 1'000 : 50'000));
    results.push_back(order_burst_case(quick ? 2'000 : 100'000, false));
    results.push_back(order_burst_case(quick ? 2'000 : 100'000, true));
    results.push_back(wal_recovery_case(quick ? 500 : 20'000));
    results.push_back(reconnect_case());
    print(results, quick);
    return 0;
}

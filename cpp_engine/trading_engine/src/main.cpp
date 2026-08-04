#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "adapter/mock_order_adapter.h"
#include "feed/feed_handler.h"
#include "feed/protocol.h"
#include "net/latency.h"
#include "runtime/runtime.h"

using namespace te;

namespace {

void print_usage(const char* program) {
    std::printf(
        "usage: %s <feed_host> <feed_port> <gateway_host> <gateway_port> "
        "[duration_seconds] [--send-demo-order] [--mode=low-latency|balanced|power-save] "
        "[--feed-cpu=N] [--core-cpu=N] [--realtime]\n",
        program);
}

bool parse_port(const char* text, uint16_t& port) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0 || value > 65535) return false;
    port = static_cast<uint16_t>(value);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        print_usage(argv[0]);
        return argc == 1 ? 0 : 1;
    }

    uint16_t feed_port = 0;
    uint16_t gateway_port = 0;
    if (!parse_port(argv[2], feed_port) || !parse_port(argv[4], gateway_port)) {
        std::fprintf(stderr, "invalid TCP port\n");
        return 1;
    }

    int duration_seconds = 10;
    if (argc >= 6 && std::strcmp(argv[5], "--send-demo-order") != 0) {
        duration_seconds = std::max(1, std::atoi(argv[5]));
    }
    const bool send_demo_order =
        (argc >= 6 && std::strcmp(argv[5], "--send-demo-order") == 0) ||
        (argc >= 7 && std::strcmp(argv[6], "--send-demo-order") == 0);
    RunMode run_mode = RunMode::BALANCED;
    int feed_cpu = -1;
    int core_cpu = -1;
    bool realtime = false;
    for (int index = 5; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument.starts_with("--mode=")) {
            if (!parse_run_mode(argument.substr(7), run_mode)) {
                std::fprintf(stderr, "invalid run mode\n");
                return 1;
            }
        } else if (argument.starts_with("--feed-cpu=")) {
            feed_cpu = std::atoi(argv[index] + 11);
        } else if (argument.starts_with("--core-cpu=")) {
            core_cpu = std::atoi(argv[index] + 11);
        } else if (argument == "--realtime") {
            realtime = true;
        }
    }

    FeedHandler feed;
    MockOrderAdapter orders([&feed](engine_common::SymbolId symbol_id) {
        return std::string(feed.symbol(symbol_id));
    });
    OrderGateway& gateway = orders.gateway();
    LatencyRecorder latency;
    LatencyRecorder queue_latency;
    LatencyRecorder end_to_end_latency;
    std::atomic<bool> running{true};
    std::atomic<uint64_t> reports{0};
    ThreadHealth feed_health;
    ThreadHealth core_health;

    orders.set_on_execution([&](const engine_common::ExecutionEvent& report) {
        reports.fetch_add(1, std::memory_order_relaxed);
        std::printf("exec report: order=%lld status=%u fill=%lld\n",
                    static_cast<long long>(report.client_order_id),
                    static_cast<unsigned>(report.status),
                    static_cast<long long>(report.last_quantity));
    });

    if (!feed.connect(argv[1], feed_port)) {
        std::fprintf(stderr, "failed to connect feed %s:%u\n", argv[1], feed_port);
        return 1;
    }
    if (!orders.connect(argv[3], gateway_port)) {
        std::fprintf(stderr, "failed to connect gateway %s:%u\n", argv[3], gateway_port);
        feed.stop();
        return 1;
    }

    std::thread feed_thread([&] {
        feed_health.started();
        if (feed_cpu >= 0) pin_current_thread(static_cast<unsigned>(feed_cpu));
        set_current_thread_realtime(realtime && run_mode == RunMode::LOW_LATENCY);
        uint32_t idle_count = 0;
        while (running.load(std::memory_order_acquire) && feed.running()) {
            feed_health.beat();
            if (feed.poll_once() == 0) idle_wait(run_mode, ++idle_count);
            else idle_count = 0;
        }
        feed_health.stopped();
    });

    core_health.started();
    if (core_cpu >= 0) pin_current_thread(static_cast<unsigned>(core_cpu));
    set_current_thread_realtime(realtime && run_mode == RunMode::LOW_LATENCY);

    bool demo_order_sent = false;
    uint64_t updates = 0;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(duration_seconds);
    uint32_t idle_count = 0;
    while (std::chrono::steady_clock::now() < deadline && feed.running()) {
        core_health.beat();
        orders.poll();

        engine_common::MarketEvent update{};
        if (!feed.try_pop(update)) {
            idle_wait(run_mode, ++idle_count);
            continue;
        }
        idle_count = 0;

        auto timer = latency.scoped();
        const int64_t processing_started = steady_now_ns();
        if (update.enqueue_timestamp > 0) {
            queue_latency.record(processing_started - update.enqueue_timestamp);
        }
        ++updates;
        if (send_demo_order && !demo_order_sent && update.bid > 0) {
            engine_common::OrderIntent request{};
            request.symbol_id = update.symbol_id;
            request.side = engine_common::Side::BUY;
            request.type = engine_common::OrderType::LIMIT;
            request.quantity = 1;
            request.limit_price = update.bid;
            const int64_t order_id = orders.submit(request);
            demo_order_sent = order_id > 0;
            if (demo_order_sent) {
                std::printf("sent demo order: id=%lld symbol=%s price=%.4f\n",
                            static_cast<long long>(order_id),
                            std::string(feed.symbol(update.symbol_id)).c_str(),
                            to_price(update.bid));
            }
        }
        if (update.enqueue_timestamp > 0) {
            end_to_end_latency.record(steady_now_ns() - update.enqueue_timestamp);
        }
    }

    running.store(false, std::memory_order_release);
    feed.stop();
    feed_thread.join();
    core_health.stopped();

    std::printf("updates=%llu reports=%llu bytes=%llu queue_drops=%llu gaps=%llu duplicates=%llu\n",
                static_cast<unsigned long long>(updates),
                static_cast<unsigned long long>(reports.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(feed.bytes_received()),
                static_cast<unsigned long long>(feed.queue_drops()),
                static_cast<unsigned long long>(feed.sequence_gaps()),
                static_cast<unsigned long long>(feed.duplicate_messages()));
    std::printf("feed_state=%s trusted=%s queue_depth=%zu queue_high_watermark=%zu "
                "consecutive_full=%llu max_consecutive_full=%llu permanent_gaps=%llu "
                "recovered_gaps=%llu last_drop_ns=%lld recovery_ns=%lld\n",
                to_string(feed.state()), feed.market_data_trusted() ? "true" : "false",
                feed.queue_depth(), feed.queue_high_watermark(),
                static_cast<unsigned long long>(feed.consecutive_full_count()),
                static_cast<unsigned long long>(feed.max_consecutive_full_count()),
                static_cast<unsigned long long>(feed.permanent_sequence_gaps()),
                static_cast<unsigned long long>(feed.recovered_sequence_gaps()),
                static_cast<long long>(feed.last_drop_time_ns()),
                static_cast<long long>(feed.last_recovery_duration_ns()));
    std::printf("latency(ns): count=%zu mean=%.2f p50=%lld p99=%lld max=%lld\n",
                latency.count(), latency.mean(),
                static_cast<long long>(latency.percentile(0.50)),
                static_cast<long long>(latency.percentile(0.99)),
                static_cast<long long>(latency.max()));
    const LatencySnapshot decode = feed.decode_latency();
    const LatencySnapshot gateway_send = gateway.send_latency();
    const LatencySnapshot ack = gateway.ack_latency();
    std::printf("phase_latency_p99_ns decode=%lld queue=%lld strategy=%lld "
                "gateway_send=%lld ack_rtt=%lld end_to_end=%lld\n",
                static_cast<long long>(decode.p99),
                static_cast<long long>(queue_latency.percentile(0.99)),
                static_cast<long long>(latency.percentile(0.99)),
                static_cast<long long>(gateway_send.p99),
                static_cast<long long>(ack.p99),
                static_cast<long long>(end_to_end_latency.percentile(0.99)));
    const ProcessMetrics process = process_metrics();
    const int64_t now = steady_now_ns();
    std::printf("runtime_mode=%s cpu_seconds=%.6f rss_bytes=%llu "
                "feed_alive=%s feed_responsive=%s core_alive=%s core_responsive=%s\n",
                to_string(run_mode), process.cpu_seconds,
                static_cast<unsigned long long>(process.resident_bytes),
                feed_health.alive.load() ? "true" : "false",
                feed_health.responsive(now, 5'000'000'000LL) ? "true" : "false",
                core_health.alive.load() ? "true" : "false",
                core_health.responsive(now, 5'000'000'000LL) ? "true" : "false");
    return 0;
}

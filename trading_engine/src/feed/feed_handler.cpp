#include "feed/feed_handler.h"

#include <atomic>
#include <chrono>

#include "adapter/mock_market_data_adapter.h"
#include "transport/tcp_transport.h"

namespace te {

namespace {

int64_t steady_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

FeedHandler::FeedHandler(size_t queue_pow2, BackpressurePolicy policy,
                         size_t spin_limit,
                         std::unique_ptr<ITransport> transport,
                         std::unique_ptr<IMarketDataAdapter> adapter)
    : transport_(transport ? std::move(transport)
                           : std::make_unique<TcpTransport>()),
      adapter_(adapter ? std::move(adapter)
                       : std::make_unique<MockMarketDataAdapter>()),
      queue_(queue_pow2), policy_(policy), spin_limit_(spin_limit) {
    adapter_->set_on_event(
        [this](const engine_common::MarketEvent& event) { on_event(event); });
}

bool FeedHandler::connect(const std::string& host, uint16_t port) {
    if (!transport_->connect(host, port)) return false;
    adapter_->reset();
    consecutive_full_count_.store(0, std::memory_order_relaxed);
    state_.store(FeedState::RUNNING, std::memory_order_release);
    running_.store(true, std::memory_order_release);
    return true;
}

size_t FeedHandler::poll_once() {
    if (!running() || !transport_->is_open()) return 0;
    uint8_t buffer[4096];
    const ssize_t received = transport_->receive(buffer, sizeof(buffer));
    receive_calls_.fetch_add(1, std::memory_order_relaxed);
    if (received < 0) {
        enter_failure_state(FeedState::DEGRADED);
        return 0;
    }
    if (received == 0) return 0;
    bytes_received_.fetch_add(static_cast<uint64_t>(received), std::memory_order_relaxed);
    return ingest_bytes(buffer, static_cast<size_t>(received));
}

size_t FeedHandler::ingest_bytes(const uint8_t* data, size_t len) {
    if (state() == FeedState::STOPPED || state() == FeedState::DEGRADED) return 0;
    if (capture_ && data != nullptr && len != 0) {
        capture_->append(engine_common::ReplayRecordType::MARKET_BYTES,
                         steady_time_ns(),
                         {reinterpret_cast<const std::byte*>(data), len});
    }
    auto timer = decode_latency_.scoped();
    return adapter_->feed(data, len);
}

void FeedHandler::enable_capture(const std::string& path) {
    capture_ = std::make_unique<engine_common::ReplayWriter>(path);
}

void FeedHandler::on_event(const engine_common::MarketEvent& event) {
    if (state() == FeedState::RESYNC_REQUIRED) {
        queue_drops_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    engine_common::MarketEvent queued = event;
    queued.enqueue_timestamp = steady_time_ns();
    if (queue_.push(queued)) {
        consecutive_full_count_.store(0, std::memory_order_relaxed);
        update_high_watermark(queue_.size());
        return;
    }

    if (policy_ == BackpressurePolicy::SPIN_THEN_FAIL) {
        for (size_t attempt = 0; attempt < spin_limit_; ++attempt) {
            std::atomic_signal_fence(std::memory_order_seq_cst);
            if (queue_.push(queued)) {
                consecutive_full_count_.store(0, std::memory_order_relaxed);
                update_high_watermark(queue_.size());
                return;
            }
        }
    }

    queue_drops_.fetch_add(1, std::memory_order_relaxed);
    last_drop_time_ns_.store(steady_time_ns(), std::memory_order_relaxed);
    const uint64_t consecutive =
        consecutive_full_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t observed = max_consecutive_full_count_.load(std::memory_order_relaxed);
    while (observed < consecutive &&
           !max_consecutive_full_count_.compare_exchange_weak(
               observed, consecutive, std::memory_order_relaxed)) {}

    if (policy_ == BackpressurePolicy::DROP_AND_RESYNC) {
        enter_failure_state(FeedState::RESYNC_REQUIRED);
    } else {
        enter_failure_state(FeedState::DEGRADED);
    }
}

void FeedHandler::enter_failure_state(FeedState new_state) {
    FeedState expected = FeedState::RUNNING;
    if (state_.compare_exchange_strong(expected, new_state,
                                       std::memory_order_acq_rel)) {
        failure_started_ns_.store(steady_time_ns(), std::memory_order_relaxed);
    } else {
        state_.store(new_state, std::memory_order_release);
    }
    if (new_state == FeedState::DEGRADED) {
        running_.store(false, std::memory_order_release);
    }
}

void FeedHandler::mark_resynchronized() {
    if (state() != FeedState::RESYNC_REQUIRED) return;
    adapter_->reset();
    engine_common::MarketEvent discarded{};
    while (queue_.pop(discarded)) {}
    consecutive_full_count_.store(0, std::memory_order_relaxed);
    const int64_t started = failure_started_ns_.load(std::memory_order_relaxed);
    if (started != 0) {
        last_recovery_duration_ns_.store(steady_time_ns() - started,
                                         std::memory_order_relaxed);
    }
    state_.store(FeedState::RUNNING, std::memory_order_release);
}

void FeedHandler::update_high_watermark(size_t depth) {
    size_t observed = queue_high_watermark_.load(std::memory_order_relaxed);
    while (observed < depth &&
           !queue_high_watermark_.compare_exchange_weak(
               observed, depth, std::memory_order_relaxed)) {}
}

void FeedHandler::stop() {
    running_.store(false, std::memory_order_release);
    state_.store(FeedState::STOPPED, std::memory_order_release);
    transport_->close();
}

}  // namespace te

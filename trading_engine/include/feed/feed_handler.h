#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include "adapter/market_data_adapter.h"
#include "engine_common/replay.h"
#include "net/latency.h"
#include "net/spsc_ring_buffer.h"
#include "transport/transport.h"

namespace te {

enum class BackpressurePolicy : uint8_t {
    FAIL_FAST,
    DROP_AND_RESYNC,
    SPIN_THEN_FAIL,
};

enum class FeedState : uint8_t {
    RUNNING,
    RESYNC_REQUIRED,
    DEGRADED,
    STOPPED,
};

inline const char* to_string(FeedState state) {
    switch (state) {
        case FeedState::RUNNING: return "RUNNING";
        case FeedState::RESYNC_REQUIRED: return "RESYNC_REQUIRED";
        case FeedState::DEGRADED: return "DEGRADED";
        case FeedState::STOPPED: return "STOPPED";
    }
    return "UNKNOWN";
}

// Feed Handler:行情接入层。
//
// 数据流:
//   Transport ──receive──> MarketDataAdapter ──> MarketEvent ──push──> SPSC 队列
//
// 从回测的 CSV 回放升级为实时接入:连接行情源,非阻塞收包,
// 线协议仅存在于 Adapter 内部,策略和风控只消费统一 MarketEvent。
class FeedHandler {
public:
    explicit FeedHandler(size_t queue_pow2 = 1 << 14,
                         BackpressurePolicy policy = BackpressurePolicy::FAIL_FAST,
                         size_t spin_limit = 1'000,
                         std::unique_ptr<ITransport> transport = nullptr,
                         std::unique_ptr<IMarketDataAdapter> adapter = nullptr);

    // 连接行情源
    bool connect(const std::string& host, uint16_t port);

    // 单次轮询:非阻塞收包 → 解码 → 入队。由接收线程循环调用。
    // 返回本次处理的字节数(0 表示暂无数据)。
    size_t poll_once();
    size_t ingest_bytes(const uint8_t* data, size_t len);

    // 消费端接口:撮合线程从这里取解码后的行情
    bool try_pop(engine_common::MarketEvent& out) { return queue_.pop(out); }
    std::string_view symbol(engine_common::SymbolId symbol_id) const {
        return adapter_->symbol(symbol_id);
    }

    void stop();
    bool running() const { return running_.load(std::memory_order_acquire); }
    FeedState state() const { return state_.load(std::memory_order_acquire); }
    bool market_data_trusted() const {
        return state() == FeedState::RUNNING && !adapter_->has_unresolved_gap();
    }
    void mark_resynchronized();
    void enable_capture(const std::string& path);

    uint64_t bytes_received() const { return bytes_received_.load(std::memory_order_relaxed); }
    uint64_t receive_calls() const { return receive_calls_.load(std::memory_order_relaxed); }
    uint64_t queue_drops() const { return queue_drops_.load(std::memory_order_relaxed); }
    size_t queue_depth() const { return queue_.size(); }
    size_t queue_high_watermark() const {
        return queue_high_watermark_.load(std::memory_order_relaxed);
    }
    uint64_t consecutive_full_count() const {
        return consecutive_full_count_.load(std::memory_order_relaxed);
    }
    uint64_t max_consecutive_full_count() const {
        return max_consecutive_full_count_.load(std::memory_order_relaxed);
    }
    int64_t last_drop_time_ns() const {
        return last_drop_time_ns_.load(std::memory_order_relaxed);
    }
    int64_t last_recovery_duration_ns() const {
        return last_recovery_duration_ns_.load(std::memory_order_relaxed);
    }
    uint64_t sequence_gaps() const { return adapter_->gap_count(); }
    uint64_t recovered_sequence_gaps() const { return adapter_->recovered_gap_count(); }
    uint64_t permanent_sequence_gaps() const { return adapter_->permanent_gap_count(); }
    uint64_t duplicate_messages() const { return adapter_->duplicate_count(); }
    uint64_t malformed_messages() const { return adapter_->malformed_count(); }
    LatencySnapshot decode_latency() const { return decode_latency_.snapshot(); }

private:
    void on_event(const engine_common::MarketEvent& event);
    void enter_failure_state(FeedState state);
    void update_high_watermark(size_t depth);

    std::unique_ptr<ITransport> transport_;
    std::unique_ptr<IMarketDataAdapter> adapter_;
    SpscRingBuffer<engine_common::MarketEvent> queue_;
    BackpressurePolicy policy_;
    size_t spin_limit_;
    std::atomic<bool> running_{true};
    std::atomic<FeedState> state_{FeedState::RUNNING};
    std::atomic<uint64_t> bytes_received_{0};
    std::atomic<uint64_t> receive_calls_{0};
    std::atomic<uint64_t> queue_drops_{0};
    std::atomic<size_t> queue_high_watermark_{0};
    std::atomic<uint64_t> consecutive_full_count_{0};
    std::atomic<uint64_t> max_consecutive_full_count_{0};
    std::atomic<int64_t> last_drop_time_ns_{0};
    std::atomic<int64_t> failure_started_ns_{0};
    std::atomic<int64_t> last_recovery_duration_ns_{0};
    std::unique_ptr<engine_common::ReplayWriter> capture_;
    LatencyRecorder decode_latency_;
};

}  // namespace te

#pragma once

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine_common/types.h"
#include "net/latency.h"
#include "oms/order_state.h"
#include "transport/transport.h"

namespace te {

enum class GatewayState : uint8_t {
    DISCONNECTED,
    CONNECTING,
    LOGGING_IN,
    RECOVERING,
    READY,
    DEGRADED,
};

enum class WalDurability : uint8_t {
    ASYNC,
    GROUP_COMMIT,
    SYNC_EACH,
};

inline const char* to_string(GatewayState state) {
    switch (state) {
        case GatewayState::DISCONNECTED: return "DISCONNECTED";
        case GatewayState::CONNECTING: return "CONNECTING";
        case GatewayState::LOGGING_IN: return "LOGGING_IN";
        case GatewayState::RECOVERING: return "RECOVERING";
        case GatewayState::READY: return "READY";
        case GatewayState::DEGRADED: return "DEGRADED";
    }
    return "UNKNOWN";
}

struct GatewayConfig {
    size_t max_retries = 5;
    std::chrono::milliseconds reconnect_base_delay{10};
    std::chrono::milliseconds reconnect_max_delay{1'000};
    std::chrono::milliseconds heartbeat_interval{1'000};
    std::chrono::milliseconds heartbeat_timeout{5'000};
    std::string wal_path;
    WalDurability wal_durability = WalDurability::SYNC_EACH;
    size_t wal_group_size = 32;
    size_t send_batch_bytes = 16 * 1024;
    bool flush_on_enqueue = false;
    std::function<int64_t()> clock_now_ns;
    std::function<void(std::chrono::milliseconds)> sleep_for;
};

struct NewOrderRequest {
    int64_t client_order_id = 0;
    std::string symbol;
    uint8_t side = 0;
    int64_t quantity = 0;
    engine_common::PriceTicks price = 0;
};

struct ExecReport {
    int64_t client_order_id = 0;
    int64_t exec_id = 0;
    OrderStatus status = OrderStatus::NEW;
    int64_t fill_qty = 0;
    int64_t cumulative_qty = 0;
    engine_common::PriceTicks fill_price = 0;
    engine_common::RejectReason reject_reason = engine_common::RejectReason::NONE;
};

struct ExchangeOrderSnapshot {
    int64_t client_order_id = 0;
    int64_t quantity = 0;
    int64_t cumulative_qty = 0;
    OrderStatus status = OrderStatus::UNKNOWN;
};

struct ReconciliationResult {
    size_t matched = 0;
    size_t recovered = 0;
    size_t local_unknown = 0;
    size_t reports_applied = 0;
};

class OrderGateway {
public:
    using OnExecReport = std::function<void(const ExecReport&)>;

    explicit OrderGateway(GatewayConfig config = {},
                          std::unique_ptr<ITransport> transport = nullptr);
    ~OrderGateway();
    bool connect(const std::string& host, uint16_t port);
    int64_t send_order(const NewOrderRequest& request);
    bool cancel_order(int64_t client_order_id);
    size_t poll_once();
    void set_on_exec_report(OnExecReport callback) { on_report_ = std::move(callback); }
    bool reconnect();
    void disconnect();
    void process_report(const ExecReport& report);
    size_t recover_from_wal();
    bool flush_wal();
    bool checkpoint_wal();
    bool flush();
    ReconciliationResult reconcile(
        std::span<const ExchangeOrderSnapshot> exchange_orders,
        std::span<const ExecReport> missed_reports = {});

    const OrderState* find_order(int64_t client_order_id) const;
    GatewayState state() const { return state_; }
    bool ready() const { return state_ == GatewayState::READY; }
    uint64_t session_id() const { return session_id_; }
    uint64_t sent_sequence() const { return sent_sequence_; }
    uint64_t received_sequence() const { return received_sequence_; }
    uint64_t reconnect_count() const { return reconnect_count_; }
    uint64_t rejected_not_ready() const { return rejected_not_ready_; }
    uint64_t duplicate_reports() const { return duplicate_reports_; }
    uint64_t malformed_reports() const { return malformed_reports_; }
    uint64_t wal_recovered_orders() const { return wal_recovered_orders_; }
    uint64_t reconciliation_count() const { return reconciliation_count_; }
    uint64_t send_would_block_count() const { return send_would_block_count_; }
    size_t send_queue_high_watermark() const { return send_queue_high_watermark_; }
    int64_t last_heartbeat_ns() const { return last_heartbeat_ns_; }
    size_t active_order_count() const { return active_orders_.size(); }
    size_t archived_order_count() const { return archived_orders_.size(); }
    LatencySnapshot send_latency() const { return send_latency_.snapshot(); }
    LatencySnapshot ack_latency() const { return ack_latency_.snapshot(); }

private:
    bool connect_once();
    void enter_state(GatewayState state);
    void handle_report(const ExecReport& report);
    void archive_if_terminal(int64_t client_order_id);
    bool enqueue_message(const void* data, size_t len);
    bool flush_outgoing();
    bool enqueue_recovery_queries();
    bool maybe_send_heartbeat();
    void compact_receive_buffer();
    void compact_send_buffer();
    bool append_wal(const char* event, int64_t order_id, int64_t value = 0,
                    OrderStatus status = OrderStatus::UNKNOWN);
    bool open_wal();
    void close_wal();
    bool durable_wal_flush();
    int64_t now_ns() const;

    static constexpr size_t kReceiveCapacity = 256 * 1024;
    static constexpr size_t kSendCapacity = 256 * 1024;

    std::unique_ptr<ITransport> transport_;
    GatewayConfig config_;
    GatewayState state_ = GatewayState::DISCONNECTED;
    std::string host_;
    uint16_t port_ = 0;
    std::unordered_map<int64_t, OrderState> active_orders_;
    std::unordered_map<int64_t, OrderState> archived_orders_;
    int64_t next_client_order_id_ = 1;
    OnExecReport on_report_;
    std::array<uint8_t, kReceiveCapacity> receive_buffer_{};
    size_t receive_read_ = 0;
    size_t receive_write_ = 0;
    std::array<uint8_t, kSendCapacity> send_buffer_{};
    size_t send_read_ = 0;
    size_t send_write_ = 0;
    uint64_t session_id_ = 0;
    uint64_t sent_sequence_ = 0;
    uint64_t received_sequence_ = 0;
    uint64_t reconnect_count_ = 0;
    uint64_t rejected_not_ready_ = 0;
    uint64_t duplicate_reports_ = 0;
    uint64_t malformed_reports_ = 0;
    uint64_t wal_recovered_orders_ = 0;
    uint64_t reconciliation_count_ = 0;
    uint64_t send_would_block_count_ = 0;
    size_t send_queue_high_watermark_ = 0;
    int64_t last_send_ns_ = 0;
    int64_t last_receive_ns_ = 0;
    int64_t last_heartbeat_ns_ = 0;
    LatencyRecorder send_latency_;
    LatencyRecorder ack_latency_;
    std::FILE* wal_file_ = nullptr;
    std::vector<uint8_t> wal_buffer_;
    uint64_t wal_sequence_ = 0;
    size_t wal_pending_records_ = 0;
};

}  // namespace te

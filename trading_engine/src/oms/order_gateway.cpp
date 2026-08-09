#include "oms/order_gateway.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_set>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "feed/protocol.h"
#include "transport/tcp_transport.h"

namespace te {
namespace {

enum class OrderWireType : uint8_t {
    HEARTBEAT = 0,
    NEW_ORDER = 1,
    CANCEL_ORDER = 2,
    QUERY_ORDER = 3,
    EXEC_REPORT = 101,
};

#pragma pack(push, 1)
struct OrderWireHeader { uint8_t type; uint8_t version; uint16_t length; };
struct NewOrderWire {
    OrderWireHeader header; int64_t client_order_id; char symbol[8]; uint8_t side;
    uint8_t reserved[3]; int64_t quantity; int64_t price;
};
struct OrderIdWire { OrderWireHeader header; int64_t client_order_id; };
struct ExecReportWire {
    OrderWireHeader header; int64_t client_order_id; int64_t exec_id; uint8_t status;
    uint8_t reject_reason; uint8_t reserved[2]; int64_t fill_qty; int64_t fill_price;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct WalRecord {
    uint32_t magic;
    uint16_t version;
    uint16_t size;
    uint32_t crc;
    int64_t timestamp_ns;
    uint64_t session_id;
    uint64_t sequence;
    uint8_t event;
    uint8_t status;
    uint16_t reserved;
    int64_t order_id;
    int64_t value;
};
#pragma pack(pop)

constexpr uint32_t kWalMagic = 0x51425457;
constexpr uint16_t kWalVersion = 1;

uint32_t wal_crc(const uint8_t* data, size_t size) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> values{};
        for (uint32_t index = 0; index < values.size(); ++index) {
            uint32_t value = index;
            for (int bit = 0; bit < 8; ++bit)
                value = (value >> 1) ^ (0xEDB88320u & -(value & 1u));
            values[index] = value;
        }
        return values;
    }();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t index = 0; index < size; ++index)
        crc = (crc >> 8) ^ table[(crc ^ data[index]) & 0xFFU];
    return ~crc;
}

enum class WalEvent : uint8_t { INTENT_NEW = 1, INTENT_CANCEL = 2, REPORT = 3 };

WalEvent wal_event_from_name(const char* event) {
    if (std::strcmp(event, "INTENT_CANCEL") == 0) return WalEvent::INTENT_CANCEL;
    if (std::strcmp(event, "REPORT") == 0) return WalEvent::REPORT;
    return WalEvent::INTENT_NEW;
}

OrderWireHeader header(OrderWireType type, uint16_t length) {
    return {static_cast<uint8_t>(type), 1, host_to_be16(length)};
}

}  // namespace

OrderGateway::OrderGateway(GatewayConfig config,
                           std::unique_ptr<ITransport> transport)
    : transport_(transport ? std::move(transport)
                           : std::make_unique<TcpTransport>()),
      config_(std::move(config)) {
    recover_from_wal();
    open_wal();
}

OrderGateway::~OrderGateway() {
    close_wal();
}

int64_t OrderGateway::now_ns() const {
    if (config_.clock_now_ns) return config_.clock_now_ns();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void OrderGateway::enter_state(GatewayState state) { state_ = state; }

bool OrderGateway::connect(const std::string& host, uint16_t port) {
    host_ = host;
    port_ = port;
    return connect_once();
}

bool OrderGateway::connect_once() {
    if (host_.empty() || port_ == 0) return false;
    enter_state(GatewayState::CONNECTING);
    if (!transport_->connect(host_, port_)) {
        enter_state(GatewayState::DISCONNECTED);
        return false;
    }
    ++session_id_;
    receive_read_ = receive_write_ = send_read_ = send_write_ = 0;
    last_send_ns_ = last_receive_ns_ = last_heartbeat_ns_ = now_ns();
    enter_state(GatewayState::LOGGING_IN);
    enter_state(GatewayState::RECOVERING);
    if (!enqueue_recovery_queries() || !flush_outgoing()) {
        enter_state(GatewayState::DEGRADED);
        return false;
    }
    enter_state(GatewayState::READY);
    return true;
}

int64_t OrderGateway::send_order(const NewOrderRequest& request) {
    if (!ready()) {
        ++rejected_not_ready_;
        return -1;
    }
    if (request.symbol.empty() || request.symbol.size() > 8 || request.side > 1 ||
        request.quantity <= 0 || request.price < 0) return -1;
    const int64_t id = request.client_order_id > 0
        ? request.client_order_id : next_client_order_id_++;
    if (active_orders_.contains(id) || archived_orders_.contains(id)) return -1;
    next_client_order_id_ = std::max(next_client_order_id_, id + 1);
    if (!append_wal("INTENT_NEW", id, request.quantity)) return -1;
    NewOrderWire wire{};
    wire.header = header(OrderWireType::NEW_ORDER, sizeof(wire));
    wire.client_order_id = host_to_be64(id);
    std::memcpy(wire.symbol, request.symbol.data(), request.symbol.size());
    wire.side = request.side;
    wire.quantity = host_to_be64(request.quantity);
    wire.price = host_to_be64(request.price);
    const int64_t sent_at = now_ns();
    {
        auto timer = send_latency_.scoped();
        if (!enqueue_message(&wire, sizeof(wire))) return -1;
    }
    auto [position, inserted] = active_orders_.emplace(
        id, OrderState(id, request.quantity, sent_at));
    if (!inserted || !position->second.transition(OrderStatus::PENDING)) return -1;
    return id;
}

bool OrderGateway::cancel_order(int64_t client_order_id) {
    if (!ready()) {
        ++rejected_not_ready_;
        return false;
    }
    auto found = active_orders_.find(client_order_id);
    if (found == active_orders_.end() || found->second.is_terminal()) return false;
    if (!append_wal("INTENT_CANCEL", client_order_id)) return false;
    OrderIdWire wire{header(OrderWireType::CANCEL_ORDER, sizeof(OrderIdWire)),
                     host_to_be64(client_order_id)};
    if (!enqueue_message(&wire, sizeof(wire))) return false;
    return found->second.transition(OrderStatus::CANCEL_PENDING);
}

size_t OrderGateway::poll_once() {
    if (!transport_->is_open() || state_ == GatewayState::DISCONNECTED) return 0;
    const int64_t current = now_ns();
    if (current - last_receive_ns_ >
        std::chrono::duration_cast<std::chrono::nanoseconds>(config_.heartbeat_timeout).count()) {
        enter_state(GatewayState::DEGRADED);
        return 0;
    }
    if (!maybe_send_heartbeat() || !flush_outgoing()) {
        enter_state(GatewayState::DEGRADED);
        return 0;
    }
    if (receive_buffer_.size() - receive_write_ < 4096) compact_receive_buffer();
    const ssize_t received = transport_->receive(
        receive_buffer_.data() + receive_write_,
        receive_buffer_.size() - receive_write_);
    if (received < 0) {
        enter_state(GatewayState::DEGRADED);
        return 0;
    }
    if (received == 0) return 0;
    receive_write_ += static_cast<size_t>(received);
    last_receive_ns_ = now_ns();
    size_t reports = 0;
    while (receive_write_ - receive_read_ >= sizeof(OrderWireHeader)) {
        const uint8_t* message = receive_buffer_.data() + receive_read_;
        const uint16_t length = be16_to_host(
            read_unaligned<uint16_t>(message + offsetof(OrderWireHeader, length)));
        if (length < sizeof(OrderWireHeader) || length > 64 * 1024) {
            ++malformed_reports_;
            ++receive_read_;
            continue;
        }
        if (receive_write_ - receive_read_ < length) break;
        if (message[0] == static_cast<uint8_t>(OrderWireType::EXEC_REPORT) &&
            length == sizeof(ExecReportWire)) {
            ExecReport report;
            report.client_order_id = be64_to_host(
                read_unaligned<int64_t>(message + offsetof(ExecReportWire, client_order_id)));
            report.exec_id = be64_to_host(
                read_unaligned<int64_t>(message + offsetof(ExecReportWire, exec_id)));
            const uint8_t status = message[offsetof(ExecReportWire, status)];
            const uint8_t reason = message[offsetof(ExecReportWire, reject_reason)];
            if (status <= static_cast<uint8_t>(OrderStatus::DONE) &&
                reason <= static_cast<uint8_t>(engine_common::RejectReason::SESSION_UNAVAILABLE)) {
                report.status = static_cast<OrderStatus>(status);
                report.reject_reason = static_cast<engine_common::RejectReason>(reason);
                report.fill_qty = be64_to_host(
                    read_unaligned<int64_t>(message + offsetof(ExecReportWire, fill_qty)));
                report.fill_price = be64_to_host(
                    read_unaligned<int64_t>(message + offsetof(ExecReportWire, fill_price)));
                handle_report(report);
                ++reports;
                ++received_sequence_;
            } else {
                ++malformed_reports_;
            }
        }
        receive_read_ += length;
    }
    if (receive_read_ == receive_write_) receive_read_ = receive_write_ = 0;
    else if (receive_read_ >= receive_buffer_.size() / 2) compact_receive_buffer();
    return reports;
}

void OrderGateway::process_report(const ExecReport& report) { handle_report(report); }

void OrderGateway::handle_report(const ExecReport& report) {
    auto found = active_orders_.find(report.client_order_id);
    if (found == active_orders_.end()) {
        ++duplicate_reports_;
        return;
    }
    OrderState& order = found->second;
    if (order.sent_at_ns() > 0) ack_latency_.record(now_ns() - order.sent_at_ns());
    bool changed = false;
    if (report.fill_qty > 0) {
        if (order.status() == OrderStatus::PENDING) order.transition(OrderStatus::ACK);
        const int64_t cumulative = report.cumulative_qty > 0
            ? report.cumulative_qty : order.filled() + report.fill_qty;
        changed = order.on_cumulative_fill(cumulative, report.exec_id);
    }
    if (report.fill_qty == 0 ||
        (report.status != OrderStatus::PARTIALLY_FILLED && report.status != OrderStatus::FILLED)) {
        changed = order.transition(report.status) || changed;
    }
    if (!changed) ++duplicate_reports_;
    if (!append_wal("REPORT", report.client_order_id, order.filled(), order.status())) {
        enter_state(GatewayState::DEGRADED);
    }
    if (on_report_) on_report_(report);
    archive_if_terminal(report.client_order_id);
}

void OrderGateway::archive_if_terminal(int64_t client_order_id) {
    auto found = active_orders_.find(client_order_id);
    if (found == active_orders_.end() || !found->second.is_terminal()) return;
    archived_orders_.insert_or_assign(client_order_id, std::move(found->second));
    active_orders_.erase(found);
}

bool OrderGateway::reconnect() {
    if (host_.empty() || port_ == 0) return false;
    disconnect();
    std::mt19937_64 random(session_id_ + reconnect_count_ + 1);
    auto delay = config_.reconnect_base_delay;
    for (size_t attempt = 0; attempt < config_.max_retries; ++attempt) {
        ++reconnect_count_;
        if (connect_once()) return true;
        const auto jitter = std::chrono::milliseconds(random() % std::max<int64_t>(1, delay.count() / 4 + 1));
        if (config_.sleep_for) config_.sleep_for(delay + jitter);
        else std::this_thread::sleep_for(delay + jitter);
        delay = std::min(delay * 2, config_.reconnect_max_delay);
    }
    enter_state(GatewayState::DEGRADED);
    return false;
}

void OrderGateway::disconnect() {
    transport_->close();
    enter_state(GatewayState::DISCONNECTED);
    receive_read_ = receive_write_ = send_read_ = send_write_ = 0;
}

const OrderState* OrderGateway::find_order(int64_t client_order_id) const {
    auto active = active_orders_.find(client_order_id);
    if (active != active_orders_.end()) return &active->second;
    auto archived = archived_orders_.find(client_order_id);
    return archived == archived_orders_.end() ? nullptr : &archived->second;
}

size_t OrderGateway::recover_from_wal() {
    if (config_.wal_path.empty()) return 0;
    active_orders_.clear();
    archived_orders_.clear();
    wal_sequence_ = 0;

    std::ifstream binary_input(config_.wal_path, std::ios::binary);
    if (!binary_input) return 0;
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(binary_input)),
                               std::istreambuf_iterator<char>());
    binary_input.close();
    if (bytes.size() >= sizeof(uint32_t)) {
        uint32_t magic = 0;
        std::memcpy(&magic, bytes.data(), sizeof(magic));
        if (magic == kWalMagic) {
            size_t offset = 0;
            while (offset + sizeof(WalRecord) <= bytes.size()) {
                WalRecord record{};
                std::memcpy(&record, bytes.data() + offset, sizeof(record));
                if (record.magic != kWalMagic || record.version != kWalVersion ||
                    record.size != sizeof(WalRecord)) break;
                const uint32_t expected_crc = record.crc;
                record.crc = 0;
                if (wal_crc(reinterpret_cast<const uint8_t*>(&record), sizeof(record)) !=
                    expected_crc) break;
                wal_sequence_ = std::max(wal_sequence_, record.sequence);
                const auto event = static_cast<WalEvent>(record.event);
                if (event == WalEvent::INTENT_NEW && record.value > 0) {
                    auto [position, inserted] = active_orders_.try_emplace(
                        record.order_id, record.order_id, record.value);
                    if (inserted) position->second.transition(OrderStatus::PENDING);
                    next_client_order_id_ = std::max(next_client_order_id_, record.order_id + 1);
                } else if (event == WalEvent::INTENT_CANCEL) {
                    auto found = active_orders_.find(record.order_id);
                    if (found != active_orders_.end())
                        found->second.transition(OrderStatus::CANCEL_PENDING);
                } else if (event == WalEvent::REPORT) {
                    auto found = active_orders_.find(record.order_id);
                    if (found != active_orders_.end()) {
                        const auto status = static_cast<OrderStatus>(record.status);
                        if (record.value > found->second.filled()) {
                            if (found->second.status() == OrderStatus::PENDING)
                                found->second.transition(OrderStatus::ACK);
                            found->second.on_cumulative_fill(record.value);
                        }
                        if (status != OrderStatus::UNKNOWN && !found->second.is_terminal())
                            found->second.transition(status);
                        archive_if_terminal(record.order_id);
                    }
                }
                offset += sizeof(WalRecord);
            }
            if (offset != bytes.size()) {
                std::error_code error;
                std::filesystem::resize_file(config_.wal_path, offset, error);
            }
            wal_recovered_orders_ = active_orders_.size() + archived_orders_.size();
            return static_cast<size_t>(wal_recovered_orders_);
        }
    }

    std::ifstream input(config_.wal_path);
    if (!input) return 0;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream stream(line);
        std::string token;
        std::array<std::string, 6> fields{};
        size_t count = 0;
        while (count < fields.size() && std::getline(stream, token, ',')) {
            fields[count++] = token;
        }
        if (count < 5) continue;
        try {
            const std::string& event = fields[2];
            const int64_t id = std::stoll(fields[3]);
            const int64_t value = std::stoll(fields[4]);
            if (event == "INTENT_NEW" && value > 0) {
                auto [position, inserted] = active_orders_.try_emplace(id, id, value);
                if (inserted) position->second.transition(OrderStatus::PENDING);
                next_client_order_id_ = std::max(next_client_order_id_, id + 1);
            } else if (event == "INTENT_CANCEL") {
                auto found = active_orders_.find(id);
                if (found != active_orders_.end()) {
                    found->second.transition(OrderStatus::CANCEL_PENDING);
                }
            } else if (event == "REPORT") {
                auto found = active_orders_.find(id);
                if (found == active_orders_.end()) continue;
                OrderStatus status = OrderStatus::UNKNOWN;
                if (count >= 6) {
                    const int parsed = std::stoi(fields[5]);
                    if (parsed >= 0 && parsed <= static_cast<int>(OrderStatus::DONE)) {
                        status = static_cast<OrderStatus>(parsed);
                    }
                }
                if (value > found->second.filled()) {
                    if (found->second.status() == OrderStatus::PENDING) {
                        found->second.transition(OrderStatus::ACK);
                    }
                    found->second.on_cumulative_fill(value);
                }
                if (status != OrderStatus::UNKNOWN && !found->second.is_terminal()) {
                    found->second.transition(status);
                }
                archive_if_terminal(id);
            }
        } catch (const std::exception&) {
        }
    }
    wal_recovered_orders_ = active_orders_.size() + archived_orders_.size();
    return static_cast<size_t>(wal_recovered_orders_);
}

ReconciliationResult OrderGateway::reconcile(
    std::span<const ExchangeOrderSnapshot> exchange_orders,
    std::span<const ExecReport> missed_reports) {
    ReconciliationResult result;
    std::unordered_set<int64_t> exchange_ids;
    exchange_ids.reserve(exchange_orders.size());
    for (const ExchangeOrderSnapshot& snapshot : exchange_orders) {
        if (snapshot.client_order_id <= 0 || snapshot.quantity <= 0) continue;
        exchange_ids.insert(snapshot.client_order_id);
        auto found = active_orders_.find(snapshot.client_order_id);
        if (found == active_orders_.end()) {
            auto [position, inserted] = active_orders_.try_emplace(
                snapshot.client_order_id, snapshot.client_order_id, snapshot.quantity);
            found = position;
            if (inserted) {
                found->second.transition(OrderStatus::PENDING);
                found->second.transition(OrderStatus::RECOVERING);
                ++result.recovered;
                next_client_order_id_ = std::max(next_client_order_id_,
                                                 snapshot.client_order_id + 1);
            }
        } else {
            ++result.matched;
        }
        if (snapshot.cumulative_qty > found->second.filled()) {
            found->second.on_cumulative_fill(snapshot.cumulative_qty);
        }
        if (!found->second.is_terminal()) found->second.transition(snapshot.status);
        archive_if_terminal(snapshot.client_order_id);
    }
    for (auto& [id, order] : active_orders_) {
        if (!exchange_ids.contains(id) && !order.is_terminal()) {
            order.transition(OrderStatus::UNKNOWN);
            ++result.local_unknown;
        }
    }
    for (const ExecReport& report : missed_reports) {
        const OrderState* before = find_order(report.client_order_id);
        const int64_t filled = before == nullptr ? -1 : before->filled();
        handle_report(report);
        const OrderState* after = find_order(report.client_order_id);
        if (after != nullptr && after->filled() != filled) ++result.reports_applied;
    }
    ++reconciliation_count_;
    return result;
}

bool OrderGateway::enqueue_recovery_queries() {
    for (auto& [id, order] : active_orders_) {
        order.transition(OrderStatus::RECOVERING);
        OrderIdWire wire{header(OrderWireType::QUERY_ORDER, sizeof(OrderIdWire)), host_to_be64(id)};
        if (!enqueue_message(&wire, sizeof(wire))) return false;
    }
    return true;
}

bool OrderGateway::maybe_send_heartbeat() {
    const int64_t current = now_ns();
    const int64_t interval = std::chrono::duration_cast<std::chrono::nanoseconds>(
        config_.heartbeat_interval).count();
    if (current - last_send_ns_ < interval) return true;
    const OrderWireHeader heartbeat = header(OrderWireType::HEARTBEAT, sizeof(OrderWireHeader));
    if (!enqueue_message(&heartbeat, sizeof(heartbeat))) return false;
    last_heartbeat_ns_ = current;
    return true;
}

bool OrderGateway::enqueue_message(const void* data, size_t len) {
    if (!transport_->is_open() || data == nullptr || len == 0 ||
        len > send_buffer_.size()) return false;
    if (send_buffer_.size() - send_write_ < len) compact_send_buffer();
    if (send_buffer_.size() - send_write_ < len && !flush_outgoing()) return false;
    if (send_buffer_.size() - send_write_ < len) compact_send_buffer();
    if (send_buffer_.size() - send_write_ < len) {
        enter_state(GatewayState::DEGRADED);
        return false;
    }
    std::memcpy(send_buffer_.data() + send_write_, data, len);
    send_write_ += len;
    send_queue_high_watermark_ = std::max(
        send_queue_high_watermark_, send_write_ - send_read_);
    ++sent_sequence_;
    const size_t threshold = std::max<size_t>(1, config_.send_batch_bytes);
    return (!config_.flush_on_enqueue && send_write_ - send_read_ < threshold) ||
           flush_outgoing();
}

bool OrderGateway::flush_outgoing() {
    while (send_read_ < send_write_) {
        const ssize_t sent = transport_->transmit(
            send_buffer_.data() + send_read_, send_write_ - send_read_);
        if (sent < 0) return false;
        if (sent == 0) {
            ++send_would_block_count_;
            return true;
        }
        send_read_ += static_cast<size_t>(sent);
        last_send_ns_ = now_ns();
    }
    send_read_ = send_write_ = 0;
    return true;
}

bool OrderGateway::flush() {
    return flush_outgoing() && flush_wal();
}

void OrderGateway::compact_receive_buffer() {
    const size_t remaining = receive_write_ - receive_read_;
    if (remaining != 0) std::memmove(receive_buffer_.data(), receive_buffer_.data() + receive_read_, remaining);
    receive_read_ = 0;
    receive_write_ = remaining;
}

void OrderGateway::compact_send_buffer() {
    const size_t remaining = send_write_ - send_read_;
    if (remaining != 0) std::memmove(send_buffer_.data(), send_buffer_.data() + send_read_, remaining);
    send_read_ = 0;
    send_write_ = remaining;
}

bool OrderGateway::open_wal() {
    if (config_.wal_path.empty() || wal_file_ != nullptr) return true;
    wal_file_ = std::fopen(config_.wal_path.c_str(), "ab");
    if (wal_file_ != nullptr) {
        std::setvbuf(wal_file_, nullptr, _IOFBF, 1 << 20);
        wal_buffer_.reserve(std::max<size_t>(config_.wal_group_size, 1024) *
                            sizeof(WalRecord));
    }
    return wal_file_ != nullptr;
}

void OrderGateway::close_wal() {
    if (wal_file_ == nullptr) return;
    flush_wal();
    std::fclose(wal_file_);
    wal_file_ = nullptr;
}

bool OrderGateway::durable_wal_flush() {
    if (wal_file_ == nullptr) return true;
    if (std::fflush(wal_file_) != 0) return false;
#ifdef _WIN32
    return _commit(_fileno(wal_file_)) == 0;
#else
    return ::fsync(fileno(wal_file_)) == 0;
#endif
}

bool OrderGateway::flush_wal() {
    if (wal_file_ == nullptr) return true;
    if (!wal_buffer_.empty()) {
        for (size_t offset = 0; offset < wal_buffer_.size(); offset += sizeof(WalRecord)) {
            WalRecord record{};
            std::memcpy(&record, wal_buffer_.data() + offset, sizeof(record));
            record.crc = 0;
            record.crc = wal_crc(reinterpret_cast<const uint8_t*>(&record), sizeof(record));
            std::memcpy(wal_buffer_.data() + offset, &record, sizeof(record));
        }
        if (std::fwrite(wal_buffer_.data(), wal_buffer_.size(), 1, wal_file_) != 1)
            return false;
    }
    const bool ok = durable_wal_flush();
    if (ok) {
        wal_pending_records_ = 0;
        wal_buffer_.clear();
    }
    return ok;
}

bool OrderGateway::append_wal(const char* event, int64_t order_id, int64_t value,
                              OrderStatus status) {
    if (config_.wal_path.empty()) return true;
    if (!open_wal()) return false;
    WalRecord record{};
    record.magic = kWalMagic;
    record.version = kWalVersion;
    record.size = sizeof(WalRecord);
    record.timestamp_ns = now_ns();
    record.session_id = session_id_;
    record.sequence = ++wal_sequence_;
    record.event = static_cast<uint8_t>(wal_event_from_name(event));
    record.status = static_cast<uint8_t>(status);
    record.order_id = order_id;
    record.value = value;
    record.crc = 0;
    const auto* bytes = reinterpret_cast<const uint8_t*>(&record);
    wal_buffer_.insert(wal_buffer_.end(), bytes, bytes + sizeof(record));
    ++wal_pending_records_;
    if (config_.wal_durability == WalDurability::SYNC_EACH ||
        (config_.wal_durability == WalDurability::GROUP_COMMIT &&
         wal_pending_records_ >= std::max<size_t>(1, config_.wal_group_size))) {
        return flush_wal();
    }
    return true;
}

bool OrderGateway::checkpoint_wal() {
    if (config_.wal_path.empty()) return true;
    close_wal();
    std::error_code error;
    std::filesystem::remove(config_.wal_path, error);
    wal_sequence_ = 0;
    if (!open_wal()) return false;
    auto write_snapshot = [this](const auto& entries) {
        for (const auto& [id, order] : entries) {
            if (!append_wal("INTENT_NEW", id, order.quantity())) return false;
            if (!append_wal("REPORT", id, order.filled(), order.status())) return false;
        }
        return true;
    };
    if (!write_snapshot(active_orders_) || !write_snapshot(archived_orders_)) return false;
    return flush_wal();
}

}  // namespace te

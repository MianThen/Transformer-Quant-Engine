#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace te {

// 订单状态机。
//
// 生命周期:NEW → PENDING → ACK ─┬─> FILLED ──> DONE
//                                 ├─> PARTIALLY_FILLED ──> FILLED/CANCELED
//                                 └─> REJECTED ──> DONE
//                    PENDING ──> CANCELED ──> DONE
//
// 面试要点:状态机保证幂等——重复回报不会导致状态错乱,
// 非法转移(如 DONE → ACK)被拒绝。
enum class OrderStatus : uint8_t {
    NEW,                // 本地创建,未发送
    PENDING,            // 已发送,等待交易所确认
    ACK,                // 交易所已接受
    PARTIALLY_FILLED,   // 部分成交
    CANCEL_PENDING,
    REPLACE_PENDING,
    FILLED,             // 全部成交
    CANCELED,           // 已撤单
    REJECTED,           // 被拒绝
    EXPIRED,
    SUSPENDED,
    RECOVERING,
    UNKNOWN,
    DONE,               // 终态(成交完/撤完/拒绝后归档)
};

inline const char* to_string(OrderStatus s) {
    switch (s) {
        case OrderStatus::NEW: return "NEW";
        case OrderStatus::PENDING: return "PENDING";
        case OrderStatus::ACK: return "ACK";
        case OrderStatus::PARTIALLY_FILLED: return "PARTIALLY_FILLED";
        case OrderStatus::CANCEL_PENDING: return "CANCEL_PENDING";
        case OrderStatus::REPLACE_PENDING: return "REPLACE_PENDING";
        case OrderStatus::FILLED: return "FILLED";
        case OrderStatus::CANCELED: return "CANCELED";
        case OrderStatus::REJECTED: return "REJECTED";
        case OrderStatus::EXPIRED: return "EXPIRED";
        case OrderStatus::SUSPENDED: return "SUSPENDED";
        case OrderStatus::RECOVERING: return "RECOVERING";
        case OrderStatus::UNKNOWN: return "UNKNOWN";
        case OrderStatus::DONE: return "DONE";
    }
    return "?";
}

// 单个订单的状态机
class OrderState {
public:
    OrderState(int64_t client_order_id, int64_t quantity, int64_t sent_at_ns = 0)
        : client_order_id_(client_order_id), quantity_(quantity), sent_at_ns_(sent_at_ns) {}

    // 尝试状态转移。非法转移返回 false(不改变当前状态)。
    bool transition(OrderStatus next);

    // 处理成交回报,累加已成交量,自动切换 PARTIALLY_FILLED/FILLED
    bool on_fill(int64_t fill_qty, int64_t exec_id = 0);
    bool on_cumulative_fill(int64_t cumulative_qty, int64_t exec_id = 0);

    OrderStatus status() const { return status_; }
    int64_t client_order_id() const { return client_order_id_; }
    int64_t quantity() const { return quantity_; }
    int64_t filled() const { return filled_qty_; }
    int64_t last_fill() const { return last_fill_qty_; }
    int64_t leaves() const { return quantity_ - filled_qty_; }
    int64_t sent_at_ns() const { return sent_at_ns_; }
    bool is_terminal() const {
        return status_ == OrderStatus::DONE || status_ == OrderStatus::FILLED ||
               status_ == OrderStatus::CANCELED || status_ == OrderStatus::REJECTED ||
               status_ == OrderStatus::EXPIRED;
    }

private:
    int64_t client_order_id_;
    int64_t quantity_;
    int64_t filled_qty_ = 0;
    int64_t last_fill_qty_ = 0;
    int64_t sent_at_ns_ = 0;
    OrderStatus status_ = OrderStatus::NEW;
    static constexpr size_t kExecDedupWindow = 256;
    std::array<int64_t, kExecDedupWindow> seen_exec_ids_{};
    size_t seen_exec_count_ = 0;
    size_t seen_exec_next_ = 0;
};

}  // namespace te

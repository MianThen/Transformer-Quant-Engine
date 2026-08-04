#pragma once

#include <cstdint>
#include <string>

#include "types.h"

namespace qbt {

// 订单
struct Order {
    int64_t id = 0;
    std::uint64_t decision_id = 0;
    std::string symbol;
    Side side = Side::BUY;
    OrderType type = OrderType::MARKET;
    Quantity quantity = 0;
    Price limit_price = 0.0;  // 仅 LIMIT 有效
    Timestamp timestamp = 0;
};

// 成交回报
struct Fill {
    int64_t order_id = 0;
    std::string symbol;
    Side side = Side::BUY;
    Quantity quantity = 0;
    Price price = 0.0;
    Price commission = 0.0;
    Timestamp timestamp = 0;
};

enum class OrderStatus {
    ACCEPTED,
    PARTIALLY_FILLED,
    FILLED,
    CANCELED,
    REJECTED,
    EXPIRED,
};

enum class RejectReason {
    NONE,
    INVALID_ORDER,
    UNKNOWN_SYMBOL,
    NOT_LISTED,
    INVALID_LOT_SIZE,
    INSUFFICIENT_CASH,
    INSUFFICIENT_POSITION,
    STALE_MARKET_DATA,
};

struct OrderRecord {
    Order order;
    Quantity filled_quantity = 0;
    Price avg_fill_price = 0.0;
    OrderStatus status = OrderStatus::ACCEPTED;
    RejectReason reject_reason = RejectReason::NONE;
    Timestamp updated_timestamp = 0;
    std::string message;
};

}  // namespace qbt

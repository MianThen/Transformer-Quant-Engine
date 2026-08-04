#pragma once

#include <cstdint>
#include <string>

namespace engine_common {

using SymbolId = uint32_t;
using PriceTicks = int64_t;
using MoneyMinor = int64_t;
using Quantity = int64_t;
using TimestampNs = int64_t;

enum class Side : uint8_t { BUY, SELL };
enum class OrderType : uint8_t { MARKET, LIMIT };
enum class TimeInForce : uint8_t { DAY, IOC, FOK, GTC };
enum class RejectReason : uint8_t {
    NONE,
    INVALID_ORDER,
    NOT_READY,
    UNKNOWN_SYMBOL,
    NOT_LISTED,
    INVALID_LOT_SIZE,
    INSUFFICIENT_CASH,
    INSUFFICIENT_POSITION,
    STALE_MARKET_DATA,
    RISK_REJECTED,
    EXCHANGE_REJECTED,
    SESSION_UNAVAILABLE,
};

enum class ExecutionStatus : uint8_t {
    NEW,
    PENDING_NEW,
    ACKNOWLEDGED,
    PARTIALLY_FILLED,
    PENDING_CANCEL,
    PENDING_REPLACE,
    FILLED,
    CANCELED,
    REJECTED,
    EXPIRED,
    SUSPENDED,
    RECOVERING,
    UNKNOWN,
};

enum ExecutionAuditFlags : uint32_t {
    EXECUTION_HAS_SYMBOL = 1U << 0,
    EXECUTION_HAS_SIDE = 1U << 1,
    EXECUTION_HAS_PRICE_SCALE = 1U << 2,
    EXECUTION_HAS_EXPLICIT_FEE = 1U << 3,
    EXECUTION_HAS_DECISION_ID = 1U << 4,
};

struct MarketEvent {
    SymbolId symbol_id = 0;
    TimestampNs timestamp = 0;
    PriceTicks bid = 0;
    PriceTicks ask = 0;
    Quantity bid_quantity = 0;
    Quantity ask_quantity = 0;
    uint64_t sequence = 0;
    TimestampNs enqueue_timestamp = 0;
};

struct OrderIntent {
    int64_t client_order_id = 0;
    SymbolId symbol_id = 0;
    Side side = Side::BUY;
    OrderType type = OrderType::MARKET;
    TimeInForce time_in_force = TimeInForce::DAY;
    Quantity quantity = 0;
    PriceTicks limit_price = 0;
    TimestampNs timestamp = 0;
};

struct ExecutionEvent {
    int64_t client_order_id = 0;
    int64_t execution_id = 0;
    uint64_t decision_id = 0;
    SymbolId symbol_id = 0;
    Side side = Side::BUY;
    ExecutionStatus status = ExecutionStatus::NEW;
    RejectReason reject_reason = RejectReason::NONE;
    Quantity last_quantity = 0;
    Quantity cumulative_quantity = 0;
    PriceTicks last_price = 0;
    int64_t price_scale = 0;
    MoneyMinor explicit_fee = 0;
    int64_t fee_scale = 0;
    uint32_t audit_flags = 0;
    TimestampNs timestamp = 0;
};

}  // namespace engine_common

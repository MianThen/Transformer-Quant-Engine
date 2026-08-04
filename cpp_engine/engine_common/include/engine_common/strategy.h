#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "model_types.h"
#include "types.h"

namespace engine_common {

enum class StrategyStatus : uint8_t {
    OK,
    NOT_READY,
    INVALID_CONFIGURATION,
    DATA_UNTRUSTED,
    MODEL_ERROR,
    OUTPUT_OVERFLOW,
    STOPPED,
};

enum class ResetReason : uint8_t {
    DATA_GAP,
    TRADING_DAY,
    MODEL_SWITCH,
    REPLAY_SEEK,
    MANUAL,
};

struct MarketBar {
    SymbolId symbol_id = 0;
    TimestampNs timestamp = 0;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    double signal_open = 0.0;
    double signal_high = 0.0;
    double signal_low = 0.0;
    double signal_close = 0.0;
    Quantity volume = 0;
    Quantity lot_size = 1;
    uint32_t flags = 0;
};

enum MarketBarFlags : uint32_t {
    MARKET_LISTED = 1U << 0,
    MARKET_SUSPENDED = 1U << 1,
    MARKET_ST = 1U << 2,
    MARKET_DATA_TRUSTED = 1U << 3,
};

struct MarketFrameBatchView {
    TimestampNs asof_timestamp = 0;
    std::span<const MarketBar> bars;
    bool data_trusted = true;
};

struct PortfolioItem {
    SymbolId symbol_id = 0;
    Quantity position_quantity = 0;
    Quantity sellable_quantity = 0;
    double average_cost = 0.0;
    double mark_price = 0.0;
    Quantity active_buy_quantity = 0;
    Quantity active_sell_quantity = 0;
};

struct PortfolioView {
    std::span<const PortfolioItem> items;
    double cash = 0.0;
    double equity = 0.0;
    double gross_exposure = 0.0;
    double net_exposure = 0.0;
};

struct OrderIntentBuffer {
    std::span<OrderIntent> values;
    size_t size = 0;

    [[nodiscard]] bool push(const OrderIntent& value) noexcept {
        if (size >= values.size()) return false;
        values[size++] = value;
        return true;
    }

    void clear() noexcept { size = 0; }
};

struct StrategySessionContext {
    bool live = false;
    bool shadow = false;
    bool allow_orders = true;
    uint64_t feature_schema_hash = 0;
    uint64_t model_version_hash = 0;
};

struct StrategyDecisionView {
    uint64_t decision_id = 0;
    TimestampNs decision_at = 0;
    std::span<const TargetPosition> targets;

    [[nodiscard]] bool valid() const noexcept {
        return decision_id != 0 && decision_at > 0;
    }
};

class IStrategyRuntime {
public:
    virtual ~IStrategyRuntime() = default;
    virtual StrategyStatus start(const StrategySessionContext& context) = 0;
    virtual StrategyStatus on_market_batch(
        const MarketFrameBatchView& market,
        const PortfolioView& portfolio,
        OrderIntentBuffer& output) noexcept = 0;
    virtual void on_execution(const ExecutionEvent& execution) noexcept = 0;
    virtual void on_reset(ResetReason reason, TimestampNs timestamp) noexcept = 0;
    virtual StrategyDecisionView last_decision() const noexcept { return {}; }
    virtual void stop() noexcept = 0;
};

}  // namespace engine_common

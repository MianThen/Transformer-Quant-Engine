#pragma once

#include <cstdint>

#include "engine_common/strategy.h"

namespace qbt::strategy {

enum class RiskDecision : uint8_t {
    APPROVED,
    KILL_SWITCH,
    DATA_UNTRUSTED,
    UNKNOWN_SYMBOL,
    NOT_TRADABLE,
    INVALID_QUANTITY,
    ORDER_TOO_LARGE,
    INVALID_TIMESTAMP,
};

struct RiskConfig {
    bool kill_switch = false;
    bool require_trusted_market = true;
    engine_common::Quantity max_order_quantity = 1'000'000;
};

class BasicRiskManager {
public:
    explicit BasicRiskManager(RiskConfig config);
    RiskDecision evaluate(
        const engine_common::OrderIntent& intent,
        const engine_common::MarketFrameBatchView& market) const noexcept;

private:
    RiskConfig config_;
};

}  // namespace qbt::strategy

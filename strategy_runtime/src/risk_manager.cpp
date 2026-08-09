#include "strategy_runtime/risk_manager.h"

#include <algorithm>
#include <stdexcept>

namespace qbt::strategy {

BasicRiskManager::BasicRiskManager(RiskConfig config) : config_(config) {
    if (config_.max_order_quantity <= 0) {
        throw std::invalid_argument("max_order_quantity must be positive");
    }
}

RiskDecision BasicRiskManager::evaluate(
    const engine_common::OrderIntent& intent,
    const engine_common::MarketFrameBatchView& market) const noexcept {
    if (config_.kill_switch) return RiskDecision::KILL_SWITCH;
    if (config_.require_trusted_market && !market.data_trusted) {
        return RiskDecision::DATA_UNTRUSTED;
    }
    const auto found = std::find_if(
        market.bars.begin(), market.bars.end(), [&](const auto& bar) {
            return bar.symbol_id == intent.symbol_id;
        });
    if (found == market.bars.end()) return RiskDecision::UNKNOWN_SYMBOL;
    if ((found->flags & engine_common::MARKET_LISTED) == 0 ||
        (found->flags & engine_common::MARKET_SUSPENDED) != 0 ||
        (config_.require_trusted_market &&
         (found->flags & engine_common::MARKET_DATA_TRUSTED) == 0)) {
        return RiskDecision::NOT_TRADABLE;
    }
    if (intent.timestamp != market.asof_timestamp) return RiskDecision::INVALID_TIMESTAMP;
    if (intent.quantity <= 0 || found->lot_size <= 0 ||
        intent.quantity % found->lot_size != 0) {
        return RiskDecision::INVALID_QUANTITY;
    }
    if (intent.quantity > config_.max_order_quantity) {
        return RiskDecision::ORDER_TOO_LARGE;
    }
    return RiskDecision::APPROVED;
}

}  // namespace qbt::strategy

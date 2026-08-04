#include "strategy_runtime/order_planner.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace qbt::strategy {

LongOnlyOrderPlanner::LongOnlyOrderPlanner(size_t maximum_intents)
    : maximum_intents_(maximum_intents) {
    if (maximum_intents_ == 0) {
        throw std::invalid_argument("maximum order intents must be positive");
    }
    intents_.reserve(maximum_intents_);
}

std::span<const engine_common::OrderIntent> LongOnlyOrderPlanner::build(
    std::span<const engine_common::TargetPosition> targets,
    const engine_common::PortfolioView& portfolio,
    engine_common::TimestampNs timestamp, uint64_t decision_id) {
    if (timestamp <= 0 || decision_id == 0) {
        throw std::invalid_argument("order timestamp and decision id must be positive");
    }
    intents_.clear();
    size_t lookup_size = 0;
    for (const auto& item : portfolio.items) {
        lookup_size = std::max(lookup_size, static_cast<size_t>(item.symbol_id) + 1);
    }
    portfolio_by_symbol_.assign(lookup_size, nullptr);
    for (const auto& item : portfolio.items) portfolio_by_symbol_[item.symbol_id] = &item;
    for (const auto& target : targets) {
        const auto* found = target.symbol_id < portfolio_by_symbol_.size()
            ? portfolio_by_symbol_[target.symbol_id] : nullptr;
        const engine_common::Quantity current = found == nullptr
            ? 0 : found->position_quantity;
        const engine_common::Quantity active_buy = found == nullptr
            ? 0 : found->active_buy_quantity;
        const engine_common::Quantity active_sell = found == nullptr
            ? 0 : found->active_sell_quantity;
        const auto difference = target.target_quantity -
            (current + active_buy - active_sell);
        if (difference == 0) continue;
        if (intents_.size() >= maximum_intents_) {
            throw std::overflow_error("order intent plan exceeds configured maximum");
        }
        engine_common::OrderIntent intent;
        intent.decision_id = decision_id;
        intent.symbol_id = target.symbol_id;
        intent.side = difference > 0 ? engine_common::Side::BUY
                                     : engine_common::Side::SELL;
        intent.quantity = std::abs(difference);
        intent.timestamp = timestamp;
        intents_.push_back(intent);
    }
    std::sort(intents_.begin(), intents_.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.side != rhs.side) return lhs.side == engine_common::Side::SELL;
        return lhs.symbol_id < rhs.symbol_id;
    });
    return intents_;
}

}  // namespace qbt::strategy

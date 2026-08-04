#include "position.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "engine_common/fixed_point.h"

namespace qbt {

namespace {
constexpr int64_t kMoneyScale = 10'000;
}

Position& PositionTracker::position_for(const std::string& symbol) {
    const SymbolId id = symbols_->intern(symbol);
    if (positions_.size() <= id) {
        positions_.resize(static_cast<size_t>(id) + 1);
        occupied_.resize(static_cast<size_t>(id) + 1, 0);
    }
    if (occupied_[id] == 0) {
        occupied_[id] = 1;
        positions_[id].symbol = symbol;
        ++position_count_;
    }
    return positions_[id];
}

const Position* PositionTracker::find_position(const std::string& symbol) const {
    SymbolId id = 0;
    if (!symbols_->find(symbol, id) || id >= positions_.size() || occupied_[id] == 0) {
        return nullptr;
    }
    return &positions_[id];
}

void PositionTracker::apply_fill(const Fill& fill) {
    if (fill.quantity <= 0 || fill.symbol.empty() || fill.price <= 0.0) return;

    Position& pos = position_for(fill.symbol);
    const Quantity delta = fill.side == Side::BUY ? fill.quantity : -fill.quantity;
    if (fill.side == Side::SELL) {
        pos.sellable_quantity = std::max<Quantity>(
            0, pos.sellable_quantity - fill.quantity);
    }

    if (pos.quantity == 0 || (pos.quantity > 0) == (delta > 0)) {
        const Quantity old_abs = std::abs(pos.quantity);
        const Quantity new_abs = old_abs + std::abs(delta);
        pos.avg_cost = (pos.avg_cost * static_cast<Price>(old_abs) +
                        fill.price * static_cast<Price>(std::abs(delta))) /
                       static_cast<Price>(new_abs);
        pos.quantity += delta;
        return;
    }

    const Quantity closed = std::min(std::abs(pos.quantity), std::abs(delta));
    const Price direction = pos.quantity > 0 ? 1.0 : -1.0;
    pos.realized_pnl_minor += engine_common::quantize_money(
        (fill.price - pos.avg_cost) * static_cast<Price>(closed) * direction,
        kMoneyScale);
    pos.realized_pnl = engine_common::money_from_minor(pos.realized_pnl_minor,
                                                       kMoneyScale);

    const Quantity new_quantity = pos.quantity + delta;
    if (new_quantity == 0) {
        pos.avg_cost = 0.0;
    } else if ((new_quantity > 0) != (pos.quantity > 0)) {
        pos.avg_cost = fill.price;
    }
    pos.quantity = new_quantity;
}

void PositionTracker::roll_trading_day(Timestamp timestamp) {
    constexpr Timestamp kNanosecondsPerDay = 86'400'000'000'000LL;
    constexpr Timestamp kShanghaiOffset = 8LL * 3'600'000'000'000LL;
    const int64_t day = (timestamp + kShanghaiOffset) / kNanosecondsPerDay;
    if (current_trading_day_ == -1) {
        current_trading_day_ = day;
        return;
    }
    if (day == current_trading_day_) return;
    current_trading_day_ = day;
    for (size_t id = 0; id < positions_.size(); ++id) {
        if (occupied_[id] != 0) {
            positions_[id].sellable_quantity =
                std::max<Quantity>(positions_[id].quantity, 0);
        }
    }
}

CorporateActionResult PositionTracker::apply_corporate_action(
    const CorporateAction& action) {
    if (action.symbol.empty() || action.timestamp < 0 ||
        action.cash_dividend_per_share < 0.0 ||
        !std::isfinite(action.cash_dividend_per_share) ||
        action.share_multiplier <= 0.0 ||
        !std::isfinite(action.share_multiplier)) {
        throw std::invalid_argument("invalid corporate action");
    }
    CorporateActionResult result;
    result.symbol = action.symbol;
    result.timestamp = action.timestamp;
    Position* position_ptr = const_cast<Position*>(find_position(action.symbol));
    if (position_ptr == nullptr || position_ptr->quantity <= 0) return result;

    Position& position = *position_ptr;
    result.old_quantity = position.quantity;
    result.cash_dividend = static_cast<Price>(position.quantity) *
                           action.cash_dividend_per_share;
    const double adjusted = static_cast<double>(position.quantity) *
                            action.share_multiplier;
    const double adjusted_sellable =
        static_cast<double>(position.sellable_quantity) * action.share_multiplier;
    if (std::abs(adjusted - std::round(adjusted)) > 1e-9 ||
        std::abs(adjusted_sellable - std::round(adjusted_sellable)) > 1e-9) {
        throw std::invalid_argument(
            "corporate action produces fractional share quantity");
    }
    if (action.share_multiplier != 1.0) {
        position.quantity = static_cast<Quantity>(std::llround(adjusted));
        position.sellable_quantity =
            static_cast<Quantity>(std::llround(adjusted_sellable));
        position.avg_cost /= action.share_multiplier;
    }
    result.new_quantity = position.quantity;
    return result;
}

Quantity PositionTracker::available_to_sell(const std::string& symbol) const {
    const Position* position = find_position(symbol);
    return position == nullptr ? 0 : position->sellable_quantity;
}

Position PositionTracker::get_position(const std::string& symbol) const {
    const Position* position = find_position(symbol);
    if (position == nullptr) {
        Position empty;
        empty.symbol = symbol;
        return empty;
    }
    return *position;
}

std::vector<Position> PositionTracker::all_positions() const {
    std::vector<Position> result;
    result.reserve(position_count_);
    for (size_t id = 0; id < positions_.size(); ++id) {
        if (occupied_[id] != 0) result.push_back(positions_[id]);
    }
    std::sort(result.begin(), result.end(),
              [](const Position& lhs, const Position& rhs) {
                  return lhs.symbol < rhs.symbol;
              });
    return result;
}

Price PositionTracker::market_value(
    const std::unordered_map<std::string, Price>& prices) const {
    Price total = 0.0;
    for (size_t id = 0; id < positions_.size(); ++id) {
        if (occupied_[id] == 0) continue;
        const Position& pos = positions_[id];
        const auto it = prices.find(pos.symbol);
        if (it != prices.end()) {
            total += static_cast<Price>(pos.quantity) * it->second;
        }
    }
    return total;
}

Price PositionTracker::market_value(std::span<const Price> prices,
                                    std::span<const uint8_t> valid_prices) const {
    Price total = 0.0;
    const size_t count = std::min({positions_.size(), prices.size(), valid_prices.size()});
    for (size_t id = 0; id < count; ++id) {
        if (occupied_[id] != 0 && valid_prices[id] != 0) {
            total += static_cast<Price>(positions_[id].quantity) * prices[id];
        }
    }
    return total;
}

Price PositionTracker::total_realized_pnl() const {
    Price total = 0.0;
    for (size_t id = 0; id < positions_.size(); ++id) {
        if (occupied_[id] != 0) total += positions_[id].realized_pnl;
    }
    return total;
}

}  // namespace qbt

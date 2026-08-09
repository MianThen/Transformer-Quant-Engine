#include "order_book.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace qbt {

OrderBook::OrderBook(std::string symbol, ExecutionConfig config)
    : symbol_(std::move(symbol)), config_(config) {}

std::vector<Fill> OrderBook::submit_order(const Order& order) {
    FillBuffer fills;
    match_order(order, fills);
    return fills;
}

void OrderBook::submit_order(const Order& order, FillBuffer& fills) {
    match_order(order, fills);
}

std::vector<Fill> OrderBook::update_market_data(const MarketSnapshot& md) {
    begin_market_data(md);
    FillBuffer fills;
    match_resting_orders(md, fills);
    return fills;
}

void OrderBook::update_market_data(const MarketSnapshot& md, FillBuffer& fills) {
    begin_market_data(md);
    match_resting_orders(md, fills);
}

void OrderBook::begin_market_data(const MarketSnapshot& md) {
    last_price_ = md.ref_price();
    upper_limit_ = md.upper_limit;
    lower_limit_ = md.lower_limit;
    is_suspended_ = md.is_suspended;
    is_listed_ = md.is_listed;
    has_market_data_ = true;
    if (!is_listed_ || is_suspended_ || md.volume <= 0) {
        remaining_volume_ = 0;
    } else {
        remaining_volume_ = static_cast<Quantity>(std::floor(
            static_cast<double>(md.volume) * config_.max_volume_participation));
    }
}

std::vector<Fill> OrderBook::match_resting_orders(const MarketSnapshot& md) {
    FillBuffer fills;
    match_resting_orders(md, fills);
    return fills;
}

void OrderBook::match_resting_orders(const MarketSnapshot& md, FillBuffer& fills) {
    if (!is_listed_ || is_suspended_ || remaining_volume_ <= 0) return;
    auto fill_orders = [&](auto& levels, auto is_triggered, auto execution_price) {
        auto level = levels.begin();
        while (level != levels.end() && remaining_volume_ > 0) {
            if (!is_triggered(level->first)) {
                ++level;
                continue;
            }
            auto& orders = level->second;
            while (!orders.empty() && remaining_volume_ > 0) {
                Order& order = orders.front();
                const Price price = execution_price(level->first);
                const bool blocked_at_limit = config_.enforce_price_limits &&
                    ((order.side == Side::BUY && upper_limit_ > 0.0 &&
                      price >= upper_limit_) ||
                     (order.side == Side::SELL && lower_limit_ > 0.0 &&
                      price <= lower_limit_));
                if (blocked_at_limit) break;
                const Quantity quantity = std::min(order.quantity, remaining_volume_);
                fills.push_back({order.id, order.symbol, order.side, quantity,
                                 price, 0.0, md.timestamp});
                order.quantity -= quantity;
                remaining_volume_ -= quantity;
                if (order.quantity == 0) orders.pop_front();
            }
            if (orders.empty()) level = levels.erase(level);
            else ++level;
        }
    };
    fill_orders(bids_, [&](Price limit) { return md.low <= limit; },
                [&](Price limit) { return md.open <= limit ? md.open : limit; });
    fill_orders(asks_, [&](Price limit) { return md.high >= limit; },
                [&](Price limit) { return md.open >= limit ? md.open : limit; });
}

Quantity OrderBook::cancel_order(int64_t order_id) {
    Quantity canceled = 0;
    auto cancel_from = [&](auto& levels) {
        for (auto level = levels.begin(); level != levels.end();) {
            auto& orders = level->second;
            for (auto order = orders.begin(); order != orders.end();) {
                if (order->id == order_id) {
                    canceled += order->quantity;
                    order = orders.erase(order);
                } else {
                    ++order;
                }
            }
            if (orders.empty()) level = levels.erase(level);
            else ++level;
        }
    };
    cancel_from(bids_);
    cancel_from(asks_);
    return canceled;
}

Price OrderBook::best_bid() const {
    return bids_.empty() ? 0.0 : bids_.begin()->first;
}

Price OrderBook::best_ask() const {
    return asks_.empty() ? 0.0 : asks_.begin()->first;
}

Price OrderBook::mid_price() const {
    Price bid = best_bid();
    Price ask = best_ask();
    if (bid > 0.0 && ask > 0.0) return (bid + ask) / 2.0;
    return last_price_;
}

std::vector<Fill> OrderBook::match_order(const Order& order) {
    FillBuffer fills;
    match_order(order, fills);
    return fills;
}

void OrderBook::match_order(const Order& order, FillBuffer& fills) {
    if (order.quantity <= 0 || order.symbol != symbol_ ||
        (order.side != Side::BUY && order.side != Side::SELL) ||
        (order.type != OrderType::MARKET && order.type != OrderType::LIMIT) ||
        (order.type == OrderType::LIMIT &&
         (!std::isfinite(order.limit_price) || order.limit_price <= 0.0))) {
        return;
    }

    Quantity remaining = order.quantity;
    const Price reference_price = last_price_;
    auto record_fill = [&](Price price, Quantity quantity) -> Quantity {
        if (!std::isfinite(price) || price <= 0.0 || !is_listed_ ||
            is_suspended_ || (has_market_data_ && remaining_volume_ <= 0)) {
            return 0;
        }
        const bool blocked_at_limit = config_.enforce_price_limits &&
            ((order.side == Side::BUY && upper_limit_ > 0.0 && price >= upper_limit_) ||
             (order.side == Side::SELL && lower_limit_ > 0.0 && price <= lower_limit_));
        if (blocked_at_limit) return 0;
        Quantity executable = quantity;
        if (has_market_data_) executable = std::min(executable, remaining_volume_);
        if (executable <= 0) return 0;
        fills.push_back({order.id, order.symbol, order.side, executable, price,
                         0.0, order.timestamp});
        last_price_ = price;
        remaining -= executable;
        if (has_market_data_) remaining_volume_ -= executable;
        return executable;
    };

    if (order.type == OrderType::MARKET && reference_price > 0.0) {
        const double direction = order.side == Side::BUY ? 1.0 : -1.0;
        const Price slipped = reference_price *
            (1.0 + direction * config_.slippage_bps / 10'000.0);
        record_fill(slipped, remaining);
        return;
    }

    const bool can_fill = reference_price > 0.0 &&
        ((order.side == Side::BUY && reference_price <= order.limit_price) ||
         (order.side == Side::SELL && reference_price >= order.limit_price));
    if (can_fill) record_fill(reference_price, remaining);

    if (remaining > 0) {
        Order resting = order;
        resting.quantity = remaining;
        if (order.side == Side::BUY) {
            bids_[order.limit_price].push_back(std::move(resting));
        } else {
            asks_[order.limit_price].push_back(std::move(resting));
        }
    }
}

}  // namespace qbt

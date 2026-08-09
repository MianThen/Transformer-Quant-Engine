#include "pnl_tracker.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace qbt {

void PnLTracker::record_snapshot(Timestamp time, Price equity, Price cash) {
    constexpr Timestamp kNanosecondsPerDay = 86'400'000'000'000LL;
    const Timestamp day = time / kNanosecondsPerDay;
    if (!has_snapshot_) {
        has_snapshot_ = true;
        first_timestamp_ = time;
        first_equity_ = equity;
        peak_equity_ = equity;
        current_day_ = day;
        daily_equity_.push_back(equity);
    } else if (day == current_day_) {
        daily_equity_.back() = equity;
    } else {
        current_day_ = day;
        daily_equity_.push_back(equity);
    }
    last_timestamp_ = time;
    last_equity_ = equity;
    peak_equity_ = std::max(peak_equity_, equity);
    if (peak_equity_ > 0.0) {
        online_max_drawdown_ = std::max(online_max_drawdown_,
                                       (peak_equity_ - equity) / peak_equity_);
    }
    if (equity_sampling_ == EquitySampling::DISABLED ||
        equity_sampling_ == EquitySampling::ON_FILL) return;
    if (equity_sampling_ == EquitySampling::DAILY && !equity_curve_.empty() &&
        equity_curve_.back().timestamp / kNanosecondsPerDay == day) {
        equity_curve_.back() = {time, equity, cash};
    } else {
        equity_curve_.push_back({time, equity, cash});
    }
}

void PnLTracker::record_trade(const Fill& fill) {
    ++trade_count_;
    TradeRecord rec;
    rec.order_id = fill.order_id;
    rec.symbol = fill.symbol;
    rec.side = fill.side;
    rec.quantity = fill.quantity;
    rec.price = fill.price;
    rec.commission = fill.commission;
    rec.timestamp = fill.timestamp;
    if (record_trades_) trades_.push_back(rec);
    match_round_trip(fill);
}

void PnLTracker::configure_history(bool record_trades, bool record_round_trips,
                                   EquitySampling equity_sampling) {
    record_trades_ = record_trades;
    record_round_trips_ = record_round_trips;
    equity_sampling_ = equity_sampling;
}

void PnLTracker::match_round_trip(const Fill& fill) {
    OpenRoundTripPosition& position = open_round_trips_[fill.symbol];
    const Quantity delta = fill.side == Side::BUY ? fill.quantity : -fill.quantity;
    if (position.quantity == 0 || (position.quantity > 0) == (delta > 0)) {
        const Quantity old_quantity = std::abs(position.quantity);
        const Quantity added = std::abs(delta);
        if (old_quantity == 0) position.opened_at = fill.timestamp;
        position.average_price =
            (position.average_price * static_cast<Price>(old_quantity) +
             fill.price * static_cast<Price>(added)) /
            static_cast<Price>(old_quantity + added);
        position.quantity += delta;
        position.open_commission += fill.commission;
        return;
    }

    const Quantity position_quantity = std::abs(position.quantity);
    const Quantity closed = std::min(position_quantity, std::abs(delta));
    const Price direction = position.quantity > 0 ? 1.0 : -1.0;
    const Price opening_fee = position.open_commission *
                              static_cast<Price>(closed) /
                              static_cast<Price>(position_quantity);
    const Price closing_fee = fill.commission * static_cast<Price>(closed) /
                              static_cast<Price>(fill.quantity);
    const Price gross_pnl = (fill.price - position.average_price) *
                            static_cast<Price>(closed) * direction;
    const Price total_commission = opening_fee + closing_fee;
    RoundTripRecord round_trip{
        fill.symbol,
        position.quantity > 0 ? Side::BUY : Side::SELL,
        closed,
        position.average_price,
        fill.price,
        position.opened_at,
        fill.timestamp,
        gross_pnl,
        total_commission,
        gross_pnl - total_commission,
    };
    ++round_trip_count_;
    if (round_trip.net_pnl > 0.0) ++winning_round_trip_count_;
    if (record_round_trips_) round_trips_.push_back(std::move(round_trip));

    position.open_commission = std::max(position.open_commission - opening_fee, 0.0);
    const Quantity old_quantity = position.quantity;
    position.quantity += delta;
    if (position.quantity == 0) {
        open_round_trips_.erase(fill.symbol);
    } else if ((position.quantity > 0) != (old_quantity > 0)) {
        position.average_price = fill.price;
        position.open_commission = std::max(fill.commission - closing_fee, 0.0);
        position.opened_at = fill.timestamp;
    }
}

void PnLTracker::apply_corporate_action(const std::string& symbol,
                                        double share_multiplier) {
    auto it = open_round_trips_.find(symbol);
    if (it == open_round_trips_.end() || share_multiplier == 1.0) return;
    OpenRoundTripPosition& position = it->second;
    position.quantity = static_cast<Quantity>(std::llround(
        static_cast<double>(position.quantity) * share_multiplier));
    position.average_price /= share_multiplier;
}

double PnLTracker::sharpe_ratio(double risk_free_rate,
                                Price initial_equity) const {
    if (daily_equity_.empty()) return 0.0;
    std::vector<Price> daily_equity = daily_equity_;
    if (initial_equity >= 0.0) {
        daily_equity.insert(daily_equity.begin(), initial_equity);
    }
    if (daily_equity.size() < 3) return 0.0;

    std::vector<double> returns;
    returns.reserve(daily_equity.size() - 1);
    for (size_t i = 1; i < daily_equity.size(); ++i) {
        const Price previous = daily_equity[i - 1];
        if (previous > 0.0) {
            returns.push_back(daily_equity[i] / previous - 1.0);
        }
    }
    if (returns.size() < 2) return 0.0;

    double mean = 0.0;
    for (double value : returns) mean += value;
    mean /= static_cast<double>(returns.size());

    double squared_deviation = 0.0;
    for (double value : returns) {
        const double deviation = value - mean;
        squared_deviation += deviation * deviation;
    }
    const double stdev = std::sqrt(
        squared_deviation / static_cast<double>(returns.size() - 1));
    if (stdev == 0.0) return 0.0;
    return (mean - risk_free_rate / 252.0) / stdev * std::sqrt(252.0);
}

double PnLTracker::max_drawdown(Price initial_equity) const {
    if (!has_snapshot_) return 0.0;
    if (initial_equity < 0.0 || initial_equity <= peak_equity_) return online_max_drawdown_;
    return std::max(online_max_drawdown_, (initial_equity - last_equity_) / initial_equity);
}

double PnLTracker::total_return() const {
    if (!has_snapshot_ || first_equity_ == 0.0) return 0.0;
    return last_equity_ / first_equity_ - 1.0;
}

double PnLTracker::annual_return() const {
    if (!has_snapshot_ || first_equity_ <= 0.0 || last_equity_ < 0.0) return 0.0;
    constexpr double kNanosecondsPerDay = 86'400'000'000'000.0;
    const Timestamp elapsed = last_timestamp_ - first_timestamp_;
    if (elapsed <= 0) return 0.0;
    const double days = static_cast<double>(elapsed) / kNanosecondsPerDay;
    return std::pow(last_equity_ / first_equity_,
                    365.0 / days) - 1.0;
}

double PnLTracker::win_rate() const {
    return round_trip_count_ == 0 ? 0.0 :
        static_cast<double>(winning_round_trip_count_) /
        static_cast<double>(round_trip_count_);
}

}  // namespace qbt

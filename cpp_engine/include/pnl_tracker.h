#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "order.h"
#include "types.h"

namespace qbt {

enum class EquitySampling {
    EVERY_BAR,
    DAILY,
    ON_FILL,
    DISABLED,
};

// 权益曲线上的一个点
struct EquityPoint {
    Timestamp timestamp = 0;
    Price equity = 0.0;  // 总权益 = 现金 + 持仓市值
    Price cash = 0.0;
};

// 逐笔交易记录(供绩效统计和落库)
struct TradeRecord {
    int64_t order_id = 0;
    std::string symbol;
    Side side = Side::BUY;
    Quantity quantity = 0;
    Price price = 0.0;
    Price commission = 0.0;
    Timestamp timestamp = 0;
};

struct RoundTripRecord {
    std::string symbol;
    Side entry_side = Side::BUY;
    Quantity quantity = 0;
    Price entry_price = 0.0;
    Price exit_price = 0.0;
    Timestamp opened_at = 0;
    Timestamp closed_at = 0;
    Price gross_pnl = 0.0;
    Price commission = 0.0;
    Price net_pnl = 0.0;
};

// 盈亏跟踪:记录权益曲线和成交,计算绩效指标
class PnLTracker {
public:
    void record_snapshot(Timestamp time, Price equity, Price cash);
    void record_trade(const Fill& fill);
    void apply_corporate_action(const std::string& symbol,
                                double share_multiplier);
    void configure_history(bool record_trades, bool record_round_trips,
                           EquitySampling equity_sampling);
    EquitySampling equity_sampling() const { return equity_sampling_; }

    // 绩效指标
    double sharpe_ratio(double risk_free_rate = 0.02,
                        Price initial_equity = -1.0) const;
    double max_drawdown(Price initial_equity = -1.0) const;
    double total_return() const;
    double annual_return() const;
    double win_rate() const;

    const std::vector<EquityPoint>& equity_curve() const { return equity_curve_; }
    const std::vector<TradeRecord>& trades() const { return trades_; }
    const std::vector<RoundTripRecord>& round_trips() const {
        return round_trips_;
    }
    std::size_t trade_count() const { return trade_count_; }
    std::size_t equity_point_count() const { return equity_curve_.size(); }

private:
    struct OpenRoundTripPosition {
        Quantity quantity = 0;
        Price average_price = 0.0;
        Price open_commission = 0.0;
        Timestamp opened_at = 0;
    };

    void match_round_trip(const Fill& fill);

    std::vector<EquityPoint> equity_curve_;
    std::vector<TradeRecord> trades_;
    std::vector<RoundTripRecord> round_trips_;
    std::unordered_map<std::string, OpenRoundTripPosition> open_round_trips_;
    std::vector<Price> daily_equity_;
    Timestamp current_day_ = 0;
    Timestamp first_timestamp_ = 0;
    Timestamp last_timestamp_ = 0;
    Price first_equity_ = 0.0;
    Price last_equity_ = 0.0;
    Price peak_equity_ = 0.0;
    double online_max_drawdown_ = 0.0;
    size_t round_trip_count_ = 0;
    size_t trade_count_ = 0;
    size_t winning_round_trip_count_ = 0;
    bool record_trades_ = true;
    bool record_round_trips_ = true;
    bool has_snapshot_ = false;
    EquitySampling equity_sampling_ = EquitySampling::EVERY_BAR;
};

}  // namespace qbt

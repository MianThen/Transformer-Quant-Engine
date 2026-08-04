#pragma once

#include <deque>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "market_data.h"
#include "execution.h"
#include "order.h"
#include "types.h"

namespace qbt {

using FillBuffer = std::vector<Fill>;

// Bar 级撮合器。策略挂单只与 OHLCV 所代表的外部流动性成交，
// 不会互相成为对手方。
class OrderBook {
public:
    explicit OrderBook(std::string symbol,
                       ExecutionConfig config = ExecutionConfig{});

    // 提交订单,返回与当前 Bar 外部流动性的成交。
    std::vector<Fill> submit_order(const Order& order);
    void submit_order(const Order& order, FillBuffer& fills);

    // 更新行情快照,并返回本 bar 价格范围内触发的挂单成交
    std::vector<Fill> update_market_data(const MarketSnapshot& md);
    void update_market_data(const MarketSnapshot& md, FillBuffer& fills);
    void begin_market_data(const MarketSnapshot& md);
    std::vector<Fill> match_resting_orders(const MarketSnapshot& md);
    void match_resting_orders(const MarketSnapshot& md, FillBuffer& fills);
    Quantity cancel_order(int64_t order_id);

    // 设定参考价(如用某根 bar 的 open 价成交延迟到本 bar 的市价单)
    void set_reference_price(Price price) { if (price > 0.0) last_price_ = price; }

    // 最优价查询
    Price best_bid() const;
    Price best_ask() const;
    Price mid_price() const;
    Price last_price() const { return last_price_; }

private:
    // 撮合逻辑:市价单按参考价成交，限价单未成交部分进入等待队列。
    std::vector<Fill> match_order(const Order& order);
    void match_order(const Order& order, FillBuffer& fills);

    std::string symbol_;
    Price last_price_ = 0.0;
    ExecutionConfig config_;
    Quantity remaining_volume_ = 0;
    Price upper_limit_ = 0.0;
    Price lower_limit_ = 0.0;
    bool is_suspended_ = false;
    bool is_listed_ = true;
    bool has_market_data_ = false;

    std::map<Price, std::deque<Order>, std::greater<>> bids_;
    std::map<Price, std::deque<Order>> asks_;
};

}  // namespace qbt

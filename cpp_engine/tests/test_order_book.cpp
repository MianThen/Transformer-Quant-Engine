// 订单簿撮合逻辑单元测试。
// 用极简断言宏,无需引入第三方框架;后续可换成 Catch2/GoogleTest。
//
#include <cassert>
#include <cmath>
#include <iostream>

#include "order.h"
#include "order_book.h"

using namespace qbt;

static int g_failures = 0;

#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::cerr << "FAIL: " << #cond << " @ " << __LINE__ << "\n"; \
            ++g_failures;                                              \
        }                                                              \
    } while (0)

void test_strategy_orders_do_not_supply_counterparty_liquidity() {
    OrderBook book("TEST");

    Order sell1;
    sell1.symbol = "TEST";
    sell1.side = Side::SELL;
    sell1.type = OrderType::LIMIT;
    sell1.quantity = 50;
    sell1.limit_price = 10.0;
    book.submit_order(sell1);

    Order sell2 = sell1;
    sell2.limit_price = 10.1;
    book.submit_order(sell2);

    Order buy;
    buy.symbol = "TEST";
    buy.side = Side::BUY;
    buy.type = OrderType::MARKET;
    buy.quantity = 100;

    std::vector<Fill> fills = book.submit_order(buy);

    CHECK(fills.empty());
    CHECK(std::abs(book.best_ask() - 10.0) < 1e-9);
}

void test_crossed_strategy_limits_both_remain_external_orders() {
    OrderBook book("TEST");
    Order buy;
    buy.symbol = "TEST";
    buy.side = Side::BUY;
    buy.type = OrderType::LIMIT;
    buy.quantity = 25;
    buy.limit_price = 9.9;
    CHECK(book.submit_order(buy).empty());
    CHECK(std::abs(book.best_bid() - 9.9) < 1e-9);

    Order sell;
    sell.symbol = "TEST";
    sell.side = Side::SELL;
    sell.type = OrderType::LIMIT;
    sell.quantity = 25;
    sell.limit_price = 9.8;
    const auto fills = book.submit_order(sell);
    CHECK(fills.empty());
    CHECK(std::abs(book.best_bid() - 9.9) < 1e-9);
    CHECK(std::abs(book.best_ask() - 9.8) < 1e-9);
}

void test_invalid_order_is_rejected() {
    OrderBook book("TEST");
    Order order;
    order.symbol = "OTHER";
    order.quantity = 10;
    CHECK(book.submit_order(order).empty());
}

void test_resting_limit_triggers_on_later_bar() {
    OrderBook book("TEST");
    Order buy;
    buy.id = 42;
    buy.symbol = "TEST";
    buy.side = Side::BUY;
    buy.type = OrderType::LIMIT;
    buy.quantity = 10;
    buy.limit_price = 9.5;
    CHECK(book.submit_order(buy).empty());

    MarketSnapshot md;
    md.symbol = "TEST";
    md.timestamp = 100;
    md.open = 10.0;
    md.high = 10.2;
    md.low = 9.4;
    md.close = 10.1;
    md.volume = 1'000;
    const auto fills = book.update_market_data(md);
    CHECK(fills.size() == 1);
    CHECK(fills.front().order_id == 42);
    CHECK(std::abs(fills.front().price - 9.5) < 1e-9);
    CHECK(fills.front().timestamp == 100);
}

void test_a_share_execution_constraints() {
    ExecutionConfig config;
    config.max_volume_participation = 0.10;
    config.slippage_bps = 10.0;
    OrderBook book("TEST", config);

    MarketSnapshot md;
    md.symbol = "TEST";
    md.timestamp = 100;
    md.open = md.high = md.low = md.close = 10.0;
    md.volume = 100;
    book.update_market_data(md);

    Order buy;
    buy.id = 1;
    buy.symbol = "TEST";
    buy.side = Side::BUY;
    buy.type = OrderType::MARKET;
    buy.quantity = 25;
    buy.timestamp = 100;
    const auto partial = book.submit_order(buy);
    CHECK(partial.size() == 1);
    CHECK(partial.front().quantity == 10);
    CHECK(std::abs(partial.front().price - 10.01) < 1e-9);

    md.timestamp = 200;
    md.volume = 0;
    book.update_market_data(md);
    CHECK(book.submit_order(buy).empty());

    md.timestamp = 300;
    md.volume = 1'000;
    md.is_suspended = true;
    book.update_market_data(md);
    CHECK(book.submit_order(buy).empty());
}

void test_price_limit_blocks_conservative_side() {
    OrderBook book("TEST");
    MarketSnapshot md;
    md.symbol = "TEST";
    md.timestamp = 100;
    md.open = md.high = md.low = md.close = 11.0;
    md.volume = 1'000;
    md.upper_limit = 11.0;
    md.lower_limit = 9.0;
    book.update_market_data(md);

    Order buy;
    buy.symbol = "TEST";
    buy.side = Side::BUY;
    buy.type = OrderType::MARKET;
    buy.quantity = 10;
    CHECK(book.submit_order(buy).empty());

    md.timestamp = 200;
    md.open = md.high = md.low = md.close = 9.0;
    book.update_market_data(md);
    Order sell = buy;
    sell.side = Side::SELL;
    CHECK(book.submit_order(sell).empty());
}

int main() {
    test_strategy_orders_do_not_supply_counterparty_liquidity();
    test_crossed_strategy_limits_both_remain_external_orders();
    test_invalid_order_is_rejected();
    test_resting_limit_triggers_on_later_bar();
    test_a_share_execution_constraints();
    test_price_limit_blocks_conservative_side();

    if (g_failures == 0) {
        std::cout << "test_order_book: all checks passed\n";
        return 0;
    }
    std::cerr << "test_order_book: " << g_failures << " failure(s)\n";
    return 1;
}

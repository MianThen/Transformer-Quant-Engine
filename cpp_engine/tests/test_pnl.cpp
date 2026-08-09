#include <cmath>
#include <iostream>

#include "pnl_tracker.h"
#include "position.h"

using namespace qbt;

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::cerr << "FAIL: " << #cond << " @ " << __LINE__ << "\n"; \
    ++g_failures; } } while (0)

Fill fill(Side side, Quantity quantity, Price price,
          Price commission = 0.0, Timestamp timestamp = 0) {
    Fill result;
    result.symbol = "TEST";
    result.side = side;
    result.quantity = quantity;
    result.price = price;
    result.commission = commission;
    result.timestamp = timestamp;
    return result;
}

void test_round_trip_uses_entry_and_exit_commission() {
    PnLTracker tracker;
    tracker.record_trade(fill(Side::BUY, 10, 10.0, 1.0, 1));
    tracker.record_trade(fill(Side::SELL, 10, 10.15, 1.0, 2));
    const auto& round_trips = tracker.round_trips();
    CHECK(round_trips.size() == 1);
    if (!round_trips.empty()) {
        CHECK(std::abs(round_trips.front().gross_pnl - 1.5) < 1e-12);
        CHECK(std::abs(round_trips.front().commission - 2.0) < 1e-12);
        CHECK(std::abs(round_trips.front().net_pnl + 0.5) < 1e-12);
    }
    CHECK(tracker.win_rate() == 0.0);

    tracker.record_snapshot(1, 99.0, 99.0);
    CHECK(std::abs(tracker.max_drawdown(100.0) - 0.01) < 1e-12);
}

void test_performance_metrics() {
    constexpr Timestamp day = 86'400'000'000'000LL;
    PnLTracker tracker;
    tracker.record_snapshot(0, 100.0, 100.0);
    tracker.record_snapshot(day, 110.0, 110.0);
    tracker.record_snapshot(2 * day, 99.0, 99.0);

    CHECK(std::abs(tracker.total_return() + 0.01) < 1e-12);
    CHECK(std::abs(tracker.max_drawdown() - 0.1) < 1e-12);
    CHECK(tracker.annual_return() < 0.0);
    CHECK(std::isfinite(tracker.sharpe_ratio(0.0)));
}

void test_position_lifecycle_and_win_rate() {
    PositionTracker positions;
    positions.apply_fill(fill(Side::BUY, 10, 10.0));
    positions.apply_fill(fill(Side::BUY, 10, 12.0));
    Position pos = positions.get_position("TEST");
    CHECK(pos.quantity == 20);
    CHECK(std::abs(pos.avg_cost - 11.0) < 1e-12);

    positions.apply_fill(fill(Side::SELL, 25, 13.0));
    pos = positions.get_position("TEST");
    CHECK(pos.quantity == -5);
    CHECK(std::abs(pos.avg_cost - 13.0) < 1e-12);
    CHECK(std::abs(pos.realized_pnl - 40.0) < 1e-12);

    std::unordered_map<std::string, Price> prices{{"TEST", 12.0}};
    CHECK(std::abs(positions.market_value(prices) + 60.0) < 1e-12);

    PnLTracker tracker;
    tracker.record_trade(fill(Side::BUY, 10, 10.0));
    tracker.record_trade(fill(Side::SELL, 10, 11.0));
    tracker.record_trade(fill(Side::SELL, 5, 12.0));
    tracker.record_trade(fill(Side::BUY, 5, 13.0));
    CHECK(std::abs(tracker.win_rate() - 0.5) < 1e-12);
}

int main() {
    test_performance_metrics();
    test_position_lifecycle_and_win_rate();
    test_round_trip_uses_entry_and_exit_commission();
    if (g_failures != 0) return 1;
    std::cout << "test_pnl: all checks passed\n";
    return 0;
}

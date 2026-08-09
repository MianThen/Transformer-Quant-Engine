#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

#include "engine.h"

using namespace qbt;

class TestStrategyRuntime final : public engine_common::IStrategyRuntime {
public:
    engine_common::StrategyStatus start(
        const engine_common::StrategySessionContext&) override {
        started = true;
        return engine_common::StrategyStatus::OK;
    }
    engine_common::StrategyStatus on_market_batch(
        const engine_common::MarketFrameBatchView& market,
        const engine_common::PortfolioView&,
        engine_common::OrderIntentBuffer& output) noexcept override {
        ++batches;
        if (batches != 1) return engine_common::StrategyStatus::OK;
        engine_common::OrderIntent order;
        order.symbol_id = market.bars.front().symbol_id;
        order.quantity = 10;
        order.timestamp = market.asof_timestamp;
        target.symbol_id = order.symbol_id;
        target.target_quantity = order.quantity;
        decision_at = market.asof_timestamp;
        return output.push(order) ? engine_common::StrategyStatus::OK
                                  : engine_common::StrategyStatus::OUTPUT_OVERFLOW;
    }
    void on_execution(const engine_common::ExecutionEvent& execution) noexcept override {
        ++executions;
        last_execution = execution;
    }
    void on_reset(engine_common::ResetReason,
                  engine_common::TimestampNs) noexcept override {}
    engine_common::StrategyDecisionView last_decision() const noexcept override {
        if (decision_at == 0) return {};
        return {1, decision_at, std::span<const engine_common::TargetPosition>(&target, 1)};
    }
    void stop() noexcept override { stopped = true; }

    bool started = false;
    bool stopped = false;
    int batches = 0;
    int executions = 0;
    engine_common::TimestampNs decision_at = 0;
    engine_common::TargetPosition target;
    engine_common::ExecutionEvent last_execution;
};

class TestReplayAnalyticsSink final : public engine_common::IReplayAnalyticsSink {
public:
    engine_common::ReplayAnalyticsStatus on_decision(
        const engine_common::ReplayDecisionEvent& event) noexcept override {
        ++decisions;
        decision_id = event.decision.decision_id;
        target_count = event.decision.targets.size();
        return engine_common::ReplayAnalyticsStatus::OK;
    }

    engine_common::ReplayAnalyticsStatus on_execution(
        const engine_common::ExecutionEvent& event) noexcept override {
        ++executions;
        execution = event;
        return engine_common::ReplayAnalyticsStatus::OK;
    }

    engine_common::ReplayAnalyticsStatus on_replay_end(
        const engine_common::ReplayEndEvent& event) noexcept override {
        ++ends;
        ended_at = event.ended_at;
        ending_positions = event.portfolio.items.size();
        return engine_common::ReplayAnalyticsStatus::OK;
    }

    int decisions = 0;
    int executions = 0;
    int ends = 0;
    std::uint64_t decision_id = 0;
    std::size_t target_count = 0;
    engine_common::TimestampNs ended_at = 0;
    std::size_t ending_positions = 0;
    engine_common::ExecutionEvent execution;
};

static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::cerr << "FAIL: " << #cond << " @ " << __LINE__ << "\n"; \
    ++g_failures; } } while (0)

MarketSnapshot bar(Timestamp timestamp, Price open, Price close) {
    MarketSnapshot md;
    md.symbol = "TEST";
    md.timestamp = timestamp;
    md.open = open;
    md.high = std::max(open, close);
    md.low = std::min(open, close);
    md.close = close;
    md.volume = 1'000;
    return md;
}

MarketSnapshot symbol_bar(const char* symbol, Timestamp timestamp,
                          Price open, Price close) {
    MarketSnapshot md = bar(timestamp, open, close);
    md.symbol = symbol;
    return md;
}

void test_close_fill_updates_cash_and_equity() {
    BacktestEngine engine(1'000.0, FillTiming::CLOSE);
    bool ordered = false;
    engine.set_on_market_data([&](const MarketSnapshot&) {
        if (ordered) return std::vector<Order>{};
        ordered = true;
        Order order;
        order.side = Side::BUY;
        order.type = OrderType::MARKET;
        order.quantity = 10;
        return std::vector<Order>{order};
    });
    engine.push_market_data(bar(1, 10.0, 10.0));
    engine.push_market_data(bar(2, 11.0, 11.0));
    engine.run();

    CHECK(std::abs(engine.get_cash() - 900.0) < 1e-9);
    CHECK(std::abs(engine.get_equity() - 1'010.0) < 1e-9);
    CHECK(engine.get_trade_history().size() == 1);
    CHECK(engine.get_trade_history().front().timestamp == 1);
    CHECK(engine.get_equity_curve().size() == 2);
}

void test_next_open_uses_following_bar_open() {
    BacktestEngine engine(1'000.0, FillTiming::NEXT_OPEN);
    bool ordered = false;
    engine.set_on_market_data([&](const MarketSnapshot&) {
        if (ordered) return std::vector<Order>{};
        ordered = true;
        Order order;
        order.side = Side::BUY;
        order.quantity = 10;
        return std::vector<Order>{order};
    });
    engine.push_market_data(bar(1, 10.0, 10.0));
    engine.push_market_data(bar(2, 12.0, 13.0));
    engine.run();

    const auto trades = engine.get_trade_history();
    CHECK(trades.size() == 1);
    CHECK(std::abs(trades.front().price - 12.0) < 1e-9);
    CHECK(trades.front().timestamp == 2);
    CHECK(std::abs(engine.get_cash() - 880.0) < 1e-9);
    CHECK(std::abs(engine.get_equity() - 1'010.0) < 1e-9);
}

void test_next_open_limit_order_waits_for_following_bar() {
    BacktestEngine engine(1'000.0, FillTiming::NEXT_OPEN);
    bool ordered = false;
    engine.set_on_market_data([&](const MarketSnapshot&) {
        if (ordered) return std::vector<Order>{};
        ordered = true;
        Order order;
        order.side = Side::BUY;
        order.type = OrderType::LIMIT;
        order.limit_price = 10.0;
        order.quantity = 10;
        return std::vector<Order>{order};
    });
    engine.process_market_data(bar(1, 10.0, 10.0));
    CHECK(engine.get_trade_history().empty());
    engine.process_market_data(bar(2, 9.0, 9.0));
    const auto trades = engine.get_trade_history();
    CHECK(trades.size() == 1);
    if (!trades.empty()) {
        CHECK(trades.front().timestamp == 2);
        CHECK(std::abs(trades.front().price - 9.0) < 1e-9);
    }
}

void test_cross_section_is_atomic_and_deterministic() {
    BacktestEngine engine(10'000.0, FillTiming::CLOSE);
    int callback_count = 0;
    engine.set_on_cross_section([&](const std::vector<MarketSnapshot>& batch) {
        ++callback_count;
        CHECK(batch.size() == 2);
        CHECK(batch[0].symbol == "AAA");
        CHECK(batch[1].symbol == "BBB");
        return std::vector<Order>{};
    });
    engine.process_market_data_batch({
        symbol_bar("BBB", 1, 20.0, 20.0),
        symbol_bar("AAA", 1, 10.0, 10.0),
    });
    CHECK(callback_count == 1);
    CHECK(engine.get_equity_curve().size() == 1);
}

void test_commission_callback_matches_cash_and_return() {
    BacktestEngine engine(1'000.0, FillTiming::CLOSE);
    engine.set_commission_fn([](Price notional, bool) { return notional * 0.001; });
    engine.set_on_market_data([](const MarketSnapshot&) {
        Order order;
        order.side = Side::BUY;
        order.quantity = 10;
        return std::vector<Order>{order};
    });
    engine.process_market_data(bar(1, 10.0, 10.0));
    CHECK(std::abs(engine.get_cash() - 899.9) < 1e-9);
    CHECK(std::abs(engine.get_equity() - 999.9) < 1e-9);
    CHECK(std::abs(engine.get_total_return() + 0.0001) < 1e-12);
}

void test_partial_market_order_continues_on_next_bar() {
    ExecutionConfig config;
    config.max_volume_participation = 0.10;
    BacktestEngine engine(10'000.0, FillTiming::NEXT_OPEN, config);
    bool ordered = false;
    engine.set_on_market_data([&](const MarketSnapshot&) {
        if (ordered) return std::vector<Order>{};
        ordered = true;
        Order order;
        order.side = Side::BUY;
        order.quantity = 15;
        return std::vector<Order>{order};
    });
    MarketSnapshot first = bar(1, 10.0, 10.0);
    MarketSnapshot second = bar(2, 11.0, 11.0);
    MarketSnapshot third = bar(3, 12.0, 12.0);
    first.volume = second.volume = third.volume = 100;
    engine.process_market_data(first);
    engine.process_market_data(second);
    engine.process_market_data(third);

    const auto trades = engine.get_trade_history();
    CHECK(trades.size() == 2);
    if (trades.size() == 2) {
        CHECK(trades[0].quantity == 10);
        CHECK(trades[1].quantity == 5);
        CHECK(std::abs(trades[0].price - 11.0) < 1e-9);
        CHECK(std::abs(trades[1].price - 12.0) < 1e-9);
    }
}

void test_t_plus_one_and_sellable_position() {
    constexpr Timestamp kDay1MorningUtc = 1'752'714'000'000'000'000LL;
    constexpr Timestamp kDay1AfternoonUtc = kDay1MorningUtc + 3'600'000'000'000LL;
    constexpr Timestamp kDay2MorningUtc = kDay1MorningUtc + 86'400'000'000'000LL;
    BacktestEngine engine(10'000.0, FillTiming::CLOSE);
    engine.set_on_market_data([&](const MarketSnapshot& md) {
        Order order;
        order.quantity = 100;
        if (md.timestamp == kDay1MorningUtc) {
            order.side = Side::BUY;
            return std::vector<Order>{order};
        }
        order.side = Side::SELL;
        return std::vector<Order>{order};
    });

    engine.process_market_data(bar(kDay1MorningUtc, 10.0, 10.0));
    CHECK(engine.get_position("TEST").quantity == 100);
    CHECK(engine.get_position("TEST").sellable_quantity == 0);
    engine.process_market_data(bar(kDay1AfternoonUtc, 10.0, 10.0));
    CHECK(engine.get_trade_history().size() == 1);
    engine.process_market_data(bar(kDay2MorningUtc, 11.0, 11.0));
    CHECK(engine.get_trade_history().size() == 2);
    CHECK(engine.get_position("TEST").quantity == 0);
    CHECK(engine.get_position("TEST").sellable_quantity == 0);
}

void test_t_plus_one_reserves_open_sell_orders() {
    constexpr Timestamp kDay1 = 1'752'714'000'000'000'000LL;
    constexpr Timestamp kDay2 = kDay1 + 86'400'000'000'000'000LL;
    BacktestEngine engine(10'000.0, FillTiming::CLOSE);
    int phase = 0;
    engine.set_on_market_data([&](const MarketSnapshot&) {
        ++phase;
        Order order;
        order.quantity = 100;
        if (phase == 1) {
            order.side = Side::BUY;
            return std::vector<Order>{order};
        }
        order.side = Side::SELL;
        order.type = OrderType::LIMIT;
        order.limit_price = 20.0;
        return std::vector<Order>{order, order};
    });
    engine.process_market_data(bar(kDay1, 10.0, 10.0));
    engine.process_market_data(bar(kDay2, 10.0, 10.0));
    CHECK(engine.get_trade_history().size() == 1);
}

void test_t_zero_and_explicit_short_selling() {
    ExecutionConfig config;
    config.enforce_t_plus_one = false;
    BacktestEngine naked(10'000.0, FillTiming::CLOSE, config);
    naked.set_on_market_data([](const MarketSnapshot&) {
        Order order;
        order.side = Side::SELL;
        order.quantity = 100;
        return std::vector<Order>{order};
    });
    naked.process_market_data(bar(1, 10.0, 10.0));
    CHECK(naked.get_trade_history().empty());
    CHECK(naked.get_order_history().front().reject_reason ==
          RejectReason::INSUFFICIENT_POSITION);

    config.allow_short = true;
    BacktestEngine short_engine(10'000.0, FillTiming::CLOSE, config);
    short_engine.set_on_market_data([](const MarketSnapshot&) {
        Order order;
        order.side = Side::SELL;
        order.quantity = 100;
        return std::vector<Order>{order};
    });
    short_engine.process_market_data(bar(1, 10.0, 10.0));
    CHECK(short_engine.get_trade_history().size() == 1);
    CHECK(short_engine.get_position("TEST").quantity == -100);
}

void test_native_fee_schedule_uses_fill_timestamp() {
    BacktestEngine engine(10'000.0, FillTiming::CLOSE);
    FeeSchedule before;
    before.effective_from = 0;
    before.effective_to = 2;
    before.commission_rate = 0.001;
    FeeSchedule after;
    after.effective_from = 2;
    after.commission_rate = 0.002;
    engine.set_fee_schedules({before, after});
    engine.set_on_market_data([](const MarketSnapshot&) {
        Order order;
        order.quantity = 10;
        return std::vector<Order>{order};
    });
    engine.process_market_data(bar(1, 10.0, 10.0));
    engine.process_market_data(bar(2, 10.0, 10.0));
    const auto trades = engine.get_trade_history();
    CHECK(trades.size() == 2);
    if (trades.size() == 2) {
        CHECK(std::abs(trades[0].commission - 0.1) < 1e-9);
        CHECK(std::abs(trades[1].commission - 0.2) < 1e-9);
    }
    CHECK(std::abs(engine.get_cash() - 9'799.7) < 1e-9);

    bool rejected = false;
    try {
        engine.set_commission_fn([](Price, bool) { return 0.0; });
    } catch (const std::logic_error&) {
        rejected = true;
    }
    CHECK(rejected);
}

void test_invalid_execution_config_is_rejected() {
    ExecutionConfig config;
    config.max_volume_participation = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try {
        BacktestEngine engine(10'000.0, FillTiming::CLOSE, config);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);

    config = ExecutionConfig{};
    config.slippage_bps = 10'000.0;
    rejected = false;
    try {
        BacktestEngine engine(10'000.0, FillTiming::CLOSE, config);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);

    rejected = false;
    try {
        BacktestEngine engine(-1.0, FillTiming::CLOSE);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
}

void test_invalid_market_data_and_commission_are_rejected() {
    BacktestEngine engine(10'000.0, FillTiming::CLOSE);
    MarketSnapshot invalid = bar(1, 10.0, 10.0);
    invalid.high = 9.0;
    bool rejected = false;
    try {
        engine.process_market_data(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);

    BacktestEngine fee_engine(10'000.0, FillTiming::CLOSE);
    fee_engine.set_commission_fn([](Price, bool) { return -1.0; });
    fee_engine.set_on_market_data([](const MarketSnapshot&) {
        Order order;
        order.quantity = 100;
        return std::vector<Order>{order};
    });
    rejected = false;
    try {
        fee_engine.process_market_data(bar(1, 10.0, 10.0));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
}

void test_order_rejections_are_auditable() {
    BacktestEngine lot_engine(10'000.0, FillTiming::CLOSE);
    lot_engine.set_on_market_data([](const MarketSnapshot&) {
        Order order;
        order.quantity = 10;
        return std::vector<Order>{order};
    });
    MarketSnapshot lot_bar = bar(1, 10.0, 10.0);
    lot_bar.lot_size = 100;
    lot_bar.min_buy_quantity = 100;
    lot_engine.process_market_data(lot_bar);
    const auto lot_orders = lot_engine.get_order_history();
    CHECK(lot_orders.size() == 1);
    CHECK(lot_orders.front().status == OrderStatus::REJECTED);
    CHECK(lot_orders.front().reject_reason == RejectReason::INVALID_LOT_SIZE);

    BacktestEngine cash_engine(500.0, FillTiming::CLOSE);
    cash_engine.set_on_market_data([](const MarketSnapshot&) {
        Order order;
        order.quantity = 100;
        return std::vector<Order>{order};
    });
    cash_engine.process_market_data(bar(1, 10.0, 10.0));
    const auto cash_orders = cash_engine.get_order_history();
    CHECK(cash_orders.size() == 1);
    CHECK(cash_orders.front().reject_reason == RejectReason::INSUFFICIENT_CASH);
    CHECK(cash_engine.get_cash() == 500.0);
}

void test_partial_order_expires_and_limit_order_can_cancel() {
    BacktestEngine engine(10'000.0, FillTiming::CLOSE);
    bool ordered = false;
    engine.set_on_market_data([&](const MarketSnapshot&) {
        if (ordered) return std::vector<Order>{};
        ordered = true;
        Order order;
        order.quantity = 15;
        return std::vector<Order>{order};
    });
    MarketSnapshot low_volume = bar(1, 10.0, 10.0);
    low_volume.volume = 100;
    engine.process_market_data(low_volume);
    CHECK(engine.get_order_history().front().status == OrderStatus::PARTIALLY_FILLED);
    CHECK(engine.get_order_history().front().filled_quantity == 10);
    engine.finalize(2);
    CHECK(engine.get_order_history().front().status == OrderStatus::EXPIRED);

    BacktestEngine cancel_engine(10'000.0, FillTiming::CLOSE);
    cancel_engine.set_on_market_data([](const MarketSnapshot&) {
        Order order;
        order.type = OrderType::LIMIT;
        order.limit_price = 9.0;
        order.quantity = 100;
        return std::vector<Order>{order};
    });
    cancel_engine.process_market_data(bar(1, 10.0, 10.0));
    CHECK(cancel_engine.cancel_order(1, 2));
    CHECK(cancel_engine.get_order_history().front().status == OrderStatus::CANCELED);
    CHECK(!cancel_engine.cancel_order(1, 3));
}

void test_corporate_action_and_portfolio_risk() {
    BacktestEngine engine(10'000.0, FillTiming::CLOSE);
    bool ordered = false;
    engine.set_on_market_data([&](const MarketSnapshot&) {
        if (ordered) return std::vector<Order>{};
        ordered = true;
        Order order;
        order.quantity = 100;
        return std::vector<Order>{order};
    });
    MarketSnapshot md = bar(1, 10.0, 10.0);
    md.industry = "Bank";
    md.factor_exposures["size"] = 0.8;
    engine.process_market_data(md);

    CorporateAction action;
    action.symbol = "TEST";
    action.timestamp = 2;
    action.cash_dividend_per_share = 0.2;
    action.share_multiplier = 1.5;
    const auto result = engine.apply_corporate_action(action);
    CHECK(result.old_quantity == 100);
    CHECK(result.new_quantity == 150);
    CHECK(std::abs(result.cash_dividend - 20.0) < 1e-9);
    CHECK(engine.get_position("TEST").quantity == 150);
    CHECK(std::abs(engine.get_position("TEST").avg_cost - 10.0 / 1.5) < 1e-9);
    CHECK(std::abs(engine.get_cash() - 9'020.0) < 1e-9);

    const auto portfolio = engine.get_portfolio_snapshot();
    CHECK(portfolio.position_count == 1);
    CHECK(portfolio.gross_exposure > 0.0);
    CHECK(portfolio.net_exposure == portfolio.gross_exposure);
    CHECK(portfolio.industry_exposure.count("Bank") == 1);
    CHECK(portfolio.factor_exposure.count("size") == 1);
    CHECK(std::abs(engine.get_equity() - portfolio.equity) < 1e-9);
}

void test_external_order_and_fill_events_are_rejected() {
    BacktestEngine engine(10'000.0, FillTiming::CLOSE);
    Event order_event;
    order_event.type = EventType::ORDER;
    order_event.data = Order{};
    bool rejected = false;
    try {
        engine.add_event(order_event);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);

    Event fill_event;
    fill_event.type = EventType::FILL;
    fill_event.data = Fill{};
    rejected = false;
    try {
        engine.add_event(fill_event);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CHECK(rejected);
}

void test_order_update_callback_can_cancel_pending_remainder() {
    ExecutionConfig config;
    config.max_volume_participation = 0.10;
    BacktestEngine engine(10'000.0, FillTiming::NEXT_OPEN, config);
    bool ordered = false;
    engine.set_on_market_data([&](const MarketSnapshot&) {
        if (ordered) return std::vector<Order>{};
        ordered = true;
        Order order;
        order.side = Side::BUY;
        order.quantity = 15;
        return std::vector<Order>{order};
    });
    engine.set_on_order_update([&](const OrderRecord& record) {
        if (record.status == OrderStatus::PARTIALLY_FILLED) {
            CHECK(engine.cancel_order(record.order.id, record.updated_timestamp));
        }
    });
    MarketSnapshot first = bar(1, 10.0, 10.0);
    MarketSnapshot second = bar(2, 11.0, 11.0);
    first.volume = second.volume = 100;
    engine.process_market_data(first);
    engine.process_market_data(second);
    CHECK(engine.get_trade_history().size() == 1);
    CHECK(engine.get_order_history().front().status == OrderStatus::CANCELED);
}

void test_sorted_span_matches_safe_batch() {
    ExecutionConfig config;
    config.enforce_t_plus_one = false;
    BacktestEngine safe(10'000.0, FillTiming::CLOSE, config);
    BacktestEngine sorted(10'000.0, FillTiming::CLOSE, config);
    auto install = [](BacktestEngine& engine) {
        engine.set_on_cross_section([ordered = false](
                                        const std::vector<MarketSnapshot>& batch) mutable {
            if (ordered) return std::vector<Order>{};
            ordered = true;
            Order order;
            order.symbol = batch.front().symbol;
            order.quantity = 10;
            return std::vector<Order>{order};
        });
    };
    install(safe);
    install(sorted);
    std::vector<MarketSnapshot> unordered{bar(1, 10.0, 10.0), bar(1, 20.0, 20.0)};
    unordered[0].symbol = "BBB";
    unordered[1].symbol = "AAA";
    std::vector<MarketSnapshot> ordered{unordered[1], unordered[0]};
    safe.process_market_data_batch(unordered);
    sorted.process_market_data_batch_sorted(ordered);
    CHECK(std::abs(safe.get_cash() - sorted.get_cash()) < 1e-9);
    CHECK(std::abs(safe.get_equity() - sorted.get_equity()) < 1e-9);
    CHECK(safe.get_trade_count() == sorted.get_trade_count());
    CHECK(safe.get_order_count() == sorted.get_order_count());
}

void test_lightweight_history_preserves_online_metrics() {
    ExecutionConfig config;
    config.enforce_t_plus_one = false;
    BacktestEngine engine(10'000.0, FillTiming::CLOSE, config);
    HistoryConfig history;
    history.record_orders = false;
    history.record_trades = false;
    history.record_round_trips = false;
    history.equity_sampling = EquitySampling::DISABLED;
    engine.set_history_config(history);
    bool ordered = false;
    engine.set_on_market_data([&ordered](const MarketSnapshot&) {
        if (ordered) return std::vector<Order>{};
        ordered = true;
        Order order;
        order.quantity = 10;
        return std::vector<Order>{order};
    });
    engine.process_market_data(bar(1, 10.0, 10.0));
    CHECK(engine.get_order_count() == 1);
    CHECK(engine.get_trade_count() == 1);
    CHECK(engine.get_order_history().empty());
    CHECK(engine.get_trade_history().empty());
    CHECK(engine.get_round_trip_history().empty());
    CHECK(engine.get_equity_curve().empty());
    CHECK(engine.get_position("TEST").quantity == 10);
}

void test_order_history_page_is_direct_and_ordered() {
    BacktestEngine engine(10'000.0, FillTiming::CLOSE);
    engine.set_on_market_data([](const MarketSnapshot& market) {
        Order order;
        order.symbol = market.symbol;
        order.quantity = 100;
        order.type = OrderType::LIMIT;
        order.limit_price = 1.0;
        return std::vector<Order>{order};
    });
    engine.process_market_data(bar(1, 10.0, 10.0));
    engine.set_on_market_data([](const MarketSnapshot&) { return std::vector<Order>{}; });
    CHECK(engine.cancel_order(1, 2));
    const auto first = engine.get_order_history_page(0, 1);
    const auto second = engine.get_order_history_page(1, 1);
    CHECK(first.size() == 1);
    CHECK(second.empty());
    CHECK(first.front().order.id == 1);
    CHECK(first.front().status == OrderStatus::CANCELED);
}

void test_strategy_runtime_is_mutually_exclusive_and_uses_execution_rules() {
    ExecutionConfig config;
    config.enforce_t_plus_one = false;
    BacktestEngine engine(10'000.0, FillTiming::CLOSE, config);
    auto runtime = std::make_shared<TestStrategyRuntime>();
    auto analytics = std::make_shared<TestReplayAnalyticsSink>();
    engine.set_strategy_runtime(runtime);
    engine.set_replay_analytics_sink(analytics);
    bool rejected_callback = false;
    try {
        engine.set_on_market_data([](const MarketSnapshot&) {
            return std::vector<Order>{};
        });
    } catch (const std::logic_error&) {
        rejected_callback = true;
    }
    CHECK(rejected_callback);
    engine.process_market_data(bar(1, 10.0, 10.0));
    CHECK(runtime->started);
    CHECK(runtime->batches == 1);
    CHECK(runtime->executions == 1);
    CHECK(runtime->last_execution.cumulative_quantity == 10);
    CHECK(runtime->last_execution.decision_id == 1);
    CHECK(runtime->last_execution.symbol_id == 0);
    CHECK(runtime->last_execution.side == engine_common::Side::BUY);
    CHECK(runtime->last_execution.price_scale == 10'000);
    CHECK(runtime->last_execution.fee_scale == 10'000);
    CHECK((runtime->last_execution.audit_flags &
           engine_common::EXECUTION_HAS_SYMBOL) != 0);
    CHECK((runtime->last_execution.audit_flags &
           engine_common::EXECUTION_HAS_EXPLICIT_FEE) != 0);
    CHECK((runtime->last_execution.audit_flags &
           engine_common::EXECUTION_HAS_DECISION_ID) != 0);
    CHECK(engine.get_order_history().front().order.decision_id == 1);
    CHECK(engine.get_position("TEST").quantity == 10);
    CHECK(analytics->decisions == 1);
    CHECK(analytics->decision_id == 1);
    CHECK(analytics->target_count == 1);
    CHECK(analytics->executions == 1);
    CHECK(analytics->execution.symbol_id == 0);
    engine.finalize(2);
    CHECK(analytics->ends == 1);
    CHECK(analytics->ended_at == 2);
    CHECK(analytics->ending_positions == 1);

    BacktestEngine callback_engine;
    callback_engine.set_on_market_data([](const MarketSnapshot&) {
        return std::vector<Order>{};
    });
    bool rejected_runtime = false;
    try {
        callback_engine.set_strategy_runtime(std::make_shared<TestStrategyRuntime>());
    } catch (const std::logic_error&) {
        rejected_runtime = true;
    }
    CHECK(rejected_runtime);
}

int main() {
    test_close_fill_updates_cash_and_equity();
    test_next_open_uses_following_bar_open();
    test_next_open_limit_order_waits_for_following_bar();
    test_cross_section_is_atomic_and_deterministic();
    test_commission_callback_matches_cash_and_return();
    test_partial_market_order_continues_on_next_bar();
    test_t_plus_one_and_sellable_position();
    test_t_plus_one_reserves_open_sell_orders();
    test_t_zero_and_explicit_short_selling();
    test_native_fee_schedule_uses_fill_timestamp();
    test_invalid_execution_config_is_rejected();
    test_invalid_market_data_and_commission_are_rejected();
    test_order_rejections_are_auditable();
    test_partial_order_expires_and_limit_order_can_cancel();
    test_corporate_action_and_portfolio_risk();
    test_external_order_and_fill_events_are_rejected();
    test_order_update_callback_can_cancel_pending_remainder();
    test_sorted_span_matches_safe_batch();
    test_lightweight_history_preserves_online_metrics();
    test_order_history_page_is_direct_and_ordered();
    test_strategy_runtime_is_mutually_exclusive_and_uses_execution_rules();
    if (g_failures != 0) return 1;
    std::cout << "test_engine: all checks passed\n";
    return 0;
}

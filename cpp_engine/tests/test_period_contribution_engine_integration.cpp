#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "engine.h"
#include "performance_analytics/period_contribution_replay_sink.h"

namespace {

qbt::MarketSnapshot bar(qbt::Timestamp timestamp, double close) {
    qbt::MarketSnapshot value;
    value.symbol = "TEST";
    value.timestamp = timestamp;
    value.open = close;
    value.high = close;
    value.low = close;
    value.close = close;
    value.volume = 10'000;
    value.lot_size = 1;
    value.industry = "Bank";
    return value;
}

qbt::MarketSnapshot symbol_bar(const char* symbol, qbt::Timestamp timestamp,
                               double close) {
    auto value = bar(timestamp, close);
    value.symbol = symbol;
    return value;
}

performance_analytics::PerformanceSpecV1 spec() {
    performance_analytics::PerformanceSpecV1 value;
    value.frequency = performance_analytics::ReturnFrequency::DAILY;
    value.calendar_id = "XSHG_TRADING_DAY_V1";
    value.calendar_periods_per_year = 242.0;
    value.benchmark_id = "INTERNAL_FROZEN_CONTROL";
    value.config_hash = 48;
    return value;
}

}  // namespace

int main() {
    qbt::ExecutionConfig execution;
    execution.enforce_t_plus_one = false;
    qbt::BacktestEngine engine(1'000.0, qbt::FillTiming::CLOSE, execution);
    auto analytics =
        std::make_shared<performance_analytics::PeriodContributionReplaySink>(
            spec());
    engine.set_replay_analytics_sink(analytics);
    engine.set_on_market_data([](const qbt::MarketSnapshot&) {
        return std::vector<qbt::Order>{};
    });
    engine.process_market_data(bar(100, 10.0));
    engine.open_performance_period(20260801, 100);

    bool ordered = false;
    engine.set_on_market_data([&](const qbt::MarketSnapshot&) {
        if (ordered) return std::vector<qbt::Order>{};
        ordered = true;
        qbt::Order order;
        order.side = qbt::Side::BUY;
        order.quantity = 10;
        return std::vector<qbt::Order>{order};
    });
    engine.process_market_data(bar(150, 10.0));

    qbt::CorporateAction action;
    action.symbol = "TEST";
    action.timestamp = 160;
    action.cash_dividend_per_share = 1.0;
    action.share_multiplier = 2.0;
    const auto action_result = engine.apply_corporate_action(action);
    if (action_result.action_id != 1 || action_result.old_quantity != 10 ||
        action_result.new_quantity != 20 ||
        std::abs(action_result.cash_dividend - 10.0) > 1e-10) {
        return 1;
    }

    engine.set_on_market_data([](const qbt::MarketSnapshot&) {
        return std::vector<qbt::Order>{};
    });
    engine.process_market_data(bar(200, 5.0));
    engine.close_performance_period(20260801, 200);
    engine.finalize(201);

    const auto records = analytics->ledger().records();
    if (analytics->failed() || records.size() != 1 ||
        records.front().securities.size() != 1 ||
        records.front().securities.front().corporate_action_quantity_delta != 10 ||
        records.front().securities.front().pit_industry_id == 0 ||
        std::abs(records.front().executed_gross_pnl) > 1e-10 ||
        std::abs(records.front().corporate_action_cash - 10.0) > 1e-10 ||
        std::abs(records.front().security_net_contribution - 10.0) > 1e-10 ||
        std::abs(records.front().accounting_residual) > 1e-10) {
        return 1;
    }
    const auto return_records = analytics->return_ledger().records();
    if (return_records.size() != 1 ||
        std::abs(return_records.front().period_return - 0.01) > 1e-10 ||
        analytics->return_ledger().ledger_hash() == 0) {
        return 1;
    }

    qbt::BacktestEngine unclosed_engine(1'000.0, qbt::FillTiming::CLOSE,
                                        execution);
    auto unclosed_analytics =
        std::make_shared<performance_analytics::PeriodContributionReplaySink>(
            spec());
    unclosed_engine.set_replay_analytics_sink(unclosed_analytics);
    unclosed_engine.process_market_data(bar(100, 10.0));
    unclosed_engine.open_performance_period(20260801, 100);
    bool unclosed_rejected = false;
    try {
        unclosed_engine.finalize(101);
    } catch (const std::logic_error&) {
        unclosed_rejected = true;
    }
    if (!unclosed_rejected || !unclosed_analytics->ledger().records().empty()) {
        return 1;
    }

    qbt::BacktestEngine stale_engine(2'000.0, qbt::FillTiming::CLOSE,
                                     execution);
    auto stale_analytics =
        std::make_shared<performance_analytics::PeriodContributionReplaySink>(
            spec());
    stale_engine.set_replay_analytics_sink(stale_analytics);
    stale_engine.process_market_data_batch({
        symbol_bar("AAA", 100, 10.0),
        symbol_bar("BBB", 100, 20.0),
    });
    stale_engine.open_performance_period(20260801, 100);
    stale_engine.process_market_data(symbol_bar("AAA", 200, 11.0));
    bool stale_mark_rejected = false;
    try {
        stale_engine.close_performance_period(20260801, 200);
    } catch (const std::runtime_error&) {
        stale_mark_rejected = true;
    }
    if (!stale_mark_rejected || !stale_analytics->ledger().records().empty()) {
        return 1;
    }

    std::printf("test_period_contribution_engine_integration: all checks passed\n");
    return 0;
}

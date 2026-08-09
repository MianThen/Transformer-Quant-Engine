#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "engine.h"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {
using namespace qbt;
using Clock = std::chrono::steady_clock;
struct Result { std::string name, category; std::uint64_t scale, operations, elapsed_ns, checksum, peak_rss_bytes; };

std::uint64_t peak_rss_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    return GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))
        ? static_cast<std::uint64_t>(counters.PeakWorkingSetSize) : 0;
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
#endif
}

std::string symbol(std::size_t index) {
    std::ostringstream out; out << 'S' << std::setw(7) << std::setfill('0') << index; return out.str();
}
MarketSnapshot market(std::string name, Timestamp time, Price price, Quantity volume = 1'000'000) {
    MarketSnapshot md; md.symbol = std::move(name); md.timestamp = time;
    md.open = md.close = price; md.high = price * 1.01; md.low = price * 0.99;
    md.volume = volume; md.lot_size = md.min_buy_quantity = 1; return md;
}
ExecutionConfig config(double participation = 1.0) {
    ExecutionConfig value; value.max_volume_participation = participation;
    value.enforce_price_limits = value.enforce_t_plus_one = false;
    value.enforce_board_lot = value.enforce_cash = false; return value;
}
template <class Function>
Result timed(std::string name, std::string category, std::uint64_t scale,
             std::uint64_t operations, Function function) {
    const auto start = Clock::now(); const auto checksum = function();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    return {std::move(name), std::move(category), scale, operations,
            static_cast<std::uint64_t>(elapsed), checksum, peak_rss_bytes()};
}

Result single_symbol(bool quick) {
    const std::uint64_t count = quick ? 10'000 : 10'000'000;
    return timed("single_symbol_market_data", "market_data", 1, count, [&] {
        BacktestEngine engine;
        HistoryConfig history;
        history.record_orders = false;
        history.record_trades = false;
        history.record_round_trips = false;
        history.equity_sampling = EquitySampling::DISABLED;
        engine.set_history_config(history);
        for (std::uint64_t index = 0; index < count; ++index)
            engine.process_market_data(market("SINGLE", index + 1, 100.0));
        return static_cast<std::uint64_t>(engine.get_position_count());
    });
}
Result cross_section(std::size_t count, bool quick, bool direct_sorted) {
    const std::uint64_t repeats = quick ? 2 : (count == 1'000 ? 10 : 5);
    std::vector<MarketSnapshot> batch; batch.reserve(count);
    for (std::size_t index = 0; index < count; ++index) batch.push_back(market(symbol(index), 1, 50.0));
    const std::string prefix = direct_sorted ? "sorted_span_cross_section_" : "safe_sorted_cross_section_";
    return timed(prefix + std::to_string(count), "market_data", count, count * repeats, [&] {
        BacktestEngine engine;
        for (std::uint64_t repeat = 0; repeat < repeats; ++repeat) {
            for (auto& md : batch) md.timestamp = repeat + 1;
            if (direct_sorted) engine.process_market_data_batch_sorted(batch);
            else engine.process_market_data_batch(batch);
        }
        return static_cast<std::uint64_t>(engine.get_equity_point_count());
    });
}
Result unsorted_cross_section(std::size_t count, bool quick) {
    const std::uint64_t repeats = quick ? 2 : 5;
    std::vector<MarketSnapshot> batch; batch.reserve(count);
    for (std::size_t index = 0; index < count; ++index) batch.push_back(market(symbol(count - index - 1), 1, 50.0));
    return timed("safe_unsorted_cross_section_" + std::to_string(count), "market_data", count,
                 count * repeats, [&] {
        BacktestEngine engine;
        for (std::uint64_t repeat = 0; repeat < repeats; ++repeat) {
            for (auto& md : batch) md.timestamp = repeat + 1;
            engine.process_market_data_batch(batch);
        }
        return static_cast<std::uint64_t>(engine.get_equity_point_count());
    });
}
Result open_orders(std::size_t count) {
    ExecutionConfig cash_config = config();
    cash_config.enforce_cash = true;
    BacktestEngine engine(1e12, FillTiming::CLOSE, cash_config);
    engine.set_on_market_data([count](const MarketSnapshot& md) {
        std::vector<Order> orders(count);
        for (std::size_t index = 0; index < count; ++index) {
            orders[index].symbol = md.symbol; orders[index].type = OrderType::LIMIT;
            orders[index].quantity = 1; orders[index].limit_price = 1.0 + index * 1e-7;
        }
        return orders;
    });
    return timed("open_buy_orders_" + std::to_string(count), "orders", count, count, [&] {
        engine.process_market_data(market("ORDERS", 1, 100.0));
        return static_cast<std::uint64_t>(engine.get_order_count());
    });
}
Result fill_partial_cancel(bool quick) {
    const std::uint64_t count = quick ? 1'000 : 20'000;
    return timed("high_frequency_fill_partial_cancel", "orders", count, count, [&] {
        BacktestEngine engine(1e12, FillTiming::CLOSE, config(0.1)); bool emit = false; std::int64_t order_id = 0;
        engine.set_on_order_update([&](const OrderRecord& record) { order_id = std::max(order_id, record.order.id); });
        engine.set_on_market_data([&](const MarketSnapshot& md) {
            if (!emit) return std::vector<Order>{};
            emit = false;
            Order order; order.symbol = md.symbol; order.quantity = 100; return std::vector<Order>{order};
        });
        for (std::uint64_t index = 0; index < count; ++index) {
            emit = true; engine.process_market_data(market("FILL", index + 1, 10.0, 100));
            engine.cancel_order(order_id, index + 1);
        }
        return static_cast<std::uint64_t>(engine.get_trade_count());
    });
}
Result positions(std::size_t count) {
    std::vector<MarketSnapshot> batch; batch.reserve(count);
    for (std::size_t index = 0; index < count; ++index) batch.push_back(market(symbol(index), 1, 20.0, 10));
    BacktestEngine engine(1e12, FillTiming::CLOSE, config()); bool emit = true;
    engine.set_on_cross_section([&](const std::vector<MarketSnapshot>& snapshots) {
        if (!emit) return std::vector<Order>{};
        emit = false;
        std::vector<Order> orders(snapshots.size());
        for (std::size_t index = 0; index < snapshots.size(); ++index) {
            orders[index].symbol = snapshots[index].symbol; orders[index].quantity = 1;
        }
        return orders;
    });
    engine.process_market_data_batch(batch);
    for (auto& md : batch) { md.timestamp = 2; md.open += .1; md.high += .1; md.low += .1; md.close += .1; }
    return timed("positions_mark_to_market_" + std::to_string(count), "positions", count, count, [&] {
        engine.process_market_data_batch(batch); return static_cast<std::uint64_t>(engine.get_position_count());
    });
}
Result limit_levels(std::size_t count) {
    BacktestEngine engine(1e12, FillTiming::CLOSE, config());
    engine.set_on_market_data([count](const MarketSnapshot& md) {
        std::vector<Order> orders(count);
        for (std::size_t index = 0; index < count; ++index) {
            orders[index].symbol = md.symbol; orders[index].type = OrderType::LIMIT;
            orders[index].quantity = 1; orders[index].limit_price = 1.0 + index * 1e-5;
        }
        return orders;
    });
    return timed("untriggered_limit_levels_" + std::to_string(count), "orders", count, count, [&] {
        engine.process_market_data(market("LEVELS", 1, 100.0));
        return static_cast<std::uint64_t>(engine.get_order_count());
    });
}
Result million_history(bool quick) {
    const std::size_t count = quick ? 10'000 : 1'000'000;
    BacktestEngine engine(1e12, FillTiming::CLOSE, config());
    engine.set_on_market_data([count](const MarketSnapshot& md) {
        std::vector<Order> orders(count); for (auto& order : orders) order.symbol = md.symbol; return orders;
    });
    engine.process_market_data(market("HISTORY", 1, 10.0)); engine.set_on_market_data({});
    return timed("million_order_history_delist_corporate_action", "orders", count, 2, [&] {
        auto delisted = market("HISTORY", 2, 10.0); delisted.is_listed = false; engine.process_market_data(delisted);
        CorporateAction action; action.symbol = "HISTORY"; action.timestamp = 3; engine.apply_corporate_action(action);
        return static_cast<std::uint64_t>(engine.get_order_count());
    });
}
Result market_rules(bool quick) {
    const std::uint64_t count = quick ? 1'000 : 50'000;
    BacktestEngine engine(100'000.0, FillTiming::CLOSE);
    engine.set_on_market_data([](const MarketSnapshot& md) {
        Order order;
        order.symbol = md.symbol;
        order.quantity = md.lot_size;
        order.limit_price = md.upper_limit;
        order.type = OrderType::LIMIT;
        return std::vector<Order>{order};
    });
    return timed("corporate_action_and_market_rules", "behavior", count, count, [&] {
        for (std::uint64_t index = 0; index < count; ++index) {
            auto md = market("RULES", index + 1, 10.0, 10'000);
            md.lot_size = md.min_buy_quantity = 100;
            md.upper_limit = 11.0;
            md.lower_limit = 9.0;
            md.is_suspended = index % 17 == 0;
            engine.process_market_data(md);
        }
        CorporateAction action;
        action.symbol = "RULES";
        action.timestamp = count + 1;
        action.share_multiplier = 1.0;
        engine.apply_corporate_action(action);
        return static_cast<std::uint64_t>(engine.get_order_count());
    });
}
void print_results(const std::vector<Result>& results, bool quick) {
    std::cout << std::setprecision(17) << "{\"schema_version\":1,\"dataset\":{\"name\":\"qbt-m0-deterministic\","
              << "\"version\":1,\"seed\":0,\"quick\":" << (quick ? "true" : "false") << "},"
              << "\"build\":{\"type\":\"" << QBT_BUILD_TYPE << "\",\"compiler_id\":\"" << QBT_COMPILER_ID
              << "\",\"compiler_version\":\"" << QBT_COMPILER_VERSION << "\",\"lto\":"
              << (QBT_LTO_ENABLED ? "true" : "false") << "},\"results\":[";
    for (std::size_t index = 0; index < results.size(); ++index) {
        if (index) std::cout << ',';
        const auto& result = results[index];
        std::cout << "{\"name\":\"" << result.name << "\",\"category\":\"" << result.category
                  << "\",\"scale\":" << result.scale << ",\"operations\":" << result.operations
                  << ",\"elapsed_ns\":" << result.elapsed_ns << ",\"ns_per_operation\":"
                   << static_cast<double>(result.elapsed_ns) / result.operations
                   << ",\"checksum\":" << result.checksum
                   << ",\"peak_rss_bytes\":" << result.peak_rss_bytes << '}';
    }
    std::cout << "]}\n";
}
}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
    std::vector<Result> results{single_symbol(quick),
        cross_section(1'000, quick, false), cross_section(1'000, quick, true),
        cross_section(10'000, quick, false), cross_section(10'000, quick, true),
        unsorted_cross_section(quick ? 1'000 : 10'000, quick)};
    for (std::size_t count : {1'000U, 2'000U, 4'000U, 8'000U}) results.push_back(open_orders(count));
    results.push_back(fill_partial_cancel(quick));
    for (std::size_t count : {1'000U, 10'000U, 50'000U}) results.push_back(positions(quick ? std::min(count, std::size_t{2'000}) : count));
    for (std::size_t count : {10'000U, 50'000U, 100'000U}) results.push_back(limit_levels(quick ? std::min(count, std::size_t{5'000}) : count));
    results.push_back(million_history(quick));
    results.push_back(market_rules(quick));
    print_results(results, quick);
}

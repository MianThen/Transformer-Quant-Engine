#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "engine.h"

namespace {
using namespace qbt;
constexpr Timestamp kDay = 86'400'000'000'000LL;

const char* status_name(OrderStatus status) {
    switch (status) {
        case OrderStatus::ACCEPTED: return "ACCEPTED";
        case OrderStatus::PARTIALLY_FILLED: return "PARTIALLY_FILLED";
        case OrderStatus::FILLED: return "FILLED";
        case OrderStatus::CANCELED: return "CANCELED";
        case OrderStatus::REJECTED: return "REJECTED";
        case OrderStatus::EXPIRED: return "EXPIRED";
    }
    return "UNKNOWN";
}

MarketSnapshot market(Timestamp time, Price open, Price high, Price low,
                      Price close, Quantity volume) {
    MarketSnapshot md; md.symbol = "AAA"; md.timestamp = time;
    md.open = open; md.high = high; md.low = low; md.close = close; md.volume = volume;
    md.lot_size = md.min_buy_quantity = 1; return md;
}

std::string replay() {
    ExecutionConfig config; config.max_volume_participation = 0.5;
    config.enforce_price_limits = config.enforce_t_plus_one = false;
    config.enforce_board_lot = false;
    BacktestEngine engine(100'000.0, FillTiming::CLOSE, config);
    std::vector<MarketSnapshot> inputs;
    std::vector<Fill> fills;
    bool emit = true;
    engine.set_on_fill([&](const Fill& fill) { fills.push_back(fill); });
    engine.set_on_market_data([&](const MarketSnapshot& md) {
        if (!emit) return std::vector<Order>{};
        emit = false;
        Order market_order; market_order.symbol = md.symbol; market_order.quantity = 100;
        Order limit_order; limit_order.symbol = md.symbol; limit_order.type = OrderType::LIMIT;
        limit_order.quantity = 50; limit_order.limit_price = 9.5;
        return std::vector<Order>{market_order, limit_order};
    });
    inputs.push_back(market(kDay, 10.0, 10.2, 9.8, 10.0, 100)); engine.process_market_data(inputs.back());
    inputs.push_back(market(2 * kDay, 9.4, 10.0, 9.0, 9.6, 200)); engine.process_market_data(inputs.back());
    CorporateAction action; action.symbol = "AAA"; action.timestamp = 3 * kDay;
    action.cash_dividend_per_share = 0.1; action.share_multiplier = 2.0;
    action.description = "2-for-1 split plus cash dividend"; engine.apply_corporate_action(action);
    inputs.push_back(market(4 * kDay, 4.9, 5.1, 4.8, 5.0, 1'000)); engine.process_market_data(inputs.back());
    engine.finalize(5 * kDay);

    const auto orders = engine.get_order_history(); const auto positions = engine.get_positions();
    const auto curve = engine.get_equity_curve(); const auto actions = engine.get_corporate_action_history();
    std::ostringstream out; out << std::setprecision(17);
    out << "{\"schema_version\":1,\"dataset\":\"qbt-m0-golden-v1\",\"inputs\":[";
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (i) out << ',';
        const auto& md = inputs[i];
        out << "{\"symbol\":\"" << md.symbol << "\",\"timestamp\":" << md.timestamp
            << ",\"open\":" << md.open << ",\"high\":" << md.high << ",\"low\":" << md.low
            << ",\"close\":" << md.close << ",\"volume\":" << md.volume << '}';
    }
    out << "],\"orders\":[";
    for (std::size_t i = 0; i < orders.size(); ++i) {
        if (i) out << ',';
        const auto& record = orders[i];
        out << "{\"id\":" << record.order.id << ",\"symbol\":\"" << record.order.symbol
            << "\",\"side\":\"" << to_string(record.order.side) << "\",\"type\":\""
            << (record.order.type == OrderType::MARKET ? "MARKET" : "LIMIT")
            << "\",\"quantity\":" << record.order.quantity << ",\"limit_price\":" << record.order.limit_price
            << ",\"filled_quantity\":" << record.filled_quantity << ",\"avg_fill_price\":" << record.avg_fill_price
            << ",\"status\":\"" << status_name(record.status) << "\"}";
    }
    out << "],\"fills\":[";
    for (std::size_t i = 0; i < fills.size(); ++i) {
        if (i) out << ',';
        const auto& fill = fills[i];
        out << "{\"order_id\":" << fill.order_id << ",\"symbol\":\"" << fill.symbol
            << "\",\"side\":\"" << to_string(fill.side) << "\",\"quantity\":" << fill.quantity
            << ",\"price\":" << fill.price << ",\"commission\":" << fill.commission
            << ",\"timestamp\":" << fill.timestamp << '}';
    }
    out << "],\"cash\":" << engine.get_cash() << ",\"positions\":[";
    for (std::size_t i = 0; i < positions.size(); ++i) {
        if (i) out << ',';
        const auto& position = positions[i];
        out << "{\"symbol\":\"" << position.symbol << "\",\"quantity\":" << position.quantity
            << ",\"sellable_quantity\":" << position.sellable_quantity << ",\"avg_cost\":" << position.avg_cost
            << ",\"realized_pnl\":" << position.realized_pnl << '}';
    }
    out << "],\"equity_curve\":[";
    for (std::size_t i = 0; i < curve.size(); ++i) {
        if (i) out << ',';
        const auto& point = curve[i];
        out << "{\"timestamp\":" << point.timestamp << ",\"equity\":" << point.equity
            << ",\"cash\":" << point.cash << '}';
    }
    out << "],\"corporate_actions\":[";
    for (std::size_t i = 0; i < actions.size(); ++i) {
        if (i) out << ',';
        const auto& result = actions[i];
        out << "{\"symbol\":\"" << result.symbol << "\",\"timestamp\":" << result.timestamp
            << ",\"cash_dividend\":" << result.cash_dividend << ",\"old_quantity\":" << result.old_quantity
            << ",\"new_quantity\":" << result.new_quantity << '}';
    }
    out << "],\"final_metrics\":{\"equity\":" << engine.get_equity()
        << ",\"total_return\":" << engine.get_total_return() << ",\"annual_return\":" << engine.get_annual_return()
        << ",\"sharpe_ratio\":" << engine.get_sharpe_ratio() << ",\"max_drawdown\":" << engine.get_max_drawdown()
        << ",\"win_rate\":" << engine.get_win_rate() << "}}\n";
    return out.str();
}

std::string read(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
}  // namespace

int main(int argc, char** argv) {
    std::string output_path, verify_path;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (argument == "--verify" && i + 1 < argc) verify_path = argv[++i];
        else { std::cerr << "usage: qbt_golden_replay [--output path] [--verify path]\n"; return 2; }
    }
    try {
        const std::string value = replay();
        if (!verify_path.empty() && read(verify_path) != value) {
            std::cerr << "golden replay differs from " << verify_path << '\n'; return 1;
        }
        if (!output_path.empty()) {
            std::ofstream output(output_path, std::ios::binary); output << value;
        } else if (verify_path.empty()) {
            std::cout << value;
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

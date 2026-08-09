#include "engine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "engine_common/fixed_point.h"

namespace qbt {

namespace {

constexpr int64_t kMoneyScale = 10'000;

std::uint64_t stable_text_hash(std::string_view value) {
    if (value.empty()) return 0;
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

MoneyMinor to_money(Price value) {
    return engine_common::quantize_money(value, kMoneyScale);
}

Price from_money(MoneyMinor value) {
    return engine_common::money_from_minor(value, kMoneyScale);
}

void validate_market_snapshot(const MarketSnapshot& md) {
    const bool prices_valid = std::isfinite(md.open) && md.open > 0.0 &&
        std::isfinite(md.high) && md.high > 0.0 &&
        std::isfinite(md.low) && md.low > 0.0 &&
        std::isfinite(md.close) && md.close > 0.0 &&
        md.high >= std::max({md.open, md.low, md.close}) &&
        md.low <= std::min({md.open, md.high, md.close});
    const bool state_valid = md.timestamp > 0 && !md.symbol.empty() &&
        md.volume >= 0 && md.lot_size > 0 && md.min_buy_quantity > 0 &&
        std::isfinite(md.upper_limit) && md.upper_limit >= 0.0 &&
        std::isfinite(md.lower_limit) && md.lower_limit >= 0.0 &&
        (md.upper_limit == 0.0 || md.lower_limit <= md.upper_limit) &&
        std::isfinite(md.adjustment_factor) && md.adjustment_factor > 0.0;
    if (!prices_valid || !state_valid) {
        throw std::invalid_argument("invalid market snapshot");
    }
}

}  // namespace

BacktestEngine::BacktestEngine(Price initial_cash, FillTiming fill_timing,
                               ExecutionConfig execution_config)
    : positions_(&symbol_registry_), initial_cash_(initial_cash),
      fill_timing_(fill_timing), execution_config_(execution_config) {
    fill_buffer_.reserve(16);
    if (!std::isfinite(initial_cash_) || initial_cash_ < 0.0 ||
        !std::isfinite(execution_config_.max_volume_participation) ||
        !std::isfinite(execution_config_.slippage_bps) ||
        !std::isfinite(execution_config_.market_order_price_buffer_bps) ||
        execution_config_.max_volume_participation < 0.0 ||
        execution_config_.max_volume_participation > 1.0 ||
        execution_config_.slippage_bps < 0.0 ||
        execution_config_.slippage_bps >= 10'000.0 ||
        execution_config_.market_order_price_buffer_bps < 0.0) {
        throw std::invalid_argument("invalid execution config");
    }
    cash_minor_ = to_money(initial_cash);
    initial_cash_ = from_money(cash_minor_);
}

BacktestEngine::~BacktestEngine() {
    if (strategy_runtime_) strategy_runtime_->stop();
}

void BacktestEngine::add_event(const Event& event) {
    ensure_usable();
    if (event.type == EventType::ORDER || event.type == EventType::FILL) {
        throw std::invalid_argument(
            "external ORDER and FILL events are not supported by the backtest engine");
    }
    event_queue_.push(event);
}

void BacktestEngine::push_market_data(const MarketSnapshot& md) {
    ensure_usable();
    Event ev;
    ev.type = EventType::MARKET_DATA;
    ev.timestamp = md.timestamp;
    ev.symbol = md.symbol;
    ev.data = md;  // variant 存入 MarketSnapshot alternative
    event_queue_.push(ev);
}

void BacktestEngine::process_market_data(const MarketSnapshot& md) {
    process_market_data_batch_sorted(std::span<const MarketSnapshot>(&md, 1));
}

void BacktestEngine::process_market_data_batch(
    const std::vector<MarketSnapshot>& batch) {
    ensure_usable();
    if (batch.empty()) return;
    const auto symbol_less = [](const MarketSnapshot& lhs,
                                const MarketSnapshot& rhs) {
        return lhs.symbol < rhs.symbol;
    };
    std::vector<MarketSnapshot> sorted_batch;
    const std::vector<MarketSnapshot>* ordered_ptr = &batch;
    if (!std::is_sorted(batch.begin(), batch.end(), symbol_less)) {
        sorted_batch = batch;
        std::sort(sorted_batch.begin(), sorted_batch.end(), symbol_less);
        ordered_ptr = &sorted_batch;
    }
    process_market_data_batch_sorted(*ordered_ptr);
}

void BacktestEngine::process_market_data_batch_sorted(
    std::span<const MarketSnapshot> ordered) {
    ensure_usable();
    if (ordered.empty()) return;
    const Timestamp timestamp = ordered.front().timestamp;
    if (has_processed_data_ && timestamp <= last_processed_time_) {
        throw std::invalid_argument(
            "market data batch timestamps must be strictly increasing");
    }
    const std::string* previous_symbol = nullptr;
    for (const MarketSnapshot& md : ordered) {
        validate_market_snapshot(md);
        if (md.timestamp != timestamp) {
            throw std::invalid_argument("market data batch must share one timestamp");
        }
        if (md.symbol.empty()) {
            throw std::invalid_argument("market data symbol cannot be empty");
        }
        if (previous_symbol != nullptr && *previous_symbol >= md.symbol) {
            throw std::invalid_argument(
                "sorted market data batch must be unique and symbol ordered");
        }
        previous_symbol = &md.symbol;
    }
    try {
        current_time_ = timestamp;
        positions_.roll_trading_day(timestamp);

    // 先完成整个截面的成交和行情更新,策略不会看到半个截面。
    for (const MarketSnapshot& md : ordered) {
        const SymbolId symbol_id = symbol_id_for(md.symbol);
        latest_market_data_[symbol_id] = md;
        has_market_data_[symbol_id] = 1;
        OrderBook& book = *order_books_[symbol_id];
        book.begin_market_data(md);
        if (!md.is_listed) {
            const auto& active = active_order_ids_by_symbol_[symbol_id];
            std::vector<int64_t> open_orders(active.begin(), active.end());
            for (int64_t order_id : open_orders) {
                cancel_open_order(
                    order_id, timestamp, RejectReason::NOT_LISTED,
                    "order canceled because the symbol is no longer listed");
            }
        }
        if (fill_timing_ == FillTiming::NEXT_OPEN) {
            auto& pending = pending_orders_[symbol_id];
            if (!pending.empty()) {
                std::vector<Order> orders = std::move(pending);
                pending.clear();
                book.set_reference_price(md.open);
                std::vector<Order> remaining_orders;
                for (Order order : orders) {
                    order.timestamp = timestamp;
                    const Quantity executed = handle_order(order);
                    order.quantity -= executed;
                    if (order.type == OrderType::MARKET &&
                        order.quantity > 0 && is_order_open(order.id)) {
                        remaining_orders.push_back(order);
                    }
                }
                if (!remaining_orders.empty()) {
                    pending = std::move(remaining_orders);
                }
            }
        } else {
            // CLOSE 模式下，上一根 Bar 因参与率不足留下的市价单在下一根 open 继续。
            auto& pending = pending_orders_[symbol_id];
            if (!pending.empty()) {
                std::vector<Order> orders = std::move(pending);
                pending.clear();
                book.set_reference_price(md.open);
                std::vector<Order> remaining_orders;
                for (Order order : orders) {
                    order.timestamp = timestamp;
                    const Quantity executed = handle_order(order);
                    order.quantity -= executed;
                    if (order.quantity > 0 && is_order_open(order.id)) {
                        remaining_orders.push_back(order);
                    }
                }
                if (!remaining_orders.empty()) {
                    pending = std::move(remaining_orders);
                }
            }
        }
        book.set_reference_price(md.ref_price());
        fill_buffer_.clear();
        book.match_resting_orders(md, fill_buffer_);
        for (const Fill& fill : fill_buffer_) handle_fill(fill);
        last_prices_[symbol_id] = md.ref_price();
        has_last_price_[symbol_id] = 1;
    }

    if (strategy_runtime_) {
        const std::string default_symbol = ordered.size() == 1 ? ordered.front().symbol : "";
        submit_strategy_orders(run_strategy_runtime(ordered, timestamp),
                               default_symbol, timestamp);
    } else if (on_cross_section_view_) {
        const std::string default_symbol = ordered.size() == 1 ? ordered.front().symbol : "";
        submit_strategy_orders(on_cross_section_view_(ordered), default_symbol, timestamp);
    } else if (on_cross_section_) {
        const std::string default_symbol = ordered.size() == 1 ? ordered.front().symbol : "";
        const std::vector<MarketSnapshot> callback_batch(ordered.begin(), ordered.end());
        submit_strategy_orders(on_cross_section_(callback_batch), default_symbol, timestamp);
    } else if (on_market_data_) {
        for (const MarketSnapshot& md : ordered) {
            submit_strategy_orders(on_market_data_(md), md.symbol, timestamp);
        }
    }

    mark_to_market(timestamp);
    last_processed_time_ = timestamp;
    has_processed_data_ = true;
    } catch (...) {
        poisoned_ = true;
        throw;
    }
}

void BacktestEngine::set_on_market_data(OnMarketDataCallback cb) {
    if (cb && strategy_runtime_) {
        throw std::logic_error("callback mode and strategy runtime mode are mutually exclusive");
    }
    on_market_data_ = std::move(cb);
}

void BacktestEngine::set_on_cross_section(OnCrossSectionCallback cb) {
    if (cb && strategy_runtime_) {
        throw std::logic_error("callback mode and strategy runtime mode are mutually exclusive");
    }
    on_cross_section_ = std::move(cb);
}

void BacktestEngine::set_on_cross_section_view(OnCrossSectionViewCallback cb) {
    if (cb && strategy_runtime_) {
        throw std::logic_error("callback mode and strategy runtime mode are mutually exclusive");
    }
    on_cross_section_view_ = std::move(cb);
}

void BacktestEngine::set_strategy_runtime(
    std::shared_ptr<engine_common::IStrategyRuntime> runtime,
    const engine_common::StrategySessionContext& context) {
    ensure_usable();
    if (has_processed_data_) {
        throw std::logic_error("strategy runtime must be configured before market data");
    }
    if (runtime && (on_market_data_ || on_cross_section_ || on_cross_section_view_)) {
        throw std::logic_error("callback mode and strategy runtime mode are mutually exclusive");
    }
    if (strategy_runtime_) strategy_runtime_->stop();
    strategy_runtime_ = std::move(runtime);
    if (!strategy_runtime_) return;
    const auto status = strategy_runtime_->start(context);
    if (status != engine_common::StrategyStatus::OK) {
        strategy_runtime_.reset();
        throw std::runtime_error("strategy runtime failed to start");
    }
}

void BacktestEngine::set_replay_analytics_sink(
    std::shared_ptr<engine_common::IReplayAnalyticsSink> sink) {
    ensure_usable();
    if (has_processed_data_) {
        throw std::logic_error(
            "replay analytics sink must be configured before market data");
    }
    replay_analytics_sink_ = std::move(sink);
    last_analytics_decision_id_ = 0;
    open_performance_session_id_ = 0;
    open_performance_period_start_ = 0;
    performance_period_open_ = false;
    replay_analytics_finalized_ = false;
}

void BacktestEngine::open_performance_period(std::uint64_t session_id,
                                             Timestamp timestamp) {
    ensure_usable();
    try {
        const Timestamp boundary = timestamp == 0 ? current_time_ : timestamp;
        if (!replay_analytics_sink_ || replay_analytics_finalized_ ||
            performance_period_open_ || !has_processed_data_ || session_id == 0 ||
            boundary != current_time_) {
            throw std::logic_error("invalid performance period open boundary");
        }
        engine_common::MarketFrameBatchView market_view;
        engine_common::PortfolioView portfolio_view;
        prepare_replay_analytics_snapshot(boundary, market_view, portfolio_view);
        if (!std::all_of(market_view.bars.begin(), market_view.bars.end(),
                         [&](const auto& bar) {
                return bar.timestamp == boundary;
            })) {
            throw std::runtime_error(
                "performance period boundary contains stale marks");
        }
        const engine_common::ReplayPeriodOpenEvent event{
            boundary, session_id, portfolio_view.equity, portfolio_view.cash,
            analytics_period_buffer_};
        if (replay_analytics_sink_->on_period_open(event) !=
            engine_common::ReplayAnalyticsStatus::OK) {
            throw std::runtime_error("replay analytics rejected period open");
        }
        open_performance_session_id_ = session_id;
        open_performance_period_start_ = boundary;
        performance_period_open_ = true;
    } catch (...) {
        poisoned_ = true;
        throw;
    }
}

void BacktestEngine::close_performance_period(std::uint64_t session_id,
                                              Timestamp timestamp,
                                              double cash_interest,
                                              double external_cash_flow) {
    ensure_usable();
    try {
        const Timestamp boundary = timestamp == 0 ? current_time_ : timestamp;
        if (!replay_analytics_sink_ || replay_analytics_finalized_ ||
            !performance_period_open_ || session_id != open_performance_session_id_ ||
            boundary != current_time_ || boundary <= open_performance_period_start_ ||
            !std::isfinite(cash_interest) || !std::isfinite(external_cash_flow)) {
            throw std::logic_error("invalid performance period close boundary");
        }
        engine_common::MarketFrameBatchView market_view;
        engine_common::PortfolioView portfolio_view;
        prepare_replay_analytics_snapshot(boundary, market_view, portfolio_view);
        if (!std::all_of(market_view.bars.begin(), market_view.bars.end(),
                         [&](const auto& bar) {
                return bar.timestamp == boundary;
            })) {
            throw std::runtime_error(
                "performance period boundary contains stale marks");
        }
        const engine_common::ReplayPeriodCloseEvent event{
            boundary, session_id, portfolio_view.equity, portfolio_view.cash,
            cash_interest, external_cash_flow, analytics_period_buffer_};
        if (replay_analytics_sink_->on_period_close(event) !=
            engine_common::ReplayAnalyticsStatus::OK) {
            throw std::runtime_error("replay analytics rejected period close");
        }
        open_performance_session_id_ = 0;
        open_performance_period_start_ = 0;
        performance_period_open_ = false;
    } catch (...) {
        poisoned_ = true;
        throw;
    }
}

std::vector<Order> BacktestEngine::run_strategy_runtime(
    std::span<const MarketSnapshot> ordered, Timestamp timestamp) {
    strategy_market_buffer_.clear();
    strategy_portfolio_buffer_.clear();
    strategy_market_buffer_.reserve(ordered.size());
    strategy_portfolio_buffer_.reserve(ordered.size());
    for (const MarketSnapshot& market : ordered) {
        SymbolId symbol_id = 0;
        if (!find_symbol_id(market.symbol, symbol_id)) {
            throw std::logic_error("strategy market symbol is not registered");
        }
        uint32_t flags = engine_common::MARKET_DATA_TRUSTED;
        if (market.is_listed) flags |= engine_common::MARKET_LISTED;
        if (market.is_suspended) flags |= engine_common::MARKET_SUSPENDED;
        if (market.is_st) flags |= engine_common::MARKET_ST;
        strategy_market_buffer_.push_back({
            symbol_id, timestamp, market.open, market.high, market.low, market.close,
            market.signal_open > 0.0 ? market.signal_open : market.open,
            market.signal_high > 0.0 ? market.signal_high : market.high,
            market.signal_low > 0.0 ? market.signal_low : market.low,
            market.signal_close > 0.0 ? market.signal_close : market.close,
            market.volume, market.lot_size, flags});
        const Position position = positions_.get_position(market.symbol);
        engine_common::Quantity active_buy_quantity = 0;
        engine_common::Quantity active_sell_quantity = 0;
        for (const int64_t order_id : active_order_ids_by_symbol_[symbol_id]) {
            const OrderCore* active = find_active_order(order_id);
            if (active == nullptr) continue;
            const Quantity remaining = std::max<Quantity>(
                active->quantity - active->filled_quantity, 0);
            if (active->side == Side::BUY) {
                active_buy_quantity += remaining;
            } else {
                active_sell_quantity += remaining;
            }
        }
        strategy_portfolio_buffer_.push_back({
            symbol_id, position.quantity, position.sellable_quantity,
            position.avg_cost, market.close,
            active_buy_quantity, active_sell_quantity});
    }
    const PortfolioSnapshot snapshot = get_portfolio_snapshot();
    engine_common::MarketFrameBatchView market_view{
        timestamp, strategy_market_buffer_, true};
    engine_common::PortfolioView portfolio_view{
        strategy_portfolio_buffer_, snapshot.cash, snapshot.equity,
        snapshot.gross_exposure, snapshot.net_exposure};
    strategy_order_buffer_.resize(std::max<size_t>(16, ordered.size() * 2));
    engine_common::OrderIntentBuffer output{strategy_order_buffer_, 0};
    const auto status = strategy_runtime_->on_market_batch(
        market_view, portfolio_view, output);
    if (status != engine_common::StrategyStatus::OK) {
        throw std::runtime_error("strategy runtime failed during backtest");
    }
    const auto decision = strategy_runtime_->last_decision();
    if (replay_analytics_sink_) {
        if (decision.valid() && decision.decision_id != last_analytics_decision_id_) {
            if (decision.decision_id < last_analytics_decision_id_ ||
                decision.decision_at != timestamp) {
                throw std::runtime_error("invalid strategy decision sequence");
            }
            const engine_common::ReplayDecisionEvent event{
                decision, market_view, portfolio_view};
            if (replay_analytics_sink_->on_decision(event) !=
                engine_common::ReplayAnalyticsStatus::OK) {
                throw std::runtime_error("replay analytics rejected strategy decision");
            }
            last_analytics_decision_id_ = decision.decision_id;
        }
    }
    std::vector<Order> orders;
    orders.reserve(output.size);
    for (size_t index = 0; index < output.size; ++index) {
        const auto& intent = output.values[index];
        if (intent.client_order_id != 0) {
            throw std::invalid_argument("strategy runtime must not assign order ids");
        }
        if (intent.time_in_force != engine_common::TimeInForce::DAY) {
            throw std::invalid_argument("backtest strategy runtime only supports DAY orders");
        }
        Order order;
        order.decision_id = decision.valid() ? decision.decision_id : 0;
        order.symbol = symbol_registry_.symbol(intent.symbol_id);
        order.side = intent.side == engine_common::Side::BUY ? Side::BUY : Side::SELL;
        order.type = intent.type == engine_common::OrderType::MARKET
            ? OrderType::MARKET : OrderType::LIMIT;
        order.quantity = intent.quantity;
        order.limit_price = static_cast<Price>(intent.limit_price) / kMoneyScale;
        order.timestamp = intent.timestamp;
        orders.push_back(std::move(order));
    }
    return orders;
}

void BacktestEngine::set_on_fill(OnFillCallback cb) {
    on_fill_ = std::move(cb);
}

void BacktestEngine::set_on_order_update(OnOrderUpdateCallback cb) {
    on_order_update_ = std::move(cb);
}

void BacktestEngine::set_commission_fn(CommissionCallback cb) {
    if (!fee_schedules_.empty()) {
        throw std::logic_error(
            "commission callback and fee schedules cannot both be configured");
    }
    commission_fn_ = std::move(cb);
}

void BacktestEngine::set_fee_schedules(std::vector<FeeSchedule> schedules) {
    if (symbol_registry_.size() != 0) {
        throw std::logic_error(
            "fee schedules must be set before processing market data");
    }
    if (commission_fn_) {
        throw std::logic_error(
            "commission callback and fee schedules cannot both be configured");
    }
    if (schedules.empty()) {
        throw std::invalid_argument("fee schedules cannot be empty");
    }
    std::sort(schedules.begin(), schedules.end(),
              [](const FeeSchedule& lhs, const FeeSchedule& rhs) {
                  return lhs.effective_from < rhs.effective_from;
              });
    for (size_t index = 0; index < schedules.size(); ++index) {
        const FeeSchedule& value = schedules[index];
        const bool invalid = value.effective_from < 0 ||
            (value.effective_to && *value.effective_to <= value.effective_from) ||
            !std::isfinite(value.commission_rate) || value.commission_rate < 0.0 ||
            !std::isfinite(value.min_commission) || value.min_commission < 0.0 ||
            !std::isfinite(value.stamp_tax_rate) || value.stamp_tax_rate < 0.0 ||
            !std::isfinite(value.transfer_fee_rate) || value.transfer_fee_rate < 0.0;
        if (invalid || (index > 0 &&
            (!schedules[index - 1].effective_to ||
             *schedules[index - 1].effective_to > value.effective_from))) {
            throw std::invalid_argument("invalid or overlapping fee schedules");
        }
    }
    fee_schedules_ = std::move(schedules);
}

void BacktestEngine::set_execution_config(const ExecutionConfig& config) {
    if (symbol_registry_.size() != 0) {
        throw std::logic_error(
            "execution config must be set before processing market data");
    }
    if (!std::isfinite(config.max_volume_participation) ||
        !std::isfinite(config.slippage_bps) ||
        !std::isfinite(config.market_order_price_buffer_bps) ||
        config.max_volume_participation < 0.0 ||
        config.max_volume_participation > 1.0 || config.slippage_bps < 0.0 ||
        config.slippage_bps >= 10'000.0 ||
        config.market_order_price_buffer_bps < 0.0) {
        throw std::invalid_argument("invalid execution config");
    }
    execution_config_ = config;
}

void BacktestEngine::set_history_config(const HistoryConfig& config) {
    ensure_usable();
    record_order_history_ = config.record_orders;
    pnl_tracker_.configure_history(config.record_trades, config.record_round_trips,
                                   config.equity_sampling);
}

void BacktestEngine::run() {
    ensure_usable();
    try {
        // 主事件循环:按时间戳弹出事件并分发
        while (!event_queue_.empty()) {
            Event ev = event_queue_.pop();
            current_time_ = ev.timestamp;

            switch (ev.type) {
                case EventType::MARKET_DATA: {
                    std::vector<MarketSnapshot> batch{
                        std::get<MarketSnapshot>(std::move(ev.data))};
                    while (!event_queue_.empty() &&
                           event_queue_.top().type == EventType::MARKET_DATA &&
                           event_queue_.top().timestamp == current_time_) {
                        Event next = event_queue_.pop();
                        batch.push_back(
                            std::get<MarketSnapshot>(std::move(next.data)));
                    }
                    process_market_data_batch(batch);
                    break;
                }
                case EventType::ORDER:
                case EventType::FILL:
                    throw std::invalid_argument(
                        "external ORDER and FILL events are not supported by the backtest engine");
                case EventType::TIMER:
                case EventType::SIGNAL:
                    // 回测核心当前仅消费行情事件；定时器和信号由策略适配层处理。
                    break;
            }
        }
    } catch (...) {
        poisoned_ = true;
        throw;
    }
}

bool BacktestEngine::cancel_order(int64_t order_id, Timestamp timestamp) {
    ensure_usable();
    try {
        return cancel_order_impl(order_id, timestamp);
    } catch (...) {
        poisoned_ = true;
        throw;
    }
}

bool BacktestEngine::cancel_order_impl(int64_t order_id, Timestamp timestamp) {
    return cancel_open_order(order_id, timestamp, RejectReason::NONE,
                             "canceled by user");
}

bool BacktestEngine::cancel_open_order(int64_t order_id, Timestamp timestamp,
                                        RejectReason reason,
                                        const std::string& message) {
    if (!is_order_open(order_id)) return false;
    OrderCore* record = find_active_order(order_id);
    if (record == nullptr) return false;
    const SymbolId symbol_id = record->symbol_id;
    auto& orders = pending_orders_[symbol_id];
    if (!orders.empty()) {
        orders.erase(std::remove_if(orders.begin(), orders.end(),
                    [&](const Order& order) { return order.id == order_id; }),
                     orders.end());
    }
    if (order_books_[symbol_id]) order_books_[symbol_id]->cancel_order(order_id);
    release_order_reservations(order_id, 0, true);
    record->status = OrderStatus::CANCELED;
    record->reject_reason = reason;
    record->updated_timestamp = timestamp == 0 ? current_time_ : timestamp;
    record->message_code = encode_order_message(message);
    remove_active_symbol_order(*record);
    notify_order_update(*record);
    archive_order(order_id);
    return true;
}

void BacktestEngine::finalize(Timestamp timestamp) {
    ensure_usable();
    try {
        finalize_impl(timestamp);
    } catch (...) {
        poisoned_ = true;
        throw;
    }
}

void BacktestEngine::finalize_impl(Timestamp timestamp) {
    const Timestamp final_time = timestamp == 0 ? current_time_ : timestamp;
    std::vector<int64_t> open_orders;
    for (const auto& active : active_order_ids_by_symbol_) {
        open_orders.insert(open_orders.end(), active.begin(), active.end());
    }
    for (int64_t order_id : open_orders) {
        OrderCore* record = find_active_order(order_id);
        if (record == nullptr) continue;
        const SymbolId symbol_id = record->symbol_id;
        auto& orders = pending_orders_[symbol_id];
        if (!orders.empty()) {
            orders.erase(std::remove_if(orders.begin(), orders.end(),
                        [&](const Order& order) { return order.id == order_id; }),
                         orders.end());
        }
        if (order_books_[symbol_id]) order_books_[symbol_id]->cancel_order(order_id);
        release_order_reservations(order_id, 0, true);
        record->status = OrderStatus::EXPIRED;
        record->updated_timestamp = final_time;
        record->message_code = OrderMessageCode::EXPIRED;
        remove_active_symbol_order(*record);
        notify_order_update(*record);
        archive_order(order_id);
    }
    notify_replay_end(final_time);
}

CorporateActionResult BacktestEngine::apply_corporate_action(
    const CorporateAction& action) {
    ensure_usable();
    try {
        return apply_corporate_action_impl(action);
    } catch (...) {
        poisoned_ = true;
        throw;
    }
}

CorporateActionResult BacktestEngine::apply_corporate_action_impl(
    const CorporateAction& action) {
    std::vector<int64_t> open_orders;
    SymbolId symbol_id = 0;
    if (find_symbol_id(action.symbol, symbol_id)) {
        const auto& active = active_order_ids_by_symbol_[symbol_id];
        open_orders.assign(active.begin(), active.end());
    }
    for (int64_t order_id : open_orders) {
        cancel_open_order(order_id, action.timestamp, RejectReason::NONE,
                          "canceled by user");
    }
    CorporateActionResult result = positions_.apply_corporate_action(action);
    result.action_id = next_corporate_action_id_++;
    pnl_tracker_.apply_corporate_action(action.symbol, action.share_multiplier);
    cash_minor_ += to_money(result.cash_dividend);
    corporate_action_history_.push_back(result);
    if (replay_analytics_sink_ && find_symbol_id(action.symbol, symbol_id)) {
        const engine_common::ReplayCorporateActionEvent event{
            result.action_id, symbol_id, action.timestamp,
            result.new_quantity - result.old_quantity, result.cash_dividend};
        if (replay_analytics_sink_->on_corporate_action(event) !=
            engine_common::ReplayAnalyticsStatus::OK) {
            throw std::runtime_error("replay analytics rejected corporate action");
        }
    }
    return result;
}

void BacktestEngine::handle_market_data(const MarketSnapshot& md) {
    process_market_data(md);
}

void BacktestEngine::submit_strategy_orders(
    std::vector<Order> orders, const std::string& default_symbol,
    Timestamp timestamp) {
    for (Order& order : orders) {
        order.id = next_order_id_++;
        if (order.symbol.empty()) order.symbol = default_symbol;
        if (order.symbol.empty()) {
            throw std::invalid_argument(
                "cross-section order must specify a symbol");
        }
        if (order.timestamp == 0) {
            order.timestamp = timestamp;
        } else if (order.timestamp != timestamp) {
            reject_order(order, RejectReason::INVALID_ORDER,
                         "order timestamp must match the strategy callback timestamp");
            continue;
        }
        SymbolId symbol_id = 0;
        if (!find_symbol_id(order.symbol, symbol_id) || has_market_data_[symbol_id] == 0) {
            reject_order(order, RejectReason::UNKNOWN_SYMBOL,
                         "symbol has no market data");
            continue;
        }
        const MarketSnapshot& market = latest_market_data_[symbol_id];
        if (market.timestamp != timestamp) {
            reject_order(order, RejectReason::STALE_MARKET_DATA,
                         "symbol has no market data in the current cross-section");
            continue;
        }
        const RejectReason validation = validate_order(order, market);
        if (validation != RejectReason::NONE) {
            reject_order(order, validation, "order validation failed");
            continue;
        }
        if (!execution_config_.allow_short && order.side == Side::SELL) {
            const Quantity available = execution_config_.enforce_t_plus_one
                ? positions_.available_to_sell(order.symbol)
                : std::max<Quantity>(positions_.get_position(order.symbol).quantity, 0);
            const Quantity reserved = reserved_sell_quantity_[symbol_id];
            if (order.quantity > std::max<Quantity>(available - reserved, 0)) {
                reject_order(order, RejectReason::INSUFFICIENT_POSITION,
                             "insufficient sellable quantity");
                continue;
            }
        }
        MoneyMinor reserved_cash = 0;
        if (execution_config_.enforce_cash && order.side == Side::BUY) {
            reserved_cash = estimate_required_cash(order, market);
            if (reserved_cash > cash_minor_ - total_reserved_cash()) {
                reject_order(order, RejectReason::INSUFFICIENT_CASH,
                             "insufficient cash");
                continue;
            }
        }
        accept_order(order, reserved_cash);
        if (!is_order_open(order.id)) continue;
        if (fill_timing_ == FillTiming::NEXT_OPEN) {
            pending_orders_[symbol_id].push_back(order);
        } else {
            const Quantity executed = handle_order(order);
            if (order.type == OrderType::MARKET && executed < order.quantity) {
                Order remaining = order;
                remaining.quantity -= executed;
                pending_orders_[symbol_id].push_back(std::move(remaining));
            }
        }
    }
}

RejectReason BacktestEngine::validate_order(
    const Order& order, const MarketSnapshot& market) const {
    if (order.quantity <= 0 ||
        (order.side != Side::BUY && order.side != Side::SELL) ||
        (order.type != OrderType::MARKET && order.type != OrderType::LIMIT) ||
        (order.type == OrderType::LIMIT &&
         (!std::isfinite(order.limit_price) || order.limit_price <= 0.0))) {
        return RejectReason::INVALID_ORDER;
    }
    if (!market.is_listed) return RejectReason::NOT_LISTED;
    if (execution_config_.enforce_board_lot) {
        const Quantity lot = std::max<Quantity>(market.lot_size, 1);
        const Quantity minimum = std::max<Quantity>(market.min_buy_quantity, 1);
        if (order.side == Side::BUY &&
            (order.quantity < minimum || order.quantity % lot != 0)) {
            return RejectReason::INVALID_LOT_SIZE;
        }
        if (order.side == Side::SELL && order.quantity % lot != 0) {
            const Position position = positions_.get_position(order.symbol);
            if (order.quantity != std::max<Quantity>(position.quantity, 0)) {
                return RejectReason::INVALID_LOT_SIZE;
            }
        }
    }
    return RejectReason::NONE;
}

MoneyMinor BacktestEngine::estimate_required_cash(
    const Order& order, const MarketSnapshot& market) const {
    if (order.side != Side::BUY) return 0;
    Price price = order.limit_price;
    if (order.type == OrderType::MARKET) {
        price = market.ref_price() *
            (1.0 + (execution_config_.slippage_bps +
                    execution_config_.market_order_price_buffer_bps) / 10'000.0);
        if (market.upper_limit > 0.0) price = std::max(price, market.upper_limit);
    }
    if (!std::isfinite(price) || price <= 0.0) {
        throw std::invalid_argument("invalid order reservation price");
    }
    const Price notional = static_cast<Price>(order.quantity) * price;
    const Price commission = calculate_commission(order.timestamp, notional, false);
    return to_money(notional + commission);
}

Price BacktestEngine::calculate_commission(Timestamp timestamp, Price notional,
                                           bool is_sell) const {
    Price commission = 0.0;
    if (commission_fn_) {
        commission = commission_fn_(notional, is_sell);
    } else if (!fee_schedules_.empty()) {
        const FeeSchedule* selected = nullptr;
        for (auto it = fee_schedules_.rbegin(); it != fee_schedules_.rend(); ++it) {
            if (timestamp >= it->effective_from &&
                (!it->effective_to || timestamp < *it->effective_to)) {
                selected = &*it;
                break;
            }
        }
        if (selected == nullptr) {
            throw std::invalid_argument("no fee schedule for timestamp");
        }
        commission = std::max(
            notional * selected->commission_rate, selected->min_commission);
        commission += notional * selected->transfer_fee_rate;
        if (is_sell) commission += notional * selected->stamp_tax_rate;
    }
    if (!std::isfinite(commission) || commission < 0.0) {
        throw std::invalid_argument("commission must be finite and non-negative");
    }
    return commission;
}

MoneyMinor BacktestEngine::total_reserved_cash() const {
    return total_reserved_buy_cash_;
}

void BacktestEngine::accept_order(const Order& order, MoneyMinor reserved_cash) {
    OrderCore record = make_order_core(order);
    record.status = OrderStatus::ACCEPTED;
    record.updated_timestamp = order.timestamp;
    uint32_t slot = 0;
    if (free_active_order_slots_.empty()) {
        slot = static_cast<uint32_t>(active_order_slots_.size());
        active_order_slots_.push_back(record);
    } else {
        slot = free_active_order_slots_.back();
        free_active_order_slots_.pop_back();
        active_order_slots_[slot] = record;
    }
    order_record_locations_.resize(
        std::max(order_record_locations_.size(), static_cast<size_t>(order.id + 1)), 0);
    order_record_locations_[order.id] = static_cast<int32_t>(slot) + 1;
    ++active_order_count_;
    ++total_order_count_;
    OrderCore& active = active_order_slots_[slot];
    add_active_symbol_order(active);
    const SymbolId symbol_id = active.symbol_id;
    if (reserved_cash > 0) {
        reserved_buy_cash_[order.id] = reserved_cash;
        total_reserved_buy_cash_ += reserved_cash;
    }
    if (!execution_config_.allow_short && order.side == Side::SELL) {
        reserved_sell_quantity_[symbol_id] += order.quantity;
    }
    notify_order_update(active);
}

void BacktestEngine::reject_order(const Order& order, RejectReason reason,
                                   const std::string& message) {
    OrderCore record = make_order_core(order);
    record.status = OrderStatus::REJECTED;
    record.reject_reason = reason;
    record.updated_timestamp = order.timestamp;
    record.message_code = encode_order_message(message);
    ++total_order_count_;
    if (record_order_history_) {
        notify_order_update(record);
        store_archived_order(std::move(record));
    } else {
        notify_order_update(record);
    }
}

void BacktestEngine::update_order_fill(const Fill& fill) {
    OrderCore* record = find_active_order(fill.order_id);
    if (record == nullptr) return;
    const Price old_notional = record->avg_fill_price *
                               static_cast<Price>(record->filled_quantity);
    record->filled_quantity += fill.quantity;
    record->avg_fill_price = (old_notional + fill.price * fill.quantity) /
                             static_cast<Price>(record->filled_quantity);
    record->status = record->filled_quantity >= record->quantity
        ? OrderStatus::FILLED : OrderStatus::PARTIALLY_FILLED;
    record->updated_timestamp = fill.timestamp;
    const bool filled = record->status == OrderStatus::FILLED;
    if (filled) remove_active_symbol_order(*record);
    notify_order_update(*record);
    if (filled && is_order_open(fill.order_id) == false) archive_order(fill.order_id);
}

void BacktestEngine::release_order_reservations(
    int64_t order_id, Quantity filled_quantity, bool release_all) {
    const OrderCore* record = find_active_order(order_id);
    if (record == nullptr) return;
    const Quantity remaining = std::max<Quantity>(
        record->quantity - record->filled_quantity, 0);
    auto cash = reserved_buy_cash_.find(order_id);
    if (cash != reserved_buy_cash_.end()) {
        if (release_all || filled_quantity >= remaining) {
            total_reserved_buy_cash_ -= cash->second;
            reserved_buy_cash_.erase(cash);
        } else if (remaining > 0 && filled_quantity > 0) {
            const MoneyMinor old_reserve = cash->second;
            cash->second = static_cast<MoneyMinor>(std::llround(
                static_cast<long double>(old_reserve) *
                static_cast<long double>(remaining - filled_quantity) /
                static_cast<long double>(remaining)));
            total_reserved_buy_cash_ += cash->second - old_reserve;
        }
    }
    if (!execution_config_.allow_short && record->side == Side::SELL) {
        const Quantity released = release_all ? remaining : filled_quantity;
        reserved_sell_quantity_[record->symbol_id] = std::max<Quantity>(
            reserved_sell_quantity_[record->symbol_id] - released, 0);
    }
}

bool BacktestEngine::is_order_open(int64_t order_id) const {
    const OrderCore* order = find_active_order(order_id);
    return order != nullptr &&
           (order->status == OrderStatus::ACCEPTED ||
            order->status == OrderStatus::PARTIALLY_FILLED);
}

void BacktestEngine::ensure_usable() const {
    if (poisoned_) {
        throw std::logic_error(
            "backtest engine is poisoned after a failed state mutation");
    }
}

void BacktestEngine::validate_fill(const Fill& fill) const {
    const OrderCore* record = find_active_order(fill.order_id);
    if (record == nullptr) {
        throw std::logic_error("fill references an unknown order");
    }
    const Quantity remaining = record->quantity - record->filled_quantity;
    if (!is_order_open(fill.order_id) || fill.symbol != symbol_registry_.symbol(record->symbol_id) ||
        fill.side != record->side || fill.quantity <= 0 ||
        fill.quantity > remaining || !std::isfinite(fill.price) ||
        fill.price <= 0.0 || !std::isfinite(fill.commission) ||
        fill.commission < 0.0 || fill.timestamp != current_time_) {
        throw std::logic_error("fill violates its originating order");
    }
}

Quantity BacktestEngine::handle_order(const Order& order) {
    if (!is_order_open(order.id)) return 0;
    SymbolId symbol_id = 0;
    if (!find_symbol_id(order.symbol, symbol_id) || has_market_data_[symbol_id] == 0) return 0;
    if (execution_config_.enforce_cash && order.side == Side::BUY) {
        MarketSnapshot market = latest_market_data_[symbol_id];
        market.close = order_book_for(order.symbol).last_price();
        market.upper_limit = 0.0;
        const MoneyMinor required = estimate_required_cash(order, market);
        const MoneyMinor own_reserve = reserved_buy_cash_[order.id];
        const MoneyMinor available = cash_minor_ - total_reserved_cash() + own_reserve;
        if (required > available) {
            OrderCore* record = find_active_order(order.id);
            if (record == nullptr) return 0;
            release_order_reservations(order.id, 0, true);
            record->status = record->filled_quantity == 0
                ? OrderStatus::REJECTED : OrderStatus::CANCELED;
            record->reject_reason = RejectReason::INSUFFICIENT_CASH;
            record->updated_timestamp = order.timestamp;
            record->message_code = OrderMessageCode::EXECUTION_PRICE_CASH;
            remove_active_symbol_order(*record);
            notify_order_update(*record);
            archive_order(order.id);
            return 0;
        }
    }
    fill_buffer_.clear();
    order_book_for(order.symbol).submit_order(order, fill_buffer_);
    Quantity executed = 0;
    for (const Fill& f : fill_buffer_) {
        executed += f.quantity;
        handle_fill(f);
    }
    return executed;
}

void BacktestEngine::handle_fill(Fill fill) {
    validate_fill(fill);
    const OrderCore* source_order = find_active_order(fill.order_id);
    const std::uint64_t execution_decision_id =
        source_order == nullptr ? 0 : source_order->decision_id;
    const Quantity cumulative_quantity = source_order == nullptr
        ? fill.quantity : source_order->filled_quantity + fill.quantity;
    const Price notional = static_cast<Price>(fill.quantity) * fill.price;
    fill.commission = from_money(to_money(calculate_commission(
        fill.timestamp, notional, fill.side == Side::SELL)));
    const MoneyMinor notional_minor = to_money(notional);
    const MoneyMinor commission_minor = to_money(fill.commission);
    if (execution_config_.enforce_cash) {
        const auto own = reserved_buy_cash_.find(fill.order_id);
        const MoneyMinor own_reserve = fill.side == Side::BUY &&
            own != reserved_buy_cash_.end() ? own->second : 0;
        const MoneyMinor available = cash_minor_ - total_reserved_cash() + own_reserve;
        const MoneyMinor required_cash = fill.side == Side::BUY
            ? notional_minor + commission_minor
            : std::max(commission_minor - notional_minor, MoneyMinor{0});
        if (required_cash > available) {
            order_book_for(fill.symbol).cancel_order(fill.order_id);
            release_order_reservations(fill.order_id, 0, true);
            OrderCore* record = find_active_order(fill.order_id);
            if (record == nullptr) return;
            record->status = record->filled_quantity == 0
                ? OrderStatus::REJECTED : OrderStatus::CANCELED;
            record->reject_reason = RejectReason::INSUFFICIENT_CASH;
            record->updated_timestamp = fill.timestamp;
            record->message_code = OrderMessageCode::FILL_COMMISSION_CASH;
            remove_active_symbol_order(*record);
            notify_order_update(*record);
            archive_order(fill.order_id);
            return;
        }
    }
    release_order_reservations(fill.order_id, fill.quantity);
    const MoneyMinor signed_notional = fill.side == Side::BUY
        ? notional_minor : -notional_minor;
    cash_minor_ -= signed_notional + commission_minor;
    positions_.apply_fill(fill);
    pnl_tracker_.record_trade(fill);
    update_order_fill(fill);
    if (strategy_runtime_ || replay_analytics_sink_) {
        SymbolId symbol_id = 0;
        if (!find_symbol_id(fill.symbol, symbol_id)) {
            throw std::logic_error("fill symbol is not registered");
        }
        engine_common::ExecutionEvent execution;
        execution.client_order_id = fill.order_id;
        execution.execution_id = static_cast<int64_t>(pnl_tracker_.trade_count());
        execution.decision_id = execution_decision_id;
        execution.symbol_id = symbol_id;
        execution.side = fill.side == Side::BUY
            ? engine_common::Side::BUY : engine_common::Side::SELL;
        execution.status = is_order_open(fill.order_id)
            ? engine_common::ExecutionStatus::PARTIALLY_FILLED
            : engine_common::ExecutionStatus::FILLED;
        execution.last_quantity = fill.quantity;
        execution.cumulative_quantity = cumulative_quantity;
        execution.last_price = to_money(fill.price);
        execution.price_scale = kMoneyScale;
        execution.explicit_fee = commission_minor;
        execution.fee_scale = kMoneyScale;
        execution.audit_flags = engine_common::EXECUTION_HAS_SYMBOL |
            engine_common::EXECUTION_HAS_SIDE |
            engine_common::EXECUTION_HAS_PRICE_SCALE |
            engine_common::EXECUTION_HAS_EXPLICIT_FEE;
        if (execution.decision_id != 0) {
            execution.audit_flags |= engine_common::EXECUTION_HAS_DECISION_ID;
        }
        execution.timestamp = fill.timestamp;
        if (strategy_runtime_) strategy_runtime_->on_execution(execution);
        if (replay_analytics_sink_ &&
            replay_analytics_sink_->on_execution(execution) !=
                engine_common::ReplayAnalyticsStatus::OK) {
            throw std::runtime_error("replay analytics rejected execution");
        }
    }
    if (pnl_tracker_.equity_sampling() == EquitySampling::ON_FILL) {
        const Price cash = get_cash();
        const Price equity = cash + positions_.market_value(last_prices_, has_last_price_);
        pnl_tracker_.record_snapshot(fill.timestamp, equity, cash);
    }
    if (on_fill_) on_fill_(fill);
}

void BacktestEngine::prepare_replay_analytics_snapshot(
    Timestamp timestamp,
    engine_common::MarketFrameBatchView& market_view,
    engine_common::PortfolioView& portfolio_view) {
    if (timestamp <= 0) {
        throw std::invalid_argument(
            "replay analytics requires a positive snapshot timestamp");
    }
    strategy_market_buffer_.clear();
    strategy_portfolio_buffer_.clear();
    analytics_period_buffer_.clear();
    strategy_market_buffer_.reserve(symbol_registry_.size());
    strategy_portfolio_buffer_.reserve(symbol_registry_.size());
    analytics_period_buffer_.reserve(symbol_registry_.size());
    for (SymbolId symbol_id = 0; symbol_id < symbol_registry_.size(); ++symbol_id) {
        if (symbol_id >= has_market_data_.size() || has_market_data_[symbol_id] == 0) {
            continue;
        }
        const auto& market = latest_market_data_[symbol_id];
        std::uint32_t flags = engine_common::MARKET_DATA_TRUSTED;
        if (market.is_listed) flags |= engine_common::MARKET_LISTED;
        if (market.is_suspended) flags |= engine_common::MARKET_SUSPENDED;
        if (market.is_st) flags |= engine_common::MARKET_ST;
        strategy_market_buffer_.push_back({
            symbol_id, market.timestamp, market.open, market.high, market.low,
            market.close,
            market.signal_open > 0.0 ? market.signal_open : market.open,
            market.signal_high > 0.0 ? market.signal_high : market.high,
            market.signal_low > 0.0 ? market.signal_low : market.low,
            market.signal_close > 0.0 ? market.signal_close : market.close,
            market.volume, market.lot_size, flags});
        const auto position = positions_.get_position(market.symbol);
        strategy_portfolio_buffer_.push_back({
            symbol_id, position.quantity, position.sellable_quantity,
            position.avg_cost, market.close, 0, 0});
        analytics_period_buffer_.push_back({
            symbol_id, stable_text_hash(market.industry),
            position.quantity, market.close});
    }
    const auto snapshot = get_portfolio_snapshot();
    market_view = engine_common::MarketFrameBatchView{
        timestamp, strategy_market_buffer_, true};
    portfolio_view = engine_common::PortfolioView{
        strategy_portfolio_buffer_, snapshot.cash, snapshot.equity,
        snapshot.gross_exposure, snapshot.net_exposure};
}

void BacktestEngine::notify_replay_end(Timestamp timestamp) {
    if (!replay_analytics_sink_ || replay_analytics_finalized_) return;
    if (performance_period_open_) {
        throw std::logic_error(
            "performance period must close before replay finalization");
    }
    engine_common::MarketFrameBatchView market_view;
    engine_common::PortfolioView portfolio_view;
    prepare_replay_analytics_snapshot(timestamp, market_view, portfolio_view);
    const engine_common::ReplayEndEvent event{
        timestamp, market_view, portfolio_view};
    if (replay_analytics_sink_->on_replay_end(event) !=
        engine_common::ReplayAnalyticsStatus::OK) {
        throw std::runtime_error("replay analytics rejected replay end");
    }
    replay_analytics_finalized_ = true;
}

OrderBook& BacktestEngine::order_book_for(const std::string& symbol) {
    return *order_books_[symbol_id_for(symbol)];
}

SymbolId BacktestEngine::symbol_id_for(const std::string& symbol) {
    const SymbolId id = symbol_registry_.intern(symbol);
    const size_t required = static_cast<size_t>(id) + 1;
    if (order_books_.size() < required) {
        order_books_.resize(required);
        last_prices_.resize(required);
        has_last_price_.resize(required, 0);
        latest_market_data_.resize(required);
        has_market_data_.resize(required, 0);
        pending_orders_.resize(required);
        reserved_sell_quantity_.resize(required, 0);
        active_order_ids_by_symbol_.resize(required);
    }
    if (!order_books_[id]) {
        order_books_[id] = std::make_unique<OrderBook>(symbol, execution_config_);
    }
    return id;
}

bool BacktestEngine::find_symbol_id(const std::string& symbol, SymbolId& id) const {
    return symbol_registry_.find(symbol, id);
}

void BacktestEngine::mark_to_market(Timestamp time) {
    if (pnl_tracker_.equity_sampling() == EquitySampling::ON_FILL) return;
    Price position_value = positions_.market_value(last_prices_, has_last_price_);
    const Price cash = get_cash();
    Price equity = cash + position_value;
    pnl_tracker_.record_snapshot(time, equity, cash);
}

Price BacktestEngine::get_equity() const {
    return get_cash() + positions_.market_value(last_prices_, has_last_price_);
}

Price BacktestEngine::get_cash() const { return from_money(cash_minor_); }

double BacktestEngine::get_sharpe_ratio() const {
    return pnl_tracker_.sharpe_ratio(0.02, initial_cash_);
}

double BacktestEngine::get_max_drawdown() const {
    return pnl_tracker_.max_drawdown(initial_cash_);
}

double BacktestEngine::get_total_return() const {
    return initial_cash_ == 0.0 ? 0.0 : get_equity() / initial_cash_ - 1.0;
}

double BacktestEngine::get_annual_return() const {
    const auto& curve = pnl_tracker_.equity_curve();
    if (curve.size() < 2 || initial_cash_ <= 0.0 || get_equity() < 0.0) return 0.0;
    constexpr double kNanosecondsPerDay = 86'400'000'000'000.0;
    const Timestamp elapsed = curve.back().timestamp - curve.front().timestamp;
    if (elapsed <= 0) return 0.0;
    const double days = static_cast<double>(elapsed) / kNanosecondsPerDay;
    return std::pow(get_equity() / initial_cash_, 365.0 / days) - 1.0;
}

double BacktestEngine::get_win_rate() const {
    return pnl_tracker_.win_rate();
}

std::vector<TradeRecord> BacktestEngine::get_trade_history() const {
    return pnl_tracker_.trades();
}

std::vector<TradeRecord> BacktestEngine::get_trade_history_page(
    size_t offset, size_t limit) const {
    const auto& source = pnl_tracker_.trades();
    if (offset >= source.size() || limit == 0) return {};
    const size_t end = std::min(source.size(), offset + limit);
    return {source.begin() + static_cast<std::ptrdiff_t>(offset),
            source.begin() + static_cast<std::ptrdiff_t>(end)};
}

std::vector<RoundTripRecord> BacktestEngine::get_round_trip_history() const {
    return pnl_tracker_.round_trips();
}

std::vector<RoundTripRecord> BacktestEngine::get_round_trip_history_page(
    size_t offset, size_t limit) const {
    const auto& source = pnl_tracker_.round_trips();
    if (offset >= source.size() || limit == 0) return {};
    const size_t end = std::min(source.size(), offset + limit);
    return {source.begin() + static_cast<std::ptrdiff_t>(offset),
            source.begin() + static_cast<std::ptrdiff_t>(end)};
}

std::vector<EquityPoint> BacktestEngine::get_equity_curve() const {
    return pnl_tracker_.equity_curve();
}

std::vector<EquityPoint> BacktestEngine::get_equity_curve_page(
    size_t offset, size_t limit) const {
    const auto& source = pnl_tracker_.equity_curve();
    if (offset >= source.size() || limit == 0) return {};
    const size_t end = std::min(source.size(), offset + limit);
    return {source.begin() + static_cast<std::ptrdiff_t>(offset),
            source.begin() + static_cast<std::ptrdiff_t>(end)};
}

Position BacktestEngine::get_position(const std::string& symbol) const {
    return positions_.get_position(symbol);
}

std::vector<Position> BacktestEngine::get_positions() const {
    return positions_.all_positions();
}

std::vector<OrderRecord> BacktestEngine::get_order_history() const {
    std::vector<OrderRecord> records;
    records.reserve(order_record_locations_.size());
    for (size_t order_id = 1; order_id < order_record_locations_.size(); ++order_id) {
        const int32_t location = order_record_locations_[order_id];
        if (location > 0) {
            records.push_back(materialize_order(active_order_slots_[location - 1]));
        } else if (location < 0) {
            records.push_back(materialize_order(order_history_[-location - 1],
                                                static_cast<int64_t>(order_id)));
        }
    }
    return records;
}

std::vector<OrderRecord> BacktestEngine::get_order_history_page(
    size_t offset, size_t limit) const {
    if (limit == 0) return {};
    std::vector<OrderRecord> page;
    page.reserve(limit);
    size_t skipped = 0;
    for (size_t order_id = 1;
         order_id < order_record_locations_.size() && page.size() < limit;
         ++order_id) {
        const int32_t location = order_record_locations_[order_id];
        if (location == 0) continue;
        if (skipped < offset) {
            ++skipped;
            continue;
        }
        if (location > 0) {
            page.push_back(materialize_order(active_order_slots_[location - 1]));
        } else {
            page.push_back(materialize_order(order_history_[-location - 1],
                                             static_cast<int64_t>(order_id)));
        }
    }
    return page;
}

void BacktestEngine::archive_order(int64_t order_id) {
    OrderCore* found = find_active_order(order_id);
    if (found == nullptr) return;
    if (record_order_history_) {
        store_archived_order(std::move(*found));
    } else if (order_id < static_cast<int64_t>(order_record_locations_.size())) {
        order_record_locations_[order_id] = 0;
    }
    found->id = 0;
    free_active_order_slots_.push_back(
        static_cast<uint32_t>(found - active_order_slots_.data()));
    --active_order_count_;
}

BacktestEngine::OrderCore* BacktestEngine::find_active_order(int64_t order_id) {
    if (order_id <= 0 || static_cast<size_t>(order_id) >= order_record_locations_.size())
        return nullptr;
    const int32_t location = order_record_locations_[order_id];
    if (location <= 0 || static_cast<size_t>(location - 1) >= active_order_slots_.size())
        return nullptr;
    OrderCore& order = active_order_slots_[location - 1];
    return order.id == order_id ? &order : nullptr;
}

const BacktestEngine::OrderCore* BacktestEngine::find_active_order(
    int64_t order_id) const {
    if (order_id <= 0 || static_cast<size_t>(order_id) >= order_record_locations_.size())
        return nullptr;
    const int32_t location = order_record_locations_[order_id];
    if (location <= 0 || static_cast<size_t>(location - 1) >= active_order_slots_.size())
        return nullptr;
    const OrderCore& order = active_order_slots_[location - 1];
    return order.id == order_id ? &order : nullptr;
}

BacktestEngine::OrderCore BacktestEngine::make_order_core(const Order& order) {
    OrderCore core;
    core.id = order.id;
    core.decision_id = order.decision_id;
    core.quantity = order.quantity;
    core.timestamp = order.timestamp;
    core.updated_timestamp = order.timestamp;
    core.limit_price = order.limit_price;
    core.side = order.side;
    core.type = order.type;
    if (!order.symbol.empty()) core.symbol_id = symbol_id_for(order.symbol);
    return core;
}

OrderRecord BacktestEngine::materialize_order(const OrderCore& core) const {
    OrderRecord record;
    record.order.id = core.id;
    record.order.decision_id = core.decision_id;
    record.order.symbol = symbol_registry_.symbol(core.symbol_id);
    record.order.side = core.side;
    record.order.type = core.type;
    record.order.quantity = core.quantity;
    record.order.limit_price = core.limit_price;
    record.order.timestamp = core.timestamp;
    record.filled_quantity = core.filled_quantity;
    record.avg_fill_price = core.avg_fill_price;
    record.status = core.status;
    record.reject_reason = core.reject_reason;
    record.updated_timestamp = core.updated_timestamp;
    record.message = decode_order_message(core.message_code);
    return record;
}

OrderRecord BacktestEngine::materialize_order(
    const HistoryOrderCore& core, int64_t order_id) const {
    OrderRecord record;
    record.order.id = order_id;
    record.order.decision_id = core.decision_id;
    record.order.symbol = symbol_registry_.symbol(core.symbol_id);
    record.order.side = static_cast<Side>(core.flags & 0x1U);
    record.order.type = static_cast<OrderType>((core.flags >> 1U) & 0x1U);
    record.order.quantity = core.quantity;
    record.order.limit_price = static_cast<Price>(core.limit_price);
    record.order.timestamp = core.timestamp;
    record.filled_quantity = core.filled_quantity;
    record.avg_fill_price = core.avg_fill_price;
    record.status = static_cast<OrderStatus>((core.flags >> 2U) & 0x7U);
    record.reject_reason = static_cast<RejectReason>((core.flags >> 5U) & 0xFU);
    record.updated_timestamp = core.updated_timestamp;
    record.message = decode_order_message(
        static_cast<OrderMessageCode>((core.flags >> 9U) & 0xFU));
    return record;
}

void BacktestEngine::notify_order_update(const OrderCore& core) {
    if (!on_order_update_) return;
    const OrderRecord record = materialize_order(core);
    on_order_update_(record);
}

void BacktestEngine::add_active_symbol_order(OrderCore& core) {
    auto& orders = active_order_ids_by_symbol_[core.symbol_id];
    core.symbol_active_index = static_cast<uint32_t>(orders.size());
    orders.push_back(core.id);
}

void BacktestEngine::remove_active_symbol_order(OrderCore& core) {
    auto& orders = active_order_ids_by_symbol_[core.symbol_id];
    if (core.symbol_active_index >= orders.size() ||
        orders[core.symbol_active_index] != core.id) return;
    const int64_t moved_id = orders.back();
    orders[core.symbol_active_index] = moved_id;
    orders.pop_back();
    if (moved_id != core.id) {
        if (OrderCore* moved = find_active_order(moved_id)) {
            moved->symbol_active_index = core.symbol_active_index;
        }
    }
}

void BacktestEngine::store_archived_order(OrderCore core) {
    const size_t history_index = order_history_.size();
    const int64_t order_id = core.id;
    HistoryOrderCore archived;
    archived.decision_id = core.decision_id;
    archived.quantity = core.quantity;
    archived.filled_quantity = core.filled_quantity;
    archived.timestamp = core.timestamp;
    archived.updated_timestamp = core.updated_timestamp;
    archived.avg_fill_price = core.avg_fill_price;
    archived.limit_price = static_cast<float>(core.limit_price);
    archived.symbol_id = core.symbol_id;
    archived.flags = static_cast<uint32_t>(core.side) |
        (static_cast<uint32_t>(core.type) << 1U) |
        (static_cast<uint32_t>(core.status) << 2U) |
        (static_cast<uint32_t>(core.reject_reason) << 5U) |
        (static_cast<uint32_t>(core.message_code) << 9U);
    order_history_.push_back(archived);
    order_record_locations_.resize(
        std::max(order_record_locations_.size(), static_cast<size_t>(order_id + 1)), 0);
    order_record_locations_[order_id] = -static_cast<int32_t>(history_index) - 1;
}

BacktestEngine::OrderMessageCode BacktestEngine::encode_order_message(
    const std::string& message) {
    if (message == "canceled by user") return OrderMessageCode::CANCELED_BY_USER;
    if (message == "order canceled because the symbol is no longer listed")
        return OrderMessageCode::SYMBOL_DELISTED;
    if (message == "expired at end of backtest") return OrderMessageCode::EXPIRED;
    if (message == "order timestamp must match the strategy callback timestamp")
        return OrderMessageCode::TIMESTAMP_MISMATCH;
    if (message == "symbol has no market data") return OrderMessageCode::UNKNOWN_SYMBOL;
    if (message == "symbol has no market data in the current cross-section")
        return OrderMessageCode::STALE_MARKET_DATA;
    if (message == "order validation failed") return OrderMessageCode::VALIDATION_FAILED;
    if (message == "insufficient sellable quantity")
        return OrderMessageCode::INSUFFICIENT_POSITION;
    if (message == "insufficient cash") return OrderMessageCode::INSUFFICIENT_CASH;
    if (message == "insufficient cash at execution price")
        return OrderMessageCode::EXECUTION_PRICE_CASH;
    if (message == "insufficient cash for fill including commission")
        return OrderMessageCode::FILL_COMMISSION_CASH;
    return OrderMessageCode::NONE;
}

const char* BacktestEngine::decode_order_message(OrderMessageCode code) {
    switch (code) {
        case OrderMessageCode::CANCELED_BY_USER: return "canceled by user";
        case OrderMessageCode::SYMBOL_DELISTED:
            return "order canceled because the symbol is no longer listed";
        case OrderMessageCode::EXPIRED: return "expired at end of backtest";
        case OrderMessageCode::TIMESTAMP_MISMATCH:
            return "order timestamp must match the strategy callback timestamp";
        case OrderMessageCode::UNKNOWN_SYMBOL: return "symbol has no market data";
        case OrderMessageCode::STALE_MARKET_DATA:
            return "symbol has no market data in the current cross-section";
        case OrderMessageCode::VALIDATION_FAILED: return "order validation failed";
        case OrderMessageCode::INSUFFICIENT_POSITION: return "insufficient sellable quantity";
        case OrderMessageCode::INSUFFICIENT_CASH: return "insufficient cash";
        case OrderMessageCode::EXECUTION_PRICE_CASH:
            return "insufficient cash at execution price";
        case OrderMessageCode::FILL_COMMISSION_CASH:
            return "insufficient cash for fill including commission";
        case OrderMessageCode::NONE: break;
    }
    return "";
}

std::vector<CorporateActionResult>
BacktestEngine::get_corporate_action_history() const {
    return corporate_action_history_;
}

PortfolioSnapshot BacktestEngine::get_portfolio_snapshot() const {
    PortfolioSnapshot result;
    result.cash = get_cash();
    result.equity = result.cash + positions_.market_value(last_prices_, has_last_price_);
    Price gross_value = 0.0;
    Price net_value = 0.0;
    Price largest_value = 0.0;
    std::unordered_map<std::string, Price> industry_values;
    std::unordered_map<std::string, Price> factor_values;
    for (const Position& position : positions_.all_positions()) {
        if (position.quantity == 0) continue;
        SymbolId symbol_id = 0;
        if (!find_symbol_id(position.symbol, symbol_id) || has_last_price_[symbol_id] == 0) continue;
        const Price value = static_cast<Price>(position.quantity) * last_prices_[symbol_id];
        gross_value += std::abs(value);
        net_value += value;
        largest_value = std::max(largest_value, std::abs(value));
        ++result.position_count;
        const bool has_market = has_market_data_[symbol_id] != 0;
        const MarketSnapshot& market = latest_market_data_[symbol_id];
        const std::string industry =
            !has_market || market.industry.empty() ? "UNKNOWN" : market.industry;
        industry_values[industry] += value;
        if (has_market) {
            for (const auto& [factor, exposure] : market.factor_exposures) {
                factor_values[factor] += value * exposure;
            }
        }
    }
    if (result.equity != 0.0) {
        result.gross_exposure = gross_value / result.equity;
        result.net_exposure = net_value / result.equity;
        result.largest_position_weight = largest_value / std::abs(result.equity);
        for (const auto& [industry, value] : industry_values) {
            result.industry_exposure[industry] = value / result.equity;
        }
        for (const auto& [factor, value] : factor_values) {
            result.factor_exposure[factor] = value / result.equity;
        }
    }
    return result;
}

}  // namespace qbt

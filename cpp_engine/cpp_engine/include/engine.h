#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "event.h"
#include "event_queue.h"
#include "execution.h"
#include "engine_common/replay_analytics.h"
#include "engine_common/symbol_registry.h"
#include "engine_common/strategy.h"
#include "corporate_action.h"
#include "market_data.h"
#include "order.h"
#include "order_book.h"
#include "pnl_tracker.h"
#include "position.h"
#include "portfolio.h"
#include "types.h"

namespace qbt {

// 策略回调:引擎收到行情时调用,返回要下的订单。
// C++ 侧用 std::function,pybind11 绑定时可接 Python 可调用对象。
using OnMarketDataCallback =
    std::function<std::vector<Order>(const MarketSnapshot&)>;
using OnCrossSectionCallback =
    std::function<std::vector<Order>(const std::vector<MarketSnapshot>&)>;
using OnCrossSectionViewCallback =
    std::function<std::vector<Order>(std::span<const MarketSnapshot>)>;
using OnFillCallback = std::function<void(const Fill&)>;
using OnOrderUpdateCallback = std::function<void(const OrderRecord&)>;
using CommissionCallback = std::function<Price(Price, bool)>;

// 回测引擎:主事件循环,管理订单簿、持仓、盈亏。
// 成交时点模型:与纯 Python 后端 core.py 的 fill_timing 语义对齐。
enum class FillTiming {
    NEXT_OPEN,  // bar t 的信号在 bar t+1 的 open 成交(默认,无前视偏差)
    CLOSE,      // bar t 的信号在 bar t 的 close 立即成交(会用到本 bar 信息)
};

struct HistoryConfig {
    bool record_orders = true;
    bool record_trades = true;
    bool record_round_trips = true;
    EquitySampling equity_sampling = EquitySampling::EVERY_BAR;
};

class BacktestEngine {
public:
    explicit BacktestEngine(Price initial_cash = 1'000'000.0,
                            FillTiming fill_timing = FillTiming::NEXT_OPEN,
                            ExecutionConfig execution_config = ExecutionConfig{});
    ~BacktestEngine();

    // 事件注入(由 Python 侧的 DataFeed 灌入)
    void add_event(const Event& event);

    // 便捷入口:等价于 add_event 一个 MARKET_DATA 事件。
    // 让 Python 侧 BacktestRunner 用统一的 "push_market_data → run" 模型驱动,
    // 无需在 Python 构造 Event/variant。
    void push_market_data(const MarketSnapshot& md);

    // 历史回测首选入口:即时处理已排序行情,不在事件队列中累积数据。
    void process_market_data(const MarketSnapshot& md);
    void process_market_data_batch(const std::vector<MarketSnapshot>& batch);
    void process_market_data_batch_sorted(
        std::span<const MarketSnapshot> ordered_batch);

    // 注册策略回调
    void set_on_market_data(OnMarketDataCallback cb);
    void set_on_cross_section(OnCrossSectionCallback cb);
    void set_on_cross_section_view(OnCrossSectionViewCallback cb);
    void set_on_fill(OnFillCallback cb);
    void set_on_order_update(OnOrderUpdateCallback cb);
    void set_strategy_runtime(
        std::shared_ptr<engine_common::IStrategyRuntime> runtime,
        const engine_common::StrategySessionContext& context = {});
    void set_replay_analytics_sink(
        std::shared_ptr<engine_common::IReplayAnalyticsSink> sink);
    void open_performance_period(std::uint64_t session_id,
                                 Timestamp timestamp = 0);
    void close_performance_period(std::uint64_t session_id,
                                  Timestamp timestamp = 0,
                                  double cash_interest = 0.0,
                                  double external_cash_flow = 0.0);
    void set_commission_fn(CommissionCallback cb);
    void set_fee_schedules(std::vector<FeeSchedule> schedules);
    void set_execution_config(const ExecutionConfig& config);
    void set_history_config(const HistoryConfig& config);

    // 运行主事件循环直到队列清空
    void run();
    bool cancel_order(int64_t order_id, Timestamp timestamp = 0);
    void finalize(Timestamp timestamp = 0);
    CorporateActionResult apply_corporate_action(const CorporateAction& action);

    // 结果查询接口(pybind11 暴露)
    Price get_equity() const;
    Price get_cash() const;
    double get_sharpe_ratio() const;
    double get_max_drawdown() const;
    double get_total_return() const;
    double get_annual_return() const;
    double get_win_rate() const;
    std::vector<TradeRecord> get_trade_history() const;
    std::vector<TradeRecord> get_trade_history_page(size_t offset, size_t limit) const;
    std::vector<RoundTripRecord> get_round_trip_history() const;
    std::vector<RoundTripRecord> get_round_trip_history_page(size_t offset,
                                                             size_t limit) const;
    std::vector<EquityPoint> get_equity_curve() const;
    std::vector<EquityPoint> get_equity_curve_page(size_t offset, size_t limit) const;
    Position get_position(const std::string& symbol) const;
    std::vector<Position> get_positions() const;
    std::vector<OrderRecord> get_order_history() const;
    std::vector<OrderRecord> get_order_history_page(size_t offset, size_t limit) const;
    std::vector<CorporateActionResult> get_corporate_action_history() const;
    PortfolioSnapshot get_portfolio_snapshot() const;
    std::size_t get_order_count() const { return total_order_count_; }
    std::size_t get_trade_count() const { return pnl_tracker_.trade_count(); }
    std::size_t get_position_count() const { return positions_.size(); }
    std::size_t get_equity_point_count() const {
        return pnl_tracker_.equity_point_count();
    }
    const std::vector<TradeRecord>& trade_history_view() const {
        return pnl_tracker_.trades();
    }
    const std::vector<RoundTripRecord>& round_trip_history_view() const {
        return pnl_tracker_.round_trips();
    }
    const std::vector<EquityPoint>& equity_curve_view() const {
        return pnl_tracker_.equity_curve();
    }

private:
    enum class OrderMessageCode : uint8_t {
        NONE,
        CANCELED_BY_USER,
        SYMBOL_DELISTED,
        EXPIRED,
        TIMESTAMP_MISMATCH,
        UNKNOWN_SYMBOL,
        STALE_MARKET_DATA,
        VALIDATION_FAILED,
        INSUFFICIENT_POSITION,
        INSUFFICIENT_CASH,
        EXECUTION_PRICE_CASH,
        FILL_COMMISSION_CASH,
    };

    struct OrderCore {
        int64_t id = 0;
        std::uint64_t decision_id = 0;
        Quantity quantity = 0;
        Quantity filled_quantity = 0;
        Timestamp timestamp = 0;
        Timestamp updated_timestamp = 0;
        Price limit_price = 0.0;
        Price avg_fill_price = 0.0;
        SymbolId symbol_id = 0;
        uint32_t symbol_active_index = 0;
        Side side = Side::BUY;
        OrderType type = OrderType::MARKET;
        OrderStatus status = OrderStatus::ACCEPTED;
        RejectReason reject_reason = RejectReason::NONE;
        OrderMessageCode message_code = OrderMessageCode::NONE;
    };

#pragma pack(push, 1)
    struct HistoryOrderCore {
        std::uint64_t decision_id = 0;
        Quantity quantity = 0;
        Quantity filled_quantity = 0;
        Timestamp timestamp = 0;
        Timestamp updated_timestamp = 0;
        Price avg_fill_price = 0.0;
        float limit_price = 0.0F;
        SymbolId symbol_id = 0;
        uint32_t flags = 0;
    };
#pragma pack(pop)

    // 事件分发
    void handle_market_data(const MarketSnapshot& md);
    Quantity handle_order(const Order& order);
    void handle_fill(Fill fill);
    void submit_strategy_orders(std::vector<Order> orders,
                                const std::string& default_symbol,
                                Timestamp timestamp);
    RejectReason validate_order(const Order& order,
                                const MarketSnapshot& market) const;
    MoneyMinor estimate_required_cash(const Order& order,
                                      const MarketSnapshot& market) const;
    Price calculate_commission(Timestamp timestamp, Price notional,
                               bool is_sell) const;
    MoneyMinor total_reserved_cash() const;
    void accept_order(const Order& order, MoneyMinor reserved_cash = 0);
    void reject_order(const Order& order, RejectReason reason,
                      const std::string& message);
    void update_order_fill(const Fill& fill);
    void release_order_reservations(int64_t order_id, Quantity filled_quantity,
                                    bool release_all = false);
    bool is_order_open(int64_t order_id) const;
    void ensure_usable() const;
    void validate_fill(const Fill& fill) const;
    bool cancel_order_impl(int64_t order_id, Timestamp timestamp);
    bool cancel_open_order(int64_t order_id, Timestamp timestamp,
                           RejectReason reason, const std::string& message);
    void finalize_impl(Timestamp timestamp);
    void archive_order(int64_t order_id);
    OrderCore* find_active_order(int64_t order_id);
    const OrderCore* find_active_order(int64_t order_id) const;
    OrderCore make_order_core(const Order& order);
    OrderRecord materialize_order(const OrderCore& core) const;
    OrderRecord materialize_order(const HistoryOrderCore& core,
                                  int64_t order_id) const;
    void notify_order_update(const OrderCore& core);
    void add_active_symbol_order(OrderCore& core);
    void remove_active_symbol_order(OrderCore& core);
    void store_archived_order(OrderCore core);
    static OrderMessageCode encode_order_message(const std::string& message);
    static const char* decode_order_message(OrderMessageCode code);
    CorporateActionResult apply_corporate_action_impl(
        const CorporateAction& action);

    // 按需获取/创建某标的的订单簿
    OrderBook& order_book_for(const std::string& symbol);
    SymbolId symbol_id_for(const std::string& symbol);
    bool find_symbol_id(const std::string& symbol, SymbolId& id) const;

    // 计算当前总权益并记录快照
    void mark_to_market(Timestamp time);
    std::vector<Order> run_strategy_runtime(
        std::span<const MarketSnapshot> ordered, Timestamp timestamp);
    void prepare_replay_analytics_snapshot(
        Timestamp timestamp,
        engine_common::MarketFrameBatchView& market_view,
        engine_common::PortfolioView& portfolio_view);
    void notify_replay_end(Timestamp timestamp);

    EventQueue event_queue_;
    engine_common::SymbolRegistry symbol_registry_;
    std::vector<std::unique_ptr<OrderBook>> order_books_;
    std::vector<Price> last_prices_;
    std::vector<uint8_t> has_last_price_;
    std::vector<MarketSnapshot> latest_market_data_;
    std::vector<uint8_t> has_market_data_;

    PnLTracker pnl_tracker_;
    PositionTracker positions_;

    Price initial_cash_ = 0.0;
    MoneyMinor cash_minor_ = 0;
    Timestamp current_time_ = 0;
    Timestamp last_processed_time_ = 0;
    bool has_processed_data_ = false;
    bool poisoned_ = false;
    int64_t next_order_id_ = 1;
    FillTiming fill_timing_ = FillTiming::NEXT_OPEN;
    ExecutionConfig execution_config_;

    // next_open 模式下待下一 bar 激活的订单，以及参与率不足的市价单。
    std::vector<std::vector<Order>> pending_orders_;
    // 已接受但尚未成交的卖单数量，避免多张订单合计穿透 T+1 可卖数量。
    std::vector<Quantity> reserved_sell_quantity_;
    std::unordered_map<int64_t, MoneyMinor> reserved_buy_cash_;
    MoneyMinor total_reserved_buy_cash_ = 0;
    std::vector<OrderCore> active_order_slots_;
    std::vector<uint32_t> free_active_order_slots_;
    std::deque<HistoryOrderCore> order_history_;
    // 0=无记录，正数=active slot + 1，负数=history index + 1 的相反数。
    std::vector<int32_t> order_record_locations_{0};
    size_t active_order_count_ = 0;
    size_t total_order_count_ = 0;
    bool record_order_history_ = true;
    std::vector<std::vector<int64_t>> active_order_ids_by_symbol_;
    std::vector<CorporateActionResult> corporate_action_history_;

    OnMarketDataCallback on_market_data_;
    OnCrossSectionCallback on_cross_section_;
    OnCrossSectionViewCallback on_cross_section_view_;
    OnFillCallback on_fill_;
    OnOrderUpdateCallback on_order_update_;
    CommissionCallback commission_fn_;
    std::shared_ptr<engine_common::IStrategyRuntime> strategy_runtime_;
    std::shared_ptr<engine_common::IReplayAnalyticsSink> replay_analytics_sink_;
    std::vector<engine_common::MarketBar> strategy_market_buffer_;
    std::vector<engine_common::PortfolioItem> strategy_portfolio_buffer_;
    std::vector<engine_common::OrderIntent> strategy_order_buffer_;
    std::vector<engine_common::ReplayPeriodSecurity> analytics_period_buffer_;
    std::vector<FeeSchedule> fee_schedules_;
    FillBuffer fill_buffer_;
    std::uint64_t last_analytics_decision_id_{0};
    std::uint64_t next_corporate_action_id_{1};
    std::uint64_t open_performance_session_id_{0};
    Timestamp open_performance_period_start_{0};
    bool performance_period_open_{false};
    bool replay_analytics_finalized_{false};
};

}  // namespace qbt

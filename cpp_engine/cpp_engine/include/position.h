#pragma once

#include <string>
#include <span>
#include <unordered_map>
#include <vector>

#include "corporate_action.h"
#include "engine_common/symbol_registry.h"
#include "order.h"
#include "types.h"

namespace qbt {

// 单个标的的持仓
struct Position {
    std::string symbol;
    Quantity quantity = 0;      // 正为多头,负为空头
    Price avg_cost = 0.0;       // 持仓均价
    Price realized_pnl = 0.0;   // 已实现盈亏
    MoneyMinor realized_pnl_minor = 0;
    Quantity sellable_quantity = 0;  // A股 T+1:当日可卖数量
};

// 持仓管理:根据成交更新各标的持仓和已实现盈亏
class PositionTracker {
public:
    explicit PositionTracker(engine_common::SymbolRegistry* symbols = nullptr)
        : symbols_(symbols == nullptr ? &owned_symbols_ : symbols) {}
    // 根据成交更新持仓
    void apply_fill(const Fill& fill);
    void roll_trading_day(Timestamp timestamp);
    CorporateActionResult apply_corporate_action(const CorporateAction& action);
    Quantity available_to_sell(const std::string& symbol) const;

    // 查询某标的持仓(不存在返回空持仓)
    Position get_position(const std::string& symbol) const;
    std::vector<Position> all_positions() const;

    // 全部持仓的浮动市值(需传入各标的最新价)
    Price market_value(const std::unordered_map<std::string, Price>& prices) const;
    Price market_value(std::span<const Price> prices,
                       std::span<const uint8_t> valid_prices) const;

    // 全部已实现盈亏
    Price total_realized_pnl() const;
    std::size_t size() const { return position_count_; }

private:
    Position& position_for(const std::string& symbol);
    const Position* find_position(const std::string& symbol) const;

    engine_common::SymbolRegistry owned_symbols_;
    engine_common::SymbolRegistry* symbols_;
    std::vector<Position> positions_;
    std::vector<uint8_t> occupied_;
    size_t position_count_ = 0;
    int64_t current_trading_day_ = -1;
};

}  // namespace qbt

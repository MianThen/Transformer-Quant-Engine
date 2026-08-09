#pragma once

#include <cstdint>
#include <string>

#include "engine_common/types.h"

namespace qbt {

// 纳秒时间戳
using Timestamp = engine_common::TimestampNs;

// 数量(股/手),整数避免浮点误差
using Quantity = engine_common::Quantity;

using PriceTicks = engine_common::PriceTicks;
using MoneyMinor = engine_common::MoneyMinor;
using SymbolId = engine_common::SymbolId;

// 价格,回测场景用 double 足够
using Price = double;

// 买卖方向
enum class Side {
    BUY,
    SELL,
};

// 订单类型
enum class OrderType {
    MARKET,  // 市价单:直接吃对手盘
    LIMIT,   // 限价单:插入盘口等待成交
};

inline const char* to_string(Side s) {
    return s == Side::BUY ? "BUY" : "SELL";
}

}  // namespace qbt

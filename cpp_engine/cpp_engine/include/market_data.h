#pragma once

#include <unordered_map>

#include "types.h"

namespace qbt {

// 行情快照:一条 bar 或 tick 数据
struct MarketSnapshot {
    std::string symbol;
    Timestamp timestamp = 0;

    // OHLCV
    Price open = 0.0;
    Price high = 0.0;
    Price low = 0.0;
    Price close = 0.0;
    Quantity volume = 0;

    // 可选的 point-in-time 交易状态。0 表示未提供涨跌停价格。
    Price upper_limit = 0.0;
    Price lower_limit = 0.0;
    bool is_suspended = false;
    bool is_listed = true;
    bool is_st = false;
    Quantity lot_size = 1;
    Quantity min_buy_quantity = 1;
    std::string board;
    std::string industry;
    std::unordered_map<std::string, double> factor_exposures;

    // 原始价格用于成交；signal_* 仅供策略研究使用。
    Price adjustment_factor = 1.0;
    Price signal_open = 0.0;
    Price signal_high = 0.0;
    Price signal_low = 0.0;
    Price signal_close = 0.0;

    // 便捷:用收盘价作为参考价
    Price ref_price() const { return close; }
    Price signal_ref_price() const {
        return signal_close > 0.0 ? signal_close : close;
    }
};

}  // namespace qbt

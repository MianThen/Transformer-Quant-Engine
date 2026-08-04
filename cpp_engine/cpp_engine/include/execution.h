#pragma once

#include <optional>

#include "types.h"

namespace qbt {

struct FeeSchedule {
    Timestamp effective_from = 0;
    std::optional<Timestamp> effective_to;
    double commission_rate = 0.0;
    Price min_commission = 0.0;
    double stamp_tax_rate = 0.0;
    double transfer_fee_rate = 0.0;
};

// Bar 级执行模型配置。价格单位为基点(1bp = 0.01%)。
struct ExecutionConfig {
    double max_volume_participation = 0.10;
    double slippage_bps = 0.0;
    bool enforce_price_limits = true;
    bool enforce_t_plus_one = true;
    bool allow_short = false;
    bool enforce_board_lot = true;
    bool enforce_cash = true;
    double market_order_price_buffer_bps = 0.0;
};

}  // namespace qbt

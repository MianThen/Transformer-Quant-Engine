#pragma once

#include <string>

#include "types.h"

namespace qbt {

struct CorporateAction {
    std::string symbol;
    Timestamp timestamp = 0;
    Price cash_dividend_per_share = 0.0;
    double share_multiplier = 1.0;
    std::string description;
};

struct CorporateActionResult {
    std::uint64_t action_id = 0;
    std::string symbol;
    Timestamp timestamp = 0;
    Price cash_dividend = 0.0;
    Quantity old_quantity = 0;
    Quantity new_quantity = 0;
};

}  // namespace qbt

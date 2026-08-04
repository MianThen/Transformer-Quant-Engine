#pragma once

#include <string>
#include <unordered_map>

#include "types.h"

namespace qbt {

struct PortfolioSnapshot {
    Price cash = 0.0;
    Price equity = 0.0;
    Price gross_exposure = 0.0;
    Price net_exposure = 0.0;
    double largest_position_weight = 0.0;
    int64_t position_count = 0;
    std::unordered_map<std::string, double> industry_exposure;
    std::unordered_map<std::string, double> factor_exposure;
};

}  // namespace qbt

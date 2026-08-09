#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "engine_common/types.h"

namespace engine_common {

inline PriceTicks quantize_price(double price, double tick_size) {
    if (!std::isfinite(price) || !std::isfinite(tick_size) || tick_size <= 0.0)
        throw std::invalid_argument("invalid price or tick size");
    const long double ticks = std::round(static_cast<long double>(price) / tick_size);
    const long double minimum =
        static_cast<long double>(std::numeric_limits<PriceTicks>::min());
    const long double maximum_exclusive = -minimum;
    if (ticks < minimum || ticks >= maximum_exclusive)
        throw std::overflow_error("price ticks overflow");
    return static_cast<PriceTicks>(ticks);
}

inline double price_from_ticks(PriceTicks ticks, double tick_size) {
    if (!std::isfinite(tick_size) || tick_size <= 0.0)
        throw std::invalid_argument("invalid tick size");
    return static_cast<double>(ticks) * tick_size;
}

inline MoneyMinor quantize_money(double amount, int64_t scale = 100) {
    if (!std::isfinite(amount) || scale <= 0) throw std::invalid_argument("invalid money");
    const double limit = static_cast<double>(std::numeric_limits<MoneyMinor>::max()) /
                         static_cast<double>(scale);
    if (amount < -limit || amount > limit)
        throw std::overflow_error("money overflow");
    return static_cast<MoneyMinor>(std::llround(amount * static_cast<double>(scale)));
}

inline double money_from_minor(MoneyMinor amount, int64_t scale = 100) {
    if (scale <= 0) throw std::invalid_argument("invalid money scale");
    return static_cast<double>(amount) / static_cast<double>(scale);
}

}  // namespace engine_common

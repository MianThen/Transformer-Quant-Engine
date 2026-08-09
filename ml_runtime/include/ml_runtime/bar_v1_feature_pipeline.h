#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <string_view>
#include <vector>

#include "engine_common/strategy.h"

namespace qbt::ml {

inline constexpr uint32_t kBarV1FeatureCount = 23;
inline constexpr std::array<std::string_view, kBarV1FeatureCount>
    kBarV1FeatureNames{
        "log_return_1", "log_return_5", "log_return_10", "log_return_20",
        "intraday_range", "close_open_return", "overnight_gap", "log_volume",
        "volume_zscore_20", "volatility_5", "volatility_10",
        "volatility_20", "volatility_60", "ma_deviation_5",
        "ma_deviation_10", "ma_deviation_20", "price_position_20",
        "breakout_20", "cross_section_return_rank", "is_suspended",
        "is_listed", "is_st", "is_tradable",
    };

struct BarV1FeatureRows {
    std::span<const engine_common::SymbolId> symbols;
    std::span<const float> values;
    std::span<const uint8_t> valid;
};

class BarV1FeaturePipeline {
public:
    BarV1FeatureRows update(
        const engine_common::MarketFrameBatchView& market);
    void reset() noexcept;

private:
    struct History {
        std::deque<double> log_close;
        std::deque<double> log_volume;
        std::deque<double> returns;
        engine_common::TimestampNs last_timestamp = 0;
    };

    static void push(std::deque<double>& values, double value, size_t limit);
    static double difference(const std::deque<double>& values, size_t periods);
    static double mean(const std::deque<double>& values, size_t window);
    static double price_log_mean(const std::deque<double>& values, size_t window);
    static double standard_deviation(
        const std::deque<double>& values, size_t window);
    static double minimum(const std::deque<double>& values, size_t window);
    static double maximum(const std::deque<double>& values, size_t window);

    std::vector<History> histories_;
    std::vector<uint64_t> last_seen_batch_;
    uint64_t batch_sequence_ = 0;
    std::vector<engine_common::SymbolId> symbols_;
    std::vector<float> values_;
    std::vector<uint8_t> valid_;
};

}  // namespace qbt::ml

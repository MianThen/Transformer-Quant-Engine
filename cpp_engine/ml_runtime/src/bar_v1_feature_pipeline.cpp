#include "ml_runtime/bar_v1_feature_pipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qbt::ml {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double selected_price(double signal, double raw) {
    return std::isfinite(signal) && signal > 0.0 ? signal : raw;
}

bool finite_price(double value) {
    return std::isfinite(value) && value > 0.0;
}

}  // namespace

BarV1FeatureRows BarV1FeaturePipeline::update(
    const engine_common::MarketFrameBatchView& market) {
    if (market.asof_timestamp <= 0 || market.bars.empty()) {
        throw std::invalid_argument("BAR_V1 market batch must be non-empty");
    }
    ++batch_sequence_;
    if (batch_sequence_ == 0) throw std::overflow_error("BAR_V1 batch sequence overflow");
    symbols_.clear();
    symbols_.reserve(market.bars.size());
    values_.assign(market.bars.size() * kBarV1FeatureCount, 0.0F);
    valid_.assign(market.bars.size(), 0);
    std::vector<std::array<double, kBarV1FeatureCount>> rows(market.bars.size());
    std::vector<double> returns(market.bars.size(), kNaN);

    for (size_t row = 0; row < market.bars.size(); ++row) {
        const auto& bar = market.bars[row];
        const double open = selected_price(bar.signal_open, bar.open);
        const double high = selected_price(bar.signal_high, bar.high);
        const double low = selected_price(bar.signal_low, bar.low);
        const double close = selected_price(bar.signal_close, bar.close);
        if (bar.timestamp != market.asof_timestamp || !finite_price(open) ||
            !finite_price(high) || !finite_price(low) || !finite_price(close) ||
            high < std::max({open, low, close}) ||
            low > std::min({open, high, close}) || bar.volume < 0) {
            throw std::invalid_argument("invalid BAR_V1 market values");
        }
        const size_t required = static_cast<size_t>(bar.symbol_id) + 1;
        if (histories_.size() < required) {
            histories_.resize(required);
            last_seen_batch_.resize(required, 0);
        }
        auto& history = histories_[bar.symbol_id];
        if (last_seen_batch_[bar.symbol_id] != 0 &&
            last_seen_batch_[bar.symbol_id] + 1 != batch_sequence_) {
            history = {};
        }
        if (history.last_timestamp >= market.asof_timestamp) {
            throw std::invalid_argument("BAR_V1 timestamps must increase per symbol");
        }
        const double log_open = std::log(open);
        const double log_high = std::log(high);
        const double log_low = std::log(low);
        const double log_close = std::log(close);
        const double log_volume = std::log1p(static_cast<double>(bar.volume));
        const double return_1 = history.log_close.empty()
            ? kNaN : log_close - history.log_close.back();
        push(history.log_close, log_close, 61);
        push(history.log_volume, log_volume, 20);
        push(history.returns, return_1, 60);
        history.last_timestamp = market.asof_timestamp;
        last_seen_batch_[bar.symbol_id] = batch_sequence_;
        symbols_.push_back(bar.symbol_id);
        returns[row] = return_1;

        const double volume_mean = mean(history.log_volume, 20);
        const double volume_std = standard_deviation(history.log_volume, 20);
        const double min_20 = minimum(history.log_close, 20);
        const double max_20 = maximum(history.log_close, 20);
        rows[row] = {
            return_1, difference(history.log_close, 5),
            difference(history.log_close, 10), difference(history.log_close, 20),
            log_high - log_low, log_close - log_open,
            history.log_close.size() < 2
                ? kNaN : log_open - history.log_close[history.log_close.size() - 2],
            log_volume, volume_std > 0.0 ? (log_volume - volume_mean) / volume_std : kNaN,
            standard_deviation(history.returns, 5),
            standard_deviation(history.returns, 10),
            standard_deviation(history.returns, 20),
            standard_deviation(history.returns, 60),
            log_close - price_log_mean(history.log_close, 5),
            log_close - price_log_mean(history.log_close, 10),
            log_close - price_log_mean(history.log_close, 20),
            std::isfinite(min_20) && max_20 > min_20
                ? (close - std::exp(min_20)) /
                    (std::exp(max_20) - std::exp(min_20)) : kNaN,
            std::isfinite(max_20) ? log_close - max_20 : kNaN,
            kNaN,
            (bar.flags & engine_common::MARKET_SUSPENDED) != 0 ? 1.0 : 0.0,
            (bar.flags & engine_common::MARKET_LISTED) != 0 ? 1.0 : 0.0,
            (bar.flags & engine_common::MARKET_ST) != 0 ? 1.0 : 0.0,
            ((bar.flags & engine_common::MARKET_LISTED) != 0 &&
             (bar.flags & engine_common::MARKET_SUSPENDED) == 0) ? 1.0 : 0.0,
        };
    }

    std::vector<std::pair<double, size_t>> ranked;
    for (size_t row = 0; row < returns.size(); ++row) {
        if (std::isfinite(returns[row])) ranked.emplace_back(returns[row], row);
    }
    std::sort(ranked.begin(), ranked.end());
    for (size_t begin = 0; begin < ranked.size();) {
        size_t end = begin + 1;
        while (end < ranked.size() && ranked[end].first == ranked[begin].first) ++end;
        const double average_rank =
            (static_cast<double>(begin + 1) + static_cast<double>(end)) / 2.0;
        for (size_t index = begin; index < end; ++index) {
            rows[ranked[index].second][18] = average_rank / ranked.size();
        }
        begin = end;
    }

    for (size_t row = 0; row < rows.size(); ++row) {
        const size_t offset = row * kBarV1FeatureCount;
        bool finite = true;
        for (size_t column = 0; column < kBarV1FeatureCount; ++column) {
            finite = finite && std::isfinite(rows[row][column]);
            values_[offset + column] = std::isfinite(rows[row][column])
                ? static_cast<float>(rows[row][column]) : 0.0F;
        }
        const auto flags = market.bars[row].flags;
        const bool tradable = (flags & engine_common::MARKET_LISTED) != 0 &&
            (flags & engine_common::MARKET_SUSPENDED) == 0;
        valid_[row] = finite && tradable ? 1 : 0;
    }
    return {symbols_, values_, valid_};
}

void BarV1FeaturePipeline::reset() noexcept {
    histories_.clear();
    last_seen_batch_.clear();
    batch_sequence_ = 0;
    symbols_.clear();
    values_.clear();
    valid_.clear();
}

void BarV1FeaturePipeline::push(
    std::deque<double>& values, double value, size_t limit) {
    values.push_back(value);
    if (values.size() > limit) values.pop_front();
}

double BarV1FeaturePipeline::difference(
    const std::deque<double>& values, size_t periods) {
    return values.size() <= periods ? kNaN
        : values.back() - values[values.size() - periods - 1];
}

double BarV1FeaturePipeline::mean(
    const std::deque<double>& values, size_t window) {
    if (values.size() < window) return kNaN;
    double total = 0.0;
    for (size_t index = values.size() - window; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) return kNaN;
        total += values[index];
    }
    return total / static_cast<double>(window);
}

double BarV1FeaturePipeline::price_log_mean(
    const std::deque<double>& values, size_t window) {
    if (values.size() < window) return kNaN;
    double total = 0.0;
    for (size_t index = values.size() - window; index < values.size(); ++index) {
        if (!std::isfinite(values[index])) return kNaN;
        total += std::exp(values[index]);
    }
    return std::log(total / static_cast<double>(window));
}

double BarV1FeaturePipeline::standard_deviation(
    const std::deque<double>& values, size_t window) {
    const double average = mean(values, window);
    if (!std::isfinite(average)) return kNaN;
    double total = 0.0;
    for (size_t index = values.size() - window; index < values.size(); ++index) {
        const double delta = values[index] - average;
        total += delta * delta;
    }
    return std::sqrt(total / static_cast<double>(window));
}

double BarV1FeaturePipeline::minimum(
    const std::deque<double>& values, size_t window) {
    if (values.size() < window) return kNaN;
    return *std::min_element(
        values.end() - static_cast<std::ptrdiff_t>(window), values.end());
}

double BarV1FeaturePipeline::maximum(
    const std::deque<double>& values, size_t window) {
    if (values.size() < window) return kNaN;
    return *std::max_element(
        values.end() - static_cast<std::ptrdiff_t>(window), values.end());
}

}  // namespace qbt::ml

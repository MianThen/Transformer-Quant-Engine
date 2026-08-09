#include "ml_runtime/feature_window_store.h"

#include <algorithm>
#include <stdexcept>

namespace qbt::ml {

FeatureWindowStore::FeatureWindowStore(
    uint32_t lookback, uint32_t feature_count, uint32_t static_feature_count)
    : lookback_(lookback), feature_count_(feature_count),
      static_feature_count_(static_feature_count) {
    if (lookback_ == 0 || feature_count_ == 0) {
        throw std::invalid_argument("lookback and feature_count must be positive");
    }
}

FeatureWindowStore::Window& FeatureWindowStore::window_for(
    engine_common::SymbolId symbol) {
    if (windows_.size() <= symbol) windows_.resize(static_cast<size_t>(symbol) + 1);
    Window& window = windows_[symbol];
    if (window.values.empty()) {
        window.values.assign(static_cast<size_t>(lookback_) * feature_count_, 0.0F);
        window.mask.assign(lookback_, 0);
        window.static_values.assign(static_feature_count_, 0.0F);
    }
    return window;
}

void FeatureWindowStore::update(
    engine_common::SymbolId symbol, engine_common::TimestampNs timestamp,
    std::span<const float> features, bool valid,
    std::span<const float> static_features) {
    if (timestamp <= 0 || features.size() != feature_count_ ||
        static_features.size() != static_feature_count_) {
        throw std::invalid_argument("invalid feature row");
    }
    Window& window = window_for(symbol);
    if (window.last_timestamp >= timestamp) {
        throw std::invalid_argument("feature timestamps must be strictly increasing");
    }
    const size_t offset = static_cast<size_t>(window.next) * feature_count_;
    std::copy(features.begin(), features.end(), window.values.begin() + offset);
    window.mask[window.next] = valid ? 1 : 0;
    std::copy(static_features.begin(), static_features.end(),
              window.static_values.begin());
    window.next = (window.next + 1) % lookback_;
    window.size = std::min<uint32_t>(window.size + 1, lookback_);
    window.last_timestamp = timestamp;
}

engine_common::FeatureBatchView FeatureWindowStore::batch(
    engine_common::TimestampNs asof_timestamp, uint64_t feature_schema_hash,
    std::span<const engine_common::SymbolId> symbols) {
    if (asof_timestamp <= 0 || feature_schema_hash == 0 || symbols.empty()) {
        throw std::invalid_argument("invalid feature batch request");
    }
    batch_symbols_.assign(symbols.begin(), symbols.end());
    batch_values_.assign(
        symbols.size() * static_cast<size_t>(lookback_) * feature_count_, 0.0F);
    batch_mask_.assign(symbols.size() * static_cast<size_t>(lookback_), 0);
    batch_static_values_.assign(
        symbols.size() * static_cast<size_t>(static_feature_count_), 0.0F);
    for (size_t row = 0; row < symbols.size(); ++row) {
        if (symbols[row] >= windows_.size()) continue;
        const Window& window = windows_[symbols[row]];
        if (window.values.empty()) continue;
        const uint32_t padding = lookback_ - window.size;
        const uint32_t oldest = window.size == lookback_ ? window.next : 0;
        for (uint32_t index = 0; index < window.size; ++index) {
            const uint32_t source = (oldest + index) % lookback_;
            const uint32_t destination = padding + index;
            const size_t source_offset = static_cast<size_t>(source) * feature_count_;
            const size_t destination_offset =
                (row * lookback_ + destination) * feature_count_;
            std::copy_n(window.values.begin() + source_offset, feature_count_,
                        batch_values_.begin() + destination_offset);
            batch_mask_[row * lookback_ + destination] = window.mask[source];
        }
        std::copy(window.static_values.begin(), window.static_values.end(),
                  batch_static_values_.begin() + row * static_feature_count_);
    }
    return {asof_timestamp, feature_schema_hash, batch_symbols_, batch_values_,
            batch_mask_, batch_static_values_,
            static_cast<uint32_t>(symbols.size()), lookback_, feature_count_,
            static_feature_count_};
}

void FeatureWindowStore::reset() noexcept {
    windows_.clear();
    batch_symbols_.clear();
    batch_values_.clear();
    batch_mask_.clear();
    batch_static_values_.clear();
}

void FeatureWindowStore::reset(engine_common::SymbolId symbol) noexcept {
    if (symbol < windows_.size()) windows_[symbol] = {};
}

}  // namespace qbt::ml

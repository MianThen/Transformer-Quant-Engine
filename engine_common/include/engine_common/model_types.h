#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "types.h"

namespace engine_common {

enum class FeatureProfile : uint8_t {
  BAR_V1 = 1,
  L1_QUOTE_V1 = 2,
  TRADE_BAR_V1 = 3,
};

enum class TensorDType : uint8_t {
  FLOAT32 = 1,
  UINT8 = 2,
};

enum class RankingScoreMode : uint8_t {
  RAW_RETURN = 1,
  RISK_ADJUSTED_RETURN = 2,
};

enum PredictionFlags : uint32_t {
  PREDICTION_VALID = 1U << 0,
  INSUFFICIENT_HISTORY = 1U << 1,
  INPUT_STALE = 1U << 2,
  MODEL_FALLBACK = 1U << 3,
};

struct FeatureBatchView {
  TimestampNs asof_timestamp = 0;
  uint64_t feature_schema_hash = 0;
  std::span<const SymbolId> symbols;
  std::span<const float> values;
  std::span<const uint8_t> valid_mask;
  std::span<const float> static_values;
  uint32_t batch_size = 0;
  uint32_t lookback = 0;
  uint32_t feature_count = 0;
  uint32_t static_feature_count = 0;

  [[nodiscard]] bool valid_shape() const noexcept {
    const auto dynamic_size =
        checked_product(batch_size, lookback, feature_count);
    const auto mask_size = checked_product(batch_size, lookback, 1);
    const auto static_size =
        checked_product(batch_size, static_feature_count, 1);
    return asof_timestamp > 0 && feature_schema_hash != 0 &&
           symbols.size() == batch_size && dynamic_size != invalid_size() &&
           values.size() == dynamic_size && mask_size != invalid_size() &&
           valid_mask.size() == mask_size && static_size != invalid_size() &&
           static_values.size() == static_size;
  }

private:
  static constexpr size_t invalid_size() noexcept {
    return std::numeric_limits<size_t>::max();
  }

  static constexpr size_t checked_product(size_t first, size_t second,
                                          size_t third) noexcept {
    if (first != 0 && second > invalid_size() / first)
      return invalid_size();
    const size_t value = first * second;
    if (value != 0 && third > invalid_size() / value)
      return invalid_size();
    return value * third;
  }
};

struct ModelPrediction {
  SymbolId symbol_id = 0;
  TimestampNs asof_timestamp = 0;
  float expected_return = 0.0F;
  float expected_volatility = 0.0F;
  float direction_probability = 0.5F;
  float lower_quantile = 0.0F;
  float upper_quantile = 0.0F;
  float confidence = 0.0F;
  uint32_t flags = 0;

  [[nodiscard]] bool finite() const noexcept {
    return std::isfinite(expected_return) &&
           std::isfinite(expected_volatility) &&
           std::isfinite(direction_probability) &&
           std::isfinite(lower_quantile) && std::isfinite(upper_quantile) &&
           std::isfinite(confidence);
  }
};

struct PredictionBatch {
  TimestampNs asof_timestamp = 0;
  uint64_t model_version_hash = 0;
  std::span<ModelPrediction> values;
  size_t size = 0;

  [[nodiscard]] bool valid_shape() const noexcept {
    return asof_timestamp > 0 && model_version_hash != 0 &&
           size <= values.size();
  }
};

struct TargetPosition {
  SymbolId symbol_id = 0;
  Quantity target_quantity = 0;
  float target_weight = 0.0F;
};

struct TargetPositionBatch {
  TimestampNs asof_timestamp = 0;
  std::span<TargetPosition> values;
  size_t size = 0;
};

} // namespace engine_common

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "engine_common/types.h"

namespace performance_analytics {

enum class FactorAttributionStatus : std::uint8_t {
  OK,
  INVALID_INPUT,
  FUTURE_DATA,
  RECONCILIATION_FAILURE,
};

struct FactorAttributionSpecV2 {
  double reconciliation_tolerance{1e-10};
  std::uint64_t factor_schema_hash{0};
  std::uint64_t pit_exposure_hash{0};
  std::uint64_t config_hash{0};
};

struct FactorExposureView {
  const double* data{nullptr};
  std::size_t rows{0};
  std::size_t cols{0};
  std::size_t row_stride{0};

  [[nodiscard]] double operator()(std::size_t row, std::size_t col) const noexcept {
    return data[row * row_stride + col];
  }
};

struct FactorAttributionProblemV2 {
  FactorExposureView exposures;
  std::span<const double> asset_returns;
  std::span<const double> factor_returns;
  std::span<const double> portfolio_weights;
  double accounting_portfolio_return{0.0};
  engine_common::TimestampNs available_at{0};
  engine_common::TimestampNs decision_at{0};
  std::uint64_t period_id{0};
  FactorAttributionSpecV2 spec;
};

struct FactorAttributionResultV2 {
  FactorAttributionStatus status{FactorAttributionStatus::INVALID_INPUT};
  std::uint64_t period_id{0};
  std::vector<double> factor_contributions;
  std::vector<double> portfolio_factor_exposure;
  double specific_contribution{0.0};
  double factor_contribution_total{0.0};
  double reconstructed_portfolio_return{0.0};
  double accounting_portfolio_return{0.0};
  double reconciliation_residual{0.0};
  std::uint64_t artifact_hash{0};
};

[[nodiscard]] FactorAttributionResultV2 compute_factor_attribution_v2(
    const FactorAttributionProblemV2& problem);

}  // namespace performance_analytics

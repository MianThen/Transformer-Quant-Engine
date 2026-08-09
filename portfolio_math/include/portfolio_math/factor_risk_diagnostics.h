#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "engine_common/types.h"
#include "portfolio_math/factor_model.h"

namespace portfolio_math {

enum class FactorRiskDiagnosticStatus : std::uint8_t {
  OK,
  INVALID_INPUT,
  FUTURE_DATA,
  NON_PSD,
  NUMERICAL_FAILURE,
};

struct FactorRiskDiagnosticInput {
  FactorRiskModelView predicted_risk_model;
  std::uint64_t predicted_risk_artifact_hash{0};
  quant_math::MatrixView realized_factor_returns;
  quant_math::MatrixView realized_asset_returns;
  std::span<const double> portfolio_weights;
  std::span<const engine_common::TimestampNs> observation_timestamps;
  engine_common::TimestampNs available_at{0};
  engine_common::TimestampNs decision_at{0};
  double tolerance{1e-10};
};

struct FactorRiskDiagnosticResult {
  FactorRiskDiagnosticStatus status{FactorRiskDiagnosticStatus::INVALID_INPUT};
  std::uint64_t predicted_risk_artifact_hash{0};
  std::size_t effective_observations{0};
  double factor_covariance_qlike{0.0};
  double predicted_factor_variance{0.0};
  double predicted_specific_variance{0.0};
  double predicted_variance_reconciliation_error{0.0};
  double predicted_portfolio_variance{0.0};
  double realized_portfolio_variance{0.0};
  double realized_predicted_variance_ratio{0.0};
  double mean_absolute_risk_contribution_error{0.0};
  double maximum_absolute_risk_contribution_error{0.0};
  std::vector<double> predicted_risk_contributions;
  std::vector<double> realized_risk_contributions;
};

[[nodiscard]] FactorRiskDiagnosticResult evaluate_factor_risk_diagnostics(
    const FactorRiskDiagnosticInput& input);

}  // namespace portfolio_math

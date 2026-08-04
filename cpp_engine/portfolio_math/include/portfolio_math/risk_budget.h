#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "portfolio_math/types.h"
#include "quant_math/matrix.h"

namespace portfolio_math {

struct RiskBudgetOptions {
    std::uint32_t max_iterations{10'000};
    double tolerance{1e-10};
    double covariance_tolerance{1e-10};
    double target_investment{1.0};
};

struct RiskContributionResult {
    OptimizationStatus status{OptimizationStatus::INVALID_INPUT};
    double portfolio_volatility{0.0};
    std::vector<double> contributions;
    std::vector<double> contribution_shares;
};

struct RiskBudgetResult {
    std::vector<double> weights;
    OptimizationDiagnostics diagnostics;
};

[[nodiscard]] RiskContributionResult risk_contributions(
    quant_math::MatrixView covariance,
    std::span<const double> weights);

[[nodiscard]] RiskBudgetResult solve_long_only_risk_budget(
    quant_math::MatrixView covariance,
    std::span<const double> risk_budgets,
    std::span<const double> current_weights = {},
    std::span<const double> warm_start = {},
    RiskBudgetOptions options = {});

[[nodiscard]] RiskBudgetResult solve_bounded_long_only_risk_budget(
    quant_math::MatrixView covariance,
    std::span<const double> risk_budgets,
    std::span<const double> lower_bounds,
    std::span<const double> upper_bounds,
    std::span<const double> current_weights = {},
    std::span<const double> warm_start = {},
    RiskBudgetOptions options = {});

}  // namespace portfolio_math

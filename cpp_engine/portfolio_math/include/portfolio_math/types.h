#pragma once

#include <cstdint>

namespace portfolio_math {

enum class OptimizationStatus : std::uint8_t {
    OK,
    INVALID_INPUT,
    NON_PSD_RISK_MODEL,
    INFEASIBLE,
    MAX_ITERATIONS,
    NUMERICAL_FAILURE,
};

struct OptimizationDiagnostics {
    OptimizationStatus status{OptimizationStatus::INVALID_INPUT};
    std::uint32_t iterations{0};
    std::uint32_t active_bound_count{0};
    double kkt_residual{0.0};
    double max_risk_budget_error{0.0};
    double predicted_risk{0.0};
    double turnover{0.0};
};

}  // namespace portfolio_math

#pragma once

#include <cstdint>
#include <span>

#include "engine_common/types.h"
#include "portfolio_math/covariance.h"
#include "quant_math/matrix.h"

namespace portfolio_math {

enum class DimensionalBranch : std::uint8_t {
    REGULAR_P_LT_N,
    SINGULAR_P_GT_N,
    NOT_APPLICABLE,
};

struct RiskPreprocessorSpec {
    CovarianceEstimator official_estimator{CovarianceEstimator::SAMPLE};
    CovarianceLossProfile covariance_loss{CovarianceLossProfile::NOT_APPLICABLE};
    std::uint32_t lookback_observations{0};
    bool demean_returns{true};
    bool uniform_observation_weights{true};
    double concentration_ratio_guard{0.0};
    double eigenvalue_floor{0.0};
    std::uint64_t balanced_panel_policy_hash{0};
    std::uint64_t quest_solver_spec_hash{0};
    std::uint64_t rmt_spec_hash{0};
    std::uint64_t config_hash{0};
};

[[nodiscard]] bool valid_risk_preprocessor_spec(
    const RiskPreprocessorSpec& spec) noexcept;

struct DenseRiskModelView {
    CovarianceEstimator estimator{CovarianceEstimator::SAMPLE};
    CovarianceLossProfile loss_profile{CovarianceLossProfile::NOT_APPLICABLE};
    engine_common::TimestampNs fit_start{0};
    engine_common::TimestampNs fit_end{0};
    engine_common::TimestampNs available_at{0};
    std::span<const engine_common::SymbolId> symbols;
    quant_math::MatrixView covariance;
    std::uint64_t symbol_mapping_hash{0};
    std::uint64_t balanced_panel_policy_hash{0};
    std::uint32_t effective_observations{0};
    double concentration_ratio{0.0};
    DimensionalBranch dimensional_branch{DimensionalBranch::NOT_APPLICABLE};
    std::uint64_t artifact_hash{0};
};

[[nodiscard]] bool valid_dense_risk_model(
    const DenseRiskModelView& model,
    engine_common::TimestampNs decision_at,
    double symmetry_tolerance = 1e-10);

}  // namespace portfolio_math

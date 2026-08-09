#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "engine_common/types.h"
#include "portfolio_math/tail_risk.h"
#include "quant_math/matrix.h"

namespace portfolio_math {

enum class ConditionalExpectileStatus : std::uint8_t {
    OK,
    INVALID_INPUT,
    INSUFFICIENT_OBSERVATIONS,
    OPTIMIZATION_FAILURE,
    NUMERICAL_FAILURE,
};

struct ConditionalExpectileSpec {
    double expectile_level{0.0};
    std::uint32_t minimum_observations{40};
    std::uint32_t maximum_iterations{200};
    double convergence_tolerance{1e-10};
    double ridge_penalty{1e-10};
    bool include_intercept{true};
    bool training_only_calibration{true};
    std::uint64_t feature_spec_hash{0};
    std::uint64_t solver_spec_hash{0};
    std::uint64_t config_hash{0};
};

struct ConditionalExpectileProblemView {
    engine_common::TimestampNs decision_at{0};
    std::span<const engine_common::TimestampNs> target_timestamps;
    std::span<const engine_common::TimestampNs> feature_available_at;
    std::span<const double> target_losses;
    std::span<const double> sample_weights;
    quant_math::MatrixView feature_history;
    std::span<const double> forecast_features;
    engine_common::TimestampNs forecast_features_available_at{0};
    ConditionalExpectileSpec spec;
};

struct ConditionalExpectileResult {
    ConditionalExpectileStatus status{ConditionalExpectileStatus::INVALID_INPUT};
    double expectile_level{0.0};
    double forecast_expectile_loss{0.0};
    double mean_asymmetric_squared_loss{0.0};
    double maximum_first_order_residual{0.0};
    std::uint32_t effective_observations{0};
    std::uint32_t feature_count{0};
    std::uint32_t optimization_iterations{0};
    std::vector<double> coefficients;
    std::uint64_t input_hash{0};
    std::uint64_t artifact_hash{0};
};

[[nodiscard]] bool valid_conditional_expectile_spec(
    const ConditionalExpectileSpec& spec) noexcept;

[[nodiscard]] ConditionalExpectileResult fit_conditional_expectile(
    const ConditionalExpectileProblemView& problem);

[[nodiscard]] std::string serialize_conditional_expectile_artifact(
    const ConditionalExpectileResult& result,
    const ConditionalExpectileSpec& spec,
    const TailRiskArtifactSpec& artifact_spec);

}  // namespace portfolio_math

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "engine_common/types.h"
#include "quant_math/matrix.h"

namespace portfolio_math {

struct DenseRiskModelView;

enum class TailRiskEstimatorKind : std::uint8_t {
    EMPIRICAL_ROCKAFELLAR_URYASEV,
    GARCH_FILTERED_HISTORICAL_SIMULATION,
    GARCH_FHS_POT_GPD,
    EXPECTILE_DIRECT,
    EXPECTILE_TAYLOR_MAPPED_ES,
};

enum class TailRiskStatus : std::uint8_t {
    OK,
    INVALID_INPUT,
    VOLATILITY_FIT_FAILURE,
    RESIDUAL_DIAGNOSTIC_FAILURE,
    INSUFFICIENT_TAIL,
    EVT_FIT_FAILURE,
    EVT_INFINITE_MEAN,
    EXPECTILE_CALIBRATION_FAILURE,
    NUMERICAL_FAILURE,
};

enum class TailScenarioModelKind : std::uint8_t {
    PORTFOLIO_RETURN_SERIES,
    ASSET_VECTOR_SYNCHRONIZED,
    FACTOR_SPECIFIC_SYNCHRONIZED,
};

struct TailRiskSpec {
    TailRiskEstimatorKind estimator{
        TailRiskEstimatorKind::EMPIRICAL_ROCKAFELLAR_URYASEV};
    TailScenarioModelKind scenario_model{
        TailScenarioModelKind::PORTFOLIO_RETURN_SERIES};
    double confidence_level{0.95};
    double expectile_level{0.0};
    std::uint32_t forecast_horizon_periods{1};
    std::uint32_t residual_block_length{1};
    std::uint32_t evt_minimum_exceedances{0};
    double evt_threshold_quantile_min{0.0};
    double evt_threshold_quantile_max{0.0};
    double evt_shape_upper_guard{0.0};
    bool synchronized_residual_rows{false};
    bool filtered_volatility_state_only{true};
    bool training_only_tail_calibration{true};
    std::uint64_t mean_model_spec_hash{0};
    std::uint64_t volatility_model_spec_hash{0};
    std::uint64_t evt_threshold_spec_hash{0};
    std::uint64_t expectile_feature_spec_hash{0};
    std::uint64_t scenario_seed{0};
    std::uint64_t config_hash{0};
};

struct TailRiskProblemView {
    engine_common::TimestampNs decision_at{0};
    std::span<const engine_common::SymbolId> symbols;
    std::span<const engine_common::TimestampNs> history_timestamps;
    std::span<const double> fixed_portfolio_weights;
    std::span<const double> portfolio_return_history;
    std::span<const double> scenario_probabilities;
    quant_math::MatrixView asset_return_history;
    quant_math::MatrixView factor_return_history;
    quant_math::MatrixView specific_return_history;
    const DenseRiskModelView* factor_risk_model{nullptr};
    TailRiskSpec spec;
};

struct TailRiskEstimate {
    TailRiskStatus status{TailRiskStatus::INVALID_INPUT};
    TailRiskEstimatorKind estimator{
        TailRiskEstimatorKind::EMPIRICAL_ROCKAFELLAR_URYASEV};
    double confidence_level{0.0};
    std::optional<double> value_at_risk_loss;
    std::optional<double> expected_shortfall_loss;
    std::optional<double> return_cvar;
    std::optional<double> expectile_loss;
    std::optional<double> calibrated_expectile_level;
    std::uint32_t effective_observations{0};
    std::uint32_t evt_exceedance_count{0};
    std::optional<double> evt_threshold;
    std::optional<double> gpd_shape;
    std::optional<double> gpd_scale;
    std::uint64_t input_hash{0};
    std::uint64_t artifact_hash{0};
};

struct TailRiskArtifactSpec {
    std::string source_dataset_fingerprint;
    std::string portfolio_weights_sha256;
    std::string return_panel_policy_hash;
    std::string reference_price_quality{"PROXY"};
    bool promotion_eligible{false};
    std::vector<std::string> limitations;
};

[[nodiscard]] bool valid_tail_risk_spec(const TailRiskSpec& spec) noexcept;

[[nodiscard]] TailRiskEstimate estimate_tail_risk(
    const TailRiskProblemView& problem);

[[nodiscard]] std::string serialize_tail_risk_artifact(
    const TailRiskEstimate& estimate, const TailRiskSpec& spec,
    const TailRiskArtifactSpec& artifact_spec);

}  // namespace portfolio_math

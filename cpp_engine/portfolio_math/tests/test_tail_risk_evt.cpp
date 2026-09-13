#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "portfolio_math/tail_risk.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
    return condition;
}

bool near(double actual, double expected, double tolerance = 1e-10) {
    return std::abs(actual - expected) <= tolerance;
}

portfolio_math::TailRiskSpec make_evt_spec() {
    portfolio_math::TailRiskSpec spec;
    spec.estimator =
        portfolio_math::TailRiskEstimatorKind::GARCH_FHS_POT_GPD;
    spec.scenario_model =
        portfolio_math::TailScenarioModelKind::PORTFOLIO_RETURN_SERIES;
    spec.confidence_level = 0.95;
    spec.evt_minimum_exceedances = 5;
    spec.evt_threshold_grid_points = 4;
    spec.evt_threshold_quantile_min = 0.75;
    spec.evt_threshold_quantile_max = 0.90;
    spec.evt_shape_upper_guard = 0.99;
    spec.evt_max_shape_spread = 0.25;
    spec.evt_max_relative_es_spread = 0.25;
    spec.mean_model_spec_hash = 301;
    spec.volatility_model_spec_hash = 302;
    spec.evt_threshold_spec_hash = 303;
    spec.config_hash = 304;
    return spec;
}

bool test_gpd_formula_oracles() {
    bool ok = true;
    const auto xi_zero = portfolio_math::evaluate_gpd_tail(
        0.1, 0.1, 0.0, 0.02, 0.95, 0.99);
    const double expected_var = 0.1 + 0.02 * std::log(2.0);
    ok &= check(xi_zero.status == portfolio_math::TailRiskStatus::OK &&
                    near(xi_zero.value_at_risk_loss, expected_var) &&
                    near(xi_zero.expected_shortfall_loss,
                         expected_var + 0.02),
                "xi zero exponential limit oracle");

    const auto xi_negative = portfolio_math::evaluate_gpd_tail(
        0.1, 0.1, -0.2, 0.03, 0.95, 0.99);
    ok &= check(xi_negative.status == portfolio_math::TailRiskStatus::OK &&
                    xi_negative.expected_shortfall_loss >=
                        xi_negative.value_at_risk_loss,
                "negative xi finite-endpoint oracle");

    const auto xi_one = portfolio_math::evaluate_gpd_tail(
        0.1, 0.1, 1.0, 0.02, 0.95, 0.99);
    ok &= check(xi_one.status ==
                    portfolio_math::TailRiskStatus::EVT_INFINITE_MEAN,
                "xi at least one fails closed");
    const auto guarded_shape = portfolio_math::evaluate_gpd_tail(
        0.1, 0.1, 0.3, 0.02, 0.95, 0.25);
    ok &= check(guarded_shape.status ==
                    portfolio_math::TailRiskStatus::EVT_INFINITE_MEAN,
                "registered shape guard fails closed");
    return ok;
}

bool test_weighted_splice_and_artifact() {
    std::vector<double> losses(100, 0.0);
    const std::vector<double> excesses{
        0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.30};
    for (std::size_t index = 0; index < excesses.size(); ++index) {
        losses[91 + index] = excesses[index];
    }
    std::vector<double> probabilities(100, 0.01);
    probabilities[91] = 0.005;
    for (std::size_t index = 92; index < probabilities.size(); ++index) {
        probabilities[index] = 0.010625;
    }
    double weighted_mean = 0.0;
    double weighted_second_moment = 0.0;
    double squared_tail_probability = 0.0;
    for (std::size_t index = 91; index < losses.size(); ++index) {
        weighted_mean += probabilities[index] * losses[index];
        weighted_second_moment +=
            probabilities[index] * losses[index] * losses[index];
        squared_tail_probability += probabilities[index] * probabilities[index];
    }
    constexpr double tail_probability = 0.09;
    weighted_mean /= tail_probability;
    weighted_second_moment /= tail_probability;
    const double weighted_variance =
        weighted_second_moment - weighted_mean * weighted_mean;
    const double expected_shape =
        0.5 * (1.0 - weighted_mean * weighted_mean / weighted_variance);
    const double expected_scale = 0.5 * weighted_mean *
        (1.0 + weighted_mean * weighted_mean / weighted_variance);
    const double expected_effective =
        tail_probability * tail_probability / squared_tail_probability;

    const auto spec = make_evt_spec();
    const auto result = portfolio_math::estimate_fhs_pot_gpd_splice(
        losses, probabilities, spec);
    bool ok = true;
    ok &= check(result.status == portfolio_math::TailRiskStatus::OK &&
                    result.evt_threshold_diagnostics.size() == 4 &&
                    result.evt_selected_threshold_quantile &&
                    near(*result.evt_selected_threshold_quantile, 0.75) &&
                    result.evt_effective_exceedances &&
                    near(*result.evt_effective_exceedances,
                         expected_effective) &&
                    result.gpd_shape && near(*result.gpd_shape, expected_shape) &&
                    result.gpd_scale && near(*result.gpd_scale, expected_scale),
                "weighted moments and four-point threshold grid oracle");
    for (const auto& diagnostic : result.evt_threshold_diagnostics) {
        ok &= check(near(diagnostic.threshold_loss, 0.0) &&
                        near(diagnostic.splice_continuity_error, 0.0) &&
                        near(diagnostic.splice_probability_error, 0.0),
                    "tail splice continuity and normalization oracle");
    }
    const auto replay = portfolio_math::estimate_fhs_pot_gpd_splice(
        losses, probabilities, spec);
    ok &= check(replay.status == portfolio_math::TailRiskStatus::OK &&
                    replay.artifact_hash == result.artifact_hash,
                "weighted splice deterministic replay");

    portfolio_math::TailRiskArtifactSpec artifact_spec;
    artifact_spec.reference_price_quality = "PROXY";
    artifact_spec.promotion_eligible = true;
    const std::string artifact = portfolio_math::serialize_tail_risk_artifact(
        result, spec, artifact_spec);
    ok &= check(
        artifact.find("WEIGHTED_MOMENTS_FOUR_POINT_GRID_V1") !=
                std::string::npos &&
            artifact.find("\"threshold_diagnostics\":[{") !=
                std::string::npos &&
            artifact.find("\"promotion_eligible\":false") !=
                std::string::npos,
        "versioned EVT artifact and proxy promotion gate");
    return ok;
}

bool test_threshold_stability_gate() {
    constexpr std::size_t observation_count = 200;
    std::vector<double> losses(observation_count);
    for (std::size_t index = 0; index < observation_count; ++index) {
        const double probability =
            (static_cast<double>(index) + 0.5) /
            (static_cast<double>(observation_count) + 1.0);
        losses[index] = -std::log1p(-probability);
    }
    auto spec = make_evt_spec();
    spec.evt_minimum_exceedances = 15;
    spec.evt_max_shape_spread = 1e-12;
    spec.evt_max_relative_es_spread = 1e-12;
    const auto result = portfolio_math::estimate_fhs_pot_gpd_splice(
        losses, {}, spec);
    return check(result.status == portfolio_math::TailRiskStatus::EVT_FIT_FAILURE &&
                     result.evt_threshold_diagnostics.size() == 4 &&
                     result.evt_shape_spread &&
                     *result.evt_shape_spread > spec.evt_max_shape_spread,
                 "threshold parameter stability fails closed");
}

}  // namespace

int main() {
    const bool ok = test_gpd_formula_oracles() &&
        test_weighted_splice_and_artifact() && test_threshold_stability_gate();
    if (!ok) return 1;
    std::printf("test_tail_risk_evt: all checks passed\n");
    return 0;
}

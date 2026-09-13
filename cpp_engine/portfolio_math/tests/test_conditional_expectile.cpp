#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "portfolio_math/conditional_expectile.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
    return condition;
}

bool near(double actual, double expected, double tolerance = 1e-7) {
    return std::abs(actual - expected) <= tolerance;
}

portfolio_math::ConditionalExpectileProblemView make_problem(
    const std::vector<double>& losses,
    const std::vector<engine_common::TimestampNs>& target_timestamps,
    const std::vector<engine_common::TimestampNs>& feature_timestamps,
    const quant_math::DenseMatrix& features,
    const std::vector<double>& forecast_features,
    const portfolio_math::ConditionalExpectileSpec& spec) {
    return {
        1000,
        target_timestamps,
        feature_timestamps,
        losses,
        {},
        quant_math::view(features),
        forecast_features,
        1000,
        spec,
    };
}

bool test_conditional_expectile_oracles() {
    constexpr std::size_t observation_count = 80;
    std::vector<double> losses(observation_count);
    std::vector<engine_common::TimestampNs> target_timestamps(observation_count);
    std::vector<engine_common::TimestampNs> feature_timestamps(observation_count);
    quant_math::DenseMatrix features(observation_count, 1);
    for (std::size_t index = 0; index < observation_count; ++index) {
        const double feature = static_cast<double>(index / 2) / 20.0 - 1.0;
        const double residual = index % 2 == 0 ? 1.0 : -4.0;
        features(static_cast<Eigen::Index>(index), 0) = feature;
        losses[index] = 1.0 + 2.0 * feature + residual;
        feature_timestamps[index] = static_cast<engine_common::TimestampNs>(
            10 + index * 2);
        target_timestamps[index] = feature_timestamps[index] + 1;
    }
    std::vector<double> forecast_features{2.0};
    portfolio_math::ConditionalExpectileSpec spec;
    spec.expectile_level = 0.8;
    spec.minimum_observations = 40;
    spec.maximum_iterations = 100;
    spec.convergence_tolerance = 1e-10;
    spec.ridge_penalty = 1e-12;
    spec.feature_spec_hash = 101;
    spec.solver_spec_hash = 102;
    spec.config_hash = 103;
    auto problem = make_problem(
        losses, target_timestamps, feature_timestamps, features,
        forecast_features, spec);

    bool ok = true;
    const auto result = portfolio_math::fit_conditional_expectile(problem);
    ok &= check(result.status ==
                    portfolio_math::ConditionalExpectileStatus::OK &&
                    result.coefficients.size() == 2 &&
                    result.effective_observations == observation_count &&
                    result.feature_count == 1 &&
                    near(result.coefficients[0], 1.0) &&
                    near(result.coefficients[1], 2.0) &&
                    near(result.forecast_expectile_loss, 5.0) &&
                    result.maximum_first_order_residual < 1e-9,
                "fixed PIT ALS recovers conditional expectile oracle");
    const auto replay = portfolio_math::fit_conditional_expectile(problem);
    ok &= check(replay.status == portfolio_math::ConditionalExpectileStatus::OK &&
                    replay.artifact_hash == result.artifact_hash,
                "conditional expectile deterministic replay");

    auto translated_losses = losses;
    for (double& value : translated_losses) value += 3.0;
    auto translated_problem = make_problem(
        translated_losses, target_timestamps, feature_timestamps, features,
        forecast_features, spec);
    const auto translated =
        portfolio_math::fit_conditional_expectile(translated_problem);
    ok &= check(translated.status ==
                    portfolio_math::ConditionalExpectileStatus::OK &&
                    near(translated.forecast_expectile_loss,
                         result.forecast_expectile_loss + 3.0),
                "conditional expectile translation equivariance");

    auto scaled_losses = losses;
    for (double& value : scaled_losses) value *= 2.0;
    auto scaled_problem = make_problem(
        scaled_losses, target_timestamps, feature_timestamps, features,
        forecast_features, spec);
    const auto scaled = portfolio_math::fit_conditional_expectile(scaled_problem);
    ok &= check(scaled.status == portfolio_math::ConditionalExpectileStatus::OK &&
                    near(scaled.forecast_expectile_loss,
                         2.0 * result.forecast_expectile_loss),
                "conditional expectile positive homogeneity");

    portfolio_math::TailRiskArtifactSpec artifact_spec;
    artifact_spec.reference_price_quality = "PROXY";
    artifact_spec.promotion_eligible = true;
    artifact_spec.limitations = {"OBSERVED_PROXY_FEATURES"};
    const std::string artifact =
        portfolio_math::serialize_conditional_expectile_artifact(
            result, spec, artifact_spec);
    ok &= check(artifact.find("\"model_kind\":\"FIXED_PIT_LINEAR_ALS\"") !=
                        std::string::npos &&
                    artifact.find("\"care_sav_claimed\":false") !=
                        std::string::npos &&
                    artifact.find("\"mapped_es_claimed\":false") !=
                        std::string::npos &&
                    artifact.find("\"promotion_eligible\":false") !=
                        std::string::npos,
                "conditional expectile artifact claim boundary");
    return ok;
}

bool test_conditional_expectile_guards() {
    constexpr std::size_t observation_count = 40;
    std::vector<double> losses(observation_count, 0.0);
    std::vector<engine_common::TimestampNs> target_timestamps(observation_count);
    std::vector<engine_common::TimestampNs> feature_timestamps(observation_count);
    quant_math::DenseMatrix features(observation_count, 1);
    for (std::size_t index = 0; index < observation_count; ++index) {
        features(static_cast<Eigen::Index>(index), 0) =
            static_cast<double>(index);
        losses[index] = 0.1 * static_cast<double>(index);
        feature_timestamps[index] = static_cast<engine_common::TimestampNs>(
            10 + index * 2);
        target_timestamps[index] = feature_timestamps[index] + 1;
    }
    std::vector<double> forecast_features{40.0};
    portfolio_math::ConditionalExpectileSpec spec;
    spec.expectile_level = 0.8;
    spec.minimum_observations = 40;
    spec.feature_spec_hash = 201;
    spec.solver_spec_hash = 202;
    spec.config_hash = 203;
    auto problem = make_problem(
        losses, target_timestamps, feature_timestamps, features,
        forecast_features, spec);

    bool ok = true;
    auto future = problem;
    std::vector<engine_common::TimestampNs> future_features = feature_timestamps;
    future_features.back() = target_timestamps.back();
    future.feature_available_at = future_features;
    ok &= check(portfolio_math::fit_conditional_expectile(future).status ==
                    portfolio_math::ConditionalExpectileStatus::INVALID_INPUT,
                "contemporaneous feature fails closed");

    auto stale_forecast = problem;
    stale_forecast.forecast_features_available_at = 999;
    ok &= check(portfolio_math::fit_conditional_expectile(stale_forecast).status ==
                    portfolio_math::ConditionalExpectileStatus::INVALID_INPUT,
                "forecast feature availability mismatch fails closed");

    auto unregistered = problem;
    unregistered.spec.feature_spec_hash = 0;
    ok &= check(portfolio_math::fit_conditional_expectile(unregistered).status ==
                    portfolio_math::ConditionalExpectileStatus::INVALID_INPUT,
                "unregistered feature spec fails closed");

    auto too_short = problem;
    too_short.spec.minimum_observations = 41;
    ok &= check(portfolio_math::fit_conditional_expectile(too_short).status ==
                    portfolio_math::ConditionalExpectileStatus::INSUFFICIENT_OBSERVATIONS,
                "short conditional expectile training fold fails closed");
    return ok;
}

}  // namespace

int main() {
    const bool ok = test_conditional_expectile_oracles() &&
        test_conditional_expectile_guards();
    if (!ok) return 1;
    std::printf("test_conditional_expectile: all checks passed\n");
    return 0;
}

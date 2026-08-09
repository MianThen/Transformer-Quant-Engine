#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "portfolio_math/tail_risk.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
    return condition;
}

double uniform_01(std::uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (static_cast<double>(state >> 11) + 0.5) /
        static_cast<double>(1ULL << 53);
}

double standard_normal(std::uint64_t& state) {
    constexpr double two_pi = 6.2831853071795864769;
    const double first = uniform_01(state);
    const double second = uniform_01(state);
    return std::sqrt(-2.0 * std::log(first)) *
        std::cos(two_pi * second);
}

bool finite_probability(double value) {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

bool test_esr_suite() {
    bool ok = true;
    constexpr std::size_t observation_count = 1200;
    constexpr double normal_var_95 = 1.6448536269514722;
    constexpr double normal_es_95 = 2.0627128075074257;
    std::vector<engine_common::TimestampNs> timestamps(observation_count);
    std::vector<double> returns(observation_count);
    std::vector<double> var_loss(observation_count);
    std::vector<double> es_loss(observation_count);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (std::size_t index = 0; index < observation_count; ++index) {
        const double scale = 0.65 +
            0.7 * static_cast<double>(index % 29) / 28.0;
        timestamps[index] = static_cast<engine_common::TimestampNs>(index + 1);
        returns[index] = scale * standard_normal(state);
        var_loss[index] = scale * normal_var_95;
        es_loss[index] = scale * normal_es_95;
    }
    portfolio_math::TailRiskEsrProblemView problem;
    problem.realization_timestamps = timestamps;
    problem.realized_returns = returns;
    problem.value_at_risk_loss = var_loss;
    problem.expected_shortfall_loss = es_loss;
    problem.available_at = static_cast<engine_common::TimestampNs>(
        observation_count);
    problem.spec.confidence_level = 0.95;
    problem.spec.minimum_observations = 40;
    problem.spec.maximum_iterations = 6000;
    problem.spec.convergence_tolerance = 1e-8;
    problem.spec.regression_spec_hash = 801;
    problem.spec.covariance_spec_hash = 802;
    problem.spec.config_hash = 803;

    const auto result = portfolio_math::backtest_tail_risk_esr(problem);
    ok &= check(result.status == portfolio_math::TailRiskEsrStatus::OK &&
                    result.effective_observations == observation_count &&
                    result.input_hash != 0 && result.artifact_hash != 0,
                "ESR suite fits all registered variants");
    const auto variant_is_valid = [](const auto& variant,
                                     std::size_t expected_quantile,
                                     std::size_t expected_es) {
        return variant.status == portfolio_math::TailRiskEsrStatus::OK &&
            variant.quantile_coefficients.size() == expected_quantile &&
            variant.expected_shortfall_coefficients.size() == expected_es &&
            std::isfinite(variant.objective) &&
            variant.density_at_quantile > 0.0 &&
            variant.truncated_residual_variance > 0.0 &&
            finite_probability(variant.two_sided_p_value) &&
            variant.artifact_hash != 0;
    };
    ok &= check(variant_is_valid(result.strict, 2, 2) &&
                    variant_is_valid(result.auxiliary, 2, 2) &&
                    variant_is_valid(result.strict_intercept, 2, 1) &&
                    result.strict_intercept.one_sided_underestimation_p_value &&
                    finite_probability(
                        *result.strict_intercept.
                            one_sided_underestimation_p_value),
                "ESR coefficients, sandwich covariance and p-values");
    ok &= check(std::abs(result.strict.expected_shortfall_coefficients[0]) <
                        0.25 &&
                    result.strict.expected_shortfall_coefficients[1] > 0.7 &&
                    result.strict.expected_shortfall_coefficients[1] < 1.3 &&
                    std::abs(result.auxiliary.
                        expected_shortfall_coefficients[0]) < 0.25 &&
                    result.auxiliary.expected_shortfall_coefficients[1] > 0.7 &&
                    result.auxiliary.expected_shortfall_coefficients[1] < 1.3 &&
                    std::abs(result.strict_intercept.
                        expected_shortfall_coefficients[0]) < 0.25,
                "correct normal forecasts recover ESR null neighborhood");

    const auto replay = portfolio_math::backtest_tail_risk_esr(problem);
    ok &= check(replay.status == portfolio_math::TailRiskEsrStatus::OK &&
                    replay.input_hash == result.input_hash &&
                    replay.artifact_hash == result.artifact_hash &&
                    replay.strict.artifact_hash == result.strict.artifact_hash &&
                    replay.auxiliary.artifact_hash ==
                        result.auxiliary.artifact_hash &&
                    replay.strict_intercept.artifact_hash ==
                        result.strict_intercept.artifact_hash,
                "ESR suite deterministic replay");

    auto correct_spec_problem = problem;
    correct_spec_problem.spec.covariance_kind =
        portfolio_math::TailRiskEsrCovarianceKind::CORRECT_SPEC_IID;
    correct_spec_problem.spec.hac_lag = 0;
    correct_spec_problem.spec.covariance_spec_hash = 804;
    correct_spec_problem.spec.config_hash = 805;
    const auto correct_spec =
        portfolio_math::backtest_tail_risk_esr(correct_spec_problem);
    ok &= check(correct_spec.status == portfolio_math::TailRiskEsrStatus::OK &&
                    correct_spec.covariance_kind ==
                        portfolio_math::TailRiskEsrCovarianceKind::
                            CORRECT_SPEC_IID &&
                    correct_spec.hac_lag == 0 &&
                    correct_spec.artifact_hash != result.artifact_hash,
                "correct-spec covariance remains an explicit reference path");

    auto changed_hac_problem = problem;
    changed_hac_problem.spec.hac_lag = 8;
    changed_hac_problem.spec.covariance_spec_hash = 806;
    changed_hac_problem.spec.config_hash = 807;
    const auto changed_hac =
        portfolio_math::backtest_tail_risk_esr(changed_hac_problem);
    ok &= check(changed_hac.status == portfolio_math::TailRiskEsrStatus::OK &&
                    changed_hac.hac_lag == 8 &&
                    changed_hac.artifact_hash != result.artifact_hash,
                "HAC lag is frozen into ESR inference artifact");

    portfolio_math::TailRiskArtifactSpec artifact_spec;
    artifact_spec.source_dataset_fingerprint = "synthetic-normal-v1";
    artifact_spec.reference_price_quality = "PROXY";
    artifact_spec.promotion_eligible = true;
    const std::string artifact =
        portfolio_math::serialize_tail_risk_esr_backtest_artifact(
            result, artifact_spec);
    ok &= check(artifact.find("\"strict\"") != std::string::npos &&
                    artifact.find("\"auxiliary\"") != std::string::npos &&
                    artifact.find("\"strict_intercept\"") !=
                        std::string::npos &&
                    artifact.find(
                        "KERNEL_BREAD_EMPIRICAL_SCORE_NEWEY_WEST_HAC_V1") !=
                        std::string::npos &&
                    artifact.find("\"hac_lag\":4") != std::string::npos &&
                    artifact.find("\"promotion_eligible\":false") !=
                        std::string::npos,
                "ESR artifact records suite and closes proxy promotion");
    const std::string correct_spec_artifact =
        portfolio_math::serialize_tail_risk_esr_backtest_artifact(
            correct_spec, artifact_spec);
    ok &= check(correct_spec_artifact.find(
                    "KERNEL_IND_CORRECT_SPEC_SANDWICH_V1") !=
                    std::string::npos,
                "correct-spec reference covariance remains serialized distinctly");

    auto future_problem = problem;
    future_problem.available_at = static_cast<engine_common::TimestampNs>(
        observation_count - 1);
    ok &= check(portfolio_math::backtest_tail_risk_esr(future_problem).status ==
                    portfolio_math::TailRiskEsrStatus::INVALID_INPUT,
                "ESR future realization fails closed");
    auto domain_problem = problem;
    std::vector<double> zero_es = es_loss;
    zero_es.front() = 0.0;
    domain_problem.expected_shortfall_loss = zero_es;
    ok &= check(portfolio_math::backtest_tail_risk_esr(domain_problem).status ==
                    portfolio_math::TailRiskEsrStatus::DOMAIN_FAILURE,
                "ESR nonpositive ES domain fails closed");
    auto short_problem = problem;
    short_problem.realization_timestamps =
        std::span<const engine_common::TimestampNs>(timestamps.data(), 20);
    short_problem.realized_returns =
        std::span<const double>(returns.data(), 20);
    short_problem.value_at_risk_loss =
        std::span<const double>(var_loss.data(), 20);
    short_problem.expected_shortfall_loss =
        std::span<const double>(es_loss.data(), 20);
    ok &= check(portfolio_math::backtest_tail_risk_esr(short_problem).status ==
                    portfolio_math::TailRiskEsrStatus::
                        INSUFFICIENT_OBSERVATIONS,
                "ESR short window fails closed");
    auto invalid_hac_problem = problem;
    invalid_hac_problem.spec.hac_lag = 0;
    ok &= check(
        portfolio_math::backtest_tail_risk_esr(invalid_hac_problem).status ==
            portfolio_math::TailRiskEsrStatus::INVALID_INPUT,
        "robust ESR requires a positive frozen HAC lag");
    return ok;
}

}  // namespace

int main() {
    if (!test_esr_suite()) return 1;
    std::printf("test_tail_risk_esr: all checks passed\n");
    return 0;
}

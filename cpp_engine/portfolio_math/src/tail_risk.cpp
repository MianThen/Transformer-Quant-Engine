#include "portfolio_math/tail_risk.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "portfolio_math/risk_model.h"

namespace portfolio_math {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_value(std::uint64_t& hash, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffU;
        hash *= kFnvPrime;
    }
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
        case '\\': output << "\\\\"; break;
        case '"': output << "\\\""; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default: output << static_cast<char>(character); break;
        }
    }
    return output.str();
}

std::string json_number(double value) {
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

bool reference_price_ready(const std::string& quality) {
    return !quality.empty() && quality != "UNKNOWN" &&
        quality != "UNAVAILABLE";
}

constexpr double kGarchVarianceFloor = 1e-12;
constexpr double kGarchDiagnosticCritical = 18.307038;

struct GarchFit {
    bool ok{false};
    double mean{0.0};
    double omega{0.0};
    double alpha{0.0};
    double beta{0.0};
    double forecast_variance{0.0};
    double nll{std::numeric_limits<double>::infinity()};
    std::vector<double> standardized_residuals;
};

double garch_negative_log_likelihood(
    std::span<const double> returns, double mean, double omega,
    double alpha, double beta, double sample_variance,
    std::vector<double>* standardized_residuals,
    double* forecast_variance) {
    if (!(omega > 0.0) || alpha < 0.0 || beta < 0.0 ||
        !(alpha + beta < 1.0) || !std::isfinite(mean) ||
        !std::isfinite(sample_variance) || !(sample_variance > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    const double unconditional_variance = omega / (1.0 - alpha - beta);
    if (!std::isfinite(unconditional_variance) ||
        !(unconditional_variance > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    double variance = std::max(
        kGarchVarianceFloor,
        std::max(sample_variance, unconditional_variance));
    double negative_log_likelihood = 0.0;
    if (standardized_residuals != nullptr) {
        standardized_residuals->clear();
        standardized_residuals->reserve(returns.size());
    }
    for (const double value : returns) {
        if (!std::isfinite(value)) {
            return std::numeric_limits<double>::infinity();
        }
        const double epsilon = value - mean;
        if (!std::isfinite(epsilon) || !(variance > 0.0) ||
            !std::isfinite(variance)) {
            return std::numeric_limits<double>::infinity();
        }
        negative_log_likelihood += 0.5 *
            (std::log(variance) + (epsilon * epsilon) / variance);
        const double standardized = epsilon / std::sqrt(variance);
        if (!std::isfinite(standardized) ||
            !std::isfinite(negative_log_likelihood)) {
            return std::numeric_limits<double>::infinity();
        }
        if (standardized_residuals != nullptr) {
            standardized_residuals->push_back(standardized);
        }
        variance = omega + alpha * epsilon * epsilon + beta * variance;
        if (!std::isfinite(variance) || !(variance > 0.0)) {
            return std::numeric_limits<double>::infinity();
        }
        variance = std::max(kGarchVarianceFloor, variance);
    }
    if (forecast_variance != nullptr) {
        *forecast_variance = variance;
    }
    return negative_log_likelihood;
}

GarchFit fit_garch11(std::span<const double> returns) {
    GarchFit best;
    if (returns.size() < 40) {
        return best;
    }
    double mean = 0.0;
    for (const double value : returns) {
        if (!std::isfinite(value)) return best;
        mean += value;
    }
    mean /= static_cast<double>(returns.size());
    double sample_variance = 0.0;
    for (const double value : returns) {
        const double centered = value - mean;
        sample_variance += centered * centered;
    }
    sample_variance /= static_cast<double>(returns.size());
    if (!std::isfinite(sample_variance) ||
        !(sample_variance > kGarchVarianceFloor)) {
        return best;
    }

    const std::array<double, 3> alpha_seeds{0.03, 0.10, 0.20};
    const std::array<double, 3> beta_seeds{0.70, 0.85, 0.95};
    for (const double alpha : alpha_seeds) {
        for (const double beta : beta_seeds) {
            if (alpha + beta >= 0.995) continue;
            const double omega = sample_variance *
                std::max(1e-3, 1.0 - alpha - beta);
            const double nll = garch_negative_log_likelihood(
                returns, mean, omega, alpha, beta, sample_variance, nullptr,
                nullptr);
            if (nll < best.nll) {
                best.ok = true;
                best.mean = mean;
                best.omega = omega;
                best.alpha = alpha;
                best.beta = beta;
                best.nll = nll;
            }
        }
    }
    if (!best.ok) return best;

    double alpha_step = 0.05;
    double beta_step = 0.05;
    for (int iteration = 0; iteration < 20; ++iteration) {
        bool improved = false;
        const std::array<double, 3> omega_candidates{
            std::max(kGarchVarianceFloor, best.omega * 0.5),
            best.omega,
            best.omega * 2.0};
        const std::array<double, 3> alpha_candidates{
            std::max(0.0, best.alpha - alpha_step), best.alpha,
            std::min(0.95, best.alpha + alpha_step)};
        const std::array<double, 3> beta_candidates{
            std::max(0.0, best.beta - beta_step), best.beta,
            std::min(0.995, best.beta + beta_step)};
        for (const double omega : omega_candidates) {
            for (const double alpha : alpha_candidates) {
                for (const double beta : beta_candidates) {
                    if (alpha + beta >= 0.999) continue;
                    const double nll = garch_negative_log_likelihood(
                        returns, mean, omega, alpha, beta, sample_variance,
                        nullptr, nullptr);
                    if (nll + 1e-10 < best.nll) {
                        best.omega = omega;
                        best.alpha = alpha;
                        best.beta = beta;
                        best.nll = nll;
                        improved = true;
                    }
                }
            }
        }
        alpha_step *= 0.5;
        beta_step *= 0.5;
        if (!improved && alpha_step < 1e-5 && beta_step < 1e-5) break;
    }
    best.standardized_residuals.clear();
    best.nll = garch_negative_log_likelihood(
        returns, mean, best.omega, best.alpha, best.beta, sample_variance,
        &best.standardized_residuals, &best.forecast_variance);
    best.ok = std::isfinite(best.nll) && std::isfinite(best.forecast_variance) &&
        best.standardized_residuals.size() == returns.size() &&
        best.omega > 0.0 && best.alpha >= 0.0 && best.beta >= 0.0 &&
        best.alpha + best.beta < 1.0;
    return best;
}

double ljung_box_statistic(std::span<const double> values, bool square_values) {
    if (values.size() < 4) return std::numeric_limits<double>::infinity();
    const std::size_t lag_count = std::min<std::size_t>(10, values.size() - 2);
    std::vector<double> transformed(values.size(), 0.0);
    double mean = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double value = square_values ? values[index] * values[index] :
            values[index];
        if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
        transformed[index] = value;
        mean += value;
    }
    mean /= static_cast<double>(values.size());
    double denominator = 0.0;
    for (const double value : transformed) {
        const double centered = value - mean;
        denominator += centered * centered;
    }
    if (!(denominator > 0.0) || !std::isfinite(denominator)) {
        return 0.0;
    }
    double statistic = 0.0;
    const double observation_count = static_cast<double>(values.size());
    for (std::size_t lag = 1; lag <= lag_count; ++lag) {
        double numerator = 0.0;
        for (std::size_t index = lag; index < transformed.size(); ++index) {
            numerator += (transformed[index] - mean) *
                (transformed[index - lag] - mean);
        }
        const double autocorrelation = numerator / denominator;
        statistic += (autocorrelation * autocorrelation) /
            (observation_count - static_cast<double>(lag));
    }
    return observation_count * (observation_count + 2.0) * statistic;
}

TailRiskEstimate estimate_fhs_scenarios(
    const TailRiskProblemView& problem, const GarchFit& fit,
    std::uint64_t input_hash) {
    TailRiskEstimate result;
    result.estimator = problem.spec.estimator;
    result.scenario_model = problem.spec.scenario_model;
    result.confidence_level = problem.spec.confidence_level;
    result.effective_observations = static_cast<std::uint32_t>(
        fit.standardized_residuals.size());
    const std::size_t scenario_count = fit.standardized_residuals.size();
    if (scenario_count == 0 || !(fit.forecast_variance > 0.0) ||
        !std::isfinite(fit.forecast_variance)) {
        result.status = TailRiskStatus::NUMERICAL_FAILURE;
        return result;
    }
    std::vector<std::pair<double, double>> losses;
    losses.reserve(scenario_count);
    double probability_sum = 0.0;
    for (std::size_t index = 0; index < scenario_count; ++index) {
        const double probability = problem.scenario_probabilities.empty()
            ? 1.0 / static_cast<double>(scenario_count)
            : problem.scenario_probabilities[index];
        const double scenario_return = fit.mean +
            std::sqrt(fit.forecast_variance) * fit.standardized_residuals[index];
        if (!std::isfinite(probability) || probability < 0.0 ||
            !std::isfinite(scenario_return)) {
            result.status = TailRiskStatus::INVALID_INPUT;
            return result;
        }
        probability_sum += probability;
        losses.emplace_back(-scenario_return, probability);
        hash_value(input_hash, std::bit_cast<std::uint64_t>(probability));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(scenario_return));
    }
    if (!(probability_sum > 0.0) ||
        (!problem.scenario_probabilities.empty() &&
            std::abs(probability_sum - 1.0) > 1e-12)) {
        result.status = TailRiskStatus::INVALID_INPUT;
        return result;
    }
    std::sort(losses.begin(), losses.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });
    double cumulative_probability = 0.0;
    double value_at_risk = losses.back().first;
    for (const auto& [loss, probability] : losses) {
        cumulative_probability += probability;
        if (cumulative_probability + 1e-15 >= problem.spec.confidence_level) {
            value_at_risk = loss;
            break;
        }
    }
    double excess_loss = 0.0;
    for (const auto& [loss, probability] : losses) {
        excess_loss += probability * std::max(loss - value_at_risk, 0.0);
    }
    const double expected_shortfall = value_at_risk + excess_loss /
        (1.0 - problem.spec.confidence_level);
    if (!std::isfinite(expected_shortfall) || expected_shortfall < value_at_risk) {
        result.status = TailRiskStatus::NUMERICAL_FAILURE;
        return result;
    }
    result.status = TailRiskStatus::OK;
    result.value_at_risk_loss = value_at_risk;
    result.expected_shortfall_loss = expected_shortfall;
    result.return_cvar = -expected_shortfall;
    result.input_hash = input_hash;
    result.artifact_hash = input_hash;
    hash_value(result.artifact_hash,
               static_cast<std::uint64_t>(result.estimator));
    hash_value(result.artifact_hash,
               static_cast<std::uint64_t>(result.scenario_model));
    hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(value_at_risk));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(expected_shortfall));
    return result;
}

bool attach_garch_diagnostics(const GarchFit& fit, TailRiskEstimate& result) {
    if (!fit.ok || fit.standardized_residuals.empty()) return false;
    result.garch_mean = fit.mean;
    result.garch_omega = fit.omega;
    result.garch_alpha = fit.alpha;
    result.garch_beta = fit.beta;
    result.garch_forecast_variance = fit.forecast_variance;
    result.garch_stationarity_margin = 1.0 - fit.alpha - fit.beta;
    double standardized_mean = 0.0;
    double standardized_second_moment = 0.0;
    double maximum_standardized_residual = 0.0;
    for (const double residual : fit.standardized_residuals) {
        standardized_mean += residual;
        standardized_second_moment += residual * residual;
        maximum_standardized_residual = std::max(
            maximum_standardized_residual, std::abs(residual));
    }
    standardized_mean /= static_cast<double>(fit.standardized_residuals.size());
    const double standardized_variance = standardized_second_moment /
        static_cast<double>(fit.standardized_residuals.size()) -
        standardized_mean * standardized_mean;
    const double residual_ljung_box = ljung_box_statistic(
        fit.standardized_residuals, false);
    const double squared_residual_ljung_box = ljung_box_statistic(
        fit.standardized_residuals, true);
    result.standardized_residual_mean = standardized_mean;
    result.standardized_residual_variance = standardized_variance;
    result.residual_ljung_box = residual_ljung_box;
    result.squared_residual_ljung_box = squared_residual_ljung_box;
    result.arch_lm_statistic = squared_residual_ljung_box;
    result.maximum_standardized_residual = maximum_standardized_residual;
    result.missing_fraction = 0.0;
    return std::isfinite(standardized_mean) &&
        std::isfinite(standardized_variance) &&
        std::isfinite(residual_ljung_box) &&
        std::isfinite(squared_residual_ljung_box) &&
        std::isfinite(maximum_standardized_residual) &&
        std::abs(standardized_mean) <= 0.25 && standardized_variance > 0.25 &&
        standardized_variance < 2.5 &&
        residual_ljung_box <= kGarchDiagnosticCritical &&
        squared_residual_ljung_box <= kGarchDiagnosticCritical;
}

std::vector<std::pair<double, double>> build_fhs_losses(
    const TailRiskProblemView& problem, const GarchFit& fit) {
    std::vector<std::pair<double, double>> losses;
    if (!fit.ok || fit.standardized_residuals.empty() ||
        !(fit.forecast_variance > 0.0) ||
        (!problem.scenario_probabilities.empty() &&
            problem.scenario_probabilities.size() != fit.standardized_residuals.size())) {
        return losses;
    }
    losses.reserve(fit.standardized_residuals.size());
    for (std::size_t index = 0; index < fit.standardized_residuals.size(); ++index) {
        const double probability = problem.scenario_probabilities.empty()
            ? 1.0 / static_cast<double>(fit.standardized_residuals.size())
            : problem.scenario_probabilities[index];
        const double scenario_return = fit.mean +
            std::sqrt(fit.forecast_variance) * fit.standardized_residuals[index];
        if (!std::isfinite(probability) || probability < 0.0 ||
            !std::isfinite(scenario_return)) {
            losses.clear();
            return losses;
        }
        losses.emplace_back(-scenario_return, probability);
    }
    return losses;
}

}  // namespace

bool valid_tail_risk_spec(const TailRiskSpec& spec) noexcept {
    if (!(spec.confidence_level >= 0.5 && spec.confidence_level < 1.0) ||
        spec.forecast_horizon_periods == 0 || spec.residual_block_length == 0 ||
        spec.config_hash == 0) {
        return false;
    }
    if (spec.estimator == TailRiskEstimatorKind::EMPIRICAL_ROCKAFELLAR_URYASEV) {
        return spec.scenario_model == TailScenarioModelKind::PORTFOLIO_RETURN_SERIES &&
            spec.expectile_level == 0.0 && spec.evt_minimum_exceedances == 0 &&
            spec.mean_model_spec_hash == 0 && spec.volatility_model_spec_hash == 0 &&
            spec.evt_threshold_spec_hash == 0 && spec.expectile_feature_spec_hash == 0;
    }
    if (spec.estimator == TailRiskEstimatorKind::EXPECTILE_DIRECT ||
        spec.estimator == TailRiskEstimatorKind::EXPECTILE_TAYLOR_MAPPED_ES) {
        return spec.expectile_level > 0.5 && spec.expectile_level < 1.0 &&
            spec.expectile_feature_spec_hash != 0 && spec.training_only_tail_calibration &&
            spec.forecast_horizon_periods == 1 && spec.residual_block_length == 1 &&
            spec.scenario_model == TailScenarioModelKind::PORTFOLIO_RETURN_SERIES &&
            spec.mean_model_spec_hash == 0 && spec.volatility_model_spec_hash == 0 &&
            spec.evt_minimum_exceedances == 0 &&
            spec.evt_threshold_quantile_min == 0.0 &&
            spec.evt_threshold_quantile_max == 0.0 &&
            spec.evt_shape_upper_guard == 0.0 &&
            spec.evt_threshold_spec_hash == 0;
    }
    if (spec.estimator == TailRiskEstimatorKind::GARCH_FILTERED_HISTORICAL_SIMULATION) {
        return spec.expectile_level == 0.0 && spec.volatility_model_spec_hash != 0 &&
            spec.mean_model_spec_hash != 0 && spec.forecast_horizon_periods == 1 &&
            spec.residual_block_length == 1 && spec.filtered_volatility_state_only &&
            spec.evt_minimum_exceedances == 0 &&
            spec.evt_threshold_quantile_min == 0.0 &&
            spec.evt_threshold_quantile_max == 0.0 &&
            spec.evt_shape_upper_guard == 0.0 &&
            spec.evt_threshold_spec_hash == 0 &&
            spec.expectile_feature_spec_hash == 0 &&
            (spec.scenario_model == TailScenarioModelKind::PORTFOLIO_RETURN_SERIES ||
             spec.synchronized_residual_rows);
    }
    return spec.expectile_level == 0.0 && spec.volatility_model_spec_hash != 0 &&
        spec.mean_model_spec_hash != 0 && spec.forecast_horizon_periods == 1 &&
        spec.residual_block_length == 1 && spec.filtered_volatility_state_only &&
        spec.expectile_feature_spec_hash == 0 &&
        spec.evt_minimum_exceedances > 0 && spec.evt_threshold_spec_hash != 0 &&
        spec.evt_shape_upper_guard > 0.0 && spec.evt_shape_upper_guard < 1.0 &&
        spec.evt_threshold_quantile_min > 0.5 &&
        spec.evt_threshold_quantile_max > spec.evt_threshold_quantile_min &&
        spec.evt_threshold_quantile_max < 1.0 &&
        (spec.scenario_model == TailScenarioModelKind::PORTFOLIO_RETURN_SERIES ||
         spec.synchronized_residual_rows);
}

TailRiskEstimate estimate_garch_fhs_tail_risk(
    const TailRiskProblemView& problem) {
    TailRiskEstimate result;
    result.estimator = problem.spec.estimator;
    result.scenario_model = problem.spec.scenario_model;
    result.confidence_level = problem.spec.confidence_level;
    const auto returns = problem.portfolio_return_history;
    const bool timestamps_ordered = std::is_sorted(
        problem.history_timestamps.begin(), problem.history_timestamps.end()) &&
        std::adjacent_find(problem.history_timestamps.begin(),
                           problem.history_timestamps.end()) ==
            problem.history_timestamps.end();
    const bool symbols_ordered = std::is_sorted(problem.symbols.begin(), problem.symbols.end()) &&
        std::adjacent_find(problem.symbols.begin(), problem.symbols.end()) ==
            problem.symbols.end();
    if (!valid_tail_risk_spec(problem.spec) ||
        problem.spec.estimator != TailRiskEstimatorKind::GARCH_FILTERED_HISTORICAL_SIMULATION ||
        problem.spec.scenario_model != TailScenarioModelKind::PORTFOLIO_RETURN_SERIES ||
        problem.decision_at <= 0 || returns.size() < 40 ||
        problem.history_timestamps.size() != returns.size() ||
        problem.symbols.empty() ||
        problem.fixed_portfolio_weights.size() != problem.symbols.size() ||
        !timestamps_ordered || !symbols_ordered ||
        problem.history_timestamps.back() > problem.decision_at ||
        problem.asset_return_history.rows != 0 ||
        problem.factor_return_history.rows != 0 ||
        problem.specific_return_history.rows != 0 ||
        problem.factor_risk_model != nullptr ||
        (!problem.scenario_probabilities.empty() &&
            problem.scenario_probabilities.size() != returns.size())) {
        return result;
    }
    double weight_sum = 0.0;
    std::uint64_t input_hash = kFnvOffset;
    hash_value(input_hash, problem.spec.config_hash);
    hash_value(input_hash, problem.spec.mean_model_spec_hash);
    hash_value(input_hash, problem.spec.volatility_model_spec_hash);
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.confidence_level));
    for (std::size_t index = 0; index < problem.fixed_portfolio_weights.size(); ++index) {
        const double weight = problem.fixed_portfolio_weights[index];
        if (!std::isfinite(weight) || weight < 0.0) return result;
        weight_sum += weight;
        hash_value(input_hash, problem.symbols[index]);
        hash_value(input_hash, std::bit_cast<std::uint64_t>(weight));
    }
    if (!(weight_sum > 0.0) || weight_sum > 1.0 + 1e-12) return result;
    for (std::size_t index = 0; index < returns.size(); ++index) {
        if (!std::isfinite(returns[index]) ||
            problem.history_timestamps[index] <= 0) {
            return result;
        }
        hash_value(input_hash, static_cast<std::uint64_t>(
            problem.history_timestamps[index]));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(returns[index]));
    }
    if (!problem.scenario_probabilities.empty()) {
        double probability_sum = 0.0;
        for (const double probability : problem.scenario_probabilities) {
            if (!std::isfinite(probability) || probability < 0.0) return result;
            probability_sum += probability;
        }
        if (std::abs(probability_sum - 1.0) > 1e-12) return result;
    }
    const GarchFit fit = fit_garch11(returns);
    if (!fit.ok) {
        result.status = TailRiskStatus::VOLATILITY_FIT_FAILURE;
        return result;
    }
    TailRiskEstimate diagnostic_result;
    if (!attach_garch_diagnostics(fit, diagnostic_result)) {
        result.status = TailRiskStatus::RESIDUAL_DIAGNOSTIC_FAILURE;
        result.garch_mean = diagnostic_result.garch_mean;
        result.garch_omega = diagnostic_result.garch_omega;
        result.garch_alpha = diagnostic_result.garch_alpha;
        result.garch_beta = diagnostic_result.garch_beta;
        result.garch_stationarity_margin = diagnostic_result.garch_stationarity_margin;
        result.standardized_residual_mean = diagnostic_result.standardized_residual_mean;
        result.standardized_residual_variance = diagnostic_result.standardized_residual_variance;
        result.residual_ljung_box = diagnostic_result.residual_ljung_box;
        result.squared_residual_ljung_box = diagnostic_result.squared_residual_ljung_box;
        return result;
    }
    hash_value(input_hash, std::bit_cast<std::uint64_t>(fit.mean));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(fit.omega));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(fit.alpha));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(fit.beta));
    result = estimate_fhs_scenarios(problem, fit, input_hash);
    attach_garch_diagnostics(fit, result);
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(*result.standardized_residual_mean));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(*result.standardized_residual_variance));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(*result.residual_ljung_box));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(*result.squared_residual_ljung_box));
    return result;
}

TailRiskEstimate estimate_garch_fhs_evt_tail_risk(
    const TailRiskProblemView& problem) {
    TailRiskEstimate result;
    result.estimator = problem.spec.estimator;
    result.scenario_model = problem.spec.scenario_model;
    result.confidence_level = problem.spec.confidence_level;
    const auto returns = problem.portfolio_return_history;
    const bool timestamps_ordered = std::is_sorted(
        problem.history_timestamps.begin(), problem.history_timestamps.end()) &&
        std::adjacent_find(problem.history_timestamps.begin(),
                           problem.history_timestamps.end()) ==
            problem.history_timestamps.end();
    const bool symbols_ordered = std::is_sorted(problem.symbols.begin(), problem.symbols.end()) &&
        std::adjacent_find(problem.symbols.begin(), problem.symbols.end()) ==
            problem.symbols.end();
    if (!valid_tail_risk_spec(problem.spec) ||
        problem.spec.estimator != TailRiskEstimatorKind::GARCH_FHS_POT_GPD ||
        problem.spec.scenario_model != TailScenarioModelKind::PORTFOLIO_RETURN_SERIES ||
        problem.decision_at <= 0 || returns.size() < 40 ||
        problem.history_timestamps.size() != returns.size() ||
        problem.symbols.empty() ||
        problem.fixed_portfolio_weights.size() != problem.symbols.size() ||
        !timestamps_ordered || !symbols_ordered ||
        problem.history_timestamps.back() > problem.decision_at ||
        problem.asset_return_history.rows != 0 ||
        problem.factor_return_history.rows != 0 ||
        problem.specific_return_history.rows != 0 ||
        problem.factor_risk_model != nullptr ||
        (!problem.scenario_probabilities.empty() &&
            problem.scenario_probabilities.size() != returns.size())) {
        return result;
    }
    double weight_sum = 0.0;
    std::uint64_t input_hash = kFnvOffset;
    hash_value(input_hash, problem.spec.config_hash);
    hash_value(input_hash, problem.spec.mean_model_spec_hash);
    hash_value(input_hash, problem.spec.volatility_model_spec_hash);
    hash_value(input_hash, problem.spec.evt_threshold_spec_hash);
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.confidence_level));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.evt_threshold_quantile_min));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.evt_threshold_quantile_max));
    for (std::size_t index = 0; index < problem.fixed_portfolio_weights.size(); ++index) {
        const double weight = problem.fixed_portfolio_weights[index];
        if (!std::isfinite(weight) || weight < 0.0) return result;
        weight_sum += weight;
        hash_value(input_hash, problem.symbols[index]);
        hash_value(input_hash, std::bit_cast<std::uint64_t>(weight));
    }
    if (!(weight_sum > 0.0) || weight_sum > 1.0 + 1e-12) return result;
    for (std::size_t index = 0; index < returns.size(); ++index) {
        if (!std::isfinite(returns[index]) ||
            problem.history_timestamps[index] <= 0) {
            return result;
        }
        hash_value(input_hash, static_cast<std::uint64_t>(
            problem.history_timestamps[index]));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(returns[index]));
    }
    if (!problem.scenario_probabilities.empty()) {
        double probability_sum = 0.0;
        for (const double probability : problem.scenario_probabilities) {
            if (!std::isfinite(probability) || probability < 0.0) return result;
            probability_sum += probability;
        }
        if (std::abs(probability_sum - 1.0) > 1e-12) return result;
    }
    const GarchFit fit = fit_garch11(returns);
    if (!fit.ok) {
        result.status = TailRiskStatus::VOLATILITY_FIT_FAILURE;
        return result;
    }
    if (!attach_garch_diagnostics(fit, result)) {
        result.status = TailRiskStatus::RESIDUAL_DIAGNOSTIC_FAILURE;
        return result;
    }
    auto losses = build_fhs_losses(problem, fit);
    if (losses.empty()) {
        result.status = TailRiskStatus::INVALID_INPUT;
        return result;
    }
    double probability_sum = 0.0;
    for (const auto& [loss, probability] : losses) {
        if (!std::isfinite(loss) || !std::isfinite(probability) || probability < 0.0) {
            result.status = TailRiskStatus::INVALID_INPUT;
            return result;
        }
        probability_sum += probability;
    }
    if (std::abs(probability_sum - 1.0) > 1e-12) {
        result.status = TailRiskStatus::INVALID_INPUT;
        return result;
    }
    std::sort(losses.begin(), losses.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });
    const double threshold_quantile = problem.spec.evt_threshold_quantile_max;
    double cumulative_probability = 0.0;
    double threshold = losses.back().first;
    for (const auto& [loss, probability] : losses) {
        cumulative_probability += probability;
        if (cumulative_probability + 1e-15 >= threshold_quantile) {
            threshold = loss;
            break;
        }
    }
    std::vector<double> excesses;
    double tail_probability = 0.0;
    for (const auto& [loss, probability] : losses) {
        if (loss > threshold) {
            excesses.push_back(loss - threshold);
            tail_probability += probability;
        }
    }
    result.evt_threshold = threshold;
    result.evt_exceedance_count = static_cast<std::uint32_t>(excesses.size());
    const double target_tail_probability = 1.0 - problem.spec.confidence_level;
    if (!std::isfinite(threshold) ||
        excesses.size() < problem.spec.evt_minimum_exceedances ||
        !(tail_probability > target_tail_probability) ||
        !(target_tail_probability > 0.0)) {
        result.status = TailRiskStatus::INSUFFICIENT_TAIL;
        return result;
    }
    double excess_mean = 0.0;
    double excess_second_moment = 0.0;
    for (const double excess : excesses) {
        if (!std::isfinite(excess) || !(excess > 0.0)) {
            result.status = TailRiskStatus::EVT_FIT_FAILURE;
            return result;
        }
        excess_mean += excess;
        excess_second_moment += excess * excess;
    }
    excess_mean /= static_cast<double>(excesses.size());
    excess_second_moment /= static_cast<double>(excesses.size());
    const double excess_variance = excess_second_moment - excess_mean * excess_mean;
    if (!(excess_mean > 0.0) || !std::isfinite(excess_mean) ||
        !std::isfinite(excess_variance)) {
        result.status = TailRiskStatus::EVT_FIT_FAILURE;
        return result;
    }
    double shape = 0.0;
    double scale = excess_mean;
    if (excess_variance > 1e-18) {
        const double mean_variance_ratio = excess_mean * excess_mean /
            excess_variance;
        shape = 0.5 * (1.0 - mean_variance_ratio);
        scale = 0.5 * excess_mean * (1.0 + mean_variance_ratio);
    }
    result.gpd_shape = shape;
    result.gpd_scale = scale;
    if (!std::isfinite(shape) || !std::isfinite(scale) || !(scale > 0.0) ||
        shape >= problem.spec.evt_shape_upper_guard) {
        result.status = shape >= problem.spec.evt_shape_upper_guard
            ? TailRiskStatus::EVT_INFINITE_MEAN
            : TailRiskStatus::EVT_FIT_FAILURE;
        return result;
    }
    const double ratio = tail_probability / target_tail_probability;
    double value_at_risk = 0.0;
    if (std::abs(shape) < 1e-8) {
        value_at_risk = threshold + scale * std::log(ratio);
    } else {
        const double support_term = std::pow(ratio, shape);
        value_at_risk = threshold + scale / shape * (support_term - 1.0);
    }
    if (!std::isfinite(value_at_risk) || value_at_risk < threshold) {
        result.status = TailRiskStatus::EVT_FIT_FAILURE;
        return result;
    }
    const double mean_excess_at_var = scale + shape * (value_at_risk - threshold);
    const double expected_shortfall = value_at_risk + mean_excess_at_var /
        (1.0 - shape);
    if (!std::isfinite(expected_shortfall) ||
        expected_shortfall < value_at_risk) {
        result.status = shape >= 1.0
            ? TailRiskStatus::EVT_INFINITE_MEAN
            : TailRiskStatus::EVT_FIT_FAILURE;
        return result;
    }
    result.status = TailRiskStatus::OK;
    result.confidence_level = problem.spec.confidence_level;
    result.effective_observations = static_cast<std::uint32_t>(returns.size());
    result.value_at_risk_loss = value_at_risk;
    result.expected_shortfall_loss = expected_shortfall;
    result.return_cvar = -expected_shortfall;
    result.input_hash = input_hash;
    hash_value(result.input_hash, std::bit_cast<std::uint64_t>(threshold));
    hash_value(result.input_hash, result.evt_exceedance_count);
    result.artifact_hash = result.input_hash;
    hash_value(result.artifact_hash,
               static_cast<std::uint64_t>(result.estimator));
    hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(shape));
    hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(scale));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(value_at_risk));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(expected_shortfall));
    return result;
}

TailRiskEstimate estimate_expectile_tail_risk(
    const TailRiskProblemView& problem) {
    TailRiskEstimate result;
    result.estimator = problem.spec.estimator;
    result.scenario_model = problem.spec.scenario_model;
    result.confidence_level = problem.spec.confidence_level;
    const auto returns = problem.portfolio_return_history;
    const bool timestamps_ordered = std::is_sorted(
        problem.history_timestamps.begin(), problem.history_timestamps.end()) &&
        std::adjacent_find(problem.history_timestamps.begin(),
                           problem.history_timestamps.end()) ==
            problem.history_timestamps.end();
    const bool symbols_ordered = std::is_sorted(problem.symbols.begin(), problem.symbols.end()) &&
        std::adjacent_find(problem.symbols.begin(), problem.symbols.end()) ==
            problem.symbols.end();
    if (!valid_tail_risk_spec(problem.spec) ||
        problem.spec.estimator != TailRiskEstimatorKind::EXPECTILE_DIRECT ||
        problem.spec.scenario_model != TailScenarioModelKind::PORTFOLIO_RETURN_SERIES ||
        problem.decision_at <= 0 || returns.empty() ||
        problem.history_timestamps.size() != returns.size() ||
        problem.symbols.empty() ||
        problem.fixed_portfolio_weights.size() != problem.symbols.size() ||
        !timestamps_ordered || !symbols_ordered ||
        problem.history_timestamps.back() > problem.decision_at ||
        problem.asset_return_history.rows != 0 ||
        problem.factor_return_history.rows != 0 ||
        problem.specific_return_history.rows != 0 ||
        problem.factor_risk_model != nullptr ||
        (!problem.scenario_probabilities.empty() &&
            problem.scenario_probabilities.size() != returns.size())) {
        return result;
    }
    double weight_sum = 0.0;
    std::uint64_t input_hash = kFnvOffset;
    hash_value(input_hash, problem.spec.config_hash);
    hash_value(input_hash, problem.spec.expectile_feature_spec_hash);
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.expectile_level));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.confidence_level));
    for (std::size_t index = 0; index < problem.fixed_portfolio_weights.size(); ++index) {
        const double weight = problem.fixed_portfolio_weights[index];
        if (!std::isfinite(weight) || weight < 0.0) return result;
        weight_sum += weight;
        hash_value(input_hash, problem.symbols[index]);
        hash_value(input_hash, std::bit_cast<std::uint64_t>(weight));
    }
    if (!(weight_sum > 0.0) || weight_sum > 1.0 + 1e-12) return result;
    std::vector<std::pair<double, double>> losses;
    losses.reserve(returns.size());
    double probability_sum = 0.0;
    for (std::size_t index = 0; index < returns.size(); ++index) {
        const double probability = problem.scenario_probabilities.empty()
            ? 1.0 / static_cast<double>(returns.size())
            : problem.scenario_probabilities[index];
        const double loss = -returns[index];
        if (problem.history_timestamps[index] <= 0 || !std::isfinite(loss) ||
            !std::isfinite(probability) || probability < 0.0) {
            return result;
        }
        probability_sum += probability;
        losses.emplace_back(loss, probability);
        hash_value(input_hash, static_cast<std::uint64_t>(
            problem.history_timestamps[index]));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(loss));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(probability));
    }
    if (std::abs(probability_sum - 1.0) > 1e-12) return result;
    auto minimum = std::min_element(losses.begin(), losses.end(),
        [](const auto& left, const auto& right) { return left.first < right.first; });
    auto maximum = std::max_element(losses.begin(), losses.end(),
        [](const auto& left, const auto& right) { return left.first < right.first; });
    if (minimum == losses.end() || maximum == losses.end() ||
        !std::isfinite(minimum->first) || !std::isfinite(maximum->first)) {
        result.status = TailRiskStatus::EXPECTILE_CALIBRATION_FAILURE;
        return result;
    }
    double lower = minimum->first;
    double upper = maximum->first;
    for (int iteration = 0; iteration < 120; ++iteration) {
        const double candidate = 0.5 * (lower + upper);
        double score = 0.0;
        for (const auto& [loss, probability] : losses) {
            const double residual = loss - candidate;
            const double asymmetric_weight = residual >= 0.0
                ? problem.spec.expectile_level
                : 1.0 - problem.spec.expectile_level;
            score += probability * asymmetric_weight * residual;
        }
        if (!std::isfinite(score)) {
            result.status = TailRiskStatus::EXPECTILE_CALIBRATION_FAILURE;
            return result;
        }
        if (score > 0.0) {
            lower = candidate;
        } else {
            upper = candidate;
        }
    }
    const double expectile = 0.5 * (lower + upper);
    if (!std::isfinite(expectile)) {
        result.status = TailRiskStatus::EXPECTILE_CALIBRATION_FAILURE;
        return result;
    }
    result.status = TailRiskStatus::OK;
    result.effective_observations = static_cast<std::uint32_t>(returns.size());
    result.expectile_loss = expectile;
    result.calibrated_expectile_level = problem.spec.expectile_level;
    result.input_hash = input_hash;
    result.artifact_hash = input_hash;
    hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(expectile));
    hash_value(result.artifact_hash,
               static_cast<std::uint64_t>(result.estimator));
    return result;
}

TailRiskEstimate estimate_tail_risk(const TailRiskProblemView& problem) {
    TailRiskEstimate result;
    result.estimator = problem.spec.estimator;
    result.scenario_model = problem.spec.scenario_model;
    result.confidence_level = problem.spec.confidence_level;
    if (problem.spec.estimator == TailRiskEstimatorKind::GARCH_FILTERED_HISTORICAL_SIMULATION) {
        return estimate_garch_fhs_tail_risk(problem);
    }
    if (problem.spec.estimator == TailRiskEstimatorKind::GARCH_FHS_POT_GPD) {
        return estimate_garch_fhs_evt_tail_risk(problem);
    }
    if (problem.spec.estimator == TailRiskEstimatorKind::EXPECTILE_DIRECT) {
        return estimate_expectile_tail_risk(problem);
    }
    const auto scenario_returns = problem.portfolio_return_history;
    const auto probabilities = problem.scenario_probabilities;
    const bool timestamps_ordered = std::is_sorted(
        problem.history_timestamps.begin(), problem.history_timestamps.end()) &&
        std::adjacent_find(problem.history_timestamps.begin(),
                           problem.history_timestamps.end()) ==
            problem.history_timestamps.end();
    const bool symbols_ordered = std::is_sorted(problem.symbols.begin(), problem.symbols.end()) &&
        std::adjacent_find(problem.symbols.begin(), problem.symbols.end()) ==
            problem.symbols.end();
    if (!valid_tail_risk_spec(problem.spec) ||
        problem.spec.estimator != TailRiskEstimatorKind::EMPIRICAL_ROCKAFELLAR_URYASEV ||
        problem.decision_at <= 0 || scenario_returns.empty() ||
        problem.history_timestamps.size() != scenario_returns.size() ||
        problem.symbols.empty() || problem.fixed_portfolio_weights.size() != problem.symbols.size() ||
        !timestamps_ordered || !symbols_ordered ||
        problem.history_timestamps.back() > problem.decision_at ||
        problem.asset_return_history.rows != 0 || problem.factor_return_history.rows != 0 ||
        problem.specific_return_history.rows != 0 || problem.factor_risk_model != nullptr ||
        (!probabilities.empty() && probabilities.size() != scenario_returns.size())) {
        return result;
    }
    std::vector<std::pair<double, double>> losses;
    losses.reserve(scenario_returns.size());
    double probability_sum = 0.0;
    std::uint64_t input_hash = kFnvOffset;
    hash_value(input_hash, problem.spec.config_hash);
    hash_value(input_hash, std::bit_cast<std::uint64_t>(problem.spec.confidence_level));
    double fixed_weight_sum = 0.0;
    for (std::size_t index = 0; index < problem.fixed_portfolio_weights.size(); ++index) {
        const double weight = problem.fixed_portfolio_weights[index];
        if (!std::isfinite(weight) || weight < 0.0) return result;
        fixed_weight_sum += weight;
        hash_value(input_hash, problem.symbols[index]);
        hash_value(input_hash, std::bit_cast<std::uint64_t>(weight));
    }
    if (!(fixed_weight_sum > 0.0) || fixed_weight_sum > 1.0 + 1e-12) return result;
    for (std::size_t index = 0; index < scenario_returns.size(); ++index) {
        const double probability = probabilities.empty()
            ? 1.0 / static_cast<double>(scenario_returns.size())
            : probabilities[index];
        if (!std::isfinite(scenario_returns[index]) || !std::isfinite(probability) ||
            probability < 0.0) {
            return result;
        }
        probability_sum += probability;
        losses.emplace_back(-scenario_returns[index], probability);
        hash_value(input_hash,
                   static_cast<std::uint64_t>(problem.history_timestamps[index]));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(scenario_returns[index]));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(probability));
    }
    if (!(probability_sum > 0.0) ||
        (!probabilities.empty() && std::abs(probability_sum - 1.0) > 1e-12)) {
        return result;
    }
    std::sort(losses.begin(), losses.end(), [](const auto& left, const auto& right) {
        return left.first < right.first;
    });

    double cumulative_probability = 0.0;
    double value_at_risk = losses.back().first;
    for (const auto& [loss, probability] : losses) {
        cumulative_probability += probability;
        if (cumulative_probability + 1e-15 >= problem.spec.confidence_level) {
            value_at_risk = loss;
            break;
        }
    }
    const double tail_mass = 1.0 - problem.spec.confidence_level;
    double excess_loss = 0.0;
    for (const auto& [loss, probability] : losses) {
        excess_loss += probability * std::max(loss - value_at_risk, 0.0);
    }
    const double expected_shortfall = value_at_risk + excess_loss / tail_mass;
    if (!std::isfinite(expected_shortfall)) {
        result.status = TailRiskStatus::NUMERICAL_FAILURE;
        return result;
    }
    result.status = TailRiskStatus::OK;
    result.value_at_risk_loss = value_at_risk;
    result.expected_shortfall_loss = expected_shortfall;
    result.return_cvar = -expected_shortfall;
    result.effective_observations = static_cast<std::uint32_t>(scenario_returns.size());
    result.input_hash = input_hash;
    result.artifact_hash = input_hash;
    hash_value(result.artifact_hash,
               static_cast<std::uint64_t>(TailRiskEstimatorKind::EMPIRICAL_ROCKAFELLAR_URYASEV));
    hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(value_at_risk));
    hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(expected_shortfall));
    return result;
}

std::string serialize_tail_risk_artifact(
    const TailRiskEstimate& estimate, const TailRiskSpec& spec,
    const TailRiskArtifactSpec& artifact_spec) {
    const bool proxy = artifact_spec.reference_price_quality == "PROXY" ||
        artifact_spec.reference_price_quality == "ARRIVAL_PROXY";
    const bool promotion = artifact_spec.promotion_eligible &&
        reference_price_ready(artifact_spec.reference_price_quality) && !proxy &&
        estimate.status == TailRiskStatus::OK;
    const auto optional_number = [](const std::optional<double>& value) {
        return value ? json_number(*value) : std::string("null");
    };
    std::ostringstream output;
    output << "{\"schema_version\":1,\"role\":\"fixed_portfolio_tail_risk\""
           << ",\"estimator\":" << static_cast<int>(estimate.estimator)
           << ",\"scenario_model\":" << static_cast<int>(estimate.scenario_model)
           << ",\"confidence_level\":" << json_number(spec.confidence_level)
           << ",\"effective_observations\":" << estimate.effective_observations
           << ",\"status\":" << static_cast<int>(estimate.status)
           << ",\"input_hash\":" << estimate.input_hash
           << ",\"artifact_hash\":" << estimate.artifact_hash
           << ",\"missing_fraction\":" << json_number(estimate.missing_fraction)
           << ",\"garch\":{\"mean\":"
           << optional_number(estimate.garch_mean)
           << ",\"omega\":" << optional_number(estimate.garch_omega)
           << ",\"alpha\":" << optional_number(estimate.garch_alpha)
           << ",\"beta\":" << optional_number(estimate.garch_beta)
           << ",\"forecast_variance\":"
           << optional_number(estimate.garch_forecast_variance)
           << ",\"stationarity_margin\":"
           << optional_number(estimate.garch_stationarity_margin)
           << ",\"standardized_residual_mean\":"
           << optional_number(estimate.standardized_residual_mean)
           << ",\"standardized_residual_variance\":"
           << optional_number(estimate.standardized_residual_variance)
           << ",\"residual_ljung_box\":"
           << optional_number(estimate.residual_ljung_box)
           << ",\"squared_residual_ljung_box\":"
           << optional_number(estimate.squared_residual_ljung_box)
           << ",\"arch_lm_statistic\":"
           << optional_number(estimate.arch_lm_statistic)
           << ",\"maximum_standardized_residual\":"
           << optional_number(estimate.maximum_standardized_residual)
           << "}"
           << ",\"evt\":{\"exceedance_count\":"
           << estimate.evt_exceedance_count
           << ",\"threshold\":" << optional_number(estimate.evt_threshold)
           << ",\"shape\":" << optional_number(estimate.gpd_shape)
           << ",\"scale\":" << optional_number(estimate.gpd_scale)
           << "}"
           << ",\"expectile_loss\":"
           << optional_number(estimate.expectile_loss)
           << ",\"calibrated_expectile_level\":"
           << optional_number(estimate.calibrated_expectile_level)
           << ",\"var_loss\":";
    if (estimate.value_at_risk_loss) {
        output << json_number(*estimate.value_at_risk_loss);
    } else {
        output << "null";
    }
    output << ",\"expected_shortfall_loss\":";
    if (estimate.expected_shortfall_loss) {
        output << json_number(*estimate.expected_shortfall_loss);
    } else {
        output << "null";
    }
    output << ",\"return_cvar\":";
    if (estimate.return_cvar) {
        output << json_number(*estimate.return_cvar);
    } else {
        output << "null";
    }
    output << ",\"manifest\":{\"source_dataset_fingerprint\":\""
           << json_escape(artifact_spec.source_dataset_fingerprint)
           << "\",\"portfolio_weights_sha256\":\""
           << json_escape(artifact_spec.portfolio_weights_sha256)
           << "\",\"return_panel_policy_hash\":\""
           << json_escape(artifact_spec.return_panel_policy_hash)
           << "\",\"reference_price_quality\":\""
           << json_escape(artifact_spec.reference_price_quality)
           << "\",\"promotion_eligible\":"
           << (promotion ? "true" : "false") << ",\"limitations\":[";
    for (std::size_t index = 0; index < artifact_spec.limitations.size(); ++index) {
        if (index != 0) output << ',';
        output << '"' << json_escape(artifact_spec.limitations[index]) << '"';
    }
    output << "]}}";
    return output.str();
}

namespace {

double bernoulli_log_likelihood(std::uint32_t successes,
                                std::uint32_t trials, double probability) {
    if (trials == 0) return 0.0;
    if (!(probability >= 0.0 && probability <= 1.0)) {
        return -std::numeric_limits<double>::infinity();
    }
    const std::uint32_t failures = trials - successes;
    if ((probability == 0.0 && successes != 0) ||
        (probability == 1.0 && failures != 0)) {
        return -std::numeric_limits<double>::infinity();
    }
    double result = 0.0;
    if (successes != 0) result += static_cast<double>(successes) * std::log(probability);
    if (failures != 0) result += static_cast<double>(failures) * std::log1p(-probability);
    return result;
}

double chi_square_one_p_value(double statistic) {
    if (!std::isfinite(statistic) || statistic < 0.0) return 0.0;
    return std::erfc(std::sqrt(statistic * 0.5));
}

}  // namespace

TailRiskBacktestResult backtest_tail_risk(
    const TailRiskBacktestProblemView& problem) {
    TailRiskBacktestResult result;
    result.confidence_level = problem.confidence_level;
    const std::size_t observation_count = problem.realized_returns.size();
    if (!(problem.confidence_level >= 0.5 && problem.confidence_level < 1.0) ||
        problem.available_at <= 0 || problem.config_hash == 0 ||
        observation_count < 3 ||
        problem.realization_timestamps.size() != observation_count ||
        problem.value_at_risk_loss.size() != observation_count ||
        problem.expected_shortfall_loss.size() != observation_count) {
        result.status = observation_count < 3
            ? TailRiskBacktestStatus::INSUFFICIENT_OBSERVATIONS
            : TailRiskBacktestStatus::INVALID_INPUT;
        return result;
    }
    std::uint64_t input_hash = kFnvOffset;
    hash_value(input_hash, problem.config_hash);
    hash_value(input_hash, std::bit_cast<std::uint64_t>(problem.confidence_level));
    for (std::size_t index = 0; index < observation_count; ++index) {
        if (problem.realization_timestamps[index] <= 0 ||
            (index != 0 && problem.realization_timestamps[index] <=
                problem.realization_timestamps[index - 1]) ||
            problem.realization_timestamps[index] > problem.available_at ||
            !std::isfinite(problem.realized_returns[index]) ||
            !std::isfinite(problem.value_at_risk_loss[index]) ||
            !std::isfinite(problem.expected_shortfall_loss[index]) ||
            problem.expected_shortfall_loss[index] <
                problem.value_at_risk_loss[index]) {
            result.status = TailRiskBacktestStatus::INVALID_INPUT;
            return result;
        }
        hash_value(input_hash, static_cast<std::uint64_t>(
            problem.realization_timestamps[index]));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(
            problem.realized_returns[index]));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(
            problem.value_at_risk_loss[index]));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(
            problem.expected_shortfall_loss[index]));
    }

    constexpr double kTolerance = 1e-12;
    double exceedance_sum = 0.0;
    double es_excess_sum = 0.0;
    std::vector<bool> exceptions(observation_count, false);
    for (std::size_t index = 0; index < observation_count; ++index) {
        const double loss = -problem.realized_returns[index];
        if (loss > problem.value_at_risk_loss[index] + kTolerance) {
            exceptions[index] = true;
            ++result.exception_count;
            exceedance_sum += loss - problem.value_at_risk_loss[index];
        }
        if (loss > problem.expected_shortfall_loss[index] + kTolerance) {
            ++result.es_violation_count;
            es_excess_sum += loss - problem.expected_shortfall_loss[index];
        }
    }
    for (std::size_t index = 1; index < exceptions.size(); ++index) {
        if (!exceptions[index - 1] && !exceptions[index]) ++result.transition_00;
        if (!exceptions[index - 1] && exceptions[index]) ++result.transition_01;
        if (exceptions[index - 1] && !exceptions[index]) ++result.transition_10;
        if (exceptions[index - 1] && exceptions[index]) ++result.transition_11;
    }
    result.effective_observations = static_cast<std::uint32_t>(observation_count);
    result.exception_rate = static_cast<double>(result.exception_count) /
        static_cast<double>(observation_count);
    result.es_violation_rate = static_cast<double>(result.es_violation_count) /
        static_cast<double>(observation_count);
    result.mean_exceedance_loss = result.exception_count == 0
        ? 0.0 : exceedance_sum / static_cast<double>(result.exception_count);
    result.mean_es_excess_loss = result.es_violation_count == 0
        ? 0.0 : es_excess_sum / static_cast<double>(result.es_violation_count);

    const double expected_exception_probability = 1.0 - problem.confidence_level;
    const double null_log_likelihood = bernoulli_log_likelihood(
        result.exception_count, result.effective_observations,
        expected_exception_probability);
    const double observed_probability = result.exception_rate;
    const double unrestricted_log_likelihood = bernoulli_log_likelihood(
        result.exception_count, result.effective_observations,
        observed_probability);
    const double kupiec_statistic = 2.0 * std::max(
        0.0, unrestricted_log_likelihood - null_log_likelihood);
    result.kupiec_lr = std::isfinite(kupiec_statistic)
        ? kupiec_statistic : std::numeric_limits<double>::infinity();
    result.kupiec_p_value = chi_square_one_p_value(result.kupiec_lr);

    const std::uint32_t transition_count = result.transition_00 +
        result.transition_01 + result.transition_10 + result.transition_11;
    const std::uint32_t transition_ones = result.transition_01 + result.transition_11;
    const double iid_probability = transition_count == 0
        ? 0.0 : static_cast<double>(transition_ones) /
            static_cast<double>(transition_count);
    const std::uint32_t from_zero = result.transition_00 + result.transition_01;
    const std::uint32_t from_one = result.transition_10 + result.transition_11;
    const double probability_01 = from_zero == 0
        ? 0.0 : static_cast<double>(result.transition_01) /
            static_cast<double>(from_zero);
    const double probability_11 = from_one == 0
        ? 0.0 : static_cast<double>(result.transition_11) /
            static_cast<double>(from_one);
    const double iid_log_likelihood = bernoulli_log_likelihood(
        transition_ones, transition_count, iid_probability);
    const double independent_log_likelihood =
        bernoulli_log_likelihood(result.transition_01, from_zero, probability_01) +
        bernoulli_log_likelihood(result.transition_11, from_one, probability_11);
    const double christoffersen_statistic = 2.0 * std::max(
        0.0, independent_log_likelihood - iid_log_likelihood);
    result.christoffersen_lr = std::isfinite(christoffersen_statistic)
        ? christoffersen_statistic : std::numeric_limits<double>::infinity();
    result.christoffersen_p_value = chi_square_one_p_value(result.christoffersen_lr);
    result.status = TailRiskBacktestStatus::OK;
    result.input_hash = input_hash;
    result.artifact_hash = input_hash;
    hash_value(result.artifact_hash, result.exception_count);
    hash_value(result.artifact_hash, result.es_violation_count);
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(result.kupiec_lr));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(result.christoffersen_lr));
    return result;
}

std::string serialize_tail_risk_backtest_artifact(
    const TailRiskBacktestResult& result,
    const TailRiskArtifactSpec& artifact_spec) {
    const bool proxy = artifact_spec.reference_price_quality == "PROXY" ||
        artifact_spec.reference_price_quality == "ARRIVAL_PROXY";
    const bool promotion = artifact_spec.promotion_eligible &&
        reference_price_ready(artifact_spec.reference_price_quality) && !proxy &&
        result.status == TailRiskBacktestStatus::OK;
    std::ostringstream output;
    output << "{\"schema_version\":1,\"role\":\"tail_risk_backtest\""
           << ",\"status\":" << static_cast<int>(result.status)
           << ",\"confidence_level\":" << json_number(result.confidence_level)
           << ",\"effective_observations\":" << result.effective_observations
           << ",\"exception_count\":" << result.exception_count
           << ",\"es_violation_count\":" << result.es_violation_count
           << ",\"transition_00\":" << result.transition_00
           << ",\"transition_01\":" << result.transition_01
           << ",\"transition_10\":" << result.transition_10
           << ",\"transition_11\":" << result.transition_11
           << ",\"exception_rate\":" << json_number(result.exception_rate)
           << ",\"es_violation_rate\":" << json_number(result.es_violation_rate)
           << ",\"mean_exceedance_loss\":"
           << json_number(result.mean_exceedance_loss)
           << ",\"mean_es_excess_loss\":"
           << json_number(result.mean_es_excess_loss)
           << ",\"kupiec_lr\":" << json_number(result.kupiec_lr)
           << ",\"kupiec_p_value\":" << json_number(result.kupiec_p_value)
           << ",\"christoffersen_lr\":"
           << json_number(result.christoffersen_lr)
           << ",\"christoffersen_p_value\":"
           << json_number(result.christoffersen_p_value)
           << ",\"input_hash\":" << result.input_hash
           << ",\"artifact_hash\":" << result.artifact_hash
           << ",\"manifest\":{\"source_dataset_fingerprint\":\""
           << json_escape(artifact_spec.source_dataset_fingerprint)
           << "\",\"portfolio_weights_sha256\":\""
           << json_escape(artifact_spec.portfolio_weights_sha256)
           << "\",\"return_panel_policy_hash\":\""
           << json_escape(artifact_spec.return_panel_policy_hash)
           << "\",\"reference_price_quality\":\""
           << json_escape(artifact_spec.reference_price_quality)
           << "\",\"promotion_eligible\":"
           << (promotion ? "true" : "false") << ",\"limitations\":[";
    for (std::size_t index = 0; index < artifact_spec.limitations.size(); ++index) {
        if (index != 0) output << ',';
        output << '"' << json_escape(artifact_spec.limitations[index]) << '"';
    }
    output << "]}}";
    return output.str();
}

}  // namespace portfolio_math

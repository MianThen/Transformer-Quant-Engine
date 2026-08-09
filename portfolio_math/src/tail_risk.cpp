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

struct PotGpdCandidate {
    TailRiskStatus status{TailRiskStatus::INVALID_INPUT};
    EvtThresholdDiagnostic diagnostic;
};

PotGpdCandidate fit_pot_gpd_candidate(
    const std::vector<std::pair<double, double>>& losses,
    double threshold_quantile,
    const TailRiskSpec& spec) {
    PotGpdCandidate candidate;
    candidate.diagnostic.threshold_quantile = threshold_quantile;
    const auto fail = [&candidate](TailRiskStatus status) {
        candidate.status = status;
        candidate.diagnostic.status = status;
        return candidate;
    };
    double cumulative_probability = 0.0;
    double threshold = losses.back().first;
    for (const auto& [loss, probability] : losses) {
        cumulative_probability += probability;
        if (cumulative_probability + 1e-15 >= threshold_quantile) {
            threshold = loss;
            break;
        }
    }
    candidate.diagnostic.threshold_loss = threshold;
    double tail_probability = 0.0;
    double squared_tail_probability = 0.0;
    double weighted_excess_sum = 0.0;
    double weighted_excess_second_moment = 0.0;
    std::uint32_t exceedance_count = 0;
    for (const auto& [loss, probability] : losses) {
        if (loss <= threshold || probability == 0.0) continue;
        const double excess = loss - threshold;
        if (!std::isfinite(excess) || !(excess > 0.0)) {
            return fail(TailRiskStatus::EVT_FIT_FAILURE);
        }
        tail_probability += probability;
        squared_tail_probability += probability * probability;
        weighted_excess_sum += probability * excess;
        weighted_excess_second_moment += probability * excess * excess;
        ++exceedance_count;
    }
    candidate.diagnostic.exceedance_count = exceedance_count;
    candidate.diagnostic.tail_probability = tail_probability;
    const double effective_exceedances = squared_tail_probability > 0.0
        ? tail_probability * tail_probability / squared_tail_probability
        : 0.0;
    candidate.diagnostic.effective_exceedances = effective_exceedances;
    const double target_tail_probability = 1.0 - spec.confidence_level;
    if (!std::isfinite(threshold) ||
        exceedance_count < spec.evt_minimum_exceedances ||
        effective_exceedances + 1e-12 <
            static_cast<double>(spec.evt_minimum_exceedances) ||
        !(tail_probability > target_tail_probability) ||
        !(target_tail_probability > 0.0)) {
        return fail(TailRiskStatus::INSUFFICIENT_TAIL);
    }
    const double excess_mean = weighted_excess_sum / tail_probability;
    const double excess_second_moment =
        weighted_excess_second_moment / tail_probability;
    const double excess_variance =
        excess_second_moment - excess_mean * excess_mean;
    if (!(excess_mean > 0.0) || !std::isfinite(excess_mean) ||
        !std::isfinite(excess_variance) || !(excess_variance > 1e-18)) {
        return fail(TailRiskStatus::EVT_FIT_FAILURE);
    }
    const double mean_variance_ratio =
        excess_mean * excess_mean / excess_variance;
    const double shape = 0.5 * (1.0 - mean_variance_ratio);
    const double scale =
        0.5 * excess_mean * (1.0 + mean_variance_ratio);
    candidate.diagnostic.gpd_shape = shape;
    candidate.diagnostic.gpd_scale = scale;
    if (!std::isfinite(shape) || !std::isfinite(scale) || !(scale > 0.0)) {
        return fail(TailRiskStatus::EVT_FIT_FAILURE);
    }
    for (const auto& [loss, probability] : losses) {
        if (loss <= threshold || probability == 0.0) continue;
        const double support = 1.0 + shape * (loss - threshold) / scale;
        if (!std::isfinite(support) || !(support > 0.0)) {
            return fail(TailRiskStatus::EVT_FIT_FAILURE);
        }
    }
    const GpdTailEvaluation evaluation = evaluate_gpd_tail(
        threshold, tail_probability, shape, scale, spec.confidence_level,
        spec.evt_shape_upper_guard);
    if (evaluation.status != TailRiskStatus::OK) {
        return fail(evaluation.status);
    }
    candidate.diagnostic.value_at_risk_loss = evaluation.value_at_risk_loss;
    candidate.diagnostic.expected_shortfall_loss =
        evaluation.expected_shortfall_loss;
    const double bulk_probability = 1.0 - tail_probability;
    candidate.diagnostic.splice_continuity_error =
        std::abs(tail_probability - tail_probability * 1.0);
    candidate.diagnostic.splice_probability_error =
        std::abs(bulk_probability + tail_probability - 1.0);
    candidate.status = TailRiskStatus::OK;
    candidate.diagnostic.status = TailRiskStatus::OK;
    return candidate;
}

TailRiskEstimate apply_pot_gpd_splice(
    const TailRiskProblemView& problem,
    std::vector<std::pair<double, double>> losses,
    std::uint64_t input_hash,
    TailRiskEstimate result) {
    hash_value(input_hash, problem.spec.evt_minimum_exceedances);
    hash_value(input_hash, problem.spec.evt_threshold_grid_points);
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.evt_threshold_quantile_min));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.evt_threshold_quantile_max));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.evt_shape_upper_guard));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.evt_max_shape_spread));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.evt_max_relative_es_spread));
    hash_value(input_hash,
               problem.spec.training_only_tail_calibration ? 1U : 0U);
    double probability_sum = 0.0;
    for (const auto& [loss, probability] : losses) {
        if (!std::isfinite(loss) || !std::isfinite(probability) ||
            probability < 0.0) {
            result.status = TailRiskStatus::INVALID_INPUT;
            return result;
        }
        probability_sum += probability;
        hash_value(input_hash, std::bit_cast<std::uint64_t>(loss));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(probability));
    }
    if (losses.empty() || std::abs(probability_sum - 1.0) > 1e-12) {
        result.status = TailRiskStatus::INVALID_INPUT;
        return result;
    }
    std::sort(losses.begin(), losses.end(),
        [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
    double minimum_shape = std::numeric_limits<double>::infinity();
    double maximum_shape = -std::numeric_limits<double>::infinity();
    double minimum_es = std::numeric_limits<double>::infinity();
    double maximum_es = -std::numeric_limits<double>::infinity();
    double es_sum = 0.0;
    std::size_t selected_diagnostic_index = 0;
    std::uint32_t valid_candidate_count = 0;
    TailRiskStatus last_failure = TailRiskStatus::EVT_FIT_FAILURE;
    bool infinite_mean_candidate = false;
    result.evt_threshold_diagnostics.reserve(
        problem.spec.evt_threshold_grid_points);
    for (std::uint32_t grid_index = 0;
         grid_index < problem.spec.evt_threshold_grid_points; ++grid_index) {
        const double fraction = static_cast<double>(grid_index) /
            static_cast<double>(problem.spec.evt_threshold_grid_points - 1);
        const double threshold_quantile =
            problem.spec.evt_threshold_quantile_min + fraction *
                (problem.spec.evt_threshold_quantile_max -
                 problem.spec.evt_threshold_quantile_min);
        const PotGpdCandidate candidate = fit_pot_gpd_candidate(
            losses, threshold_quantile, problem.spec);
        result.evt_threshold_diagnostics.push_back(candidate.diagnostic);
        if (candidate.status != TailRiskStatus::OK) {
            last_failure = candidate.status;
            infinite_mean_candidate = infinite_mean_candidate ||
                candidate.status == TailRiskStatus::EVT_INFINITE_MEAN;
            continue;
        }
        if (valid_candidate_count == 0) {
            selected_diagnostic_index =
                result.evt_threshold_diagnostics.size() - 1;
        }
        ++valid_candidate_count;
        minimum_shape = std::min(minimum_shape, candidate.diagnostic.gpd_shape);
        maximum_shape = std::max(maximum_shape, candidate.diagnostic.gpd_shape);
        minimum_es = std::min(
            minimum_es, candidate.diagnostic.expected_shortfall_loss);
        maximum_es = std::max(
            maximum_es, candidate.diagnostic.expected_shortfall_loss);
        es_sum += candidate.diagnostic.expected_shortfall_loss;
    }
    if (infinite_mean_candidate || valid_candidate_count < 3) {
        result.status = infinite_mean_candidate
            ? TailRiskStatus::EVT_INFINITE_MEAN
            : last_failure;
        return result;
    }
    const double shape_spread = maximum_shape - minimum_shape;
    const double mean_es = es_sum /
        static_cast<double>(valid_candidate_count);
    const double relative_es_spread = (maximum_es - minimum_es) /
        std::max(std::abs(mean_es), 1e-12);
    result.evt_shape_spread = shape_spread;
    result.evt_relative_es_spread = relative_es_spread;
    if (!std::isfinite(shape_spread) || !std::isfinite(relative_es_spread) ||
        shape_spread > problem.spec.evt_max_shape_spread ||
        relative_es_spread > problem.spec.evt_max_relative_es_spread) {
        result.status = TailRiskStatus::EVT_FIT_FAILURE;
        return result;
    }
    const EvtThresholdDiagnostic& selected =
        result.evt_threshold_diagnostics[selected_diagnostic_index];
    result.evt_selected_threshold_quantile = selected.threshold_quantile;
    result.evt_threshold = selected.threshold_loss;
    result.evt_exceedance_count = selected.exceedance_count;
    result.evt_effective_exceedances = selected.effective_exceedances;
    result.gpd_shape = selected.gpd_shape;
    result.gpd_scale = selected.gpd_scale;
    const double value_at_risk = selected.value_at_risk_loss;
    const double expected_shortfall = selected.expected_shortfall_loss;
    result.status = TailRiskStatus::OK;
    result.confidence_level = problem.spec.confidence_level;
    result.effective_observations =
        static_cast<std::uint32_t>(losses.size());
    result.value_at_risk_loss = value_at_risk;
    result.expected_shortfall_loss = expected_shortfall;
    result.return_cvar = -expected_shortfall;
    result.input_hash = input_hash;
    hash_value(result.input_hash, std::bit_cast<std::uint64_t>(
        selected.threshold_loss));
    hash_value(result.input_hash, result.evt_exceedance_count);
    result.artifact_hash = result.input_hash;
    hash_value(result.artifact_hash,
               static_cast<std::uint64_t>(result.estimator));
    hash_value(result.artifact_hash,
               static_cast<std::uint64_t>(result.scenario_model));
    hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(
        selected.gpd_shape));
    hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(
        selected.gpd_scale));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(value_at_risk));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(expected_shortfall));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(shape_spread));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(relative_es_spread));
    for (const auto& diagnostic : result.evt_threshold_diagnostics) {
        hash_value(result.artifact_hash,
                   static_cast<std::uint64_t>(diagnostic.status));
        hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(
            diagnostic.threshold_quantile));
        hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(
            diagnostic.threshold_loss));
        hash_value(result.artifact_hash, diagnostic.exceedance_count);
        hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(
            diagnostic.gpd_shape));
        hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(
            diagnostic.gpd_scale));
    }
    return result;
}

TailRiskEstimate estimate_asset_vector_garch_tail_risk(
    const TailRiskProblemView& problem) {
    TailRiskEstimate result;
    result.estimator = problem.spec.estimator;
    result.scenario_model = problem.spec.scenario_model;
    result.confidence_level = problem.spec.confidence_level;
    const auto returns = problem.asset_return_history;
    const bool timestamps_ordered = std::is_sorted(
        problem.history_timestamps.begin(), problem.history_timestamps.end()) &&
        std::adjacent_find(problem.history_timestamps.begin(),
                           problem.history_timestamps.end()) ==
            problem.history_timestamps.end();
    const bool symbols_ordered = std::is_sorted(
        problem.symbols.begin(), problem.symbols.end()) &&
        std::adjacent_find(problem.symbols.begin(), problem.symbols.end()) ==
            problem.symbols.end();
    const bool fhs_estimator = problem.spec.estimator ==
        TailRiskEstimatorKind::GARCH_FILTERED_HISTORICAL_SIMULATION;
    const bool evt_estimator = problem.spec.estimator ==
        TailRiskEstimatorKind::GARCH_FHS_POT_GPD;
    if (!valid_tail_risk_spec(problem.spec) ||
        (!fhs_estimator && !evt_estimator) ||
        problem.spec.scenario_model !=
            TailScenarioModelKind::ASSET_VECTOR_SYNCHRONIZED ||
        !problem.spec.synchronized_residual_rows || problem.decision_at <= 0 ||
        returns.data == nullptr || returns.rows < 40 || returns.cols == 0 ||
        returns.cols > 200 || returns.row_stride < returns.cols ||
        returns.rows != problem.history_timestamps.size() ||
        returns.cols != problem.symbols.size() ||
        problem.fixed_portfolio_weights.size() != problem.symbols.size() ||
        !timestamps_ordered || !symbols_ordered ||
        problem.history_timestamps.back() > problem.decision_at ||
        !problem.portfolio_return_history.empty() ||
        problem.factor_return_history.rows != 0 ||
        problem.specific_return_history.rows != 0 ||
        problem.factor_risk_model != nullptr ||
        (!problem.scenario_probabilities.empty() &&
            problem.scenario_probabilities.size() != returns.rows) ||
        !quant_math::validate_finite(returns).ok) {
        return result;
    }

    double weight_sum = 0.0;
    std::uint64_t input_hash = kFnvOffset;
    hash_value(input_hash, problem.spec.config_hash);
    hash_value(input_hash, problem.spec.mean_model_spec_hash);
    hash_value(input_hash, problem.spec.volatility_model_spec_hash);
    hash_value(input_hash, problem.spec.evt_threshold_spec_hash);
    hash_value(input_hash, problem.spec.scenario_seed);
    hash_value(input_hash, static_cast<std::uint64_t>(problem.spec.estimator));
    hash_value(input_hash,
               static_cast<std::uint64_t>(problem.spec.scenario_model));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.confidence_level));
    hash_value(input_hash, returns.rows);
    hash_value(input_hash, returns.cols);
    for (std::size_t col = 0; col < returns.cols; ++col) {
        const double weight = problem.fixed_portfolio_weights[col];
        if (!std::isfinite(weight) || weight < 0.0) return result;
        weight_sum += weight;
        hash_value(input_hash, problem.symbols[col]);
        hash_value(input_hash, std::bit_cast<std::uint64_t>(weight));
    }
    if (!(weight_sum > 0.0) || weight_sum > 1.0 + 1e-12) return result;
    for (std::size_t row = 0; row < returns.rows; ++row) {
        if (problem.history_timestamps[row] <= 0) return result;
        hash_value(input_hash, static_cast<std::uint64_t>(
            problem.history_timestamps[row]));
        for (std::size_t col = 0; col < returns.cols; ++col) {
            hash_value(input_hash, std::bit_cast<std::uint64_t>(
                returns(row, col)));
        }
    }
    double probability_sum = 0.0;
    if (!problem.scenario_probabilities.empty()) {
        for (const double probability : problem.scenario_probabilities) {
            if (!std::isfinite(probability) || probability < 0.0) return result;
            probability_sum += probability;
            hash_value(input_hash, std::bit_cast<std::uint64_t>(probability));
        }
        if (std::abs(probability_sum - 1.0) > 1e-12) return result;
    }

    std::vector<GarchFit> fits;
    fits.reserve(returns.cols);
    result.asset_garch_diagnostics.reserve(returns.cols);
    for (std::size_t col = 0; col < returns.cols; ++col) {
        std::vector<double> asset_returns(returns.rows);
        for (std::size_t row = 0; row < returns.rows; ++row) {
            asset_returns[row] = returns(row, col);
        }
        GarchFit fit = fit_garch11(asset_returns);
        if (!fit.ok) {
            result.status = TailRiskStatus::VOLATILITY_FIT_FAILURE;
            return result;
        }
        TailRiskEstimate diagnostic_result;
        if (!attach_garch_diagnostics(fit, diagnostic_result)) {
            result.status = TailRiskStatus::RESIDUAL_DIAGNOSTIC_FAILURE;
            return result;
        }
        result.asset_garch_diagnostics.push_back(AssetGarchDiagnostic{
            problem.symbols[col], fit.mean, fit.omega, fit.alpha, fit.beta,
            fit.forecast_variance, 1.0 - fit.alpha - fit.beta,
            *diagnostic_result.standardized_residual_mean,
            *diagnostic_result.standardized_residual_variance,
            *diagnostic_result.residual_ljung_box,
            *diagnostic_result.squared_residual_ljung_box,
            *diagnostic_result.maximum_standardized_residual,
        });
        hash_value(input_hash, std::bit_cast<std::uint64_t>(fit.mean));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(fit.omega));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(fit.alpha));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(fit.beta));
        fits.push_back(std::move(fit));
    }

    std::vector<std::pair<double, double>> losses;
    losses.reserve(returns.rows);
    for (std::size_t row = 0; row < returns.rows; ++row) {
        double scenario_return = 0.0;
        for (std::size_t col = 0; col < returns.cols; ++col) {
            scenario_return += problem.fixed_portfolio_weights[col] *
                (fits[col].mean + std::sqrt(fits[col].forecast_variance) *
                    fits[col].standardized_residuals[row]);
        }
        const double probability = problem.scenario_probabilities.empty()
            ? 1.0 / static_cast<double>(returns.rows)
            : problem.scenario_probabilities[row];
        if (!std::isfinite(scenario_return)) {
            result.status = TailRiskStatus::NUMERICAL_FAILURE;
            return result;
        }
        losses.emplace_back(-scenario_return, probability);
        hash_value(input_hash, std::bit_cast<std::uint64_t>(scenario_return));
    }
    if (evt_estimator) {
        return apply_pot_gpd_splice(
            problem, std::move(losses), input_hash, std::move(result));
    }
    std::sort(losses.begin(), losses.end(),
        [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
    double cumulative_probability = 0.0;
    double value_at_risk = losses.back().first;
    for (const auto& [loss, probability] : losses) {
        cumulative_probability += probability;
        if (cumulative_probability + 1e-15 >=
            problem.spec.confidence_level) {
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
    if (!std::isfinite(expected_shortfall) ||
        expected_shortfall < value_at_risk) {
        result.status = TailRiskStatus::NUMERICAL_FAILURE;
        return result;
    }
    result.status = TailRiskStatus::OK;
    result.value_at_risk_loss = value_at_risk;
    result.expected_shortfall_loss = expected_shortfall;
    result.return_cvar = -expected_shortfall;
    result.effective_observations = static_cast<std::uint32_t>(returns.rows);
    result.input_hash = input_hash;
    result.artifact_hash = input_hash;
    hash_value(result.artifact_hash,
               static_cast<std::uint64_t>(result.estimator));
    hash_value(result.artifact_hash,
               static_cast<std::uint64_t>(result.scenario_model));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(value_at_risk));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(expected_shortfall));
    return result;
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
            spec.evt_threshold_grid_points == 0 &&
            spec.evt_max_shape_spread == 0.0 &&
            spec.evt_max_relative_es_spread == 0.0 &&
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
            spec.evt_threshold_grid_points == 0 &&
            spec.evt_threshold_quantile_min == 0.0 &&
            spec.evt_threshold_quantile_max == 0.0 &&
            spec.evt_shape_upper_guard == 0.0 &&
            spec.evt_max_shape_spread == 0.0 &&
            spec.evt_max_relative_es_spread == 0.0 &&
            spec.evt_threshold_spec_hash == 0;
    }
    if (spec.estimator == TailRiskEstimatorKind::GARCH_FILTERED_HISTORICAL_SIMULATION) {
        return spec.expectile_level == 0.0 && spec.volatility_model_spec_hash != 0 &&
            spec.mean_model_spec_hash != 0 && spec.forecast_horizon_periods == 1 &&
            spec.residual_block_length == 1 && spec.filtered_volatility_state_only &&
            spec.evt_minimum_exceedances == 0 &&
            spec.evt_threshold_grid_points == 0 &&
            spec.evt_threshold_quantile_min == 0.0 &&
            spec.evt_threshold_quantile_max == 0.0 &&
            spec.evt_shape_upper_guard == 0.0 &&
            spec.evt_max_shape_spread == 0.0 &&
            spec.evt_max_relative_es_spread == 0.0 &&
            spec.evt_threshold_spec_hash == 0 &&
            spec.expectile_feature_spec_hash == 0 &&
            (spec.scenario_model == TailScenarioModelKind::PORTFOLIO_RETURN_SERIES ||
             (spec.scenario_model ==
                  TailScenarioModelKind::ASSET_VECTOR_SYNCHRONIZED &&
              spec.synchronized_residual_rows));
    }
    return spec.expectile_level == 0.0 && spec.volatility_model_spec_hash != 0 &&
        spec.mean_model_spec_hash != 0 && spec.forecast_horizon_periods == 1 &&
        spec.residual_block_length == 1 && spec.filtered_volatility_state_only &&
        spec.training_only_tail_calibration &&
        spec.expectile_feature_spec_hash == 0 &&
        spec.evt_minimum_exceedances > 0 &&
        spec.evt_threshold_grid_points == 4 &&
        spec.evt_threshold_spec_hash != 0 &&
        spec.evt_shape_upper_guard > 0.0 && spec.evt_shape_upper_guard < 1.0 &&
        spec.evt_max_shape_spread > 0.0 &&
        spec.evt_max_shape_spread <= 1.0 &&
        spec.evt_max_relative_es_spread > 0.0 &&
        spec.evt_max_relative_es_spread <= 1.0 &&
        spec.evt_threshold_quantile_min == 0.75 &&
        spec.evt_threshold_quantile_max > spec.evt_threshold_quantile_min &&
        spec.evt_threshold_quantile_max == 0.90 &&
        (spec.scenario_model == TailScenarioModelKind::PORTFOLIO_RETURN_SERIES ||
         (spec.scenario_model ==
              TailScenarioModelKind::ASSET_VECTOR_SYNCHRONIZED &&
          spec.synchronized_residual_rows));
}

GpdTailEvaluation evaluate_gpd_tail(
    double threshold_loss, double threshold_tail_probability,
    double gpd_shape, double gpd_scale, double confidence_level,
    double shape_upper_guard) noexcept {
    GpdTailEvaluation result;
    const double target_tail_probability = 1.0 - confidence_level;
    if (!std::isfinite(threshold_loss) ||
        !std::isfinite(threshold_tail_probability) ||
        !(threshold_tail_probability > target_tail_probability) ||
        threshold_tail_probability > 1.0 ||
        !(target_tail_probability > 0.0) ||
        !std::isfinite(gpd_shape) || !std::isfinite(gpd_scale) ||
        !(gpd_scale > 0.0) || !std::isfinite(shape_upper_guard) ||
        !(shape_upper_guard > 0.0) || !(shape_upper_guard < 1.0)) {
        return result;
    }
    if (gpd_shape >= 1.0 || gpd_shape >= shape_upper_guard) {
        result.status = TailRiskStatus::EVT_INFINITE_MEAN;
        return result;
    }
    const double ratio =
        threshold_tail_probability / target_tail_probability;
    double value_at_risk = 0.0;
    if (std::abs(gpd_shape) < 1e-8) {
        value_at_risk = threshold_loss + gpd_scale * std::log(ratio);
    } else {
        const double support_term = std::pow(ratio, gpd_shape);
        value_at_risk = threshold_loss +
            gpd_scale / gpd_shape * (support_term - 1.0);
    }
    const double value_at_risk_excess = value_at_risk - threshold_loss;
    const double value_at_risk_support =
        1.0 + gpd_shape * value_at_risk_excess / gpd_scale;
    if (!std::isfinite(value_at_risk) || value_at_risk < threshold_loss ||
        !std::isfinite(value_at_risk_support) ||
        !(value_at_risk_support > 0.0)) {
        result.status = TailRiskStatus::EVT_FIT_FAILURE;
        return result;
    }
    const double mean_excess_at_var =
        gpd_scale + gpd_shape * value_at_risk_excess;
    const double expected_shortfall = value_at_risk +
        mean_excess_at_var / (1.0 - gpd_shape);
    if (!std::isfinite(mean_excess_at_var) || !(mean_excess_at_var > 0.0) ||
        !std::isfinite(expected_shortfall) ||
        expected_shortfall < value_at_risk) {
        result.status = TailRiskStatus::EVT_FIT_FAILURE;
        return result;
    }
    result.status = TailRiskStatus::OK;
    result.value_at_risk_loss = value_at_risk;
    result.expected_shortfall_loss = expected_shortfall;
    return result;
}

TailRiskEstimate estimate_fhs_pot_gpd_splice(
    std::span<const double> fhs_losses,
    std::span<const double> scenario_probabilities,
    const TailRiskSpec& spec) {
    TailRiskEstimate result;
    result.estimator = spec.estimator;
    result.scenario_model = spec.scenario_model;
    result.confidence_level = spec.confidence_level;
    if (!valid_tail_risk_spec(spec) ||
        spec.estimator != TailRiskEstimatorKind::GARCH_FHS_POT_GPD ||
        fhs_losses.empty() ||
        (!scenario_probabilities.empty() &&
            scenario_probabilities.size() != fhs_losses.size())) {
        return result;
    }
    std::vector<std::pair<double, double>> losses;
    losses.reserve(fhs_losses.size());
    for (std::size_t index = 0; index < fhs_losses.size(); ++index) {
        const double probability = scenario_probabilities.empty()
            ? 1.0 / static_cast<double>(fhs_losses.size())
            : scenario_probabilities[index];
        losses.emplace_back(fhs_losses[index], probability);
    }
    TailRiskProblemView problem;
    problem.spec = spec;
    std::uint64_t input_hash = kFnvOffset;
    hash_value(input_hash, spec.config_hash);
    hash_value(input_hash, spec.mean_model_spec_hash);
    hash_value(input_hash, spec.volatility_model_spec_hash);
    hash_value(input_hash, spec.evt_threshold_spec_hash);
    hash_value(input_hash, static_cast<std::uint64_t>(spec.estimator));
    hash_value(input_hash, static_cast<std::uint64_t>(spec.scenario_model));
    hash_value(input_hash,
               std::bit_cast<std::uint64_t>(spec.confidence_level));
    return apply_pot_gpd_splice(
        problem, std::move(losses), input_hash, std::move(result));
}

TailRiskEstimate estimate_garch_fhs_tail_risk(
    const TailRiskProblemView& problem) {
    if (problem.spec.scenario_model ==
        TailScenarioModelKind::ASSET_VECTOR_SYNCHRONIZED) {
        return estimate_asset_vector_garch_tail_risk(problem);
    }
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
    if (problem.spec.scenario_model ==
        TailScenarioModelKind::ASSET_VECTOR_SYNCHRONIZED) {
        return estimate_asset_vector_garch_tail_risk(problem);
    }
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
    return apply_pot_gpd_splice(
        problem, std::move(losses), input_hash, std::move(result));
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
           << ",\"asset_garch\":[";
    for (std::size_t index = 0;
         index < estimate.asset_garch_diagnostics.size(); ++index) {
        if (index != 0) output << ',';
        const auto& diagnostic = estimate.asset_garch_diagnostics[index];
        output << "{\"symbol_id\":" << diagnostic.symbol_id
               << ",\"mean\":" << json_number(diagnostic.mean)
               << ",\"omega\":" << json_number(diagnostic.omega)
               << ",\"alpha\":" << json_number(diagnostic.alpha)
               << ",\"beta\":" << json_number(diagnostic.beta)
               << ",\"forecast_variance\":"
               << json_number(diagnostic.forecast_variance)
               << ",\"stationarity_margin\":"
               << json_number(diagnostic.stationarity_margin)
               << ",\"standardized_residual_mean\":"
               << json_number(diagnostic.standardized_residual_mean)
               << ",\"standardized_residual_variance\":"
               << json_number(diagnostic.standardized_residual_variance)
               << ",\"residual_ljung_box\":"
               << json_number(diagnostic.residual_ljung_box)
               << ",\"squared_residual_ljung_box\":"
               << json_number(diagnostic.squared_residual_ljung_box)
               << ",\"maximum_standardized_residual\":"
               << json_number(diagnostic.maximum_standardized_residual)
               << '}';
    }
    output << "]"
           << ",\"evt\":{\"method_id\":\"WEIGHTED_MOMENTS_FOUR_POINT_GRID_V1\""
           << ",\"threshold_grid_points\":"
           << spec.evt_threshold_grid_points
           << ",\"minimum_valid_grid_points\":3"
           << ",\"selection_rule\":\"LOWEST_QUANTILE_AMONG_VALID_STABLE_GRID_V1\""
           << ",\"threshold_quantile_min\":"
           << json_number(spec.evt_threshold_quantile_min)
           << ",\"threshold_quantile_max\":"
           << json_number(spec.evt_threshold_quantile_max)
           << ",\"maximum_shape_spread\":"
           << json_number(spec.evt_max_shape_spread)
           << ",\"maximum_relative_es_spread\":"
           << json_number(spec.evt_max_relative_es_spread)
           << ",\"selected_threshold_quantile\":"
           << optional_number(estimate.evt_selected_threshold_quantile)
           << ",\"exceedance_count\":"
           << estimate.evt_exceedance_count
           << ",\"effective_exceedances\":"
           << optional_number(estimate.evt_effective_exceedances)
           << ",\"threshold\":" << optional_number(estimate.evt_threshold)
           << ",\"shape\":" << optional_number(estimate.gpd_shape)
           << ",\"scale\":" << optional_number(estimate.gpd_scale)
           << ",\"shape_spread\":"
           << optional_number(estimate.evt_shape_spread)
           << ",\"relative_es_spread\":"
           << optional_number(estimate.evt_relative_es_spread)
           << ",\"threshold_diagnostics\":[";
    for (std::size_t index = 0;
         index < estimate.evt_threshold_diagnostics.size(); ++index) {
        if (index != 0) output << ',';
        const auto& diagnostic = estimate.evt_threshold_diagnostics[index];
        output << "{\"status\":" << static_cast<int>(diagnostic.status)
               << ",\"threshold_quantile\":"
               << json_number(diagnostic.threshold_quantile)
               << ",\"threshold_loss\":"
               << json_number(diagnostic.threshold_loss)
               << ",\"exceedance_count\":"
               << diagnostic.exceedance_count
               << ",\"effective_exceedances\":"
               << json_number(diagnostic.effective_exceedances)
               << ",\"tail_probability\":"
               << json_number(diagnostic.tail_probability)
               << ",\"shape\":" << json_number(diagnostic.gpd_shape)
               << ",\"scale\":" << json_number(diagnostic.gpd_scale)
               << ",\"var_loss\":"
               << json_number(diagnostic.value_at_risk_loss)
               << ",\"expected_shortfall_loss\":"
               << json_number(diagnostic.expected_shortfall_loss)
               << ",\"splice_continuity_error\":"
               << json_number(diagnostic.splice_continuity_error)
               << ",\"splice_probability_error\":"
               << json_number(diagnostic.splice_probability_error)
               << '}';
    }
    output << "]}"
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
    double fz0_score_sum = 0.0;
    const double tail_probability = 1.0 - problem.confidence_level;
    std::vector<bool> exceptions(observation_count, false);
    for (std::size_t index = 0; index < observation_count; ++index) {
        const double loss = -problem.realized_returns[index];
        const double value_at_risk = problem.value_at_risk_loss[index];
        const double expected_shortfall = problem.expected_shortfall_loss[index];
        if (!(expected_shortfall > 0.0)) {
            result.status = TailRiskBacktestStatus::FZ0_DOMAIN_FAILURE;
            return result;
        }
        if (loss > problem.value_at_risk_loss[index] + kTolerance) {
            exceptions[index] = true;
            ++result.exception_count;
            exceedance_sum += loss - problem.value_at_risk_loss[index];
        }
        if (loss > problem.expected_shortfall_loss[index] + kTolerance) {
            ++result.es_violation_count;
            es_excess_sum += loss - problem.expected_shortfall_loss[index];
        }
        const double fz0_score =
            ((loss > value_at_risk + kTolerance)
                 ? (loss - value_at_risk) /
                       (tail_probability * expected_shortfall)
                 : 0.0) +
            value_at_risk / expected_shortfall +
            std::log(expected_shortfall) - 1.0;
        if (!std::isfinite(fz0_score)) {
            result.status = TailRiskBacktestStatus::NUMERICAL_FAILURE;
            return result;
        }
        fz0_score_sum += fz0_score;
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
    result.mean_fz0_score =
        fz0_score_sum / static_cast<double>(observation_count);

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
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(result.mean_fz0_score));
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
           << ",\"mean_fz0_score\":"
           << json_number(result.mean_fz0_score)
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

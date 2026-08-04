#include "portfolio_math/tail_risk.h"

#include <algorithm>
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
            spec.expectile_feature_spec_hash != 0 && spec.training_only_tail_calibration;
    }
    if (spec.estimator == TailRiskEstimatorKind::GARCH_FILTERED_HISTORICAL_SIMULATION) {
        return spec.expectile_level == 0.0 && spec.volatility_model_spec_hash != 0 &&
            (spec.scenario_model == TailScenarioModelKind::PORTFOLIO_RETURN_SERIES ||
             spec.synchronized_residual_rows);
    }
    return spec.expectile_level == 0.0 && spec.volatility_model_spec_hash != 0 &&
        spec.evt_minimum_exceedances > 0 && spec.evt_threshold_spec_hash != 0 &&
        spec.evt_shape_upper_guard > 0.0 && spec.evt_shape_upper_guard < 1.0 &&
        spec.evt_threshold_quantile_min > 0.5 &&
        spec.evt_threshold_quantile_max > spec.evt_threshold_quantile_min &&
        spec.evt_threshold_quantile_max < 1.0 &&
        (spec.scenario_model == TailScenarioModelKind::PORTFOLIO_RETURN_SERIES ||
         spec.synchronized_residual_rows);
}

TailRiskEstimate estimate_tail_risk(const TailRiskProblemView& problem) {
    TailRiskEstimate result;
    result.estimator = problem.spec.estimator;
    result.confidence_level = problem.spec.confidence_level;
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
    std::ostringstream output;
    output << "{\"schema_version\":1,\"role\":\"fixed_portfolio_tail_risk\""
           << ",\"estimator\":" << static_cast<int>(estimate.estimator)
           << ",\"scenario_model\":" << static_cast<int>(
               TailScenarioModelKind::PORTFOLIO_RETURN_SERIES)
           << ",\"confidence_level\":" << json_number(spec.confidence_level)
           << ",\"effective_observations\":" << estimate.effective_observations
           << ",\"status\":" << static_cast<int>(estimate.status)
           << ",\"input_hash\":" << estimate.input_hash
           << ",\"artifact_hash\":" << estimate.artifact_hash
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

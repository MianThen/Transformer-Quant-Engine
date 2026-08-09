#include "portfolio_math/conditional_expectile.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include <Eigen/Cholesky>

namespace portfolio_math {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::size_t kMaximumFeatureCount = 64;

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

bool strictly_increasing(
    std::span<const engine_common::TimestampNs> timestamps) {
    return std::is_sorted(timestamps.begin(), timestamps.end()) &&
        std::adjacent_find(timestamps.begin(), timestamps.end()) ==
            timestamps.end();
}

bool solve_weighted_system(
    const ConditionalExpectileProblemView& problem,
    const Eigen::VectorXd& asymmetric_weights,
    double sample_weight_sum,
    Eigen::VectorXd& coefficients) {
    const auto features = problem.feature_history;
    const Eigen::Index parameter_count = static_cast<Eigen::Index>(features.cols + 1);
    Eigen::MatrixXd normal = Eigen::MatrixXd::Zero(parameter_count, parameter_count);
    Eigen::VectorXd right_hand_side = Eigen::VectorXd::Zero(parameter_count);
    for (std::size_t row = 0; row < features.rows; ++row) {
        const double sample_weight = problem.sample_weights.empty()
            ? 1.0 / static_cast<double>(features.rows)
            : problem.sample_weights[row] / sample_weight_sum;
        const double weight = sample_weight * asymmetric_weights[
            static_cast<Eigen::Index>(row)];
        Eigen::VectorXd design(parameter_count);
        design[0] = 1.0;
        for (std::size_t column = 0; column < features.cols; ++column) {
            design[static_cast<Eigen::Index>(column + 1)] = features(row, column);
        }
        normal.noalias() += weight * design * design.transpose();
        right_hand_side.noalias() +=
            weight * design * problem.target_losses[row];
    }
    for (Eigen::Index index = 1; index < parameter_count; ++index) {
        normal(index, index) += problem.spec.ridge_penalty;
    }
    Eigen::LDLT<Eigen::MatrixXd> solver(normal);
    if (solver.info() != Eigen::Success) return false;
    const Eigen::VectorXd candidate = solver.solve(right_hand_side);
    if (solver.info() != Eigen::Success || !candidate.allFinite()) return false;
    coefficients = candidate;
    return true;
}

}  // namespace

bool valid_conditional_expectile_spec(
    const ConditionalExpectileSpec& spec) noexcept {
    return spec.expectile_level > 0.5 && spec.expectile_level < 1.0 &&
        spec.minimum_observations >= 20 && spec.maximum_iterations > 0 &&
        spec.maximum_iterations <= 10000 &&
        std::isfinite(spec.convergence_tolerance) &&
        spec.convergence_tolerance > 0.0 &&
        spec.convergence_tolerance <= 1e-3 &&
        std::isfinite(spec.ridge_penalty) && spec.ridge_penalty > 0.0 &&
        spec.ridge_penalty <= 1e-2 && spec.include_intercept &&
        spec.training_only_calibration && spec.feature_spec_hash != 0 &&
        spec.solver_spec_hash != 0 && spec.config_hash != 0;
}

ConditionalExpectileResult fit_conditional_expectile(
    const ConditionalExpectileProblemView& problem) {
    ConditionalExpectileResult result;
    result.expectile_level = problem.spec.expectile_level;
    const auto features = problem.feature_history;
    if (!valid_conditional_expectile_spec(problem.spec) ||
        problem.decision_at <= 0 ||
        problem.forecast_features_available_at != problem.decision_at ||
        features.data == nullptr || features.rows == 0 || features.cols == 0 ||
        features.cols > kMaximumFeatureCount || features.row_stride < features.cols ||
        features.rows != problem.target_losses.size() ||
        features.rows != problem.target_timestamps.size() ||
        features.rows != problem.feature_available_at.size() ||
        features.cols != problem.forecast_features.size() ||
        (!problem.sample_weights.empty() &&
            problem.sample_weights.size() != features.rows) ||
        !strictly_increasing(problem.target_timestamps) ||
        !strictly_increasing(problem.feature_available_at) ||
        problem.target_timestamps.back() > problem.decision_at ||
        !quant_math::validate_finite(features).ok) {
        return result;
    }
    if (features.rows < problem.spec.minimum_observations ||
        features.rows <= features.cols + 1) {
        result.status = ConditionalExpectileStatus::INSUFFICIENT_OBSERVATIONS;
        return result;
    }

    double sample_weight_sum = 0.0;
    std::uint64_t input_hash = kFnvOffset;
    hash_value(input_hash, problem.spec.feature_spec_hash);
    hash_value(input_hash, problem.spec.solver_spec_hash);
    hash_value(input_hash, problem.spec.config_hash);
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.expectile_level));
    hash_value(input_hash, problem.spec.minimum_observations);
    hash_value(input_hash, problem.spec.maximum_iterations);
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.convergence_tolerance));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.ridge_penalty));
    hash_value(input_hash, static_cast<std::uint64_t>(features.rows));
    hash_value(input_hash, static_cast<std::uint64_t>(features.cols));
    hash_value(input_hash, static_cast<std::uint64_t>(problem.decision_at));
    hash_value(input_hash, static_cast<std::uint64_t>(
        problem.forecast_features_available_at));
    for (std::size_t row = 0; row < features.rows; ++row) {
        const double target = problem.target_losses[row];
        const double sample_weight = problem.sample_weights.empty()
            ? 1.0
            : problem.sample_weights[row];
        if (!std::isfinite(target) || !std::isfinite(sample_weight) ||
            sample_weight < 0.0 || problem.target_timestamps[row] <= 0 ||
            problem.feature_available_at[row] <= 0 ||
            problem.feature_available_at[row] >= problem.target_timestamps[row]) {
            return result;
        }
        sample_weight_sum += sample_weight;
        hash_value(input_hash, static_cast<std::uint64_t>(
            problem.target_timestamps[row]));
        hash_value(input_hash, static_cast<std::uint64_t>(
            problem.feature_available_at[row]));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(target));
        hash_value(input_hash, std::bit_cast<std::uint64_t>(sample_weight));
        for (std::size_t column = 0; column < features.cols; ++column) {
            hash_value(input_hash, std::bit_cast<std::uint64_t>(
                features(row, column)));
        }
    }
    for (const double feature : problem.forecast_features) {
        if (!std::isfinite(feature)) return result;
        hash_value(input_hash, std::bit_cast<std::uint64_t>(feature));
    }
    if (!std::isfinite(sample_weight_sum) || !(sample_weight_sum > 0.0)) {
        return result;
    }

    const Eigen::Index observation_count = static_cast<Eigen::Index>(features.rows);
    Eigen::VectorXd asymmetric_weights = Eigen::VectorXd::Ones(observation_count);
    Eigen::VectorXd coefficients;
    if (!solve_weighted_system(
            problem, asymmetric_weights, sample_weight_sum, coefficients)) {
        result.status = ConditionalExpectileStatus::OPTIMIZATION_FAILURE;
        return result;
    }
    bool converged = false;
    for (std::uint32_t iteration = 1;
         iteration <= problem.spec.maximum_iterations; ++iteration) {
        for (std::size_t row = 0; row < features.rows; ++row) {
            double fitted = coefficients[0];
            for (std::size_t column = 0; column < features.cols; ++column) {
                fitted += coefficients[static_cast<Eigen::Index>(column + 1)] *
                    features(row, column);
            }
            const double residual = problem.target_losses[row] - fitted;
            asymmetric_weights[static_cast<Eigen::Index>(row)] = residual >= 0.0
                ? problem.spec.expectile_level
                : 1.0 - problem.spec.expectile_level;
        }
        Eigen::VectorXd candidate;
        if (!solve_weighted_system(
                problem, asymmetric_weights, sample_weight_sum, candidate)) {
            result.status = ConditionalExpectileStatus::OPTIMIZATION_FAILURE;
            return result;
        }
        const double change = (candidate - coefficients).lpNorm<Eigen::Infinity>();
        const double scale = 1.0 + candidate.lpNorm<Eigen::Infinity>();
        coefficients = std::move(candidate);
        result.optimization_iterations = iteration;
        if (change <= problem.spec.convergence_tolerance * scale) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        result.status = ConditionalExpectileStatus::OPTIMIZATION_FAILURE;
        return result;
    }

    double objective = 0.0;
    Eigen::VectorXd first_order = Eigen::VectorXd::Zero(coefficients.size());
    for (std::size_t row = 0; row < features.rows; ++row) {
        Eigen::VectorXd design(coefficients.size());
        design[0] = 1.0;
        for (std::size_t column = 0; column < features.cols; ++column) {
            design[static_cast<Eigen::Index>(column + 1)] = features(row, column);
        }
        const double fitted = coefficients.dot(design);
        const double residual = problem.target_losses[row] - fitted;
        const double asymmetric_weight = residual >= 0.0
            ? problem.spec.expectile_level
            : 1.0 - problem.spec.expectile_level;
        const double sample_weight = problem.sample_weights.empty()
            ? 1.0 / static_cast<double>(features.rows)
            : problem.sample_weights[row] / sample_weight_sum;
        objective += sample_weight * asymmetric_weight * residual * residual;
        first_order.noalias() +=
            sample_weight * asymmetric_weight * residual * design;
    }
    double forecast = coefficients[0];
    for (std::size_t column = 0; column < features.cols; ++column) {
        forecast += coefficients[static_cast<Eigen::Index>(column + 1)] *
            problem.forecast_features[column];
    }
    if (!std::isfinite(objective) || !first_order.allFinite() ||
        !std::isfinite(forecast)) {
        result.status = ConditionalExpectileStatus::NUMERICAL_FAILURE;
        return result;
    }

    result.status = ConditionalExpectileStatus::OK;
    result.forecast_expectile_loss = forecast;
    result.mean_asymmetric_squared_loss = objective;
    result.maximum_first_order_residual =
        first_order.lpNorm<Eigen::Infinity>();
    result.effective_observations = static_cast<std::uint32_t>(features.rows);
    result.feature_count = static_cast<std::uint32_t>(features.cols);
    result.coefficients.assign(coefficients.data(),
                               coefficients.data() + coefficients.size());
    result.input_hash = input_hash;
    result.artifact_hash = input_hash;
    hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(forecast));
    hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(objective));
    hash_value(result.artifact_hash, std::bit_cast<std::uint64_t>(
        result.maximum_first_order_residual));
    for (const double coefficient : result.coefficients) {
        hash_value(result.artifact_hash,
                   std::bit_cast<std::uint64_t>(coefficient));
    }
    return result;
}

std::string serialize_conditional_expectile_artifact(
    const ConditionalExpectileResult& result,
    const ConditionalExpectileSpec& spec,
    const TailRiskArtifactSpec& artifact_spec) {
    const bool trusted_reference =
        artifact_spec.reference_price_quality != "PROXY" &&
        artifact_spec.reference_price_quality != "ARRIVAL_PROXY" &&
        artifact_spec.reference_price_quality != "UNKNOWN" &&
        artifact_spec.reference_price_quality != "UNAVAILABLE";
    const bool promotion_eligible =
        result.status == ConditionalExpectileStatus::OK && trusted_reference &&
        artifact_spec.promotion_eligible;
    std::ostringstream output;
    output << "{\"schema_version\":1"
           << ",\"role\":\"fixed_portfolio_conditional_expectile\""
           << ",\"estimator_id\":\"TAIL-EXPECTILE\""
           << ",\"model_kind\":\"FIXED_PIT_LINEAR_ALS\""
           << ",\"care_sav_claimed\":false"
           << ",\"mapped_es_claimed\":false"
           << ",\"status\":" << static_cast<int>(result.status)
           << ",\"expectile_level\":" << json_number(result.expectile_level)
           << ",\"forecast_expectile_loss\":"
           << json_number(result.forecast_expectile_loss)
           << ",\"mean_asymmetric_squared_loss\":"
           << json_number(result.mean_asymmetric_squared_loss)
           << ",\"maximum_first_order_residual\":"
           << json_number(result.maximum_first_order_residual)
           << ",\"effective_observations\":"
           << result.effective_observations
           << ",\"feature_count\":" << result.feature_count
           << ",\"optimization_iterations\":"
           << result.optimization_iterations
           << ",\"feature_spec_hash\":" << spec.feature_spec_hash
           << ",\"solver_spec_hash\":" << spec.solver_spec_hash
           << ",\"config_hash\":" << spec.config_hash
           << ",\"input_hash\":" << result.input_hash
           << ",\"artifact_hash\":" << result.artifact_hash
           << ",\"coefficients\":[";
    for (std::size_t index = 0; index < result.coefficients.size(); ++index) {
        if (index != 0) output << ',';
        output << json_number(result.coefficients[index]);
    }
    output << "]"
           << ",\"reference_price_quality\":\""
           << json_escape(artifact_spec.reference_price_quality) << "\""
           << ",\"promotion_eligible\":"
           << (promotion_eligible ? "true" : "false")
           << ",\"limitations\":[";
    for (std::size_t index = 0; index < artifact_spec.limitations.size(); ++index) {
        if (index != 0) output << ',';
        output << '"' << json_escape(artifact_spec.limitations[index]) << '"';
    }
    output << "]}";
    return output.str();
}

}  // namespace portfolio_math

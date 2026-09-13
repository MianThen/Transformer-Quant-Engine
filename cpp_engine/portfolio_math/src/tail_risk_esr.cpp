#include "portfolio_math/tail_risk.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace portfolio_math {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr double kSqrtTwo = 1.4142135623730950488;
constexpr double kSqrtTwoPi = 2.5066282746310005024;

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

struct EsrDesign {
    std::vector<double> response;
    Eigen::MatrixXd quantile_regressors;
    Eigen::MatrixXd expected_shortfall_regressors;
    std::vector<double> initial_parameters;
    double lower_tail_probability{0.0};
    double response_shift{0.0};
    double response_scale{0.0};
    double domain_margin{0.0};
};

struct OptimizationResult {
    bool converged{false};
    std::vector<double> parameters;
    double objective{std::numeric_limits<double>::infinity()};
    std::uint32_t iterations{0};
};

struct CovarianceResult {
    bool ok{false};
    Eigen::MatrixXd covariance;
    double density_at_quantile{0.0};
    double truncated_residual_variance{0.0};
};

double dot_row(const Eigen::MatrixXd& matrix, std::size_t row,
               std::span<const double> parameters, std::size_t offset) {
    double value = 0.0;
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
        value += matrix(static_cast<Eigen::Index>(row), col) *
            parameters[offset + static_cast<std::size_t>(col)];
    }
    return value;
}

double joint_fz_objective(const EsrDesign& design,
                          std::span<const double> parameters) {
    const std::size_t quantile_parameter_count =
        static_cast<std::size_t>(design.quantile_regressors.cols());
    if (parameters.size() != quantile_parameter_count +
            static_cast<std::size_t>(
                design.expected_shortfall_regressors.cols())) {
        return std::numeric_limits<double>::infinity();
    }
    double objective = 0.0;
    for (std::size_t row = 0; row < design.response.size(); ++row) {
        const double quantile = dot_row(
            design.quantile_regressors, row, parameters, 0);
        const double expected_shortfall = dot_row(
            design.expected_shortfall_regressors, row, parameters,
            quantile_parameter_count);
        if (!std::isfinite(quantile) || !std::isfinite(expected_shortfall) ||
            expected_shortfall >= -design.domain_margin) {
            return std::numeric_limits<double>::infinity();
        }
        const double response = design.response[row];
        const double hit = response <= quantile ? 1.0 : 0.0;
        const double observation_score =
            (-1.0 / expected_shortfall) *
                (expected_shortfall - quantile +
                 (quantile - response) * hit /
                     design.lower_tail_probability) +
            std::log(-expected_shortfall);
        if (!std::isfinite(observation_score)) {
            return std::numeric_limits<double>::infinity();
        }
        objective += observation_score;
    }
    return objective / static_cast<double>(design.response.size());
}

double parameter_distance(std::span<const double> left,
                          std::span<const double> right) {
    double squared_distance = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const double difference = left[index] - right[index];
        squared_distance += difference * difference;
    }
    return std::sqrt(squared_distance);
}

OptimizationResult minimize_joint_fz(
    const EsrDesign& design, const TailRiskEsrSpec& spec) {
    OptimizationResult result;
    const std::size_t dimension = design.initial_parameters.size();
    if (dimension == 0) return result;

    std::vector<std::vector<double>> simplex(
        dimension + 1, design.initial_parameters);
    for (std::size_t index = 0; index < dimension; ++index) {
        const bool intercept = index == 0 ||
            index == static_cast<std::size_t>(
                design.quantile_regressors.cols());
        const double step = intercept
            ? std::max(1e-8, 0.05 * design.response_scale)
            : 0.05;
        simplex[index + 1][index] += step;
    }
    std::vector<double> values(dimension + 1);
    for (std::size_t index = 0; index < simplex.size(); ++index) {
        values[index] = joint_fz_objective(design, simplex[index]);
    }
    if (!std::isfinite(values[0])) return result;

    constexpr double kReflection = 1.0;
    constexpr double kExpansion = 2.0;
    constexpr double kContraction = 0.5;
    constexpr double kShrink = 0.5;
    std::vector<std::size_t> order(dimension + 1);
    std::iota(order.begin(), order.end(), 0);
    for (std::uint32_t iteration = 0;
         iteration < spec.maximum_iterations; ++iteration) {
        std::stable_sort(order.begin(), order.end(), [&](std::size_t left,
                                                         std::size_t right) {
            if (values[left] == values[right]) return left < right;
            return values[left] < values[right];
        });
        std::vector<std::vector<double>> ordered_simplex;
        std::vector<double> ordered_values;
        ordered_simplex.reserve(dimension + 1);
        ordered_values.reserve(dimension + 1);
        for (const std::size_t index : order) {
            ordered_simplex.push_back(simplex[index]);
            ordered_values.push_back(values[index]);
        }
        simplex = std::move(ordered_simplex);
        values = std::move(ordered_values);
        std::iota(order.begin(), order.end(), 0);

        const double objective_spread = values.back() - values.front();
        double maximum_distance = 0.0;
        for (std::size_t index = 1; index < simplex.size(); ++index) {
            maximum_distance = std::max(
                maximum_distance,
                parameter_distance(simplex.front(), simplex[index]));
        }
        double best_norm = 0.0;
        for (const double parameter : simplex.front()) {
            best_norm += parameter * parameter;
        }
        best_norm = std::sqrt(best_norm);
        const double objective_tolerance = spec.convergence_tolerance *
            (1.0 + std::abs(values.front()));
        const double parameter_tolerance =
            std::sqrt(spec.convergence_tolerance) * (1.0 + best_norm);
        if (std::isfinite(values.front()) &&
            objective_spread <= objective_tolerance &&
            maximum_distance <= parameter_tolerance) {
            result.converged = true;
            result.parameters = simplex.front();
            result.objective = values.front();
            result.iterations = iteration;
            return result;
        }

        std::vector<double> centroid(dimension, 0.0);
        for (std::size_t vertex = 0; vertex < dimension; ++vertex) {
            for (std::size_t coordinate = 0; coordinate < dimension;
                 ++coordinate) {
                centroid[coordinate] += simplex[vertex][coordinate];
            }
        }
        for (double& coordinate : centroid) {
            coordinate /= static_cast<double>(dimension);
        }
        const auto trial = [&](double multiplier) {
            std::vector<double> candidate(dimension);
            for (std::size_t coordinate = 0; coordinate < dimension;
                 ++coordinate) {
                candidate[coordinate] = centroid[coordinate] + multiplier *
                    (centroid[coordinate] - simplex.back()[coordinate]);
            }
            return candidate;
        };
        auto reflected = trial(kReflection);
        const double reflected_value = joint_fz_objective(design, reflected);
        if (reflected_value < values.front()) {
            auto expanded = trial(kExpansion);
            const double expanded_value = joint_fz_objective(design, expanded);
            if (expanded_value < reflected_value) {
                simplex.back() = std::move(expanded);
                values.back() = expanded_value;
            } else {
                simplex.back() = std::move(reflected);
                values.back() = reflected_value;
            }
            continue;
        }
        if (reflected_value < values[dimension - 1]) {
            simplex.back() = std::move(reflected);
            values.back() = reflected_value;
            continue;
        }

        std::vector<double> contracted(dimension);
        const bool outside = reflected_value < values.back();
        for (std::size_t coordinate = 0; coordinate < dimension;
             ++coordinate) {
            const double target = outside ? reflected[coordinate] :
                simplex.back()[coordinate];
            contracted[coordinate] = centroid[coordinate] + kContraction *
                (target - centroid[coordinate]);
        }
        const double contracted_value = joint_fz_objective(design, contracted);
        if (contracted_value < (outside ? reflected_value : values.back())) {
            simplex.back() = std::move(contracted);
            values.back() = contracted_value;
            continue;
        }

        for (std::size_t vertex = 1; vertex < simplex.size(); ++vertex) {
            for (std::size_t coordinate = 0; coordinate < dimension;
                 ++coordinate) {
                simplex[vertex][coordinate] = simplex.front()[coordinate] +
                    kShrink * (simplex[vertex][coordinate] -
                               simplex.front()[coordinate]);
            }
            values[vertex] = joint_fz_objective(design, simplex[vertex]);
        }
    }
    const auto best = std::min_element(values.begin(), values.end());
    if (best != values.end() && std::isfinite(*best)) {
        const std::size_t best_index = static_cast<std::size_t>(
            std::distance(values.begin(), best));
        result.parameters = simplex[best_index];
        result.objective = *best;
        result.iterations = spec.maximum_iterations;
    }
    return result;
}

double empirical_quantile(std::vector<double> values, double probability) {
    std::sort(values.begin(), values.end());
    const double rank = probability * static_cast<double>(values.size());
    const std::size_t index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(std::ceil(rank)) - 1);
    return values[index];
}

double empirical_lower_expected_shortfall(std::vector<double> values,
                                          double probability) {
    std::sort(values.begin(), values.end());
    const double target_mass = probability * static_cast<double>(values.size());
    double remaining_mass = target_mass;
    double tail_sum = 0.0;
    for (const double value : values) {
        const double included_mass = std::min(1.0, remaining_mass);
        tail_sum += included_mass * value;
        remaining_mass -= included_mass;
        if (remaining_mass <= 0.0) break;
    }
    return tail_sum / target_mass;
}

EsrDesign make_design(const TailRiskEsrProblemView& problem,
                      TailRiskEsrVariant variant) {
    EsrDesign design;
    const std::size_t observation_count = problem.realized_returns.size();
    design.response.resize(observation_count);
    design.lower_tail_probability = 1.0 - problem.spec.confidence_level;
    const bool strict_intercept =
        variant == TailRiskEsrVariant::STRICT_INTERCEPT;
    design.quantile_regressors.resize(
        static_cast<Eigen::Index>(observation_count), 2);
    design.expected_shortfall_regressors.resize(
        static_cast<Eigen::Index>(observation_count),
        strict_intercept ? 1 : 2);
    for (std::size_t row = 0; row < observation_count; ++row) {
        const double quantile_return = -problem.value_at_risk_loss[row];
        const double expected_shortfall_return =
            -problem.expected_shortfall_loss[row];
        design.response[row] = strict_intercept
            ? problem.realized_returns[row] - expected_shortfall_return
            : problem.realized_returns[row];
        design.quantile_regressors(static_cast<Eigen::Index>(row), 0) = 1.0;
        design.quantile_regressors(static_cast<Eigen::Index>(row), 1) =
            variant == TailRiskEsrVariant::AUXILIARY
                ? quantile_return
                : expected_shortfall_return;
        design.expected_shortfall_regressors(
            static_cast<Eigen::Index>(row), 0) = 1.0;
        if (!strict_intercept) {
            design.expected_shortfall_regressors(
                static_cast<Eigen::Index>(row), 1) =
                expected_shortfall_return;
        }
    }
    design.response_shift = *std::max_element(
        design.response.begin(), design.response.end());
    double squared_scale = 0.0;
    double response_mean = 0.0;
    for (const double value : design.response) response_mean += value;
    response_mean /= static_cast<double>(observation_count);
    for (double& value : design.response) {
        const double centered = value - response_mean;
        squared_scale += centered * centered;
        value -= design.response_shift;
    }
    design.response_scale = std::sqrt(
        squared_scale / static_cast<double>(observation_count));
    design.response_scale = std::max(
        design.response_scale,
        std::numeric_limits<double>::epsilon() *
            (1.0 + std::abs(design.response_shift)));
    design.domain_margin = std::max(
        1e-14, design.response_scale * 1e-10);

    if (strict_intercept) {
        const double initial_quantile = empirical_quantile(
            design.response, design.lower_tail_probability);
        double initial_expected_shortfall = empirical_lower_expected_shortfall(
            design.response, design.lower_tail_probability);
        initial_expected_shortfall = std::min(
            initial_expected_shortfall, -10.0 * design.domain_margin);
        design.initial_parameters = {
            initial_quantile, 0.0, initial_expected_shortfall};
    } else {
        design.initial_parameters = {
            -design.response_shift, 1.0,
            -design.response_shift, 1.0};
        double maximum_expected_shortfall =
            -std::numeric_limits<double>::infinity();
        for (std::size_t row = 0; row < observation_count; ++row) {
            maximum_expected_shortfall = std::max(
                maximum_expected_shortfall,
                dot_row(design.expected_shortfall_regressors, row,
                        design.initial_parameters, 2));
        }
        if (maximum_expected_shortfall >= -design.domain_margin) {
            design.initial_parameters[2] -= maximum_expected_shortfall +
                10.0 * design.domain_margin;
        }
    }
    return design;
}

CovarianceResult estimate_iid_ind_covariance(
    const EsrDesign& design, std::span<const double> parameters) {
    CovarianceResult result;
    const std::size_t observation_count = design.response.size();
    const Eigen::Index quantile_parameter_count =
        design.quantile_regressors.cols();
    const Eigen::Index expected_shortfall_parameter_count =
        design.expected_shortfall_regressors.cols();
    const Eigen::Index parameter_count = quantile_parameter_count +
        expected_shortfall_parameter_count;
    std::vector<double> residuals(observation_count);
    std::vector<double> negative_residuals;
    negative_residuals.reserve(observation_count);
    double residual_mean = 0.0;
    for (std::size_t row = 0; row < observation_count; ++row) {
        const double quantile = dot_row(
            design.quantile_regressors, row, parameters, 0);
        residuals[row] = design.response[row] - quantile;
        residual_mean += residuals[row];
        if (residuals[row] <= 0.0) {
            negative_residuals.push_back(residuals[row]);
        }
    }
    residual_mean /= static_cast<double>(observation_count);
    double residual_variance = 0.0;
    for (const double residual : residuals) {
        const double centered = residual - residual_mean;
        residual_variance += centered * centered;
    }
    residual_variance /= static_cast<double>(observation_count - 1);
    if (!(residual_variance > 0.0) || negative_residuals.size() < 3) {
        return result;
    }
    const double residual_standard_deviation = std::sqrt(residual_variance);
    const double bandwidth = std::max(
        design.response_scale * 1e-8,
        1.06 * residual_standard_deviation *
            std::pow(static_cast<double>(observation_count), -0.2));
    double density = 0.0;
    for (const double residual : residuals) {
        const double standardized = residual / bandwidth;
        density += std::exp(-0.5 * standardized * standardized) /
            (kSqrtTwoPi * bandwidth);
    }
    density /= static_cast<double>(observation_count);

    double negative_mean = 0.0;
    for (const double residual : negative_residuals) {
        negative_mean += residual;
    }
    negative_mean /= static_cast<double>(negative_residuals.size());
    double truncated_variance = 0.0;
    for (const double residual : negative_residuals) {
        const double centered = residual - negative_mean;
        truncated_variance += centered * centered;
    }
    truncated_variance /=
        static_cast<double>(negative_residuals.size() - 1);
    if (!(density > 0.0) || !std::isfinite(density) ||
        !(truncated_variance > 0.0) || !std::isfinite(truncated_variance)) {
        return result;
    }

    Eigen::MatrixXd lambda = Eigen::MatrixXd::Zero(
        parameter_count, parameter_count);
    Eigen::MatrixXd sigma = Eigen::MatrixXd::Zero(
        parameter_count, parameter_count);
    const double alpha = design.lower_tail_probability;
    for (std::size_t row = 0; row < observation_count; ++row) {
        const Eigen::VectorXd quantile_regressor =
            design.quantile_regressors.row(
                static_cast<Eigen::Index>(row)).transpose();
        const Eigen::VectorXd expected_shortfall_regressor =
            design.expected_shortfall_regressors.row(
                static_cast<Eigen::Index>(row)).transpose();
        const double quantile = dot_row(
            design.quantile_regressors, row, parameters, 0);
        const double expected_shortfall = dot_row(
            design.expected_shortfall_regressors, row, parameters,
            static_cast<std::size_t>(quantile_parameter_count));
        if (expected_shortfall >= -design.domain_margin) return result;
        const double g2 = -1.0 / expected_shortfall;
        const double g2_prime = 1.0 /
            (expected_shortfall * expected_shortfall);
        lambda.topLeftCorner(quantile_parameter_count,
                             quantile_parameter_count) +=
            quantile_regressor * quantile_regressor.transpose() *
            (g2 / alpha) * density;
        lambda.bottomRightCorner(expected_shortfall_parameter_count,
                                 expected_shortfall_parameter_count) +=
            expected_shortfall_regressor *
            expected_shortfall_regressor.transpose() * g2_prime;

        const double quantile_score_scale = g2;
        const double cross_scale = quantile_score_scale * g2_prime *
            (1.0 - alpha) / alpha *
            (quantile - expected_shortfall);
        sigma.topLeftCorner(quantile_parameter_count,
                            quantile_parameter_count) +=
            quantile_regressor * quantile_regressor.transpose() *
            quantile_score_scale * quantile_score_scale *
            (1.0 - alpha) / alpha;
        sigma.bottomRightCorner(expected_shortfall_parameter_count,
                                expected_shortfall_parameter_count) +=
            expected_shortfall_regressor *
            expected_shortfall_regressor.transpose() *
            g2_prime * g2_prime *
            (truncated_variance / alpha +
             (1.0 - alpha) / alpha *
                 (quantile - expected_shortfall) *
                 (quantile - expected_shortfall));
        const Eigen::MatrixXd cross =
            expected_shortfall_regressor *
            quantile_regressor.transpose() * cross_scale;
        sigma.bottomLeftCorner(expected_shortfall_parameter_count,
                               quantile_parameter_count) += cross;
        sigma.topRightCorner(quantile_parameter_count,
                             expected_shortfall_parameter_count) +=
            cross.transpose();
    }
    lambda /= static_cast<double>(observation_count);
    sigma /= static_cast<double>(observation_count);
    if (!lambda.allFinite() || !sigma.allFinite()) return result;
    Eigen::LDLT<Eigen::MatrixXd> lambda_solver(lambda);
    if (lambda_solver.info() != Eigen::Success ||
        (lambda_solver.vectorD().array() <= 0.0).any()) {
        return result;
    }
    const Eigen::MatrixXd left = lambda_solver.solve(sigma);
    if (lambda_solver.info() != Eigen::Success || !left.allFinite()) {
        return result;
    }
    Eigen::MatrixXd covariance =
        lambda_solver.solve(left.transpose()).transpose() /
        static_cast<double>(observation_count);
    covariance = 0.5 * (covariance + covariance.transpose());
    if (lambda_solver.info() != Eigen::Success || !covariance.allFinite() ||
        (covariance.diagonal().array() <= 0.0).any()) {
        return result;
    }
    result.ok = true;
    result.covariance = std::move(covariance);
    result.density_at_quantile = density;
    result.truncated_residual_variance = truncated_variance;
    return result;
}

CovarianceResult estimate_misspecification_robust_hac_covariance(
    const EsrDesign& design, std::span<const double> parameters,
    std::uint32_t hac_lag) {
    CovarianceResult result;
    const std::size_t observation_count = design.response.size();
    const Eigen::Index quantile_parameter_count =
        design.quantile_regressors.cols();
    const Eigen::Index expected_shortfall_parameter_count =
        design.expected_shortfall_regressors.cols();
    const Eigen::Index parameter_count = quantile_parameter_count +
        expected_shortfall_parameter_count;
    if (hac_lag == 0 || hac_lag >= observation_count) return result;

    std::vector<double> residuals(observation_count);
    std::vector<double> negative_residuals;
    negative_residuals.reserve(observation_count);
    double residual_mean = 0.0;
    for (std::size_t row = 0; row < observation_count; ++row) {
        const double quantile = dot_row(
            design.quantile_regressors, row, parameters, 0);
        residuals[row] = design.response[row] - quantile;
        residual_mean += residuals[row];
        if (residuals[row] <= 0.0) {
            negative_residuals.push_back(residuals[row]);
        }
    }
    residual_mean /= static_cast<double>(observation_count);
    double residual_variance = 0.0;
    for (const double residual : residuals) {
        const double centered = residual - residual_mean;
        residual_variance += centered * centered;
    }
    residual_variance /= static_cast<double>(observation_count - 1);
    if (!(residual_variance > 0.0) || negative_residuals.size() < 3) {
        return result;
    }
    const double residual_standard_deviation = std::sqrt(residual_variance);
    const double bandwidth = std::max(
        design.response_scale * 1e-8,
        1.06 * residual_standard_deviation *
            std::pow(static_cast<double>(observation_count), -0.2));
    double density = 0.0;
    for (const double residual : residuals) {
        const double standardized = residual / bandwidth;
        density += std::exp(-0.5 * standardized * standardized) /
            (kSqrtTwoPi * bandwidth);
    }
    density /= static_cast<double>(observation_count);

    double negative_mean = 0.0;
    for (const double residual : negative_residuals) {
        negative_mean += residual;
    }
    negative_mean /= static_cast<double>(negative_residuals.size());
    double truncated_variance = 0.0;
    for (const double residual : negative_residuals) {
        const double centered = residual - negative_mean;
        truncated_variance += centered * centered;
    }
    truncated_variance /=
        static_cast<double>(negative_residuals.size() - 1);
    if (!(density > 0.0) || !std::isfinite(density) ||
        !(truncated_variance > 0.0) || !std::isfinite(truncated_variance)) {
        return result;
    }

    Eigen::MatrixXd bread = Eigen::MatrixXd::Zero(
        parameter_count, parameter_count);
    Eigen::MatrixXd scores(
        static_cast<Eigen::Index>(observation_count), parameter_count);
    const double alpha = design.lower_tail_probability;
    for (std::size_t row = 0; row < observation_count; ++row) {
        const Eigen::VectorXd quantile_regressor =
            design.quantile_regressors.row(
                static_cast<Eigen::Index>(row)).transpose();
        const Eigen::VectorXd expected_shortfall_regressor =
            design.expected_shortfall_regressors.row(
                static_cast<Eigen::Index>(row)).transpose();
        const double quantile = dot_row(
            design.quantile_regressors, row, parameters, 0);
        const double expected_shortfall = dot_row(
            design.expected_shortfall_regressors, row, parameters,
            static_cast<std::size_t>(quantile_parameter_count));
        if (expected_shortfall >= -design.domain_margin) return result;
        const double response = design.response[row];
        const double hit = response <= quantile ? 1.0 : 0.0;
        const double g2 = -1.0 / expected_shortfall;
        const double g2_prime = 1.0 /
            (expected_shortfall * expected_shortfall);
        const double g2_second = -2.0 /
            (expected_shortfall * expected_shortfall * expected_shortfall);
        const double tail_gap = expected_shortfall - quantile +
            (quantile - response) * hit / alpha;
        const double quantile_score_scale = g2 * (hit / alpha - 1.0);
        const double expected_shortfall_score_scale =
            g2_prime * tail_gap;
        scores.row(static_cast<Eigen::Index>(row)).head(
            quantile_parameter_count) =
            (quantile_regressor * quantile_score_scale).transpose();
        scores.row(static_cast<Eigen::Index>(row)).tail(
            expected_shortfall_parameter_count) =
            (expected_shortfall_regressor *
             expected_shortfall_score_scale).transpose();

        bread.topLeftCorner(quantile_parameter_count,
                            quantile_parameter_count) +=
            quantile_regressor * quantile_regressor.transpose() *
            (g2 / alpha) * density;
        const double cross_scale = g2_prime * (hit / alpha - 1.0);
        const Eigen::MatrixXd cross = quantile_regressor *
            expected_shortfall_regressor.transpose() * cross_scale;
        bread.topRightCorner(quantile_parameter_count,
                             expected_shortfall_parameter_count) += cross;
        bread.bottomLeftCorner(expected_shortfall_parameter_count,
                               quantile_parameter_count) += cross.transpose();
        bread.bottomRightCorner(expected_shortfall_parameter_count,
                                expected_shortfall_parameter_count) +=
            expected_shortfall_regressor *
            expected_shortfall_regressor.transpose() *
            (g2_prime + g2_second * tail_gap);
    }
    bread /= static_cast<double>(observation_count);
    bread = 0.5 * (bread + bread.transpose());
    if (!bread.allFinite() || !scores.allFinite()) return result;

    const Eigen::RowVectorXd score_mean = scores.colwise().mean();
    scores.rowwise() -= score_mean;
    Eigen::MatrixXd meat = scores.transpose() * scores /
        static_cast<double>(observation_count);
    for (std::uint32_t lag = 1; lag <= hac_lag; ++lag) {
        const Eigen::Index retained = static_cast<Eigen::Index>(
            observation_count - lag);
        const Eigen::MatrixXd lagged_covariance =
            scores.bottomRows(retained).transpose() *
            scores.topRows(retained) /
            static_cast<double>(observation_count);
        const double bartlett_weight = 1.0 -
            static_cast<double>(lag) / static_cast<double>(hac_lag + 1);
        meat += bartlett_weight *
            (lagged_covariance + lagged_covariance.transpose());
    }
    meat = 0.5 * (meat + meat.transpose());
    if (!meat.allFinite()) return result;

    Eigen::LDLT<Eigen::MatrixXd> bread_solver(bread);
    if (bread_solver.info() != Eigen::Success ||
        (bread_solver.vectorD().array() <= 0.0).any()) {
        return result;
    }
    const Eigen::MatrixXd left = bread_solver.solve(meat);
    if (bread_solver.info() != Eigen::Success || !left.allFinite()) {
        return result;
    }
    Eigen::MatrixXd covariance =
        bread_solver.solve(left.transpose()).transpose() /
        static_cast<double>(observation_count);
    covariance = 0.5 * (covariance + covariance.transpose());
    if (bread_solver.info() != Eigen::Success || !covariance.allFinite() ||
        (covariance.diagonal().array() <= 0.0).any()) {
        return result;
    }
    result.ok = true;
    result.covariance = std::move(covariance);
    result.density_at_quantile = density;
    result.truncated_residual_variance = truncated_variance;
    return result;
}

TailRiskEsrVariantResult fit_variant(
    const TailRiskEsrProblemView& problem, TailRiskEsrVariant variant,
    std::uint64_t input_hash) {
    TailRiskEsrVariantResult result;
    result.variant = variant;
    EsrDesign design = make_design(problem, variant);
    const OptimizationResult optimization = minimize_joint_fz(
        design, problem.spec);
    if (!optimization.converged || optimization.parameters.empty() ||
        !std::isfinite(optimization.objective)) {
        result.status = TailRiskEsrStatus::OPTIMIZATION_FAILURE;
        return result;
    }
    const CovarianceResult covariance = problem.spec.covariance_kind ==
            TailRiskEsrCovarianceKind::MISSPECIFICATION_ROBUST_HAC
        ? estimate_misspecification_robust_hac_covariance(
              design, optimization.parameters, problem.spec.hac_lag)
        : estimate_iid_ind_covariance(design, optimization.parameters);
    if (!covariance.ok) {
        result.status = TailRiskEsrStatus::COVARIANCE_FAILURE;
        return result;
    }
    const Eigen::Index quantile_parameter_count =
        design.quantile_regressors.cols();
    const Eigen::Index expected_shortfall_parameter_count =
        design.expected_shortfall_regressors.cols();
    result.quantile_coefficients.assign(
        optimization.parameters.begin(),
        optimization.parameters.begin() + quantile_parameter_count);
    result.expected_shortfall_coefficients.assign(
        optimization.parameters.begin() + quantile_parameter_count,
        optimization.parameters.end());
    result.quantile_coefficients.front() += design.response_shift;
    result.expected_shortfall_coefficients.front() += design.response_shift;
    result.objective = optimization.objective;
    result.density_at_quantile = covariance.density_at_quantile;
    result.truncated_residual_variance =
        covariance.truncated_residual_variance;
    result.optimization_iterations = optimization.iterations;

    const Eigen::MatrixXd expected_shortfall_covariance =
        covariance.covariance.bottomRightCorner(
            expected_shortfall_parameter_count,
            expected_shortfall_parameter_count);
    if (variant == TailRiskEsrVariant::STRICT_INTERCEPT) {
        const double variance = expected_shortfall_covariance(0, 0);
        if (!(variance > 0.0) || !std::isfinite(variance)) {
            result.status = TailRiskEsrStatus::COVARIANCE_FAILURE;
            return result;
        }
        const double statistic = result.expected_shortfall_coefficients[0] /
            std::sqrt(variance);
        result.wald_statistic = statistic;
        result.two_sided_p_value = std::erfc(
            std::abs(statistic) / kSqrtTwo);
        result.one_sided_underestimation_p_value =
            0.5 * std::erfc(-statistic / kSqrtTwo);
    } else {
        Eigen::Vector2d deviation;
        deviation << result.expected_shortfall_coefficients[0],
            result.expected_shortfall_coefficients[1] - 1.0;
        Eigen::LDLT<Eigen::Matrix2d> covariance_solver(
            expected_shortfall_covariance);
        if (covariance_solver.info() != Eigen::Success ||
            (covariance_solver.vectorD().array() <= 0.0).any()) {
            result.status = TailRiskEsrStatus::COVARIANCE_FAILURE;
            return result;
        }
        const Eigen::Vector2d solved = covariance_solver.solve(deviation);
        const double statistic = std::max(0.0, deviation.dot(solved));
        if (!std::isfinite(statistic)) {
            result.status = TailRiskEsrStatus::COVARIANCE_FAILURE;
            return result;
        }
        result.wald_statistic = statistic;
        result.two_sided_p_value = std::exp(-0.5 * statistic);
    }
    if (!std::isfinite(result.two_sided_p_value) ||
        result.two_sided_p_value < 0.0 || result.two_sided_p_value > 1.0 ||
        (result.one_sided_underestimation_p_value &&
         (!std::isfinite(*result.one_sided_underestimation_p_value) ||
          *result.one_sided_underestimation_p_value < 0.0 ||
          *result.one_sided_underestimation_p_value > 1.0))) {
        result.status = TailRiskEsrStatus::COVARIANCE_FAILURE;
        return result;
    }
    result.status = TailRiskEsrStatus::OK;
    result.artifact_hash = input_hash;
    hash_value(result.artifact_hash, static_cast<std::uint64_t>(variant));
    for (const double coefficient : result.quantile_coefficients) {
        hash_value(result.artifact_hash,
                   std::bit_cast<std::uint64_t>(coefficient));
    }
    for (const double coefficient : result.expected_shortfall_coefficients) {
        hash_value(result.artifact_hash,
                   std::bit_cast<std::uint64_t>(coefficient));
    }
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(result.objective));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(result.wald_statistic));
    hash_value(result.artifact_hash,
               std::bit_cast<std::uint64_t>(result.two_sided_p_value));
    return result;
}

std::string serialize_coefficients(std::span<const double> coefficients) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < coefficients.size(); ++index) {
        if (index != 0) output << ',';
        output << json_number(coefficients[index]);
    }
    output << ']';
    return output.str();
}

std::string serialize_variant(const TailRiskEsrVariantResult& result) {
    std::ostringstream output;
    output << "{\"status\":" << static_cast<int>(result.status)
           << ",\"variant\":" << static_cast<int>(result.variant);
    if (result.status == TailRiskEsrStatus::OK) {
        output << ",\"quantile_coefficients\":"
               << serialize_coefficients(result.quantile_coefficients)
               << ",\"expected_shortfall_coefficients\":"
               << serialize_coefficients(
                      result.expected_shortfall_coefficients)
               << ",\"objective\":" << json_number(result.objective)
               << ",\"density_at_quantile\":"
               << json_number(result.density_at_quantile)
               << ",\"truncated_residual_variance\":"
               << json_number(result.truncated_residual_variance)
               << ",\"wald_statistic\":"
               << json_number(result.wald_statistic)
               << ",\"two_sided_p_value\":"
               << json_number(result.two_sided_p_value)
               << ",\"one_sided_underestimation_p_value\":";
        if (result.one_sided_underestimation_p_value) {
            output << json_number(
                *result.one_sided_underestimation_p_value);
        } else {
            output << "null";
        }
        output << ",\"optimization_iterations\":"
               << result.optimization_iterations
               << ",\"artifact_hash\":" << result.artifact_hash;
    } else {
        output << ",\"quantile_coefficients\":null"
               << ",\"expected_shortfall_coefficients\":null"
               << ",\"objective\":null,\"density_at_quantile\":null"
               << ",\"truncated_residual_variance\":null"
               << ",\"wald_statistic\":null"
               << ",\"two_sided_p_value\":null"
               << ",\"one_sided_underestimation_p_value\":null"
               << ",\"optimization_iterations\":"
               << result.optimization_iterations
               << ",\"artifact_hash\":0";
    }
    output << '}';
    return output.str();
}

}  // namespace

TailRiskEsrBacktestResult backtest_tail_risk_esr(
    const TailRiskEsrProblemView& problem) {
    TailRiskEsrBacktestResult result;
    result.confidence_level = problem.spec.confidence_level;
    result.strict.variant = TailRiskEsrVariant::STRICT;
    result.auxiliary.variant = TailRiskEsrVariant::AUXILIARY;
    result.strict_intercept.variant = TailRiskEsrVariant::STRICT_INTERCEPT;
    result.covariance_kind = problem.spec.covariance_kind;
    result.hac_lag = problem.spec.hac_lag;
    const std::size_t observation_count = problem.realized_returns.size();
    const bool robust_covariance = problem.spec.covariance_kind ==
        TailRiskEsrCovarianceKind::MISSPECIFICATION_ROBUST_HAC;
    const bool correct_spec_covariance = problem.spec.covariance_kind ==
        TailRiskEsrCovarianceKind::CORRECT_SPEC_IID;
    if (!(problem.spec.confidence_level >= 0.5 &&
          problem.spec.confidence_level < 1.0) ||
        problem.spec.minimum_observations < 20 ||
        problem.spec.maximum_iterations < 100 ||
        !std::isfinite(problem.spec.convergence_tolerance) ||
        !(problem.spec.convergence_tolerance > 0.0) ||
        (!robust_covariance && !correct_spec_covariance) ||
        (robust_covariance && problem.spec.hac_lag == 0) ||
        (correct_spec_covariance && problem.spec.hac_lag != 0) ||
        problem.spec.regression_spec_hash == 0 ||
        problem.spec.covariance_spec_hash == 0 ||
        problem.spec.config_hash == 0 || problem.available_at <= 0 ||
        problem.realization_timestamps.size() != observation_count ||
        problem.value_at_risk_loss.size() != observation_count ||
        problem.expected_shortfall_loss.size() != observation_count) {
        return result;
    }
    if (observation_count < problem.spec.minimum_observations) {
        result.status = TailRiskEsrStatus::INSUFFICIENT_OBSERVATIONS;
        return result;
    }
    if (problem.spec.hac_lag >= observation_count) {
        result.status = TailRiskEsrStatus::INVALID_INPUT;
        return result;
    }
    std::uint64_t input_hash = kFnvOffset;
    hash_value(input_hash, problem.spec.regression_spec_hash);
    hash_value(input_hash, problem.spec.covariance_spec_hash);
    hash_value(input_hash, problem.spec.config_hash);
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.confidence_level));
    hash_value(input_hash, problem.spec.minimum_observations);
    hash_value(input_hash, problem.spec.maximum_iterations);
    hash_value(input_hash, problem.spec.hac_lag);
    hash_value(input_hash,
               static_cast<std::uint64_t>(problem.spec.covariance_kind));
    hash_value(input_hash, std::bit_cast<std::uint64_t>(
        problem.spec.convergence_tolerance));
    for (std::size_t index = 0; index < observation_count; ++index) {
        if (problem.realization_timestamps[index] <= 0 ||
            (index != 0 && problem.realization_timestamps[index] <=
                problem.realization_timestamps[index - 1]) ||
            problem.realization_timestamps[index] > problem.available_at ||
            !std::isfinite(problem.realized_returns[index]) ||
            !std::isfinite(problem.value_at_risk_loss[index]) ||
            !std::isfinite(problem.expected_shortfall_loss[index]) ||
            !(problem.expected_shortfall_loss[index] > 0.0) ||
            problem.expected_shortfall_loss[index] <
                problem.value_at_risk_loss[index]) {
            result.status =
                std::isfinite(problem.expected_shortfall_loss[index]) &&
                !(problem.expected_shortfall_loss[index] > 0.0)
                    ? TailRiskEsrStatus::DOMAIN_FAILURE
                    : TailRiskEsrStatus::INVALID_INPUT;
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
    result.effective_observations = static_cast<std::uint32_t>(
        observation_count);
    result.input_hash = input_hash;
    result.strict = fit_variant(
        problem, TailRiskEsrVariant::STRICT, input_hash);
    result.auxiliary = fit_variant(
        problem, TailRiskEsrVariant::AUXILIARY, input_hash);
    result.strict_intercept = fit_variant(
        problem, TailRiskEsrVariant::STRICT_INTERCEPT, input_hash);
    result.status = result.strict.status;
    if (result.status == TailRiskEsrStatus::OK) {
        result.status = result.auxiliary.status;
    }
    if (result.status == TailRiskEsrStatus::OK) {
        result.status = result.strict_intercept.status;
    }
    result.artifact_hash = input_hash;
    hash_value(result.artifact_hash,
               static_cast<std::uint64_t>(result.status));
    hash_value(result.artifact_hash, result.strict.artifact_hash);
    hash_value(result.artifact_hash, result.auxiliary.artifact_hash);
    hash_value(result.artifact_hash, result.strict_intercept.artifact_hash);
    return result;
}

std::string serialize_tail_risk_esr_backtest_artifact(
    const TailRiskEsrBacktestResult& result,
    const TailRiskArtifactSpec& artifact_spec) {
    const bool proxy = artifact_spec.reference_price_quality == "PROXY" ||
        artifact_spec.reference_price_quality == "ARRIVAL_PROXY";
    const bool promotion = artifact_spec.promotion_eligible &&
        reference_price_ready(artifact_spec.reference_price_quality) &&
        !proxy && result.status == TailRiskEsrStatus::OK;
    std::ostringstream output;
    output << "{\"schema_version\":1,\"role\":\"tail_risk_esr_backtest\""
           << ",\"status\":" << static_cast<int>(result.status)
           << ",\"confidence_level\":"
           << json_number(result.confidence_level)
           << ",\"effective_observations\":"
           << result.effective_observations
           << ",\"regression_method\":"
              "\"JOINT_FZ_G1_ZERO_G2_NEGATIVE_RECIPROCAL\""
           << ",\"covariance_method\":\""
           << (result.covariance_kind ==
                       TailRiskEsrCovarianceKind::MISSPECIFICATION_ROBUST_HAC
                   ? "KERNEL_BREAD_EMPIRICAL_SCORE_NEWEY_WEST_HAC_V1"
                   : "KERNEL_IND_CORRECT_SPEC_SANDWICH_V1")
           << "\""
           << ",\"hac_lag\":" << result.hac_lag
           << ",\"strict\":" << serialize_variant(result.strict)
           << ",\"auxiliary\":" << serialize_variant(result.auxiliary)
           << ",\"strict_intercept\":"
           << serialize_variant(result.strict_intercept)
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
           << (promotion ? "true" : "false")
           << ",\"limitations\":[";
    for (std::size_t index = 0; index < artifact_spec.limitations.size();
         ++index) {
        if (index != 0) output << ',';
        output << '"' << json_escape(artifact_spec.limitations[index]) << '"';
    }
    output << "]}}";
    return output.str();
}

}  // namespace portfolio_math

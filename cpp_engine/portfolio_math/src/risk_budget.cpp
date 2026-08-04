#include "portfolio_math/risk_budget.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace portfolio_math {
namespace {

using quant_math::DenseMatrix;
using quant_math::DenseVector;

DenseMatrix copy_matrix(quant_math::MatrixView matrix) {
    DenseMatrix result(static_cast<Eigen::Index>(matrix.rows),
                       static_cast<Eigen::Index>(matrix.cols));
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        for (std::size_t col = 0; col < matrix.cols; ++col) {
            result(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
                matrix(row, col);
        }
    }
    return result;
}

bool all_finite_nonnegative(std::span<const double> values) {
    return std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value) && value >= 0.0;
    });
}

DenseVector project_bounded_simplex(
    const DenseVector& values,
    const DenseVector& lower,
    const DenseVector& upper,
    double target) {
    double left = (values - upper).minCoeff();
    double right = (values - lower).maxCoeff();
    DenseVector projected(values.size());
    for (int iteration = 0; iteration < 100; ++iteration) {
        const double shift = 0.5 * (left + right);
        projected = (values.array() - shift).max(lower.array()).min(upper.array());
        if (projected.sum() > target) {
            left = shift;
        } else {
            right = shift;
        }
    }
    projected = (values.array() - 0.5 * (left + right))
                    .max(lower.array()).min(upper.array());
    return projected;
}

struct BudgetObjective {
    bool ok{false};
    double value{0.0};
    DenseVector gradient;
    DenseVector shares;
};

BudgetObjective budget_objective(
    const DenseMatrix& sigma, const DenseVector& weights, const DenseVector& budgets) {
    const DenseVector marginal = sigma * weights;
    const double variance = weights.dot(marginal);
    if (!(variance > 0.0) || !std::isfinite(variance)) return {};
    const DenseVector numerators = weights.array() * marginal.array();
    const DenseVector shares = numerators / variance;
    const DenseVector errors = shares - budgets;
    DenseVector gradient = DenseVector::Zero(weights.size());
    for (Eigen::Index asset = 0; asset < weights.size(); ++asset) {
        for (Eigen::Index coordinate = 0; coordinate < weights.size(); ++coordinate) {
            const double numerator_derivative =
                (asset == coordinate ? marginal(asset) : 0.0) +
                weights(asset) * sigma(asset, coordinate);
            const double share_derivative = numerator_derivative / variance -
                numerators(asset) * 2.0 * marginal(coordinate) /
                    (variance * variance);
            gradient(coordinate) += errors(asset) * share_derivative;
        }
    }
    if (!std::isfinite(gradient.squaredNorm())) return {};
    return {true, 0.5 * errors.squaredNorm(), std::move(gradient), shares};
}

RiskBudgetResult solve_impl(
    quant_math::MatrixView covariance,
    std::span<const double> risk_budgets,
    std::span<const double> lower_bounds,
    std::span<const double> upper_bounds,
    std::span<const double> current_weights,
    std::span<const double> warm_start,
    RiskBudgetOptions options) {
    RiskBudgetResult result;
    result.diagnostics.status = OptimizationStatus::INVALID_INPUT;
    const std::size_t asset_count = covariance.rows;
    const bool bounded = !lower_bounds.empty() || !upper_bounds.empty();
    if (asset_count == 0 || covariance.cols != asset_count ||
        risk_budgets.size() != asset_count ||
        (bounded && (lower_bounds.size() != asset_count ||
                     upper_bounds.size() != asset_count)) ||
        (!current_weights.empty() && current_weights.size() != asset_count) ||
        (!warm_start.empty() && warm_start.size() != asset_count) ||
        !all_finite_nonnegative(risk_budgets) ||
        (!current_weights.empty() && !all_finite_nonnegative(current_weights)) ||
        (!warm_start.empty() && !all_finite_nonnegative(warm_start)) ||
        (bounded && (!all_finite_nonnegative(lower_bounds) ||
                     !all_finite_nonnegative(upper_bounds))) ||
        options.max_iterations == 0 || !(options.tolerance > 0.0) ||
        !(options.covariance_tolerance >= 0.0) || !(options.target_investment > 0.0) ||
        options.target_investment > 1.0 ||
        !quant_math::validate_symmetric(covariance, options.covariance_tolerance).ok) {
        return result;
    }
    const double budget_sum = std::accumulate(risk_budgets.begin(), risk_budgets.end(), 0.0);
    if (!(budget_sum > 0.0) || std::abs(budget_sum - 1.0) > options.tolerance * 10.0) {
        return result;
    }
    DenseVector lower = DenseVector::Zero(static_cast<Eigen::Index>(asset_count));
    DenseVector upper = DenseVector::Ones(static_cast<Eigen::Index>(asset_count));
    if (bounded) {
        lower = Eigen::Map<const DenseVector>(lower_bounds.data(),
                                              static_cast<Eigen::Index>(asset_count));
        upper = Eigen::Map<const DenseVector>(upper_bounds.data(),
                                              static_cast<Eigen::Index>(asset_count));
        if ((lower.array() > upper.array()).any() ||
            lower.sum() > options.target_investment + options.tolerance ||
            upper.sum() < options.target_investment - options.tolerance) {
            result.diagnostics.status = OptimizationStatus::INFEASIBLE;
            return result;
        }
    }
    const DenseMatrix sigma = copy_matrix(covariance);
    if (!quant_math::is_positive_semidefinite(sigma, options.covariance_tolerance)) {
        result.diagnostics.status = OptimizationStatus::NON_PSD_RISK_MODEL;
        return result;
    }
    if ((sigma.diagonal().array() <= options.covariance_tolerance).any()) {
        result.diagnostics.status = OptimizationStatus::NUMERICAL_FAILURE;
        return result;
    }

    DenseVector weights(static_cast<Eigen::Index>(asset_count));
    if (warm_start.empty()) {
        for (std::size_t index = 0; index < asset_count; ++index) {
            weights(static_cast<Eigen::Index>(index)) =
                std::sqrt(risk_budgets[index] /
                          sigma(static_cast<Eigen::Index>(index),
                                static_cast<Eigen::Index>(index)));
        }
    } else {
        weights = Eigen::Map<const DenseVector>(warm_start.data(),
                                                static_cast<Eigen::Index>(asset_count));
        weights = weights.cwiseMax(std::numeric_limits<double>::epsilon());
    }

    double maximum_change = std::numeric_limits<double>::infinity();
    for (std::uint32_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
        maximum_change = 0.0;
        for (std::size_t index = 0; index < asset_count; ++index) {
            const Eigen::Index row = static_cast<Eigen::Index>(index);
            if (risk_budgets[index] == 0.0) {
                maximum_change = std::max(maximum_change, std::abs(weights(row)));
                weights(row) = 0.0;
                continue;
            }
            const double diagonal = sigma(row, row);
            const double cross = sigma.row(row).dot(weights) - diagonal * weights(row);
            const double discriminant = cross * cross + 4.0 * diagonal * risk_budgets[index];
            if (!(discriminant >= 0.0) || !std::isfinite(discriminant)) {
                result.diagnostics.status = OptimizationStatus::NUMERICAL_FAILURE;
                return result;
            }
            const double root = std::sqrt(discriminant);
            const double updated = cross >= 0.0
                ? 2.0 * risk_budgets[index] / (root + cross)
                : (root - cross) / (2.0 * diagonal);
            if (!(updated > 0.0) || !std::isfinite(updated)) {
                result.diagnostics.status = OptimizationStatus::NUMERICAL_FAILURE;
                return result;
            }
            maximum_change = std::max(maximum_change, std::abs(updated - weights(row)));
            weights(row) = updated;
        }
        result.diagnostics.iterations = iteration;
        if (maximum_change <= options.tolerance * std::max(1.0, weights.maxCoeff())) break;
    }
    if (maximum_change > options.tolerance * std::max(1.0, weights.maxCoeff())) {
        result.diagnostics.status = OptimizationStatus::MAX_ITERATIONS;
        return result;
    }

    weights *= options.target_investment / weights.sum();
    double projected_residual = 0.0;
    if (bounded && ((weights.array() < lower.array() - options.tolerance).any() ||
                    (weights.array() > upper.array() + options.tolerance).any())) {
        weights = project_bounded_simplex(weights, lower, upper, options.target_investment);
        auto objective = budget_objective(
            sigma, weights,
            Eigen::Map<const DenseVector>(risk_budgets.data(),
                                          static_cast<Eigen::Index>(asset_count)));
        if (!objective.ok) {
            result.diagnostics.status = OptimizationStatus::NUMERICAL_FAILURE;
            return result;
        }
        const double convergence_tolerance = std::sqrt(options.tolerance);
        for (std::uint32_t iteration = 1; iteration <= options.max_iterations; ++iteration) {
            const DenseVector projected_gradient = project_bounded_simplex(
                weights - objective.gradient, lower, upper, options.target_investment);
            projected_residual = (projected_gradient - weights).cwiseAbs().maxCoeff();
            result.diagnostics.iterations += 1;
            if (projected_residual <= convergence_tolerance) break;
            double step = 1.0;
            bool accepted = false;
            for (int line_search = 0; line_search < 40; ++line_search) {
                const DenseVector candidate = project_bounded_simplex(
                    weights - step * objective.gradient, lower, upper,
                    options.target_investment);
                auto candidate_objective = budget_objective(
                    sigma, candidate,
                    Eigen::Map<const DenseVector>(risk_budgets.data(),
                                                  static_cast<Eigen::Index>(asset_count)));
                if (candidate_objective.ok &&
                    candidate_objective.value <= objective.value -
                        1e-4 * (weights - candidate).squaredNorm() / step) {
                    weights = candidate;
                    objective = std::move(candidate_objective);
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) {
                result.diagnostics.status = OptimizationStatus::NUMERICAL_FAILURE;
                return result;
            }
        }
        if (projected_residual > convergence_tolerance) {
            result.diagnostics.status = OptimizationStatus::MAX_ITERATIONS;
            return result;
        }
    }

    result.weights.assign(weights.data(), weights.data() + weights.size());
    const auto contributions = risk_contributions(covariance, result.weights);
    if (contributions.status != OptimizationStatus::OK) {
        result.weights.clear();
        result.diagnostics.status = contributions.status;
        return result;
    }
    double maximum_budget_error = 0.0;
    for (std::size_t index = 0; index < asset_count; ++index) {
        maximum_budget_error = std::max(
            maximum_budget_error,
            std::abs(contributions.contribution_shares[index] - risk_budgets[index]));
        if (bounded && (std::abs(weights(static_cast<Eigen::Index>(index)) - lower(static_cast<Eigen::Index>(index))) <=
                            std::sqrt(options.tolerance) ||
                        std::abs(weights(static_cast<Eigen::Index>(index)) - upper(static_cast<Eigen::Index>(index))) <=
                            std::sqrt(options.tolerance))) {
            ++result.diagnostics.active_bound_count;
        }
    }
    result.diagnostics.max_risk_budget_error = maximum_budget_error;
    result.diagnostics.kkt_residual = bounded ? projected_residual : maximum_budget_error;
    result.diagnostics.predicted_risk = contributions.portfolio_volatility;
    if (!current_weights.empty()) {
        for (std::size_t index = 0; index < asset_count; ++index) {
            result.diagnostics.turnover += std::abs(result.weights[index] - current_weights[index]);
        }
        result.diagnostics.turnover *= 0.5;
    }
    result.diagnostics.status = (!bounded && maximum_budget_error > std::sqrt(options.tolerance))
        ? OptimizationStatus::NUMERICAL_FAILURE
        : OptimizationStatus::OK;
    if (result.diagnostics.status != OptimizationStatus::OK) result.weights.clear();
    return result;
}

}  // namespace

RiskContributionResult risk_contributions(
    quant_math::MatrixView covariance, std::span<const double> weights) {
    if (covariance.rows == 0 || covariance.rows != covariance.cols ||
        covariance.rows != weights.size() || !all_finite_nonnegative(weights) ||
        !quant_math::validate_symmetric(covariance, 1e-10).ok) {
        return {};
    }
    const DenseMatrix sigma = copy_matrix(covariance);
    if (!quant_math::is_positive_semidefinite(sigma, 1e-10)) {
        return {OptimizationStatus::NON_PSD_RISK_MODEL, 0.0, {}, {}};
    }
    const Eigen::Map<const DenseVector> weight_vector(weights.data(),
                                                       static_cast<Eigen::Index>(weights.size()));
    const DenseVector marginal = sigma * weight_vector;
    const double variance = weight_vector.dot(marginal);
    if (!(variance > 0.0) || !std::isfinite(variance)) {
        return {OptimizationStatus::NUMERICAL_FAILURE, 0.0, {}, {}};
    }
    const double volatility = std::sqrt(variance);
    std::vector<double> contributions(weights.size());
    std::vector<double> shares(weights.size());
    for (std::size_t index = 0; index < weights.size(); ++index) {
        contributions[index] = weights[index] * marginal(static_cast<Eigen::Index>(index)) /
                               volatility;
        shares[index] = contributions[index] / volatility;
    }
    return {OptimizationStatus::OK, volatility, std::move(contributions), std::move(shares)};
}

RiskBudgetResult solve_long_only_risk_budget(
    quant_math::MatrixView covariance,
    std::span<const double> risk_budgets,
    std::span<const double> current_weights,
    std::span<const double> warm_start,
    RiskBudgetOptions options) {
    return solve_impl(covariance, risk_budgets, {}, {}, current_weights, warm_start, options);
}

RiskBudgetResult solve_bounded_long_only_risk_budget(
    quant_math::MatrixView covariance,
    std::span<const double> risk_budgets,
    std::span<const double> lower_bounds,
    std::span<const double> upper_bounds,
    std::span<const double> current_weights,
    std::span<const double> warm_start,
    RiskBudgetOptions options) {
    return solve_impl(covariance, risk_budgets, lower_bounds, upper_bounds,
                      current_weights, warm_start, options);
}

}  // namespace portfolio_math

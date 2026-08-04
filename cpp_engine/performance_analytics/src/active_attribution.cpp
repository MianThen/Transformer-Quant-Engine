#include "performance_analytics/active_attribution.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace performance_analytics {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_value(std::uint64_t& hash, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffU;
        hash *= kFnvPrime;
    }
}

void hash_double(std::uint64_t& hash, double value) {
    const double normalized = value == 0.0 ? 0.0 : value;
    hash_value(hash, std::bit_cast<std::uint64_t>(normalized));
}

bool valid_group_kind(AttributionGroupKind kind) {
    return kind == AttributionGroupKind::EQUITY_INDUSTRY ||
        kind == AttributionGroupKind::CASH ||
        kind == AttributionGroupKind::UNMAPPED;
}

bool valid_effect_kind(AttributionEffectKind kind) {
    return kind == AttributionEffectKind::ALLOCATION ||
        kind == AttributionEffectKind::SELECTION ||
        kind == AttributionEffectKind::INTERACTION ||
        kind == AttributionEffectKind::OTHER;
}

bool within_tolerance(double residual, double scale, double tolerance) {
    return std::abs(residual) <= tolerance * std::max(1.0, std::abs(scale));
}

void hash_effect_key(std::uint64_t& hash, const AttributionEffectKey& key) {
    hash_value(hash, static_cast<std::uint8_t>(key.group_kind));
    hash_value(hash, key.group_id);
    hash_value(hash, static_cast<std::uint8_t>(key.effect_kind));
}

}  // namespace

bool AttributionEffectKey::operator<(
    const AttributionEffectKey& other) const noexcept {
    return std::tie(group_kind, group_id, effect_kind) <
        std::tie(other.group_kind, other.group_id, other.effect_kind);
}

BrinsonResult compute_brinson_fachler(const BrinsonProblem& problem) {
    BrinsonResult result;
    result.portfolio_return = problem.portfolio_return;
    result.benchmark_return = problem.benchmark_return;
    result.active_return = problem.portfolio_return - problem.benchmark_return;
    if (!std::isfinite(problem.portfolio_return) ||
        !std::isfinite(problem.benchmark_return) ||
        !std::isfinite(problem.spec.reconciliation_tolerance) ||
        problem.spec.reconciliation_tolerance <= 0.0 ||
        problem.spec.config_hash == 0) {
        return result;
    }
    if (!problem.benchmark_holdings_available) {
        result.status = AttributionStatus::UNAVAILABLE;
        return result;
    }
    if (problem.benchmark_provenance_hash == 0 ||
        problem.pit_classification_hash == 0 || problem.groups.empty()) {
        return result;
    }

    std::vector<BrinsonGroupInput> groups(problem.groups.begin(),
                                          problem.groups.end());
    std::sort(groups.begin(), groups.end(), [](const auto& left, const auto& right) {
        return std::tie(left.group_kind, left.group_id) <
            std::tie(right.group_kind, right.group_id);
    });
    double portfolio_weight_total = 0.0;
    double benchmark_weight_total = 0.0;
    double reconstructed_portfolio_return = 0.0;
    double reconstructed_benchmark_return = 0.0;
    for (std::size_t index = 0; index < groups.size(); ++index) {
        const auto& group = groups[index];
        const double values[] = {
            group.portfolio_weight, group.benchmark_weight,
            group.portfolio_return, group.benchmark_return,
        };
        if (!valid_group_kind(group.group_kind) ||
            !std::all_of(std::begin(values), std::end(values),
                         [](double value) { return std::isfinite(value); }) ||
            group.portfolio_return <= -1.0 || group.benchmark_return <= -1.0 ||
            (index > 0 && group.group_kind == groups[index - 1].group_kind &&
             group.group_id == groups[index - 1].group_id)) {
            return result;
        }
        portfolio_weight_total += group.portfolio_weight;
        benchmark_weight_total += group.benchmark_weight;
        reconstructed_portfolio_return +=
            group.portfolio_weight * group.portfolio_return;
        reconstructed_benchmark_return +=
            group.benchmark_weight * group.benchmark_return;
    }
    const double tolerance = problem.spec.reconciliation_tolerance;
    if (!within_tolerance(portfolio_weight_total - 1.0, 1.0, tolerance) ||
        !within_tolerance(benchmark_weight_total - 1.0, 1.0, tolerance) ||
        !within_tolerance(reconstructed_portfolio_return - problem.portfolio_return,
                          problem.portfolio_return, tolerance) ||
        !within_tolerance(reconstructed_benchmark_return - problem.benchmark_return,
                          problem.benchmark_return, tolerance)) {
        result.status = AttributionStatus::RECONCILIATION_FAILURE;
        return result;
    }

    result.groups.reserve(groups.size());
    for (const auto& group : groups) {
        const double allocation =
            (group.portfolio_weight - group.benchmark_weight) *
            (group.benchmark_return - problem.benchmark_return);
        const double selection = group.benchmark_weight *
            (group.portfolio_return - group.benchmark_return);
        const double interaction =
            (group.portfolio_weight - group.benchmark_weight) *
            (group.portfolio_return - group.benchmark_return);
        BrinsonGroupEffect effect;
        effect.group_kind = group.group_kind;
        effect.group_id = group.group_id;
        if (group.group_kind == AttributionGroupKind::EQUITY_INDUSTRY) {
            effect.allocation = allocation;
            effect.selection = selection;
            effect.interaction = interaction;
        } else {
            effect.other = allocation + selection + interaction;
        }
        effect.total = allocation + selection + interaction;
        result.effect_total += effect.total;
        result.groups.push_back(effect);
    }
    result.reconciliation_residual = result.active_return - result.effect_total;
    if (!within_tolerance(result.reconciliation_residual,
                          result.active_return, tolerance)) {
        result.status = AttributionStatus::RECONCILIATION_FAILURE;
        result.groups.clear();
        return result;
    }

    result.artifact_hash = kFnvOffset;
    hash_value(result.artifact_hash, problem.spec.config_hash);
    hash_value(result.artifact_hash, problem.benchmark_provenance_hash);
    hash_value(result.artifact_hash, problem.pit_classification_hash);
    hash_double(result.artifact_hash, problem.portfolio_return);
    hash_double(result.artifact_hash, problem.benchmark_return);
    for (const auto& group : groups) {
        hash_value(result.artifact_hash, static_cast<std::uint8_t>(group.group_kind));
        hash_value(result.artifact_hash, group.group_id);
        hash_double(result.artifact_hash, group.portfolio_weight);
        hash_double(result.artifact_hash, group.benchmark_weight);
        hash_double(result.artifact_hash, group.portfolio_return);
        hash_double(result.artifact_hash, group.benchmark_return);
    }
    result.status = AttributionStatus::OK;
    return result;
}

std::vector<AttributionEffectValue> brinson_effect_values(
    const BrinsonResult& result) {
    if (result.status != AttributionStatus::OK) return {};
    std::vector<AttributionEffectValue> effects;
    for (const auto& group : result.groups) {
        if (group.group_kind == AttributionGroupKind::EQUITY_INDUSTRY) {
            effects.push_back({{group.group_kind, group.group_id,
                                AttributionEffectKind::ALLOCATION},
                               group.allocation});
            effects.push_back({{group.group_kind, group.group_id,
                                AttributionEffectKind::SELECTION},
                               group.selection});
            effects.push_back({{group.group_kind, group.group_id,
                                AttributionEffectKind::INTERACTION},
                               group.interaction});
        } else {
            effects.push_back({{group.group_kind, group.group_id,
                                AttributionEffectKind::OTHER},
                               group.other});
        }
    }
    return effects;
}

MencheroResult menchero_link(std::span<const MencheroPeriodInput> periods,
                             const MencheroSpec& spec) {
    MencheroResult result;
    if (periods.empty() || !std::isfinite(spec.equality_tolerance) ||
        !std::isfinite(spec.degeneracy_tolerance) ||
        !std::isfinite(spec.reconciliation_tolerance) ||
        spec.equality_tolerance <= 0.0 || spec.degeneracy_tolerance < 0.0 ||
        spec.reconciliation_tolerance <= 0.0 || spec.config_hash == 0) {
        return result;
    }

    std::vector<double> active_returns;
    active_returns.reserve(periods.size());
    std::vector<std::vector<AttributionEffectValue>> sorted_effects;
    sorted_effects.reserve(periods.size());
    long double portfolio_growth = 1.0L;
    long double benchmark_growth = 1.0L;
    double active_sum = 0.0;
    double active_squared_sum = 0.0;
    std::uint64_t previous_period_id = 0;
    for (const auto& period : periods) {
        if (period.period_id == 0 || period.period_id <= previous_period_id ||
            !std::isfinite(period.portfolio_return) ||
            !std::isfinite(period.benchmark_return) ||
            period.portfolio_return <= -1.0 || period.benchmark_return <= -1.0) {
            return result;
        }
        previous_period_id = period.period_id;
        std::vector<AttributionEffectValue> effects(period.effects.begin(),
                                                    period.effects.end());
        std::sort(effects.begin(), effects.end(), [](const auto& left,
                                                     const auto& right) {
            return left.key < right.key;
        });
        double effect_sum = 0.0;
        for (std::size_t index = 0; index < effects.size(); ++index) {
            if (!valid_group_kind(effects[index].key.group_kind) ||
                !valid_effect_kind(effects[index].key.effect_kind) ||
                !std::isfinite(effects[index].value) ||
                (index > 0 && effects[index - 1].key == effects[index].key)) {
                return result;
            }
            effect_sum += effects[index].value;
        }
        const double active = period.portfolio_return - period.benchmark_return;
        if (!within_tolerance(effect_sum - active, active,
                              spec.reconciliation_tolerance)) {
            result.status = AttributionStatus::RECONCILIATION_FAILURE;
            return result;
        }
        active_returns.push_back(active);
        sorted_effects.push_back(std::move(effects));
        active_sum += active;
        active_squared_sum += active * active;
        portfolio_growth *= 1.0L + period.portfolio_return;
        benchmark_growth *= 1.0L + period.benchmark_return;
    }
    if (!std::isfinite(active_sum) || !std::isfinite(active_squared_sum) ||
        !std::isfinite(portfolio_growth) || !std::isfinite(benchmark_growth)) {
        return result;
    }
    result.cumulative_portfolio_return =
        static_cast<double>(portfolio_growth - 1.0L);
    result.cumulative_benchmark_return =
        static_cast<double>(benchmark_growth - 1.0L);
    result.cumulative_return_difference =
        result.cumulative_portfolio_return - result.cumulative_benchmark_return;
    const double period_count = static_cast<double>(periods.size());
    const double difference_scale = std::max(
        {1.0, std::abs(result.cumulative_portfolio_return),
         std::abs(result.cumulative_benchmark_return)});
    if (std::abs(result.cumulative_return_difference) <=
        spec.equality_tolerance * difference_scale) {
        result.used_equal_return_limit = true;
        result.linking_constant = std::pow(
            1.0 + result.cumulative_portfolio_return,
            (1.0 - period_count) / period_count);
    } else {
        const double portfolio_root = std::pow(
            1.0 + result.cumulative_portfolio_return, 1.0 / period_count);
        const double benchmark_root = std::pow(
            1.0 + result.cumulative_benchmark_return, 1.0 / period_count);
        result.linking_constant =
            (portfolio_root - benchmark_root) /
            result.cumulative_return_difference;
    }
    if (!std::isfinite(result.linking_constant)) return result;

    const double adjustment_residual = result.cumulative_return_difference -
        result.linking_constant * active_sum;
    double adjustment_scale = 0.0;
    if (active_squared_sum <= spec.degeneracy_tolerance) {
        if (!within_tolerance(adjustment_residual,
                              result.cumulative_return_difference,
                              spec.reconciliation_tolerance)) {
            result.status = AttributionStatus::DEGENERATE_LINKING;
            return result;
        }
    } else {
        adjustment_scale = adjustment_residual / active_squared_sum;
    }

    std::map<AttributionEffectKey, double> linked;
    result.periods.reserve(periods.size());
    for (std::size_t index = 0; index < periods.size(); ++index) {
        const double beta = result.linking_constant +
            adjustment_scale * active_returns[index];
        if (!std::isfinite(beta)) return result;
        result.periods.push_back({periods[index].period_id,
                                  active_returns[index], beta});
        for (const auto& effect : sorted_effects[index]) {
            linked[effect.key] += beta * effect.value;
        }
    }
    for (const auto& [key, value] : linked) {
        result.effects.push_back({key, value});
        result.linked_effect_total += value;
    }
    result.reconciliation_residual = result.cumulative_return_difference -
        result.linked_effect_total;
    if (!within_tolerance(result.reconciliation_residual,
                          result.cumulative_return_difference,
                          spec.reconciliation_tolerance)) {
        result.status = AttributionStatus::RECONCILIATION_FAILURE;
        return result;
    }

    result.artifact_hash = kFnvOffset;
    hash_value(result.artifact_hash, spec.config_hash);
    for (std::size_t index = 0; index < periods.size(); ++index) {
        hash_value(result.artifact_hash, periods[index].period_id);
        hash_double(result.artifact_hash, periods[index].portfolio_return);
        hash_double(result.artifact_hash, periods[index].benchmark_return);
        for (const auto& effect : sorted_effects[index]) {
            hash_effect_key(result.artifact_hash, effect.key);
            hash_double(result.artifact_hash, effect.value);
        }
    }
    result.status = AttributionStatus::OK;
    return result;
}

}  // namespace performance_analytics

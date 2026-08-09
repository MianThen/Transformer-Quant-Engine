#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include "performance_analytics/active_attribution.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
    return condition;
}

bool near(double left, double right, double tolerance = 1e-10) {
    return std::abs(left - right) < tolerance;
}

}  // namespace

int main() {
    using performance_analytics::AttributionEffectKind;
    using performance_analytics::AttributionEffectValue;
    using performance_analytics::AttributionGroupKind;
    using performance_analytics::AttributionStatus;
    using performance_analytics::BrinsonGroupInput;
    using performance_analytics::BrinsonProblem;
    using performance_analytics::BrinsonSpec;
    using performance_analytics::MencheroPeriodInput;
    using performance_analytics::MencheroSpec;

    bool ok = true;
    std::array groups{
        BrinsonGroupInput{AttributionGroupKind::EQUITY_INDUSTRY, 1,
                          0.6, 0.5, 0.10, 0.06},
        BrinsonGroupInput{AttributionGroupKind::EQUITY_INDUSTRY, 2,
                          0.4, 0.5, 0.05, 0.04},
    };
    BrinsonProblem problem{
        0.08, 0.05, true, 1001, 2001, groups, BrinsonSpec{1e-10, 49},
    };
    const auto brinson = performance_analytics::compute_brinson_fachler(problem);
    ok &= check(brinson.status == AttributionStatus::OK &&
                    brinson.groups.size() == 2 &&
                    near(brinson.groups[0].allocation, 0.001) &&
                    near(brinson.groups[0].selection, 0.020) &&
                    near(brinson.groups[0].interaction, 0.004) &&
                    near(brinson.groups[1].total, 0.005) &&
                    near(brinson.effect_total, 0.03) &&
                    near(brinson.reconciliation_residual, 0.0),
                "Brinson-Fachler hand calculation reconciles active return");
    const auto flattened =
        performance_analytics::brinson_effect_values(brinson);
    ok &= check(flattened.size() == 6,
                "industry effects flatten into three named components");

    auto reversed_groups = groups;
    std::reverse(reversed_groups.begin(), reversed_groups.end());
    BrinsonProblem reordered_problem = problem;
    reordered_problem.groups = reversed_groups;
    const auto reordered =
        performance_analytics::compute_brinson_fachler(reordered_problem);
    ok &= check(reordered.status == AttributionStatus::OK &&
                    reordered.artifact_hash == brinson.artifact_hash,
                "group input order does not change Brinson hash");

    BrinsonProblem unavailable = problem;
    unavailable.benchmark_holdings_available = false;
    unavailable.benchmark_provenance_hash = 0;
    unavailable.groups = {};
    ok &= check(performance_analytics::compute_brinson_fachler(unavailable).status ==
                    AttributionStatus::UNAVAILABLE,
                "missing benchmark holdings returns unavailable");

    const std::array cash_groups{
        BrinsonGroupInput{AttributionGroupKind::EQUITY_INDUSTRY, 1,
                          0.9, 1.0, 0.08 / 0.9, 0.05},
        BrinsonGroupInput{AttributionGroupKind::CASH, 0,
                          0.1, 0.0, 0.0, 0.0},
    };
    BrinsonProblem cash_problem{
        0.08, 0.05, true, 1002, 2002, cash_groups,
        BrinsonSpec{1e-10, 50},
    };
    const auto cash_result =
        performance_analytics::compute_brinson_fachler(cash_problem);
    ok &= check(cash_result.status == AttributionStatus::OK &&
                    cash_result.groups.front().group_kind ==
                        AttributionGroupKind::EQUITY_INDUSTRY &&
                    cash_result.groups.back().group_kind ==
                        AttributionGroupKind::CASH &&
                    near(cash_result.groups.back().other, -0.005) &&
                    near(cash_result.groups.back().selection, 0.0) &&
                    near(cash_result.effect_total, 0.03),
                "cash contribution is separate and still reconciles");

    const performance_analytics::AttributionEffectKey effect_a{
        AttributionGroupKind::EQUITY_INDUSTRY, 1,
        AttributionEffectKind::SELECTION};
    const performance_analytics::AttributionEffectKey effect_b{
        AttributionGroupKind::EQUITY_INDUSTRY, 2,
        AttributionEffectKind::SELECTION};
    std::array first_effects{
        AttributionEffectValue{effect_a, 0.02},
        AttributionEffectValue{effect_b, 0.01},
    };
    std::array second_effects{
        AttributionEffectValue{effect_a, -0.01},
        AttributionEffectValue{effect_b, -0.02},
    };
    const std::array periods{
        MencheroPeriodInput{1, 0.08, 0.05, first_effects},
        MencheroPeriodInput{2, -0.02, 0.01, second_effects},
    };
    const MencheroSpec menchero_spec{1e-12, 1e-18, 1e-10, 51};
    const auto linked =
        performance_analytics::menchero_link(periods, menchero_spec);
    ok &= check(linked.status == AttributionStatus::OK &&
                    linked.periods.size() == 2 && linked.effects.size() == 2 &&
                    near(linked.cumulative_portfolio_return, 0.0584) &&
                    near(linked.cumulative_benchmark_return, 0.0605) &&
                    near(linked.cumulative_return_difference, -0.0021) &&
                    near(linked.linked_effect_total, -0.0021) &&
                    near(linked.reconciliation_residual, 0.0),
                "Menchero links effects to geometric cumulative difference");

    std::reverse(first_effects.begin(), first_effects.end());
    std::reverse(second_effects.begin(), second_effects.end());
    const std::array reordered_periods{
        MencheroPeriodInput{1, 0.08, 0.05, first_effects},
        MencheroPeriodInput{2, -0.02, 0.01, second_effects},
    };
    const auto reordered_link =
        performance_analytics::menchero_link(reordered_periods, menchero_spec);
    ok &= check(reordered_link.status == AttributionStatus::OK &&
                    reordered_link.artifact_hash == linked.artifact_hash,
                "effect input order does not change Menchero hash");

    const std::array zero_effect{
        AttributionEffectValue{effect_a, 0.0},
    };
    const std::array equal_periods{
        MencheroPeriodInput{1, 0.01, 0.01, zero_effect},
        MencheroPeriodInput{2, 0.02, 0.02, zero_effect},
    };
    const auto equal =
        performance_analytics::menchero_link(equal_periods, menchero_spec);
    ok &= check(equal.status == AttributionStatus::OK &&
                    equal.used_equal_return_limit &&
                    near(equal.cumulative_return_difference, 0.0) &&
                    near(equal.linked_effect_total, 0.0),
                "RP equals RB uses the registered continuous limit");

    if (!ok) return 1;
    std::printf("test_active_attribution: all checks passed\n");
    return 0;
}

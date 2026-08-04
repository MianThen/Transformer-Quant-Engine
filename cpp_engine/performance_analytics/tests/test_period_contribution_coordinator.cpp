#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "performance_analytics/period_contribution_coordinator.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
    return condition;
}

bool near(double left, double right) {
    return std::abs(left - right) < 1e-10;
}

performance_analytics::PerformanceSpecV1 spec() {
    performance_analytics::PerformanceSpecV1 value;
    value.frequency = performance_analytics::ReturnFrequency::DAILY;
    value.calendar_id = "XSHG_TRADING_DAY_V1";
    value.calendar_periods_per_year = 242.0;
    value.benchmark_id = "INTERNAL_FROZEN_CONTROL";
    value.config_hash = 47;
    return value;
}

}  // namespace

int main() {
    using engine_common::Side;
    using performance_analytics::PeriodContributionCoordinator;
    using performance_analytics::PeriodCoordinatorStatus;
    using performance_analytics::PeriodSecuritySnapshot;

    bool ok = true;
    std::vector beginning{
        PeriodSecuritySnapshot{0, 7, 10, 10.0},
    };
    const std::vector ending{
        PeriodSecuritySnapshot{1, 8, 5, 22.0},
    };
    PeriodContributionCoordinator coordinator(spec());
    ok &= check(coordinator.open_period(
                    {100, 1, 1'100.0, 1'000.0, beginning}) ==
                    PeriodCoordinatorStatus::OK,
                "explicit calendar session opens a period");
    beginning[0].quantity = 999;
    beginning[0].mark_price = 99.0;
    ok &= check(coordinator.record_corporate_action(
                    {100, 0, 140, 0, 2.0}) ==
                    PeriodCoordinatorStatus::OK &&
                    coordinator.record_fill(
                        {1, 0, Side::SELL, 10, 11.0, 1.0, 150}) ==
                    PeriodCoordinatorStatus::OK &&
                    coordinator.record_fill(
                        {2, 1, Side::BUY, 5, 20.0, 1.0, 160}) ==
                    PeriodCoordinatorStatus::OK,
                "fills and corporate action are captured inside the period");
    ok &= check(coordinator.close_period(
                    {200, 1'121.0, 1'011.0, 1.0, 0.0, ending}) ==
                    PeriodCoordinatorStatus::OK &&
                    coordinator.ledger().records().size() == 1,
                "universe change closes to one reconciled record");
    const auto& first = coordinator.ledger().records().front();
    ok &= check(first.securities.size() == 2 &&
                    first.securities[0].symbol_id == 0 &&
                    first.securities[0].beginning_quantity == 10 &&
                    first.securities[0].ending_quantity == 0 &&
                    first.securities[1].beginning_quantity == 0 &&
                    first.securities[1].ending_quantity == 5,
                "frozen union retains closed and newly entered securities");
    ok &= check(first.industries.size() == 2 &&
                    first.industries[0].pit_industry_id == 7 &&
                    near(first.industries[0].net_contribution, 11.0) &&
                    first.industries[1].pit_industry_id == 8 &&
                    near(first.industries[1].net_contribution, 9.0) &&
                    coordinator.corporate_action_history().size() == 1,
                "PIT industries and action history remain auditable");
    const auto first_period_hash = coordinator.coordinator_hash();

    const std::array split_beginning{
        PeriodSecuritySnapshot{1, 9, 5, 22.0},
    };
    const std::array split_ending{
        PeriodSecuritySnapshot{1, 9, 10, 11.0},
    };
    ok &= check(coordinator.open_period(
                    {200, 2, 1'121.0, 1'011.0, split_beginning}) ==
                    PeriodCoordinatorStatus::OK &&
                    coordinator.record_corporate_action(
                        {101, 1, 250, 5, 0.0}) ==
                    PeriodCoordinatorStatus::OK &&
                    coordinator.finalize(
                        {300, 1'121.0, 1'011.0, 0.0, 0.0, split_ending}) ==
                    PeriodCoordinatorStatus::OK &&
                    coordinator.finalized() &&
                    coordinator.ledger().records().size() == 2 &&
                    near(coordinator.ledger().records()[1]
                             .securities.front().market_pnl,
                         0.0) &&
                    coordinator.coordinator_hash() != first_period_hash,
                "split period finalizes without artificial return");

    const std::vector stable_beginning{
        PeriodSecuritySnapshot{0, 7, 10, 10.0},
    };
    PeriodContributionCoordinator reordered(spec());
    ok &= check(reordered.open_period(
                    {100, 1, 1'100.0, 1'000.0, stable_beginning}) ==
                    PeriodCoordinatorStatus::OK &&
                    reordered.record_fill(
                        {2, 1, Side::BUY, 5, 20.0, 1.0, 160}) ==
                    PeriodCoordinatorStatus::OK &&
                    reordered.record_fill(
                        {1, 0, Side::SELL, 10, 11.0, 1.0, 150}) ==
                    PeriodCoordinatorStatus::OK &&
                    reordered.record_corporate_action(
                        {100, 0, 140, 0, 2.0}) ==
                    PeriodCoordinatorStatus::OK &&
                    reordered.close_period(
                        {200, 1'121.0, 1'011.0, 1.0, 0.0, ending}) ==
                    PeriodCoordinatorStatus::OK &&
                    reordered.coordinator_hash() == first_period_hash,
                "event input order does not change coordinator hash");

    PeriodContributionCoordinator duplicate_action(spec());
    ok &= check(duplicate_action.open_period(
                    {100, 1, 1'100.0, 1'000.0, stable_beginning}) ==
                    PeriodCoordinatorStatus::OK &&
                    duplicate_action.record_corporate_action(
                        {200, 0, 120, 0, 1.0}) ==
                    PeriodCoordinatorStatus::OK &&
                    duplicate_action.record_corporate_action(
                        {200, 0, 130, 0, 1.0}) ==
                    PeriodCoordinatorStatus::DUPLICATE_CORPORATE_ACTION &&
                    duplicate_action.failed() &&
                    duplicate_action.ledger().records().empty(),
                "duplicate action fails closed without partial period");

    if (!ok) return 1;
    std::printf("test_period_contribution_coordinator: all checks passed\n");
    return 0;
}

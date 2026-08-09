#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "performance_analytics/shortfall_replay_adapter.h"

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
    value.config_hash = 44;
    return value;
}

}  // namespace

int main() {
    using engine_common::Side;
    using performance_analytics::FrozenPaperTarget;
    using performance_analytics::ReferencePriceType;
    using performance_analytics::ShortfallCloseAsset;
    using performance_analytics::ShortfallCloseSnapshot;
    using performance_analytics::ShortfallDecisionSnapshot;
    using performance_analytics::ShortfallFillInput;
    using performance_analytics::ShortfallReplayAdapter;
    using performance_analytics::ShortfallReplayStatus;
    using performance_analytics::UnexecutedReason;

    bool ok = true;
    std::vector targets{
        FrozenPaperTarget{1, 100, 0, 10.0,
                          ReferencePriceType::BID_ASK_MIDPOINT, 10.0},
    };
    ShortfallReplayAdapter adapter(spec());
    ok &= check(adapter.open_interval({1, 1, 100, targets}) ==
                    ShortfallReplayStatus::OK && adapter.has_open_interval(),
                "decision opens an interval");
    targets[0].target_quantity = 999;
    targets[0].decision_reference_price = 99.0;
    ok &= check(adapter.record_fill({10, 1, Side::BUY, 40, 10.1, 1.0, 150}) ==
                    ShortfallReplayStatus::OK,
                "fill is captured inside the open interval");
    const std::array first_close{
        ShortfallCloseAsset{1, 40, 11.0, UnexecutedReason::PARTIAL_FILL},
    };
    ok &= check(adapter.close_interval({200, first_close}) ==
                    ShortfallReplayStatus::OK &&
                    adapter.ledger().records().size() == 1 &&
                    !adapter.has_open_interval(),
                "close emits exactly one ledger record");
    const auto& first = adapter.ledger().records().front();
    ok &= check(first.assets.front().target_quantity == 100 &&
                    near(first.assets.front().decision_reference_price, 10.0) &&
                    near(first.implementation_shortfall, 65.0),
                "paper target is frozen against future input mutation");

    const std::array second_target{
        FrozenPaperTarget{1, 40, 40, 11.0,
                          ReferencePriceType::BID_ASK_MIDPOINT, std::nullopt},
    };
    ok &= check(adapter.open_interval({2, 2, 200, second_target}) ==
                    ShortfallReplayStatus::OK,
                "next target starts only after previous interval closes");
    const std::array second_close{
        ShortfallCloseAsset{1, 40, 12.0, UnexecutedReason::NONE},
    };
    ok &= check(adapter.finalize({300, second_close}) ==
                    ShortfallReplayStatus::OK && adapter.finalized() &&
                    adapter.ledger().records().size() == 2 &&
                    near(adapter.ledger().records()[1].implementation_shortfall, 0.0),
                "replay finalize closes the last interval");
    ok &= check(adapter.open_interval({3, 3, 300, second_target}) ==
                    ShortfallReplayStatus::FINALIZED && adapter.failed(),
                "finalized replay rejects later decisions");

    ShortfallReplayAdapter reset_violation(spec());
    ok &= check(reset_violation.open_interval({1, 1, 100, second_target}) ==
                    ShortfallReplayStatus::OK &&
                    reset_violation.open_interval({2, 2, 150, second_target}) ==
                    ShortfallReplayStatus::INTERVAL_ALREADY_OPEN &&
                    reset_violation.failed() &&
                    reset_violation.ledger().records().empty(),
                "target reset cannot mix two decision intervals");

    ShortfallReplayAdapter unknown_fill(spec());
    ok &= check(unknown_fill.open_interval({1, 1, 100, second_target}) ==
                    ShortfallReplayStatus::OK &&
                    unknown_fill.record_fill(
                        {11, 2, Side::BUY, 1, 10.0, 0.0, 150}) ==
                    ShortfallReplayStatus::UNKNOWN_SYMBOL &&
                    unknown_fill.failed() && unknown_fill.ledger().records().empty(),
                "unknown fill symbol fails closed");

    ShortfallReplayAdapter duplicate_fill(spec());
    ok &= check(duplicate_fill.open_interval({1, 1, 100, second_target}) ==
                    ShortfallReplayStatus::OK &&
                    duplicate_fill.record_fill(
                        {12, 1, Side::BUY, 1, 11.0, 0.0, 150}) ==
                    ShortfallReplayStatus::OK &&
                    duplicate_fill.record_fill(
                        {12, 1, Side::BUY, 1, 11.0, 0.0, 160}) ==
                    ShortfallReplayStatus::DUPLICATE_EXECUTION &&
                    duplicate_fill.failed() && duplicate_fill.ledger().records().empty(),
                "duplicate execution fails closed");

    const std::array<FrozenPaperTarget, 0> cash_targets{};
    const std::array<ShortfallCloseAsset, 0> cash_close{};
    ShortfallReplayAdapter cash_adapter(spec());
    ok &= check(cash_adapter.open_interval({1, 1, 100, cash_targets}) ==
                    ShortfallReplayStatus::OK &&
                    cash_adapter.finalize({200, cash_close}) ==
                    ShortfallReplayStatus::OK &&
                    cash_adapter.ledger().records().size() == 1 &&
                    near(cash_adapter.ledger().records().front()
                             .implementation_shortfall,
                         0.0),
                "empty target preserves the all-cash replay interval");

    if (!ok) return 1;
    std::printf("test_shortfall_replay_adapter: all checks passed\n");
    return 0;
}

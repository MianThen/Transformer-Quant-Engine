#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "performance_analytics/implementation_shortfall.h"

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
    value.config_hash = 43;
    return value;
}

performance_analytics::ImplementationShortfallInput make_input(
    std::span<const performance_analytics::ShortfallAssetInput> assets,
    std::span<const performance_analytics::ShortfallFillInput> fills) {
    return {1, 1, 100, 200, assets, fills};
}

}  // namespace

int main() {
    using engine_common::Side;
    using performance_analytics::ImplementationShortfallLedger;
    using performance_analytics::ReferencePriceType;
    using performance_analytics::ShortfallAssetInput;
    using performance_analytics::ShortfallFillInput;
    using performance_analytics::ShortfallStatus;

    bool ok = true;
    const std::array buy_asset{
        ShortfallAssetInput{1, 100, 0, 100, 10.0, 11.0,
                            ReferencePriceType::BID_ASK_MIDPOINT, std::nullopt,
                            performance_analytics::UnexecutedReason::NONE},
    };
    const std::array buy_fill{
        ShortfallFillInput{10, 1, Side::BUY, 100, 10.1, 1.0, 150},
    };
    ImplementationShortfallLedger buy_ledger(spec());
    ok &= check(buy_ledger.append(make_input(buy_asset, buy_fill)) ==
                    ShortfallStatus::OK,
                "buy interval is accepted");
    const auto& buy = buy_ledger.records().front();
    ok &= check(near(buy.execution_price_cost, 10.0) &&
                    near(buy.explicit_fees, 1.0) &&
                    near(buy.opportunity_cost, 0.0) &&
                    near(buy.implementation_shortfall, 11.0) &&
                    near(buy.paper_pnl, 100.0) && near(buy.real_net_pnl, 89.0) &&
                    near(buy.identity_residual, 0.0) && buy.promotion_eligible,
                "buy direction and paper-real identity");

    const std::array sell_asset{
        ShortfallAssetInput{1, 0, 100, 0, 10.0, 9.0,
                            ReferencePriceType::BID_ASK_MIDPOINT, std::nullopt,
                            performance_analytics::UnexecutedReason::NONE},
    };
    const std::array sell_fill{
        ShortfallFillInput{11, 1, Side::SELL, 100, 9.9, 1.0, 150},
    };
    ImplementationShortfallLedger sell_ledger(spec());
    ok &= check(sell_ledger.append(make_input(sell_asset, sell_fill)) ==
                    ShortfallStatus::OK &&
                    near(sell_ledger.records().front().execution_price_cost, 10.0) &&
                    near(sell_ledger.records().front().implementation_shortfall, 11.0),
                "sell direction produces positive implementation cost");

    const std::array partial_asset{
        ShortfallAssetInput{1, 100, 0, 40, 10.0, 11.0,
                            ReferencePriceType::BID_ASK_MIDPOINT, 10.0,
                            performance_analytics::UnexecutedReason::PARTIAL_FILL},
    };
    const std::array partial_fill{
        ShortfallFillInput{12, 1, Side::BUY, 40, 10.1, 1.0, 150},
    };
    ImplementationShortfallLedger partial_ledger(spec());
    ok &= check(partial_ledger.append(make_input(partial_asset, partial_fill)) ==
                    ShortfallStatus::OK &&
                    near(partial_ledger.records().front().execution_price_cost, 4.0) &&
                    near(partial_ledger.records().front().opportunity_cost, 60.0) &&
                    near(partial_ledger.records().front().implementation_shortfall, 65.0),
                "partial fill retains unexecuted opportunity cost");

    const std::array unfilled_asset{
        ShortfallAssetInput{1, 100, 0, 0, 10.0, 11.0,
                            ReferencePriceType::ARRIVAL_PRICE_PROXY, 10.0,
                            performance_analytics::UnexecutedReason::LIQUIDITY_LIMIT},
    };
    const std::array<ShortfallFillInput, 0> no_fills{};
    ImplementationShortfallLedger unfilled_ledger(spec());
    ok &= check(unfilled_ledger.append(make_input(unfilled_asset, no_fills)) ==
                    ShortfallStatus::OK &&
                    near(unfilled_ledger.records().front().opportunity_cost, 100.0) &&
                    !unfilled_ledger.records().front().promotion_eligible,
                "unfilled target is charged and proxy reference blocks promotion");

    const std::array ordered_assets{
        ShortfallAssetInput{1, 10, 0, 10, 10.0, 11.0,
                            ReferencePriceType::BID_ASK_MIDPOINT, std::nullopt,
                            performance_analytics::UnexecutedReason::NONE},
        ShortfallAssetInput{2, 20, 0, 20, 20.0, 19.0,
                            ReferencePriceType::BID_ASK_MIDPOINT, std::nullopt,
                            performance_analytics::UnexecutedReason::NONE},
    };
    const std::array ordered_fills{
        ShortfallFillInput{21, 1, Side::BUY, 10, 10.0, 0.1, 140},
        ShortfallFillInput{22, 2, Side::BUY, 20, 20.0, 0.2, 160},
    };
    auto reversed_assets = ordered_assets;
    auto reversed_fills = ordered_fills;
    std::reverse(reversed_assets.begin(), reversed_assets.end());
    std::reverse(reversed_fills.begin(), reversed_fills.end());
    ImplementationShortfallLedger ordered_ledger(spec());
    ImplementationShortfallLedger reversed_ledger(spec());
    ok &= check(ordered_ledger.append(make_input(ordered_assets, ordered_fills)) ==
                    ShortfallStatus::OK &&
                    reversed_ledger.append(make_input(reversed_assets, reversed_fills)) ==
                    ShortfallStatus::OK &&
                    ordered_ledger.ledger_hash() == reversed_ledger.ledger_hash(),
                "logical input order does not change deterministic hash");

    auto mismatched_asset = partial_asset;
    mismatched_asset[0].actual_end_quantity = 41;
    ImplementationShortfallLedger rejected_ledger(spec());
    const auto initial_hash = rejected_ledger.ledger_hash();
    ok &= check(rejected_ledger.append(make_input(mismatched_asset, partial_fill)) ==
                    ShortfallStatus::POSITION_MISMATCH &&
                    rejected_ledger.records().empty() &&
                    rejected_ledger.ledger_hash() == initial_hash,
                "position mismatch fails closed without ledger mutation");

    auto next_input = make_input(buy_asset, buy_fill);
    next_input.decision_id = 2;
    next_input.measurement_interval_id = 2;
    next_input.decision_at = 199;
    next_input.interval_end = 300;
    ImplementationShortfallLedger overlap_ledger(spec());
    ok &= check(overlap_ledger.append(make_input(buy_asset, buy_fill)) ==
                    ShortfallStatus::OK &&
                    overlap_ledger.append(next_input) ==
                    ShortfallStatus::NON_MONOTONIC_INTERVAL &&
                    overlap_ledger.records().size() == 1,
                "overlapping paper-target intervals fail closed");

    next_input.decision_at = 200;
    next_input.interval_end = 300;
    auto repeated_fill = buy_fill;
    repeated_fill[0].timestamp = 250;
    next_input.fills = repeated_fill;
    ok &= check(overlap_ledger.append(next_input) ==
                    ShortfallStatus::DUPLICATE_EXECUTION &&
                    overlap_ledger.records().size() == 1,
                "execution cannot be counted in multiple intervals");

    if (!ok) return 1;
    std::printf("test_implementation_shortfall: all checks passed\n");
    return 0;
}

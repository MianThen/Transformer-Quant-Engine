#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "performance_analytics/security_contribution.h"

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
    value.config_hash = 46;
    return value;
}

}  // namespace

int main() {
    using engine_common::Side;
    using performance_analytics::ContributionFillInput;
    using performance_analytics::ContributionStatus;
    using performance_analytics::PeriodContributionInput;
    using performance_analytics::SecurityContributionInput;
    using performance_analytics::SecurityContributionLedger;

    bool ok = true;
    std::vector securities{
        SecurityContributionInput{0, 7, 10, 0, 0, 10.0, 12.0, 2.0},
        SecurityContributionInput{1, 7, 0, 5, 0, 20.0, 22.0, 0.0},
    };
    std::vector fills{
        ContributionFillInput{1, 0, Side::SELL, 10, 11.0, 1.0, 150},
        ContributionFillInput{2, 1, Side::BUY, 5, 20.0, 1.0, 160},
    };
    PeriodContributionInput first{
        100, 200, 1, 1'100.0, 1'121.0, 1'000.0, 1'011.0,
        1.0, 0.0, securities, fills,
    };
    SecurityContributionLedger ledger(spec());
    ok &= check(ledger.append(first) == ContributionStatus::OK,
                "closed and opened positions reconcile");
    const auto& first_record = ledger.records().front();
    ok &= check(first_record.securities.size() == 2 &&
                    first_record.securities[0].symbol_id == 0 &&
                    first_record.securities[0].ending_quantity == 0 &&
                    near(first_record.securities[0].market_pnl, 10.0) &&
                    near(first_record.securities[0].net_contribution, 11.0) &&
                    near(first_record.securities[1].net_contribution, 9.0),
                "closed position remains in security attribution");
    ok &= check(first_record.industries.size() == 1 &&
                    first_record.industries.front().pit_industry_id == 7 &&
                    near(first_record.industries.front().net_contribution, 20.0),
                "security contributions aggregate to PIT industry");
    ok &= check(near(first_record.executed_gross_pnl, 20.0) &&
                    near(first_record.explicit_fees, 2.0) &&
                    near(first_record.net_pnl, 18.0) &&
                    near(first_record.corporate_action_cash, 2.0) &&
                    near(first_record.security_net_contribution, 20.0) &&
                    near(first_record.cash_residual, 0.0) &&
                    near(first_record.beginning_balance_residual, 0.0) &&
                    near(first_record.ending_balance_residual, 0.0) &&
                    near(first_record.accounting_residual, 0.0),
                "cash and equity identities reconcile");
    const auto first_period_hash = ledger.ledger_hash();

    performance_analytics::ReturnLedger return_ledger(spec());
    ok &= check(return_ledger.append(
                    performance_analytics::period_return_input(first_record)) ==
                    performance_analytics::LedgerStatus::OK &&
                    near(return_ledger.records().front().period_return,
                         first_record.period_return),
                "contribution record feeds the authoritative return ledger");

    const std::array split_security{
        SecurityContributionInput{1, 9, 5, 10, 5, 22.0, 11.0, 0.0},
    };
    const std::array<ContributionFillInput, 0> no_fills{};
    PeriodContributionInput split{
        200, 300, 2, 1'121.0, 1'121.0, 1'011.0, 1'011.0,
        0.0, 0.0, split_security, no_fills,
    };
    ok &= check(ledger.append(split) == ContributionStatus::OK &&
                    ledger.records().size() == 2 &&
                    near(ledger.records()[1].securities.front().market_pnl, 0.0) &&
                    ledger.records()[1].industries.front().pit_industry_id == 9,
                "split is value-neutral and industry remains point-in-time");

    auto reversed_securities = securities;
    auto reversed_fills = fills;
    std::reverse(reversed_securities.begin(), reversed_securities.end());
    std::reverse(reversed_fills.begin(), reversed_fills.end());
    PeriodContributionInput reordered = first;
    reordered.securities = reversed_securities;
    reordered.fills = reversed_fills;
    SecurityContributionLedger reordered_ledger(spec());
    ok &= check(reordered_ledger.append(reordered) == ContributionStatus::OK &&
                    reordered_ledger.ledger_hash() == first_period_hash,
                "logical input order does not change contribution hash");

    PeriodContributionInput cash_mismatch = first;
    cash_mismatch.ending_cash = 1'012.0;
    SecurityContributionLedger rejected(spec());
    const auto initial_hash = rejected.ledger_hash();
    ok &= check(rejected.append(cash_mismatch) ==
                    ContributionStatus::CASH_MISMATCH &&
                    rejected.records().empty() && rejected.ledger_hash() == initial_hash,
                "cash mismatch fails closed without ledger mutation");

    auto position_mismatch_securities = securities;
    position_mismatch_securities[1].ending_quantity = 6;
    PeriodContributionInput position_mismatch = first;
    position_mismatch.securities = position_mismatch_securities;
    SecurityContributionLedger position_rejected(spec());
    ok &= check(position_rejected.append(position_mismatch) ==
                    ContributionStatus::POSITION_MISMATCH &&
                    position_rejected.records().empty(),
                "position mismatch fails closed");

    PeriodContributionInput shifted_balance = first;
    shifted_balance.starting_equity += 100.0;
    shifted_balance.ending_equity += 100.0;
    SecurityContributionLedger balance_rejected(spec());
    ok &= check(balance_rejected.append(shifted_balance) ==
                    ContributionStatus::ACCOUNTING_MISMATCH &&
                    balance_rejected.records().empty(),
                "absolute balance-sheet mismatch fails closed");

    if (!ok) return 1;
    std::printf("test_security_contribution: all checks passed\n");
    return 0;
}

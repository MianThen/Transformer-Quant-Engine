#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "performance_analytics/return_ledger.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
    return condition;
}

performance_analytics::PerformanceSpecV1 spec() {
    performance_analytics::PerformanceSpecV1 value;
    value.frequency = performance_analytics::ReturnFrequency::DAILY;
    value.calendar_id = "XSHG_TRADING_DAY_V1";
    value.calendar_periods_per_year = 242.0;
    value.benchmark_id = "INTERNAL_FROZEN_CONTROL";
    value.config_hash = 42;
    return value;
}

}  // namespace

int main() {
    bool ok = true;
    performance_analytics::ReturnLedger ledger(spec());
    performance_analytics::PeriodReturnInput first;
    first.period_start = 1;
    first.period_end = 2;
    first.session_id = 1;
    first.starting_equity = 1'000.0;
    first.ending_equity = 1'010.0;
    first.executed_gross_pnl = 12.0;
    first.explicit_fees = 2.0;
    first.net_pnl = 10.0;
    first.benchmark_return = 0.005;
    ok &= check(ledger.append(first) == performance_analytics::LedgerStatus::OK,
                "first accounting period");
    const auto first_period_hash = ledger.ledger_hash();

    auto second = first;
    second.period_start = 2;
    second.period_end = 3;
    second.session_id = 2;
    second.starting_equity = 1'010.0;
    second.ending_equity = 1'005.0;
    second.executed_gross_pnl = -4.0;
    second.explicit_fees = 1.0;
    second.net_pnl = -5.0;
    second.benchmark_return = -0.002;
    ok &= check(ledger.append(second) == performance_analytics::LedgerStatus::OK,
                "second accounting period");
    ok &= check(ledger.records().size() == 2 &&
                std::abs(ledger.records()[0].period_return - 0.01) < 1e-12 &&
                std::abs(*ledger.cumulative_return() - 0.005) < 1e-12 &&
                ledger.ledger_hash() != 0,
                "simple return, geometric linking, and replay hash");
    performance_analytics::ReturnLedgerArtifactSpec artifact_spec;
    artifact_spec.source_replay_sha256 = std::string(64, '1');
    artifact_spec.dataset_fingerprint = std::string(64, '2');
    artifact_spec.benchmark_artifact_sha256 = std::string(64, '3');
    artifact_spec.benchmark_available = true;
    artifact_spec.promotion_eligible = true;
    artifact_spec.reference_price_quality = "PROXY";
    artifact_spec.limitations.push_back("REFERENCE_PRICE_PROXY");
    const auto artifact = performance_analytics::serialize_return_ledger_artifact(
        ledger, artifact_spec);
    const bool artifact_valid = artifact.find("\"records\":[") != std::string::npos &&
        artifact.find("\"ledger_sha256\":\"aa3d9d1072bda406e947e9b6b2a9243bd66f6f99ee911c2e2d557adda9f6c2d0\"") !=
            std::string::npos &&
        artifact.find("\"promotion_eligible\":false") != std::string::npos;
    ok &= check(artifact_valid,
                "frozen ledger artifact records provenance and closes proxy promotion");
    performance_analytics::ReturnLedger changed_benchmark(spec());
    auto changed_first = first;
    changed_first.benchmark_return = 0.006;
    ok &= check(changed_benchmark.append(changed_first) ==
                    performance_analytics::LedgerStatus::OK &&
                changed_benchmark.ledger_hash() != first_period_hash,
                "benchmark participates in the replay hash");

    auto external_flow = second;
    external_flow.period_start = 3;
    external_flow.period_end = 4;
    external_flow.session_id = 3;
    external_flow.starting_equity = 1'005.0;
    external_flow.ending_equity = 1'105.0;
    external_flow.executed_gross_pnl = 0.0;
    external_flow.explicit_fees = 0.0;
    external_flow.net_pnl = 0.0;
    external_flow.external_cash_flow = 100.0;
    ok &= check(ledger.append(external_flow) ==
                    performance_analytics::LedgerStatus::EXTERNAL_CASH_FLOW &&
                ledger.records().size() == 2,
                "external flow fails closed without mutating the ledger");

    auto mismatch = external_flow;
    mismatch.external_cash_flow = 0.0;
    mismatch.ending_equity = 1'006.0;
    ok &= check(ledger.append(mismatch) ==
                    performance_analytics::LedgerStatus::ACCOUNTING_MISMATCH &&
                ledger.records().size() == 2,
                "accounting mismatch fails closed");

    auto invalid_spec = spec();
    invalid_spec.calendar_id.clear();
    bool threw = false;
    try {
        performance_analytics::ReturnLedger invalid(std::move(invalid_spec));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ok &= check(threw, "invalid PerformanceSpecV1 is rejected");
    if (!ok) return 1;
    std::printf("test_performance_analytics: all checks passed\n");
    return 0;
}

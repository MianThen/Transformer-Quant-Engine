#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "engine_common/types.h"
#include "performance_analytics/performance_spec.h"

namespace performance_analytics {

enum class LedgerStatus : std::uint8_t {
    OK,
    INVALID_SPEC,
    INVALID_INPUT,
    NON_MONOTONIC_PERIOD,
    EQUITY_DISCONTINUITY,
    EXTERNAL_CASH_FLOW,
    ACCOUNTING_MISMATCH,
};

struct PeriodReturnInput {
    engine_common::TimestampNs period_start{0};
    engine_common::TimestampNs period_end{0};
    std::uint64_t session_id{0};
    double starting_equity{0.0};
    double ending_equity{0.0};
    double executed_gross_pnl{0.0};
    double explicit_fees{0.0};
    double net_pnl{0.0};
    double corporate_action_cash{0.0};
    double cash_interest{0.0};
    double external_cash_flow{0.0};
    std::optional<double> benchmark_return;
};

struct PeriodReturnRecord {
    engine_common::TimestampNs period_start{0};
    engine_common::TimestampNs period_end{0};
    std::uint64_t session_id{0};
    double starting_equity{0.0};
    double ending_equity{0.0};
    double period_return{0.0};
    double executed_gross_pnl{0.0};
    double explicit_fees{0.0};
    double net_pnl{0.0};
    double corporate_action_cash{0.0};
    double cash_interest{0.0};
    double accounting_residual{0.0};
    std::optional<double> benchmark_return;
};

struct ReturnLedgerArtifactSpec {
    std::string source_replay_sha256;
    std::string dataset_fingerprint;
    std::string benchmark_artifact_sha256;
    bool benchmark_available{false};
    bool promotion_eligible{false};
    std::string reference_price_quality;
    std::vector<std::string> limitations;
};

class ReturnLedger {
public:
    explicit ReturnLedger(PerformanceSpecV1 spec);

    [[nodiscard]] LedgerStatus append(const PeriodReturnInput& input);
    [[nodiscard]] std::span<const PeriodReturnRecord> records() const noexcept {
        return records_;
    }
    [[nodiscard]] std::optional<double> cumulative_return() const noexcept;
    [[nodiscard]] std::string ledger_sha256() const;
    [[nodiscard]] LedgerStatus last_status() const noexcept { return last_status_; }
    [[nodiscard]] std::uint64_t ledger_hash() const noexcept { return ledger_hash_; }
    [[nodiscard]] const PerformanceSpecV1& spec() const noexcept { return spec_; }

private:
    [[nodiscard]] bool within_accounting_tolerance(double residual, double scale) const noexcept;

    PerformanceSpecV1 spec_;
    std::vector<PeriodReturnRecord> records_;
    LedgerStatus last_status_{LedgerStatus::OK};
    std::uint64_t ledger_hash_{0};
};

[[nodiscard]] std::string serialize_return_ledger_artifact(
    const ReturnLedger& ledger, const ReturnLedgerArtifactSpec& spec);

}  // namespace performance_analytics

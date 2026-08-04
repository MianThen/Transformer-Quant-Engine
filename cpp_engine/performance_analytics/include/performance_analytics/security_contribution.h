#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "engine_common/types.h"
#include "performance_analytics/performance_spec.h"
#include "performance_analytics/return_ledger.h"

namespace performance_analytics {

enum class ContributionStatus : std::uint8_t {
    OK,
    INVALID_INPUT,
    NON_MONOTONIC_PERIOD,
    EQUITY_DISCONTINUITY,
    EXTERNAL_CASH_FLOW,
    DUPLICATE_SYMBOL,
    DUPLICATE_EXECUTION,
    UNKNOWN_FILL_SYMBOL,
    POSITION_MISMATCH,
    CASH_MISMATCH,
    ACCOUNTING_MISMATCH,
};

struct SecurityContributionInput {
    engine_common::SymbolId symbol_id{0};
    std::uint64_t pit_industry_id{0};
    engine_common::Quantity beginning_quantity{0};
    engine_common::Quantity ending_quantity{0};
    engine_common::Quantity corporate_action_quantity_delta{0};
    double beginning_mark_price{0.0};
    double ending_mark_price{0.0};
    double corporate_action_cash{0.0};
};

struct ContributionFillInput {
    std::int64_t execution_id{0};
    engine_common::SymbolId symbol_id{0};
    engine_common::Side side{engine_common::Side::BUY};
    engine_common::Quantity quantity{0};
    double fill_price{0.0};
    double explicit_fee{0.0};
    engine_common::TimestampNs timestamp{0};
};

struct PeriodContributionInput {
    engine_common::TimestampNs period_start{0};
    engine_common::TimestampNs period_end{0};
    std::uint64_t session_id{0};
    double starting_equity{0.0};
    double ending_equity{0.0};
    double starting_cash{0.0};
    double ending_cash{0.0};
    double cash_interest{0.0};
    double external_cash_flow{0.0};
    std::span<const SecurityContributionInput> securities;
    std::span<const ContributionFillInput> fills;
};

struct SecurityContributionRecord {
    engine_common::SymbolId symbol_id{0};
    std::uint64_t pit_industry_id{0};
    engine_common::Quantity beginning_quantity{0};
    engine_common::Quantity ending_quantity{0};
    engine_common::Quantity corporate_action_quantity_delta{0};
    engine_common::Quantity signed_filled_quantity{0};
    double beginning_mark_price{0.0};
    double ending_mark_price{0.0};
    double beginning_market_value{0.0};
    double ending_market_value{0.0};
    double signed_trade_cash_flow{0.0};
    double market_pnl{0.0};
    double explicit_fees{0.0};
    double corporate_action_cash{0.0};
    double net_contribution{0.0};
};

struct IndustryContributionRecord {
    std::uint64_t pit_industry_id{0};
    double market_pnl{0.0};
    double explicit_fees{0.0};
    double corporate_action_cash{0.0};
    double net_contribution{0.0};
};

struct PeriodContributionRecord {
    engine_common::TimestampNs period_start{0};
    engine_common::TimestampNs period_end{0};
    std::uint64_t session_id{0};
    double starting_equity{0.0};
    double ending_equity{0.0};
    double starting_cash{0.0};
    double ending_cash{0.0};
    double beginning_market_value{0.0};
    double ending_market_value{0.0};
    double period_return{0.0};
    double executed_gross_pnl{0.0};
    double explicit_fees{0.0};
    double net_pnl{0.0};
    double corporate_action_cash{0.0};
    double security_net_contribution{0.0};
    double cash_interest{0.0};
    double cash_residual{0.0};
    double beginning_balance_residual{0.0};
    double ending_balance_residual{0.0};
    double accounting_residual{0.0};
    std::vector<SecurityContributionRecord> securities;
    std::vector<IndustryContributionRecord> industries;
    std::vector<ContributionFillInput> fills;
    std::uint64_t record_hash{0};
};

[[nodiscard]] PeriodReturnInput period_return_input(
    const PeriodContributionRecord& record) noexcept;

class SecurityContributionLedger {
public:
    explicit SecurityContributionLedger(PerformanceSpecV1 spec);

    [[nodiscard]] ContributionStatus append(const PeriodContributionInput& input);
    [[nodiscard]] std::span<const PeriodContributionRecord> records() const noexcept {
        return records_;
    }
    [[nodiscard]] ContributionStatus last_status() const noexcept {
        return last_status_;
    }
    [[nodiscard]] std::uint64_t ledger_hash() const noexcept { return ledger_hash_; }
    [[nodiscard]] const PerformanceSpecV1& spec() const noexcept { return spec_; }

private:
    [[nodiscard]] bool within_accounting_tolerance(double residual,
                                                   double scale) const noexcept;

    PerformanceSpecV1 spec_;
    std::vector<PeriodContributionRecord> records_;
    ContributionStatus last_status_{ContributionStatus::OK};
    std::uint64_t ledger_hash_{0};
};

}  // namespace performance_analytics

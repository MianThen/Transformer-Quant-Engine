#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "performance_analytics/security_contribution.h"

namespace performance_analytics {

enum class PeriodCoordinatorStatus : std::uint8_t {
    OK,
    INVALID_INPUT,
    PERIOD_ALREADY_OPEN,
    NO_OPEN_PERIOD,
    DUPLICATE_EXECUTION,
    DUPLICATE_CORPORATE_ACTION,
    LEDGER_REJECTED,
    FINALIZED,
    FAILED,
};

struct PeriodSecuritySnapshot {
    engine_common::SymbolId symbol_id{0};
    std::uint64_t pit_industry_id{0};
    engine_common::Quantity quantity{0};
    double mark_price{0.0};
};

struct PeriodOpenSnapshot {
    engine_common::TimestampNs period_start{0};
    std::uint64_t session_id{0};
    double starting_equity{0.0};
    double starting_cash{0.0};
    std::span<const PeriodSecuritySnapshot> securities;
};

struct ContributionCorporateActionInput {
    std::uint64_t action_id{0};
    engine_common::SymbolId symbol_id{0};
    engine_common::TimestampNs timestamp{0};
    engine_common::Quantity quantity_delta{0};
    double cash_amount{0.0};
};

struct PeriodCloseSnapshot {
    engine_common::TimestampNs period_end{0};
    double ending_equity{0.0};
    double ending_cash{0.0};
    double cash_interest{0.0};
    double external_cash_flow{0.0};
    std::span<const PeriodSecuritySnapshot> securities;
};

class PeriodContributionCoordinator {
public:
    explicit PeriodContributionCoordinator(PerformanceSpecV1 spec);

    [[nodiscard]] PeriodCoordinatorStatus open_period(
        const PeriodOpenSnapshot& snapshot);
    [[nodiscard]] PeriodCoordinatorStatus record_fill(
        const ContributionFillInput& fill);
    [[nodiscard]] PeriodCoordinatorStatus record_corporate_action(
        const ContributionCorporateActionInput& action);
    [[nodiscard]] PeriodCoordinatorStatus close_period(
        const PeriodCloseSnapshot& snapshot);
    [[nodiscard]] PeriodCoordinatorStatus finalize(
        const PeriodCloseSnapshot& snapshot);

    [[nodiscard]] const SecurityContributionLedger& ledger() const noexcept {
        return ledger_;
    }
    [[nodiscard]] std::span<const ContributionCorporateActionInput>
    corporate_action_history() const noexcept {
        return corporate_action_history_;
    }
    [[nodiscard]] std::uint64_t coordinator_hash() const noexcept {
        return coordinator_hash_;
    }
    [[nodiscard]] PeriodCoordinatorStatus last_status() const noexcept {
        return last_status_;
    }
    [[nodiscard]] bool has_open_period() const noexcept { return open_; }
    [[nodiscard]] bool finalized() const noexcept { return finalized_; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }

private:
    [[nodiscard]] PeriodCoordinatorStatus fail(PeriodCoordinatorStatus status);

    SecurityContributionLedger ledger_;
    engine_common::TimestampNs period_start_{0};
    std::uint64_t session_id_{0};
    double starting_equity_{0.0};
    double starting_cash_{0.0};
    std::vector<PeriodSecuritySnapshot> beginning_securities_;
    std::vector<ContributionFillInput> fills_;
    std::vector<ContributionCorporateActionInput> corporate_actions_;
    std::vector<std::int64_t> execution_ids_;
    std::vector<std::uint64_t> corporate_action_ids_;
    std::vector<ContributionCorporateActionInput> corporate_action_history_;
    std::uint64_t coordinator_hash_{0};
    PeriodCoordinatorStatus last_status_{PeriodCoordinatorStatus::OK};
    bool open_{false};
    bool finalized_{false};
    bool failed_{false};
};

}  // namespace performance_analytics

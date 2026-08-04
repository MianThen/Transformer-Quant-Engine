#include "performance_analytics/period_contribution_replay_sink.h"

#include <utility>
#include <vector>

namespace performance_analytics {

PeriodContributionReplaySink::PeriodContributionReplaySink(PerformanceSpecV1 spec)
    : coordinator_(spec), return_ledger_(std::move(spec)) {}

engine_common::ReplayAnalyticsStatus PeriodContributionReplaySink::fail() noexcept {
    failed_ = true;
    return engine_common::ReplayAnalyticsStatus::FAILED;
}

engine_common::ReplayAnalyticsStatus PeriodContributionReplaySink::on_decision(
    const engine_common::ReplayDecisionEvent&) {
    return failed_ ? engine_common::ReplayAnalyticsStatus::FAILED
                   : engine_common::ReplayAnalyticsStatus::OK;
}

engine_common::ReplayAnalyticsStatus PeriodContributionReplaySink::on_execution(
    const engine_common::ExecutionEvent& event) {
    if (failed_) return engine_common::ReplayAnalyticsStatus::FAILED;
    if (!coordinator_.has_open_period()) {
        return period_started_ ? fail()
                               : engine_common::ReplayAnalyticsStatus::OK;
    }
    constexpr std::uint32_t required_flags =
        engine_common::EXECUTION_HAS_SYMBOL |
        engine_common::EXECUTION_HAS_SIDE |
        engine_common::EXECUTION_HAS_PRICE_SCALE |
        engine_common::EXECUTION_HAS_EXPLICIT_FEE;
    if ((event.audit_flags & required_flags) != required_flags ||
        event.price_scale <= 0 || event.fee_scale <= 0) {
        return fail();
    }
    const double price = static_cast<double>(event.last_price) /
        static_cast<double>(event.price_scale);
    const double fee = static_cast<double>(event.explicit_fee) /
        static_cast<double>(event.fee_scale);
    if (coordinator_.record_fill({
            event.execution_id, event.symbol_id, event.side,
            event.last_quantity, price, fee, event.timestamp}) !=
        PeriodCoordinatorStatus::OK) {
        return fail();
    }
    return engine_common::ReplayAnalyticsStatus::OK;
}

engine_common::ReplayAnalyticsStatus PeriodContributionReplaySink::on_replay_end(
    const engine_common::ReplayEndEvent&) {
    if (failed_ || coordinator_.has_open_period()) return fail();
    return engine_common::ReplayAnalyticsStatus::OK;
}

engine_common::ReplayAnalyticsStatus PeriodContributionReplaySink::on_period_open(
    const engine_common::ReplayPeriodOpenEvent& event) {
    if (failed_) return engine_common::ReplayAnalyticsStatus::FAILED;
    std::vector<PeriodSecuritySnapshot> securities;
    securities.reserve(event.securities.size());
    for (const auto& security : event.securities) {
        securities.push_back({security.symbol_id, security.pit_industry_id,
                              security.quantity, security.mark_price});
    }
    if (coordinator_.open_period({
            event.period_start, event.session_id,
            event.starting_equity, event.starting_cash, securities}) !=
        PeriodCoordinatorStatus::OK) {
        return fail();
    }
    period_started_ = true;
    return engine_common::ReplayAnalyticsStatus::OK;
}

engine_common::ReplayAnalyticsStatus PeriodContributionReplaySink::on_period_close(
    const engine_common::ReplayPeriodCloseEvent& event) {
    if (failed_) return engine_common::ReplayAnalyticsStatus::FAILED;
    std::vector<PeriodSecuritySnapshot> securities;
    securities.reserve(event.securities.size());
    for (const auto& security : event.securities) {
        securities.push_back({security.symbol_id, security.pit_industry_id,
                              security.quantity, security.mark_price});
    }
    if (coordinator_.close_period({
            event.period_end, event.ending_equity, event.ending_cash,
            event.cash_interest, event.external_cash_flow, securities}) !=
        PeriodCoordinatorStatus::OK) {
        return fail();
    }
    const auto contribution_records = coordinator_.ledger().records();
    while (return_ledger_record_count_ < contribution_records.size()) {
        const auto& contribution = contribution_records[return_ledger_record_count_];
        if (return_ledger_.append(period_return_input(contribution)) !=
            LedgerStatus::OK) {
            return fail();
        }
        ++return_ledger_record_count_;
    }
    return engine_common::ReplayAnalyticsStatus::OK;
}

engine_common::ReplayAnalyticsStatus
PeriodContributionReplaySink::on_corporate_action(
    const engine_common::ReplayCorporateActionEvent& event) {
    if (failed_) return engine_common::ReplayAnalyticsStatus::FAILED;
    if (!coordinator_.has_open_period()) {
        return period_started_ ? fail()
                               : engine_common::ReplayAnalyticsStatus::OK;
    }
    if (coordinator_.record_corporate_action({
            event.action_id, event.symbol_id, event.timestamp,
            event.quantity_delta, event.cash_amount}) !=
        PeriodCoordinatorStatus::OK) {
        return fail();
    }
    return engine_common::ReplayAnalyticsStatus::OK;
}

}  // namespace performance_analytics

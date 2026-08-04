#pragma once

#include <cstdint>
#include <span>

#include "engine_common/strategy.h"

namespace engine_common {

enum class ReplayAnalyticsStatus : std::uint8_t {
    OK,
    INVALID_EVENT,
    FAILED,
};

struct ReplayDecisionEvent {
    StrategyDecisionView decision;
    MarketFrameBatchView market;
    PortfolioView portfolio;
};

struct ReplayEndEvent {
    TimestampNs ended_at{0};
    MarketFrameBatchView market;
    PortfolioView portfolio;
};

struct ReplayPeriodSecurity {
    SymbolId symbol_id{0};
    std::uint64_t pit_industry_id{0};
    Quantity quantity{0};
    double mark_price{0.0};
};

struct ReplayPeriodOpenEvent {
    TimestampNs period_start{0};
    std::uint64_t session_id{0};
    double starting_equity{0.0};
    double starting_cash{0.0};
    std::span<const ReplayPeriodSecurity> securities;
};

struct ReplayPeriodCloseEvent {
    TimestampNs period_end{0};
    std::uint64_t session_id{0};
    double ending_equity{0.0};
    double ending_cash{0.0};
    double cash_interest{0.0};
    double external_cash_flow{0.0};
    std::span<const ReplayPeriodSecurity> securities;
};

struct ReplayCorporateActionEvent {
    std::uint64_t action_id{0};
    SymbolId symbol_id{0};
    TimestampNs timestamp{0};
    Quantity quantity_delta{0};
    double cash_amount{0.0};
};

class IReplayAnalyticsSink {
public:
    virtual ~IReplayAnalyticsSink() = default;
    virtual ReplayAnalyticsStatus on_decision(
        const ReplayDecisionEvent& event) = 0;
    virtual ReplayAnalyticsStatus on_execution(
        const ExecutionEvent& event) = 0;
    virtual ReplayAnalyticsStatus on_replay_end(
        const ReplayEndEvent& event) = 0;
    virtual ReplayAnalyticsStatus on_period_open(
        const ReplayPeriodOpenEvent&) {
        return ReplayAnalyticsStatus::OK;
    }
    virtual ReplayAnalyticsStatus on_period_close(
        const ReplayPeriodCloseEvent&) {
        return ReplayAnalyticsStatus::OK;
    }
    virtual ReplayAnalyticsStatus on_corporate_action(
        const ReplayCorporateActionEvent&) {
        return ReplayAnalyticsStatus::OK;
    }
};

}  // namespace engine_common

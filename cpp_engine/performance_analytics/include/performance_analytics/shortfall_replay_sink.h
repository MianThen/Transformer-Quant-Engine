#pragma once

#include <cstdint>
#include <vector>

#include "engine_common/replay_analytics.h"
#include "performance_analytics/shortfall_replay_adapter.h"

namespace performance_analytics {

class ShortfallReplaySink final : public engine_common::IReplayAnalyticsSink {
public:
    explicit ShortfallReplaySink(PerformanceSpecV1 spec);

    engine_common::ReplayAnalyticsStatus on_decision(
        const engine_common::ReplayDecisionEvent& event) override;
    engine_common::ReplayAnalyticsStatus on_execution(
        const engine_common::ExecutionEvent& event) override;
    engine_common::ReplayAnalyticsStatus on_replay_end(
        const engine_common::ReplayEndEvent& event) override;

    [[nodiscard]] const ImplementationShortfallLedger& ledger() const noexcept {
        return adapter_.ledger();
    }
    [[nodiscard]] bool failed() const noexcept { return failed_; }
    [[nodiscard]] ShortfallReplayStatus last_shortfall_status() const noexcept {
        return last_shortfall_status_;
    }

private:
    [[nodiscard]] engine_common::ReplayAnalyticsStatus fail(
        ShortfallReplayStatus status) noexcept;
    [[nodiscard]] engine_common::ReplayAnalyticsStatus close_open_interval(
        engine_common::TimestampNs interval_end,
        const engine_common::MarketFrameBatchView& market,
        const engine_common::PortfolioView& portfolio,
        bool finalize);

    ShortfallReplayAdapter adapter_;
    std::vector<FrozenPaperTarget> open_targets_;
    std::vector<engine_common::SymbolId> filled_symbols_;
    std::uint64_t last_decision_id_{0};
    ShortfallReplayStatus last_shortfall_status_{ShortfallReplayStatus::OK};
    bool failed_{false};
    bool ended_{false};
};

}  // namespace performance_analytics

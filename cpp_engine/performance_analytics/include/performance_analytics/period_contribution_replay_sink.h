#pragma once

#include <utility>

#include "engine_common/replay_analytics.h"
#include "performance_analytics/period_contribution_coordinator.h"
#include "performance_analytics/return_analysis.h"
#include "performance_analytics/return_ledger.h"

namespace performance_analytics {

class PeriodContributionReplaySink final
    : public engine_common::IReplayAnalyticsSink {
public:
    explicit PeriodContributionReplaySink(PerformanceSpecV1 spec);

    engine_common::ReplayAnalyticsStatus on_decision(
        const engine_common::ReplayDecisionEvent&) override;
    engine_common::ReplayAnalyticsStatus on_execution(
        const engine_common::ExecutionEvent& event) override;
    engine_common::ReplayAnalyticsStatus on_replay_end(
        const engine_common::ReplayEndEvent&) override;
    engine_common::ReplayAnalyticsStatus on_period_open(
        const engine_common::ReplayPeriodOpenEvent& event) override;
    engine_common::ReplayAnalyticsStatus on_period_close(
        const engine_common::ReplayPeriodCloseEvent& event) override;
    engine_common::ReplayAnalyticsStatus on_corporate_action(
        const engine_common::ReplayCorporateActionEvent& event) override;

    [[nodiscard]] const SecurityContributionLedger& ledger() const noexcept {
        return coordinator_.ledger();
    }
    [[nodiscard]] const PeriodContributionCoordinator& coordinator() const noexcept {
        return coordinator_;
    }
    [[nodiscard]] const ReturnLedger& return_ledger() const noexcept {
        return return_ledger_;
    }
    [[nodiscard]] std::string serialize_return_ledger_artifact(
        const ReturnLedgerArtifactSpec& spec) const {
        return performance_analytics::serialize_return_ledger_artifact(
            return_ledger_, spec);
    }
    [[nodiscard]] std::string serialize_return_analysis_report(
        ReturnAnalysisManifest manifest) const {
        return build_return_analysis_report(return_ledger_, std::move(manifest))
            .artifact_json;
    }
    [[nodiscard]] bool failed() const noexcept { return failed_; }

private:
    [[nodiscard]] engine_common::ReplayAnalyticsStatus fail() noexcept;

    PeriodContributionCoordinator coordinator_;
    ReturnLedger return_ledger_;
    std::size_t return_ledger_record_count_{0};
    bool failed_{false};
    bool period_started_{false};
};

}  // namespace performance_analytics

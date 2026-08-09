#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "engine_common/types.h"
#include "performance_analytics/implementation_shortfall.h"

namespace performance_analytics {

enum class ShortfallReplayStatus : std::uint8_t {
    OK,
    INVALID_INPUT,
    INTERVAL_ALREADY_OPEN,
    NO_OPEN_INTERVAL,
    UNKNOWN_SYMBOL,
    DUPLICATE_EXECUTION,
    LEDGER_REJECTED,
    FINALIZED,
    FAILED,
};

struct FrozenPaperTarget {
    engine_common::SymbolId symbol_id{0};
    engine_common::Quantity target_quantity{0};
    engine_common::Quantity actual_begin_quantity{0};
    double decision_reference_price{0.0};
    ReferencePriceType reference_price_type{ReferencePriceType::BID_ASK_MIDPOINT};
    std::optional<double> arrival_price;
};

struct ShortfallDecisionSnapshot {
    std::uint64_t decision_id{0};
    std::uint64_t measurement_interval_id{0};
    engine_common::TimestampNs decision_at{0};
    std::span<const FrozenPaperTarget> targets;
};

struct ShortfallCloseAsset {
    engine_common::SymbolId symbol_id{0};
    engine_common::Quantity actual_end_quantity{0};
    double end_mark_price{0.0};
    UnexecutedReason unexecuted_reason{UnexecutedReason::NONE};
};

struct ShortfallCloseSnapshot {
    engine_common::TimestampNs interval_end{0};
    std::span<const ShortfallCloseAsset> assets;
};

class ShortfallReplayAdapter {
public:
    explicit ShortfallReplayAdapter(PerformanceSpecV1 spec);

    [[nodiscard]] ShortfallReplayStatus open_interval(
        const ShortfallDecisionSnapshot& decision);
    [[nodiscard]] ShortfallReplayStatus record_fill(const ShortfallFillInput& fill);
    [[nodiscard]] ShortfallReplayStatus close_interval(
        const ShortfallCloseSnapshot& close);
    [[nodiscard]] ShortfallReplayStatus finalize(const ShortfallCloseSnapshot& close);

    [[nodiscard]] bool has_open_interval() const noexcept { return open_; }
    [[nodiscard]] bool finalized() const noexcept { return finalized_; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }
    [[nodiscard]] ShortfallReplayStatus last_status() const noexcept {
        return last_status_;
    }
    [[nodiscard]] const ImplementationShortfallLedger& ledger() const noexcept {
        return ledger_;
    }

private:
    [[nodiscard]] ShortfallReplayStatus fail(ShortfallReplayStatus status);

    ImplementationShortfallLedger ledger_;
    std::uint64_t decision_id_{0};
    std::uint64_t measurement_interval_id_{0};
    engine_common::TimestampNs decision_at_{0};
    std::vector<FrozenPaperTarget> targets_;
    std::vector<ShortfallFillInput> fills_;
    std::vector<std::int64_t> execution_ids_;
    ShortfallReplayStatus last_status_{ShortfallReplayStatus::OK};
    bool open_{false};
    bool finalized_{false};
    bool failed_{false};
};

}  // namespace performance_analytics

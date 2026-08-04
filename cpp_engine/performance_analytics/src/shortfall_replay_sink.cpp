#include "performance_analytics/shortfall_replay_sink.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace performance_analytics {
namespace {

const engine_common::MarketBar* find_market(
    engine_common::SymbolId symbol_id,
    const engine_common::MarketFrameBatchView& market) {
    const auto found = std::find_if(market.bars.begin(), market.bars.end(),
                                    [&](const auto& bar) {
        return bar.symbol_id == symbol_id;
    });
    return found == market.bars.end() ? nullptr : &*found;
}

const engine_common::PortfolioItem* find_portfolio(
    engine_common::SymbolId symbol_id,
    const engine_common::PortfolioView& portfolio) {
    const auto found = std::find_if(portfolio.items.begin(), portfolio.items.end(),
                                    [&](const auto& item) {
        return item.symbol_id == symbol_id;
    });
    return found == portfolio.items.end() ? nullptr : &*found;
}

}  // namespace

ShortfallReplaySink::ShortfallReplaySink(PerformanceSpecV1 spec)
    : adapter_(std::move(spec)) {}

engine_common::ReplayAnalyticsStatus ShortfallReplaySink::fail(
    ShortfallReplayStatus status) noexcept {
    failed_ = true;
    last_shortfall_status_ = status;
    return engine_common::ReplayAnalyticsStatus::FAILED;
}

engine_common::ReplayAnalyticsStatus ShortfallReplaySink::close_open_interval(
    engine_common::TimestampNs interval_end,
    const engine_common::MarketFrameBatchView& market,
    const engine_common::PortfolioView& portfolio,
    bool finalize) {
    std::vector<ShortfallCloseAsset> closing;
    closing.reserve(open_targets_.size());
    for (const auto& target : open_targets_) {
        const auto* bar = find_market(target.symbol_id, market);
        if (bar == nullptr || !std::isfinite(bar->close) || bar->close <= 0.0) {
            return fail(ShortfallReplayStatus::INVALID_INPUT);
        }
        const auto* item = find_portfolio(target.symbol_id, portfolio);
        const engine_common::Quantity actual_end =
            item == nullptr ? 0 : item->position_quantity;
        UnexecutedReason reason = UnexecutedReason::NONE;
        if (actual_end != target.target_quantity) {
            reason = std::find(filled_symbols_.begin(), filled_symbols_.end(),
                               target.symbol_id) != filled_symbols_.end()
                ? UnexecutedReason::PARTIAL_FILL : UnexecutedReason::UNKNOWN;
        }
        closing.push_back({target.symbol_id, actual_end, bar->close, reason});
    }
    const ShortfallCloseSnapshot snapshot{interval_end, closing};
    last_shortfall_status_ = finalize
        ? adapter_.finalize(snapshot) : adapter_.close_interval(snapshot);
    if (last_shortfall_status_ != ShortfallReplayStatus::OK) {
        return fail(last_shortfall_status_);
    }
    open_targets_.clear();
    filled_symbols_.clear();
    return engine_common::ReplayAnalyticsStatus::OK;
}

engine_common::ReplayAnalyticsStatus ShortfallReplaySink::on_decision(
    const engine_common::ReplayDecisionEvent& event) {
    if (failed_ || ended_) return engine_common::ReplayAnalyticsStatus::FAILED;
    if (!event.decision.valid() || event.market.asof_timestamp != event.decision.decision_at ||
        event.decision.decision_id <= last_decision_id_) {
        return fail(ShortfallReplayStatus::INVALID_INPUT);
    }
    if (adapter_.has_open_interval()) {
        const auto status = close_open_interval(
            event.decision.decision_at, event.market, event.portfolio, false);
        if (status != engine_common::ReplayAnalyticsStatus::OK) return status;
    }

    std::vector<FrozenPaperTarget> targets;
    targets.reserve(event.decision.targets.size());
    for (const auto& target : event.decision.targets) {
        const auto* bar = find_market(target.symbol_id, event.market);
        if (bar == nullptr || !std::isfinite(bar->close) || bar->close <= 0.0) {
            return fail(ShortfallReplayStatus::INVALID_INPUT);
        }
        const auto* item = find_portfolio(target.symbol_id, event.portfolio);
        targets.push_back({
            target.symbol_id,
            target.target_quantity,
            item == nullptr ? 0 : item->position_quantity,
            bar->close,
            ReferencePriceType::OFFICIAL_BAR_PROXY,
            bar->close,
        });
    }
    const ShortfallDecisionSnapshot snapshot{
        event.decision.decision_id,
        event.decision.decision_id,
        event.decision.decision_at,
        targets,
    };
    last_shortfall_status_ = adapter_.open_interval(snapshot);
    if (last_shortfall_status_ != ShortfallReplayStatus::OK) {
        return fail(last_shortfall_status_);
    }
    open_targets_ = std::move(targets);
    filled_symbols_.clear();
    last_decision_id_ = event.decision.decision_id;
    return engine_common::ReplayAnalyticsStatus::OK;
}

engine_common::ReplayAnalyticsStatus ShortfallReplaySink::on_execution(
    const engine_common::ExecutionEvent& event) {
    if (failed_ || ended_ || !adapter_.has_open_interval()) {
        return fail(ShortfallReplayStatus::NO_OPEN_INTERVAL);
    }
    constexpr std::uint32_t required_flags =
        engine_common::EXECUTION_HAS_SYMBOL |
        engine_common::EXECUTION_HAS_SIDE |
        engine_common::EXECUTION_HAS_PRICE_SCALE |
        engine_common::EXECUTION_HAS_EXPLICIT_FEE |
        engine_common::EXECUTION_HAS_DECISION_ID;
    if ((event.audit_flags & required_flags) != required_flags ||
        event.price_scale <= 0 || event.fee_scale <= 0 ||
        event.decision_id != last_decision_id_) {
        return fail(ShortfallReplayStatus::INVALID_INPUT);
    }
    const double fill_price = static_cast<double>(event.last_price) /
        static_cast<double>(event.price_scale);
    const double explicit_fee = static_cast<double>(event.explicit_fee) /
        static_cast<double>(event.fee_scale);
    last_shortfall_status_ = adapter_.record_fill({
        event.execution_id,
        event.symbol_id,
        event.side,
        event.last_quantity,
        fill_price,
        explicit_fee,
        event.timestamp,
    });
    if (last_shortfall_status_ != ShortfallReplayStatus::OK) {
        return fail(last_shortfall_status_);
    }
    if (std::find(filled_symbols_.begin(), filled_symbols_.end(), event.symbol_id) ==
        filled_symbols_.end()) {
        filled_symbols_.push_back(event.symbol_id);
    }
    return engine_common::ReplayAnalyticsStatus::OK;
}

engine_common::ReplayAnalyticsStatus ShortfallReplaySink::on_replay_end(
    const engine_common::ReplayEndEvent& event) {
    if (failed_ || ended_ || event.ended_at <= 0 ||
        event.market.asof_timestamp != event.ended_at) {
        return fail(ShortfallReplayStatus::INVALID_INPUT);
    }
    if (adapter_.has_open_interval()) {
        const auto status = close_open_interval(
            event.ended_at, event.market, event.portfolio, true);
        if (status != engine_common::ReplayAnalyticsStatus::OK) return status;
    }
    ended_ = true;
    return engine_common::ReplayAnalyticsStatus::OK;
}

}  // namespace performance_analytics

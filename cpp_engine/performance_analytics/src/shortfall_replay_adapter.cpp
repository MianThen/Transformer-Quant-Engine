#include "performance_analytics/shortfall_replay_adapter.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace performance_analytics {
namespace {

bool valid_reference_type(ReferencePriceType type) {
    return type == ReferencePriceType::BID_ASK_MIDPOINT ||
        type == ReferencePriceType::ARRIVAL_PRICE_PROXY ||
        type == ReferencePriceType::OFFICIAL_BAR_PROXY;
}

bool valid_unexecuted_reason(UnexecutedReason reason) {
    return reason == UnexecutedReason::NONE ||
        reason == UnexecutedReason::PARTIAL_FILL ||
        reason == UnexecutedReason::CANCELED ||
        reason == UnexecutedReason::REJECTED ||
        reason == UnexecutedReason::LIQUIDITY_LIMIT ||
        reason == UnexecutedReason::TRADING_CONSTRAINT ||
        reason == UnexecutedReason::UNKNOWN;
}

bool valid_target(const FrozenPaperTarget& target) {
    return valid_reference_type(target.reference_price_type) &&
        std::isfinite(target.decision_reference_price) &&
        target.decision_reference_price > 0.0 &&
        (!target.arrival_price.has_value() ||
         (std::isfinite(*target.arrival_price) && *target.arrival_price > 0.0));
}

bool valid_fill(const ShortfallFillInput& fill,
                engine_common::TimestampNs decision_at) {
    const bool valid_side = fill.side == engine_common::Side::BUY ||
        fill.side == engine_common::Side::SELL;
    return fill.execution_id != 0 && valid_side &&
        fill.quantity > 0 && std::isfinite(fill.fill_price) &&
        fill.fill_price > 0.0 && std::isfinite(fill.explicit_fee) &&
        fill.explicit_fee >= 0.0 && fill.timestamp >= decision_at;
}

}  // namespace

ShortfallReplayAdapter::ShortfallReplayAdapter(PerformanceSpecV1 spec)
    : ledger_(std::move(spec)) {}

ShortfallReplayStatus ShortfallReplayAdapter::fail(ShortfallReplayStatus status) {
    targets_.clear();
    fills_.clear();
    open_ = false;
    failed_ = true;
    last_status_ = status;
    return last_status_;
}

ShortfallReplayStatus ShortfallReplayAdapter::open_interval(
    const ShortfallDecisionSnapshot& decision) {
    if (failed_) return ShortfallReplayStatus::FAILED;
    if (finalized_) return fail(ShortfallReplayStatus::FINALIZED);
    if (open_) return fail(ShortfallReplayStatus::INTERVAL_ALREADY_OPEN);
    if (decision.decision_id == 0 || decision.measurement_interval_id == 0 ||
        decision.decision_at <= 0) {
        return fail(ShortfallReplayStatus::INVALID_INPUT);
    }
    if (!ledger_.records().empty()) {
        const auto& previous = ledger_.records().back();
        if (decision.decision_id <= previous.decision_id ||
            decision.measurement_interval_id <= previous.measurement_interval_id ||
            decision.decision_at < previous.interval_end) {
            return fail(ShortfallReplayStatus::INVALID_INPUT);
        }
    }

    targets_.assign(decision.targets.begin(), decision.targets.end());
    std::sort(targets_.begin(), targets_.end(), [](const auto& left, const auto& right) {
        return left.symbol_id < right.symbol_id;
    });
    for (std::size_t index = 0; index < targets_.size(); ++index) {
        if (!valid_target(targets_[index]) ||
            (index > 0 && targets_[index - 1].symbol_id == targets_[index].symbol_id)) {
            return fail(ShortfallReplayStatus::INVALID_INPUT);
        }
    }

    decision_id_ = decision.decision_id;
    measurement_interval_id_ = decision.measurement_interval_id;
    decision_at_ = decision.decision_at;
    fills_.clear();
    open_ = true;
    last_status_ = ShortfallReplayStatus::OK;
    return last_status_;
}

ShortfallReplayStatus ShortfallReplayAdapter::record_fill(
    const ShortfallFillInput& fill) {
    if (failed_) return ShortfallReplayStatus::FAILED;
    if (finalized_) return fail(ShortfallReplayStatus::FINALIZED);
    if (!open_) return fail(ShortfallReplayStatus::NO_OPEN_INTERVAL);
    if (!valid_fill(fill, decision_at_)) {
        return fail(ShortfallReplayStatus::INVALID_INPUT);
    }
    const auto target = std::lower_bound(
        targets_.begin(), targets_.end(), fill.symbol_id,
        [](const FrozenPaperTarget& candidate, engine_common::SymbolId symbol_id) {
            return candidate.symbol_id < symbol_id;
        });
    if (target == targets_.end() || target->symbol_id != fill.symbol_id) {
        return fail(ShortfallReplayStatus::UNKNOWN_SYMBOL);
    }
    if (std::find(execution_ids_.begin(), execution_ids_.end(), fill.execution_id) !=
        execution_ids_.end()) {
        return fail(ShortfallReplayStatus::DUPLICATE_EXECUTION);
    }
    execution_ids_.push_back(fill.execution_id);
    fills_.push_back(fill);
    last_status_ = ShortfallReplayStatus::OK;
    return last_status_;
}

ShortfallReplayStatus ShortfallReplayAdapter::close_interval(
    const ShortfallCloseSnapshot& close) {
    if (failed_) return ShortfallReplayStatus::FAILED;
    if (finalized_) return fail(ShortfallReplayStatus::FINALIZED);
    if (!open_) return fail(ShortfallReplayStatus::NO_OPEN_INTERVAL);
    if (close.interval_end <= decision_at_ || close.assets.size() != targets_.size()) {
        return fail(ShortfallReplayStatus::INVALID_INPUT);
    }
    if (std::any_of(fills_.begin(), fills_.end(), [&](const auto& fill) {
            return fill.timestamp > close.interval_end;
        })) {
        return fail(ShortfallReplayStatus::INVALID_INPUT);
    }

    std::vector<ShortfallCloseAsset> close_assets(close.assets.begin(), close.assets.end());
    std::sort(close_assets.begin(), close_assets.end(),
              [](const auto& left, const auto& right) {
        return left.symbol_id < right.symbol_id;
    });
    std::vector<ShortfallAssetInput> assets;
    assets.reserve(targets_.size());
    for (std::size_t index = 0; index < targets_.size(); ++index) {
        const auto& target = targets_[index];
        const auto& closing = close_assets[index];
        if (closing.symbol_id != target.symbol_id ||
            !std::isfinite(closing.end_mark_price) || closing.end_mark_price <= 0.0 ||
            !valid_unexecuted_reason(closing.unexecuted_reason)) {
            return fail(ShortfallReplayStatus::INVALID_INPUT);
        }
        assets.push_back({
            target.symbol_id,
            target.target_quantity,
            target.actual_begin_quantity,
            closing.actual_end_quantity,
            target.decision_reference_price,
            closing.end_mark_price,
            target.reference_price_type,
            target.arrival_price,
            closing.unexecuted_reason,
        });
    }

    ImplementationShortfallInput input{
        decision_id_, measurement_interval_id_, decision_at_, close.interval_end,
        assets, fills_};
    if (ledger_.append(input) != ShortfallStatus::OK) {
        return fail(ShortfallReplayStatus::LEDGER_REJECTED);
    }
    targets_.clear();
    fills_.clear();
    open_ = false;
    last_status_ = ShortfallReplayStatus::OK;
    return last_status_;
}

ShortfallReplayStatus ShortfallReplayAdapter::finalize(
    const ShortfallCloseSnapshot& close) {
    const auto status = close_interval(close);
    if (status != ShortfallReplayStatus::OK) return status;
    finalized_ = true;
    return last_status_;
}

}  // namespace performance_analytics

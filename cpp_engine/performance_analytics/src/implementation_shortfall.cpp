#include "performance_analytics/implementation_shortfall.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace performance_analytics {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_value(std::uint64_t& hash, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffU;
        hash *= kFnvPrime;
    }
}

void hash_double(std::uint64_t& hash, double value) {
    const double normalized = value == 0.0 ? 0.0 : value;
    hash_value(hash, std::bit_cast<std::uint64_t>(normalized));
}

bool add_quantity(engine_common::Quantity& total,
                  engine_common::Quantity increment) {
    if ((increment > 0 &&
         total > std::numeric_limits<engine_common::Quantity>::max() - increment) ||
        (increment < 0 &&
         total < std::numeric_limits<engine_common::Quantity>::min() - increment)) {
        return false;
    }
    total += increment;
    return true;
}

bool eligible_reference(ReferencePriceType type) {
    return type == ReferencePriceType::BID_ASK_MIDPOINT;
}

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

bool valid_asset(const ShortfallAssetInput& asset) {
    const bool has_unexecuted_quantity =
        asset.target_quantity != asset.actual_end_quantity;
    const bool reason_matches_quantity =
        has_unexecuted_quantity == (asset.unexecuted_reason != UnexecutedReason::NONE);
    return valid_reference_type(asset.reference_price_type) &&
        valid_unexecuted_reason(asset.unexecuted_reason) &&
        std::isfinite(asset.decision_reference_price) &&
        asset.decision_reference_price > 0.0 && std::isfinite(asset.end_mark_price) &&
        asset.end_mark_price > 0.0 &&
        (!asset.arrival_price.has_value() ||
         (std::isfinite(*asset.arrival_price) && *asset.arrival_price > 0.0)) &&
        reason_matches_quantity;
}

bool valid_fill(const ShortfallFillInput& fill,
                engine_common::TimestampNs decision_at,
                engine_common::TimestampNs interval_end) {
    const bool valid_side = fill.side == engine_common::Side::BUY ||
        fill.side == engine_common::Side::SELL;
    return fill.execution_id != 0 && valid_side &&
        fill.quantity > 0 &&
        std::isfinite(fill.fill_price) && fill.fill_price > 0.0 &&
        std::isfinite(fill.explicit_fee) && fill.explicit_fee >= 0.0 &&
        fill.timestamp >= decision_at && fill.timestamp <= interval_end;
}

void hash_asset(std::uint64_t& hash, const ShortfallAssetRecord& asset) {
    hash_value(hash, asset.symbol_id);
    hash_value(hash, static_cast<std::uint64_t>(asset.target_quantity));
    hash_value(hash, static_cast<std::uint64_t>(asset.actual_begin_quantity));
    hash_value(hash, static_cast<std::uint64_t>(asset.actual_end_quantity));
    hash_double(hash, asset.decision_reference_price);
    hash_double(hash, asset.end_mark_price);
    hash_value(hash, static_cast<std::uint8_t>(asset.reference_price_type));
    hash_value(hash, asset.arrival_price.has_value() ? 1U : 0U);
    if (asset.arrival_price.has_value()) hash_double(hash, *asset.arrival_price);
    hash_value(hash, static_cast<std::uint8_t>(asset.unexecuted_reason));
}

void hash_fill(std::uint64_t& hash, const ShortfallFillInput& fill) {
    hash_value(hash, static_cast<std::uint64_t>(fill.execution_id));
    hash_value(hash, fill.symbol_id);
    hash_value(hash, static_cast<std::uint8_t>(fill.side));
    hash_value(hash, static_cast<std::uint64_t>(fill.quantity));
    hash_double(hash, fill.fill_price);
    hash_double(hash, fill.explicit_fee);
    hash_value(hash, static_cast<std::uint64_t>(fill.timestamp));
}

}  // namespace

ImplementationShortfallLedger::ImplementationShortfallLedger(PerformanceSpecV1 spec)
    : spec_(std::move(spec)) {
    if (!valid_performance_spec(spec_)) {
        throw std::invalid_argument("invalid PerformanceSpecV1");
    }
    ledger_hash_ = kFnvOffset;
    hash_value(ledger_hash_, spec_.config_hash);
}

bool ImplementationShortfallLedger::within_accounting_tolerance(
    double residual, double scale) const noexcept {
    return std::abs(residual) <= spec_.accounting_absolute_tolerance +
        spec_.accounting_relative_tolerance * std::max(1.0, std::abs(scale));
}

ShortfallStatus ImplementationShortfallLedger::append(
    const ImplementationShortfallInput& input) {
    auto fail = [&](ShortfallStatus status) {
        last_status_ = status;
        return status;
    };
    if (input.decision_id == 0 || input.measurement_interval_id == 0 ||
        input.decision_at <= 0 || input.interval_end <= input.decision_at) {
        return fail(ShortfallStatus::INVALID_INPUT);
    }
    if (!records_.empty()) {
        const auto& previous = records_.back();
        if (input.decision_id <= previous.decision_id ||
            input.measurement_interval_id <= previous.measurement_interval_id ||
            input.decision_at < previous.interval_end) {
            return fail(ShortfallStatus::NON_MONOTONIC_INTERVAL);
        }
    }

    std::vector<ShortfallAssetInput> assets(input.assets.begin(), input.assets.end());
    std::sort(assets.begin(), assets.end(), [](const auto& left, const auto& right) {
        return left.symbol_id < right.symbol_id;
    });
    for (std::size_t index = 0; index < assets.size(); ++index) {
        if (!valid_asset(assets[index])) return fail(ShortfallStatus::INVALID_INPUT);
        if (index > 0 && assets[index - 1].symbol_id == assets[index].symbol_id) {
            return fail(ShortfallStatus::DUPLICATE_SYMBOL);
        }
    }

    std::vector<ShortfallFillInput> fills(input.fills.begin(), input.fills.end());
    std::sort(fills.begin(), fills.end(), [](const auto& left, const auto& right) {
        if (left.timestamp != right.timestamp) return left.timestamp < right.timestamp;
        return left.execution_id < right.execution_id;
    });
    std::set<std::int64_t> execution_ids;
    for (std::size_t index = 0; index < fills.size(); ++index) {
        if (!valid_fill(fills[index], input.decision_at, input.interval_end)) {
            return fail(ShortfallStatus::INVALID_INPUT);
        }
        if (!execution_ids.insert(fills[index].execution_id).second) {
            return fail(ShortfallStatus::DUPLICATE_EXECUTION);
        }
        const bool seen_in_previous_interval = std::any_of(
            records_.begin(), records_.end(), [&](const auto& previous) {
                return std::any_of(previous.fills.begin(), previous.fills.end(),
                                   [&](const auto& previous_fill) {
                    return previous_fill.execution_id == fills[index].execution_id;
                });
            });
        if (seen_in_previous_interval) {
            return fail(ShortfallStatus::DUPLICATE_EXECUTION);
        }
        const auto asset = std::lower_bound(
            assets.begin(), assets.end(), fills[index].symbol_id,
            [](const ShortfallAssetInput& candidate, engine_common::SymbolId symbol_id) {
                return candidate.symbol_id < symbol_id;
            });
        if (asset == assets.end() || asset->symbol_id != fills[index].symbol_id) {
            return fail(ShortfallStatus::UNKNOWN_FILL_SYMBOL);
        }
    }

    ImplementationShortfallRecord record;
    record.decision_id = input.decision_id;
    record.measurement_interval_id = input.measurement_interval_id;
    record.decision_at = input.decision_at;
    record.interval_end = input.interval_end;
    record.fills = fills;
    record.assets.reserve(assets.size());
    record.promotion_eligible = true;

    for (const auto& asset : assets) {
        ShortfallAssetRecord asset_record;
        asset_record.symbol_id = asset.symbol_id;
        asset_record.target_quantity = asset.target_quantity;
        asset_record.actual_begin_quantity = asset.actual_begin_quantity;
        asset_record.actual_end_quantity = asset.actual_end_quantity;
        asset_record.decision_reference_price = asset.decision_reference_price;
        asset_record.end_mark_price = asset.end_mark_price;
        asset_record.reference_price_type = asset.reference_price_type;
        asset_record.arrival_price = asset.arrival_price;
        asset_record.unexecuted_reason = asset.unexecuted_reason;
        record.promotion_eligible =
            record.promotion_eligible && eligible_reference(asset.reference_price_type);

        long double execution_price_cost = 0.0L;
        long double explicit_fees = 0.0L;
        long double fill_to_end_pnl = 0.0L;
        for (const auto& fill : fills) {
            if (fill.symbol_id != asset.symbol_id) continue;
            const engine_common::Quantity signed_quantity =
                fill.side == engine_common::Side::BUY ? fill.quantity : -fill.quantity;
            if (!add_quantity(asset_record.signed_filled_quantity, signed_quantity)) {
                return fail(ShortfallStatus::INVALID_INPUT);
            }
            execution_price_cost +=
                static_cast<long double>(fill.fill_price - asset.decision_reference_price) *
                static_cast<long double>(signed_quantity);
            explicit_fees += static_cast<long double>(fill.explicit_fee);
            fill_to_end_pnl +=
                static_cast<long double>(asset.end_mark_price - fill.fill_price) *
                static_cast<long double>(signed_quantity);
        }
        engine_common::Quantity reconstructed_end = asset.actual_begin_quantity;
        if (!add_quantity(reconstructed_end, asset_record.signed_filled_quantity) ||
            reconstructed_end != asset.actual_end_quantity) {
            return fail(ShortfallStatus::POSITION_MISMATCH);
        }

        const long double reference_to_end =
            static_cast<long double>(asset.end_mark_price -
                                     asset.decision_reference_price);
        const long double opportunity_cost = reference_to_end *
            (static_cast<long double>(asset.target_quantity) -
             static_cast<long double>(asset.actual_end_quantity));
        const long double paper_pnl = reference_to_end *
            static_cast<long double>(asset.target_quantity);
        const long double real_gross_pnl = reference_to_end *
            static_cast<long double>(asset.actual_begin_quantity) + fill_to_end_pnl;
        const long double real_net_pnl = real_gross_pnl - explicit_fees;

        asset_record.execution_price_cost = static_cast<double>(execution_price_cost);
        asset_record.explicit_fees = static_cast<double>(explicit_fees);
        asset_record.opportunity_cost = static_cast<double>(opportunity_cost);
        asset_record.paper_pnl = static_cast<double>(paper_pnl);
        asset_record.real_gross_pnl = static_cast<double>(real_gross_pnl);
        asset_record.real_net_pnl = static_cast<double>(real_net_pnl);
        const double outputs[] = {
            asset_record.execution_price_cost, asset_record.explicit_fees,
            asset_record.opportunity_cost, asset_record.paper_pnl,
            asset_record.real_gross_pnl, asset_record.real_net_pnl,
        };
        if (!std::all_of(std::begin(outputs), std::end(outputs), [](double value) {
                return std::isfinite(value);
            })) {
            return fail(ShortfallStatus::INVALID_INPUT);
        }
        record.execution_price_cost += asset_record.execution_price_cost;
        record.explicit_fees += asset_record.explicit_fees;
        record.opportunity_cost += asset_record.opportunity_cost;
        record.paper_pnl += asset_record.paper_pnl;
        record.real_gross_pnl += asset_record.real_gross_pnl;
        record.real_net_pnl += asset_record.real_net_pnl;
        record.assets.push_back(asset_record);
    }

    record.implementation_shortfall = record.execution_price_cost +
        record.explicit_fees + record.opportunity_cost;
    record.identity_residual = record.paper_pnl - record.real_net_pnl -
        record.implementation_shortfall;
    const double scale = std::max(std::abs(record.paper_pnl),
                                  std::abs(record.real_net_pnl));
    if (!std::isfinite(record.implementation_shortfall) ||
        !std::isfinite(record.identity_residual) ||
        !within_accounting_tolerance(record.identity_residual, scale)) {
        return fail(ShortfallStatus::ACCOUNTING_MISMATCH);
    }

    record.record_hash = kFnvOffset;
    hash_value(record.record_hash, record.decision_id);
    hash_value(record.record_hash, record.measurement_interval_id);
    hash_value(record.record_hash, static_cast<std::uint64_t>(record.decision_at));
    hash_value(record.record_hash, static_cast<std::uint64_t>(record.interval_end));
    for (const auto& asset : record.assets) hash_asset(record.record_hash, asset);
    for (const auto& fill : record.fills) hash_fill(record.record_hash, fill);

    hash_value(ledger_hash_, record.record_hash);
    records_.push_back(std::move(record));
    last_status_ = ShortfallStatus::OK;
    return last_status_;
}

}  // namespace performance_analytics

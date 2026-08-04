#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "engine_common/types.h"
#include "performance_analytics/performance_spec.h"

namespace performance_analytics {

enum class ReferencePriceType : std::uint8_t {
    BID_ASK_MIDPOINT,
    ARRIVAL_PRICE_PROXY,
    OFFICIAL_BAR_PROXY,
};

enum class ShortfallStatus : std::uint8_t {
    OK,
    INVALID_INPUT,
    NON_MONOTONIC_INTERVAL,
    DUPLICATE_SYMBOL,
    DUPLICATE_EXECUTION,
    UNKNOWN_FILL_SYMBOL,
    POSITION_MISMATCH,
    ACCOUNTING_MISMATCH,
};

enum class UnexecutedReason : std::uint8_t {
    NONE,
    PARTIAL_FILL,
    CANCELED,
    REJECTED,
    LIQUIDITY_LIMIT,
    TRADING_CONSTRAINT,
    UNKNOWN,
};

struct ShortfallAssetInput {
    engine_common::SymbolId symbol_id{0};
    engine_common::Quantity target_quantity{0};
    engine_common::Quantity actual_begin_quantity{0};
    engine_common::Quantity actual_end_quantity{0};
    double decision_reference_price{0.0};
    double end_mark_price{0.0};
    ReferencePriceType reference_price_type{ReferencePriceType::BID_ASK_MIDPOINT};
    std::optional<double> arrival_price;
    UnexecutedReason unexecuted_reason{UnexecutedReason::NONE};
};

struct ShortfallFillInput {
    std::int64_t execution_id{0};
    engine_common::SymbolId symbol_id{0};
    engine_common::Side side{engine_common::Side::BUY};
    engine_common::Quantity quantity{0};
    double fill_price{0.0};
    double explicit_fee{0.0};
    engine_common::TimestampNs timestamp{0};
};

struct ImplementationShortfallInput {
    std::uint64_t decision_id{0};
    std::uint64_t measurement_interval_id{0};
    engine_common::TimestampNs decision_at{0};
    engine_common::TimestampNs interval_end{0};
    std::span<const ShortfallAssetInput> assets;
    std::span<const ShortfallFillInput> fills;
};

struct ShortfallAssetRecord {
    engine_common::SymbolId symbol_id{0};
    engine_common::Quantity target_quantity{0};
    engine_common::Quantity actual_begin_quantity{0};
    engine_common::Quantity actual_end_quantity{0};
    engine_common::Quantity signed_filled_quantity{0};
    double decision_reference_price{0.0};
    double end_mark_price{0.0};
    ReferencePriceType reference_price_type{ReferencePriceType::BID_ASK_MIDPOINT};
    std::optional<double> arrival_price;
    UnexecutedReason unexecuted_reason{UnexecutedReason::NONE};
    double execution_price_cost{0.0};
    double explicit_fees{0.0};
    double opportunity_cost{0.0};
    double paper_pnl{0.0};
    double real_gross_pnl{0.0};
    double real_net_pnl{0.0};
};

struct ImplementationShortfallRecord {
    std::uint64_t decision_id{0};
    std::uint64_t measurement_interval_id{0};
    engine_common::TimestampNs decision_at{0};
    engine_common::TimestampNs interval_end{0};
    std::vector<ShortfallAssetRecord> assets;
    std::vector<ShortfallFillInput> fills;
    double execution_price_cost{0.0};
    double explicit_fees{0.0};
    double opportunity_cost{0.0};
    double implementation_shortfall{0.0};
    double paper_pnl{0.0};
    double real_gross_pnl{0.0};
    double real_net_pnl{0.0};
    double identity_residual{0.0};
    bool promotion_eligible{false};
    std::uint64_t record_hash{0};
};

class ImplementationShortfallLedger {
public:
    explicit ImplementationShortfallLedger(PerformanceSpecV1 spec);

    [[nodiscard]] ShortfallStatus append(const ImplementationShortfallInput& input);
    [[nodiscard]] std::span<const ImplementationShortfallRecord> records() const noexcept {
        return records_;
    }
    [[nodiscard]] ShortfallStatus last_status() const noexcept { return last_status_; }
    [[nodiscard]] std::uint64_t ledger_hash() const noexcept { return ledger_hash_; }
    [[nodiscard]] const PerformanceSpecV1& spec() const noexcept { return spec_; }

private:
    [[nodiscard]] bool within_accounting_tolerance(double residual,
                                                   double scale) const noexcept;

    PerformanceSpecV1 spec_;
    std::vector<ImplementationShortfallRecord> records_;
    ShortfallStatus last_status_{ShortfallStatus::OK};
    std::uint64_t ledger_hash_{0};
};

}  // namespace performance_analytics

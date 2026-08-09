#include "performance_analytics/security_contribution.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
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

bool valid_security(const SecurityContributionInput& security) {
    return std::isfinite(security.beginning_mark_price) &&
        security.beginning_mark_price > 0.0 &&
        std::isfinite(security.ending_mark_price) &&
        security.ending_mark_price > 0.0 &&
        std::isfinite(security.corporate_action_cash);
}

bool valid_fill(const ContributionFillInput& fill,
                engine_common::TimestampNs period_start,
                engine_common::TimestampNs period_end) {
    const bool valid_side = fill.side == engine_common::Side::BUY ||
        fill.side == engine_common::Side::SELL;
    return fill.execution_id != 0 && valid_side && fill.quantity > 0 &&
        std::isfinite(fill.fill_price) && fill.fill_price > 0.0 &&
        std::isfinite(fill.explicit_fee) && fill.explicit_fee >= 0.0 &&
        fill.timestamp >= period_start && fill.timestamp <= period_end;
}

void hash_security(std::uint64_t& hash,
                   const SecurityContributionInput& security) {
    hash_value(hash, security.symbol_id);
    hash_value(hash, security.pit_industry_id);
    hash_value(hash, static_cast<std::uint64_t>(security.beginning_quantity));
    hash_value(hash, static_cast<std::uint64_t>(security.ending_quantity));
    hash_value(hash,
               static_cast<std::uint64_t>(security.corporate_action_quantity_delta));
    hash_double(hash, security.beginning_mark_price);
    hash_double(hash, security.ending_mark_price);
    hash_double(hash, security.corporate_action_cash);
}

void hash_fill(std::uint64_t& hash, const ContributionFillInput& fill) {
    hash_value(hash, static_cast<std::uint64_t>(fill.execution_id));
    hash_value(hash, fill.symbol_id);
    hash_value(hash, static_cast<std::uint8_t>(fill.side));
    hash_value(hash, static_cast<std::uint64_t>(fill.quantity));
    hash_double(hash, fill.fill_price);
    hash_double(hash, fill.explicit_fee);
    hash_value(hash, static_cast<std::uint64_t>(fill.timestamp));
}

}  // namespace

PeriodReturnInput period_return_input(
    const PeriodContributionRecord& record) noexcept {
    PeriodReturnInput input;
    input.period_start = record.period_start;
    input.period_end = record.period_end;
    input.session_id = record.session_id;
    input.starting_equity = record.starting_equity;
    input.ending_equity = record.ending_equity;
    input.executed_gross_pnl = record.executed_gross_pnl;
    input.explicit_fees = record.explicit_fees;
    input.net_pnl = record.net_pnl;
    input.corporate_action_cash = record.corporate_action_cash;
    input.cash_interest = record.cash_interest;
    return input;
}

SecurityContributionLedger::SecurityContributionLedger(PerformanceSpecV1 spec)
    : spec_(std::move(spec)) {
    if (!valid_performance_spec(spec_)) {
        throw std::invalid_argument("invalid PerformanceSpecV1");
    }
    ledger_hash_ = kFnvOffset;
    hash_value(ledger_hash_, spec_.config_hash);
}

bool SecurityContributionLedger::within_accounting_tolerance(
    double residual, double scale) const noexcept {
    return std::abs(residual) <= spec_.accounting_absolute_tolerance +
        spec_.accounting_relative_tolerance * std::max(1.0, std::abs(scale));
}

ContributionStatus SecurityContributionLedger::append(
    const PeriodContributionInput& input) {
    auto fail = [&](ContributionStatus status) {
        last_status_ = status;
        return status;
    };
    const double period_values[] = {
        input.starting_equity, input.ending_equity,
        input.starting_cash, input.ending_cash,
        input.cash_interest, input.external_cash_flow,
    };
    if (!std::all_of(std::begin(period_values), std::end(period_values),
                     [](double value) { return std::isfinite(value); }) ||
        input.period_start <= 0 || input.period_end <= input.period_start ||
        input.session_id == 0 || input.starting_equity <= 0.0 ||
        input.ending_equity < 0.0) {
        return fail(ContributionStatus::INVALID_INPUT);
    }
    if (!records_.empty()) {
        const auto& previous = records_.back();
        if (input.period_start < previous.period_end ||
            input.session_id <= previous.session_id) {
            return fail(ContributionStatus::NON_MONOTONIC_PERIOD);
        }
        if (!within_accounting_tolerance(
                input.starting_equity - previous.ending_equity,
                previous.ending_equity)) {
            return fail(ContributionStatus::EQUITY_DISCONTINUITY);
        }
    }
    if (!within_accounting_tolerance(input.external_cash_flow,
                                     input.starting_equity)) {
        return fail(ContributionStatus::EXTERNAL_CASH_FLOW);
    }

    std::vector<SecurityContributionInput> securities(
        input.securities.begin(), input.securities.end());
    std::sort(securities.begin(), securities.end(),
              [](const auto& left, const auto& right) {
        return left.symbol_id < right.symbol_id;
    });
    for (std::size_t index = 0; index < securities.size(); ++index) {
        if (!valid_security(securities[index])) {
            return fail(ContributionStatus::INVALID_INPUT);
        }
        if (index > 0 &&
            securities[index - 1].symbol_id == securities[index].symbol_id) {
            return fail(ContributionStatus::DUPLICATE_SYMBOL);
        }
    }

    std::vector<ContributionFillInput> fills(input.fills.begin(), input.fills.end());
    std::sort(fills.begin(), fills.end(), [](const auto& left, const auto& right) {
        if (left.timestamp != right.timestamp) return left.timestamp < right.timestamp;
        return left.execution_id < right.execution_id;
    });
    std::set<std::int64_t> execution_ids;
    for (const auto& fill : fills) {
        if (!valid_fill(fill, input.period_start, input.period_end)) {
            return fail(ContributionStatus::INVALID_INPUT);
        }
        if (!execution_ids.insert(fill.execution_id).second) {
            return fail(ContributionStatus::DUPLICATE_EXECUTION);
        }
        const bool seen_in_previous_period = std::any_of(
            records_.begin(), records_.end(), [&](const auto& previous) {
                return std::any_of(previous.fills.begin(), previous.fills.end(),
                                   [&](const auto& previous_fill) {
                    return previous_fill.execution_id == fill.execution_id;
                });
            });
        if (seen_in_previous_period) {
            return fail(ContributionStatus::DUPLICATE_EXECUTION);
        }
        const auto security = std::lower_bound(
            securities.begin(), securities.end(), fill.symbol_id,
            [](const SecurityContributionInput& candidate,
               engine_common::SymbolId symbol_id) {
                return candidate.symbol_id < symbol_id;
            });
        if (security == securities.end() || security->symbol_id != fill.symbol_id) {
            return fail(ContributionStatus::UNKNOWN_FILL_SYMBOL);
        }
    }

    PeriodContributionRecord record;
    record.period_start = input.period_start;
    record.period_end = input.period_end;
    record.session_id = input.session_id;
    record.starting_equity = input.starting_equity;
    record.ending_equity = input.ending_equity;
    record.starting_cash = input.starting_cash;
    record.ending_cash = input.ending_cash;
    record.period_return = input.ending_equity / input.starting_equity - 1.0;
    record.cash_interest = input.cash_interest;
    record.fills = fills;
    record.securities.reserve(securities.size());
    std::map<std::uint64_t, IndustryContributionRecord> industries;
    long double total_signed_trade_cash_flow = 0.0L;

    for (const auto& security : securities) {
        SecurityContributionRecord contribution;
        contribution.symbol_id = security.symbol_id;
        contribution.pit_industry_id = security.pit_industry_id;
        contribution.beginning_quantity = security.beginning_quantity;
        contribution.ending_quantity = security.ending_quantity;
        contribution.corporate_action_quantity_delta =
            security.corporate_action_quantity_delta;
        contribution.beginning_mark_price = security.beginning_mark_price;
        contribution.ending_mark_price = security.ending_mark_price;
        long double signed_trade_cash_flow = 0.0L;
        long double explicit_fees = 0.0L;
        for (const auto& fill : fills) {
            if (fill.symbol_id != security.symbol_id) continue;
            const engine_common::Quantity signed_quantity =
                fill.side == engine_common::Side::BUY ? fill.quantity : -fill.quantity;
            if (!add_quantity(contribution.signed_filled_quantity, signed_quantity)) {
                return fail(ContributionStatus::INVALID_INPUT);
            }
            signed_trade_cash_flow += static_cast<long double>(signed_quantity) *
                static_cast<long double>(fill.fill_price);
            explicit_fees += static_cast<long double>(fill.explicit_fee);
        }
        engine_common::Quantity reconstructed_quantity = security.beginning_quantity;
        if (!add_quantity(reconstructed_quantity,
                          contribution.signed_filled_quantity) ||
            !add_quantity(reconstructed_quantity,
                          security.corporate_action_quantity_delta) ||
            reconstructed_quantity != security.ending_quantity) {
            return fail(ContributionStatus::POSITION_MISMATCH);
        }
        const long double beginning_market_value =
            static_cast<long double>(security.beginning_quantity) *
            static_cast<long double>(security.beginning_mark_price);
        const long double ending_market_value =
            static_cast<long double>(security.ending_quantity) *
            static_cast<long double>(security.ending_mark_price);
        const long double market_pnl = ending_market_value - beginning_market_value -
            signed_trade_cash_flow;
        const long double net_contribution = market_pnl - explicit_fees +
            static_cast<long double>(security.corporate_action_cash);
        contribution.beginning_market_value =
            static_cast<double>(beginning_market_value);
        contribution.ending_market_value = static_cast<double>(ending_market_value);
        contribution.signed_trade_cash_flow =
            static_cast<double>(signed_trade_cash_flow);
        contribution.market_pnl = static_cast<double>(market_pnl);
        contribution.explicit_fees = static_cast<double>(explicit_fees);
        contribution.corporate_action_cash = security.corporate_action_cash;
        contribution.net_contribution = static_cast<double>(net_contribution);
        const double values[] = {
            contribution.beginning_market_value,
            contribution.ending_market_value,
            contribution.signed_trade_cash_flow,
            contribution.market_pnl,
            contribution.explicit_fees,
            contribution.net_contribution,
        };
        if (!std::all_of(std::begin(values), std::end(values),
                         [](double value) { return std::isfinite(value); })) {
            return fail(ContributionStatus::INVALID_INPUT);
        }
        record.executed_gross_pnl += contribution.market_pnl;
        record.beginning_market_value += contribution.beginning_market_value;
        record.ending_market_value += contribution.ending_market_value;
        record.explicit_fees += contribution.explicit_fees;
        record.corporate_action_cash += contribution.corporate_action_cash;
        record.security_net_contribution += contribution.net_contribution;
        total_signed_trade_cash_flow += signed_trade_cash_flow;
        auto& industry = industries[security.pit_industry_id];
        industry.pit_industry_id = security.pit_industry_id;
        industry.market_pnl += contribution.market_pnl;
        industry.explicit_fees += contribution.explicit_fees;
        industry.corporate_action_cash += contribution.corporate_action_cash;
        industry.net_contribution += contribution.net_contribution;
        record.securities.push_back(contribution);
    }
    record.net_pnl = record.executed_gross_pnl - record.explicit_fees;
    for (const auto& [industry_id, industry] : industries) {
        static_cast<void>(industry_id);
        record.industries.push_back(industry);
    }
    const long double reconstructed_cash_delta =
        -total_signed_trade_cash_flow -
        static_cast<long double>(record.explicit_fees) +
        static_cast<long double>(record.corporate_action_cash) +
        static_cast<long double>(record.cash_interest) +
        static_cast<long double>(input.external_cash_flow);
    record.cash_residual = input.ending_cash - input.starting_cash -
        static_cast<double>(reconstructed_cash_delta);
    record.beginning_balance_residual = input.starting_equity -
        input.starting_cash - record.beginning_market_value;
    record.ending_balance_residual = input.ending_equity -
        input.ending_cash - record.ending_market_value;
    const double equity_delta = input.ending_equity - input.starting_equity;
    const double reconstructed_equity_delta = record.security_net_contribution +
        record.cash_interest + input.external_cash_flow;
    record.accounting_residual = equity_delta - reconstructed_equity_delta;
    if (!within_accounting_tolerance(record.cash_residual, input.starting_cash)) {
        return fail(ContributionStatus::CASH_MISMATCH);
    }
    if (!within_accounting_tolerance(record.beginning_balance_residual,
                                     input.starting_equity) ||
        !within_accounting_tolerance(record.ending_balance_residual,
                                     input.ending_equity) ||
        !within_accounting_tolerance(record.accounting_residual,
                                     input.starting_equity)) {
        return fail(ContributionStatus::ACCOUNTING_MISMATCH);
    }

    record.record_hash = kFnvOffset;
    hash_value(record.record_hash, static_cast<std::uint64_t>(record.period_start));
    hash_value(record.record_hash, static_cast<std::uint64_t>(record.period_end));
    hash_value(record.record_hash, record.session_id);
    hash_double(record.record_hash, record.starting_equity);
    hash_double(record.record_hash, record.ending_equity);
    hash_double(record.record_hash, record.starting_cash);
    hash_double(record.record_hash, record.ending_cash);
    hash_double(record.record_hash, record.cash_interest);
    for (const auto& security : securities) hash_security(record.record_hash, security);
    for (const auto& fill : fills) hash_fill(record.record_hash, fill);
    hash_value(ledger_hash_, record.record_hash);
    records_.push_back(std::move(record));
    last_status_ = ContributionStatus::OK;
    return last_status_;
}

}  // namespace performance_analytics

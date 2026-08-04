#include "performance_analytics/return_ledger.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "performance_analytics/return_analysis.h"

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

bool finite_input(const PeriodReturnInput& input) {
    const double values[] = {
        input.starting_equity, input.ending_equity, input.executed_gross_pnl,
        input.explicit_fees, input.net_pnl, input.corporate_action_cash,
        input.cash_interest, input.external_cash_flow,
    };
    return std::all_of(std::begin(values), std::end(values), [](double value) {
        return std::isfinite(value);
    }) && (!input.benchmark_return.has_value() ||
           std::isfinite(*input.benchmark_return));
}

std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
        case '\\': output << "\\\\"; break;
        case '"': output << "\\\""; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20U) {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned int>(character) << std::dec;
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    return output.str();
}

std::string json_double(double value) {
    char buffer[64]{};
    const auto result = std::to_chars(
        buffer, buffer + sizeof(buffer), value, std::chars_format::general);
    if (result.ec != std::errc{}) {
        throw std::runtime_error("failed to format JSON number");
    }
    std::string text(buffer, result.ptr);
    if (text.find('.') == std::string::npos && text.find('e') == std::string::npos &&
        text.find('E') == std::string::npos) {
        text += ".0";
    }
    return text;
}

bool reference_price_ready(const std::string& quality) {
    return !quality.empty() && quality != "UNKNOWN" &&
        quality != "UNAVAILABLE";
}

std::string serialize_records(std::span<const PeriodReturnRecord> records) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < records.size(); ++index) {
        if (index != 0) output << ',';
        const auto& record = records[index];
        output << "{\"accounting_residual\":" << json_double(record.accounting_residual)
               << ",\"benchmark_return\":";
        if (record.benchmark_return.has_value()) {
            output << json_double(*record.benchmark_return);
        } else {
            output << "null";
        }
        output << ",\"cash_interest\":" << json_double(record.cash_interest)
               << ",\"corporate_action_cash\":"
               << json_double(record.corporate_action_cash)
               << ",\"ending_equity\":" << json_double(record.ending_equity)
               << ",\"executed_gross_pnl\":" << json_double(record.executed_gross_pnl)
               << ",\"explicit_fees\":" << json_double(record.explicit_fees)
               << ",\"net_pnl\":" << json_double(record.net_pnl)
               << ",\"period_end\":" << record.period_end
               << ",\"period_return\":" << json_double(record.period_return)
               << ",\"period_start\":" << record.period_start
               << ",\"session_id\":" << record.session_id
               << ",\"starting_equity\":" << json_double(record.starting_equity)
               << '}';
    }
    output << ']';
    return output.str();
}

}  // namespace

ReturnLedger::ReturnLedger(PerformanceSpecV1 spec) : spec_(std::move(spec)) {
    if (!valid_performance_spec(spec_)) {
        throw std::invalid_argument("invalid PerformanceSpecV1");
    }
    ledger_hash_ = kFnvOffset;
    hash_value(ledger_hash_, spec_.config_hash);
}

bool ReturnLedger::within_accounting_tolerance(double residual, double scale) const noexcept {
    return std::abs(residual) <= spec_.accounting_absolute_tolerance +
        spec_.accounting_relative_tolerance * std::max(1.0, std::abs(scale));
}

LedgerStatus ReturnLedger::append(const PeriodReturnInput& input) {
    auto fail = [&](LedgerStatus status) {
        last_status_ = status;
        return status;
    };
    if (!finite_input(input) || input.period_start <= 0 ||
        input.period_end <= input.period_start || input.session_id == 0 ||
        !(input.starting_equity > 0.0) || input.ending_equity < 0.0 ||
        input.explicit_fees < 0.0) {
        return fail(LedgerStatus::INVALID_INPUT);
    }
    if (!records_.empty()) {
        const auto& previous = records_.back();
        if (input.period_start < previous.period_end || input.session_id <= previous.session_id) {
            return fail(LedgerStatus::NON_MONOTONIC_PERIOD);
        }
        if (!within_accounting_tolerance(
                input.starting_equity - previous.ending_equity, previous.ending_equity)) {
            return fail(LedgerStatus::EQUITY_DISCONTINUITY);
        }
    }
    if (!within_accounting_tolerance(input.external_cash_flow, input.starting_equity)) {
        return fail(LedgerStatus::EXTERNAL_CASH_FLOW);
    }
    const double net_bridge_residual =
        input.net_pnl - (input.executed_gross_pnl - input.explicit_fees);
    if (!within_accounting_tolerance(net_bridge_residual, input.executed_gross_pnl)) {
        return fail(LedgerStatus::ACCOUNTING_MISMATCH);
    }
    const double equity_delta = input.ending_equity - input.starting_equity;
    const double reconstructed_delta = input.net_pnl + input.corporate_action_cash +
                                       input.cash_interest + input.external_cash_flow;
    const double accounting_residual = equity_delta - reconstructed_delta;
    if (!within_accounting_tolerance(accounting_residual, input.starting_equity)) {
        return fail(LedgerStatus::ACCOUNTING_MISMATCH);
    }

    records_.push_back({
        input.period_start, input.period_end, input.session_id,
        input.starting_equity, input.ending_equity,
        input.ending_equity / input.starting_equity - 1.0,
        input.executed_gross_pnl, input.explicit_fees, input.net_pnl,
        input.corporate_action_cash, input.cash_interest, accounting_residual,
        input.benchmark_return,
    });
    hash_value(ledger_hash_, static_cast<std::uint64_t>(input.period_start));
    hash_value(ledger_hash_, static_cast<std::uint64_t>(input.period_end));
    hash_value(ledger_hash_, input.session_id);
    hash_value(ledger_hash_, std::bit_cast<std::uint64_t>(input.starting_equity));
    hash_value(ledger_hash_, std::bit_cast<std::uint64_t>(input.ending_equity));
    hash_value(ledger_hash_, std::bit_cast<std::uint64_t>(input.executed_gross_pnl));
    hash_value(ledger_hash_, std::bit_cast<std::uint64_t>(input.explicit_fees));
    hash_value(ledger_hash_, std::bit_cast<std::uint64_t>(input.net_pnl));
    hash_value(ledger_hash_, std::bit_cast<std::uint64_t>(input.corporate_action_cash));
    hash_value(ledger_hash_, std::bit_cast<std::uint64_t>(input.cash_interest));
    hash_value(ledger_hash_, input.benchmark_return.has_value() ? 1U : 0U);
    if (input.benchmark_return.has_value()) {
        hash_value(ledger_hash_, std::bit_cast<std::uint64_t>(*input.benchmark_return));
    }
    last_status_ = LedgerStatus::OK;
    return last_status_;
}

std::optional<double> ReturnLedger::cumulative_return() const noexcept {
    if (records_.empty()) return std::nullopt;
    double growth = 1.0;
    for (const auto& record : records_) growth *= 1.0 + record.period_return;
    return growth - 1.0;
}

std::string ReturnLedger::ledger_sha256() const {
    return sha256_text(serialize_records(records_) + "\n");
}

std::string serialize_return_ledger_artifact(
    const ReturnLedger& ledger, const ReturnLedgerArtifactSpec& spec) {
    const std::string records = serialize_records(ledger.records());
    const bool proxy = spec.reference_price_quality == "PROXY" ||
        spec.reference_price_quality == "ARRIVAL_PROXY";
    const bool promotion = spec.promotion_eligible && spec.benchmark_available &&
        reference_price_ready(spec.reference_price_quality) && !proxy;
    std::ostringstream output;
    output << "{\"schema_version\":1,\"ledger_sha256\":\""
           << sha256_text(records + "\n") << "\",\"records\":" << records
           << ",\"manifest\":{\"benchmark_artifact_sha256\":\""
           << json_escape(spec.benchmark_artifact_sha256)
           << "\",\"benchmark_available\":"
           << (spec.benchmark_available ? "true" : "false")
           << ",\"dataset_fingerprint\":\""
           << json_escape(spec.dataset_fingerprint)
           << "\",\"promotion_eligible\":" << (promotion ? "true" : "false")
           << ",\"reference_price_quality\":\""
           << json_escape(spec.reference_price_quality)
           << "\",\"source_replay_sha256\":\""
           << json_escape(spec.source_replay_sha256) << "\",\"limitations\":[";
    for (std::size_t index = 0; index < spec.limitations.size(); ++index) {
        if (index != 0) output << ',';
        output << '"' << json_escape(spec.limitations[index]) << '"';
    }
    output << "]}}";
    return output.str();
}

}  // namespace performance_analytics

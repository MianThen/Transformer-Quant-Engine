#include "performance_analytics/infots_replay.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "performance_analytics/performance_spec.h"
#include "performance_analytics/return_analysis.h"
#include "performance_analytics/return_ledger.h"

namespace performance_analytics {
namespace {

bool digest_like(const std::string& value) {
    return value.size() == 64 && std::all_of(
        value.begin(), value.end(), [](char c) {
            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
}

bool finite(double value) noexcept { return std::isfinite(value); }

void append_u32(std::string& output, std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8) {
        output.push_back(static_cast<char>((value >> shift) & 0xffU));
    }
}

void append_u64(std::string& output, std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        output.push_back(static_cast<char>((value >> shift) & 0xffU));
    }
}

void append_i64(std::string& output, std::int64_t value) {
    append_u64(output, static_cast<std::uint64_t>(value));
}

void append_double(std::string& output, double value) {
    append_u64(output, std::bit_cast<std::uint64_t>(value == 0.0 ? 0.0 : value));
}

void append_string(std::string& output, const std::string& value) {
    append_u64(output, value.size());
    output.append(value);
}

void append_spec(std::string& payload, const InfoTSReplaySpecV1& spec) {
    append_u32(payload, spec.schema_version);
    append_string(payload, spec.group_id);
    append_u32(payload, spec.fold_id);
    append_string(payload, spec.policy_id);
    append_string(payload, spec.contract_sha256);
    append_string(payload, spec.dataset_fingerprint);
    append_string(payload, spec.prediction_artifact_sha256);
    append_string(payload, spec.validation_embedding_sha256);
    append_string(payload, spec.test_embedding_sha256);
    append_string(payload, spec.source_snapshot_set_sha256);
    append_string(payload, spec.calendar_id);
    append_double(payload, spec.calendar_periods_per_year);
    append_u64(payload, spec.config_hash);
    append_u32(payload, spec.minimum_tail_observations);
    append_double(payload, spec.initial_equity);
    append_string(payload, spec.claim_scope);
    append_string(payload, spec.reference_price_quality);
    append_string(payload, spec.execution_data_state);
    append_string(payload, spec.corporate_action_state);
    append_string(payload, spec.bar_reference_policy);
    append_string(payload, spec.action_policy);
    append_string(payload, spec.lot_policy);
    append_string(payload, spec.limit_policy);
    append_string(payload, spec.fee_policy);
    append_string(payload, spec.slippage_policy);
}

void append_row(std::string& payload, const InfoTSReplayRowV1& row) {
    append_u64(payload, row.session_id);
    append_i64(payload, row.prediction_available_at);
    append_i64(payload, row.decision_at);
    append_i64(payload, row.realized_at);
    append_double(payload, row.realized_proxy_return);
    for (double output : row.prediction_outputs) append_double(payload, output);
}

std::string status_json(InfoTSReplayStatus status);

std::string build_input_payload(const InfoTSReplaySpecV1& spec,
                                const std::vector<InfoTSReplayRowV1>& rows) {
    constexpr char kMagic[] = "QBT-PHASE5-INFOTS-REPLAY-V1\0";
    std::string payload{kMagic, sizeof(kMagic) - 1};
    append_spec(payload, spec);
    append_u64(payload, rows.size());
    for (const auto& row : rows) append_row(payload, row);
    return payload;
}

std::string build_report_hash_payload(const InfoTSReplayResultV1& result,
                                      const InfoTSReplaySpecV1& spec) {
    constexpr char kMagic[] = "QBT-PHASE5-INFOTS-REPLAY-REPORT-V1\0";
    std::string payload{kMagic, sizeof(kMagic) - 1};
    append_string(payload, status_json(result.status));
    append_string(payload, spec.group_id);
    append_u32(payload, spec.fold_id);
    append_u32(payload, spec.minimum_tail_observations);
    append_string(payload, spec.policy_id);
    append_u64(payload, result.row_count);
    append_u64(payload, result.observations);
    append_string(payload, result.source_replay_sha256);
    append_string(payload, result.ledger_sha256);
    append_u64(payload, result.ledger_hash);
    append_string(payload, spec.dataset_fingerprint);
    append_string(payload, spec.contract_sha256);
    append_string(payload, spec.prediction_artifact_sha256);
    append_string(payload, spec.validation_embedding_sha256);
    append_string(payload, spec.test_embedding_sha256);
    append_string(payload, spec.source_snapshot_set_sha256);
    append_double(payload, result.cumulative_return);
    append_double(payload, result.sharpe);
    append_double(payload, result.maximum_drawdown);
    append_double(payload, result.var_loss);
    append_double(payload, result.expected_shortfall_loss);
    append_double(payload, result.return_cvar);
    append_u32(payload, result.research_comparison_eligible ? 1U : 0U);
    append_u32(payload, result.phase_exit_eligible ? 1U : 0U);
    append_u32(payload, result.promotion_eligible ? 1U : 0U);
    append_u32(payload, 1U);
    append_u32(payload, 1U);
    return payload;
}

std::string escape(const std::string& value) {
    std::string output;
    output.reserve(value.size());
    for (char c : value) {
        if (c == '\\') output += "\\\\";
        else if (c == '"') output += "\\\"";
        else if (c == '\n') output += "\\n";
        else if (c == '\r') output += "\\r";
        else output += c;
    }
    return output;
}

std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

bool valid_row_shape(const InfoTSReplayRowV1& row) noexcept {
    if (row.session_id == 0 || row.prediction_available_at <= 0 ||
        row.decision_at <= 0 || row.realized_at <= row.decision_at ||
        row.prediction_available_at > row.decision_at ||
        !finite(row.realized_proxy_return) || row.realized_proxy_return <= -1.0) {
        return false;
    }
    for (double value : row.prediction_outputs) {
        if (!finite(value)) return false;
    }
    const double volatility = row.prediction_outputs[1];
    const double direction = row.prediction_outputs[2];
    const double lower = row.prediction_outputs[3];
    const double upper = row.prediction_outputs[4];
    const double confidence = row.prediction_outputs[5];
    return volatility >= 0.0 && direction >= 0.0 && direction <= 1.0 &&
           lower <= upper && confidence >= 0.0 && confidence <= 1.0;
}

bool valid_rows(const std::vector<InfoTSReplayRowV1>& rows) noexcept {
    if (rows.size() < 2) return false;
    for (std::size_t index = 0; index < rows.size(); ++index) {
        if (!valid_row_shape(rows[index])) return false;
        if (index == 0) continue;
        const auto& previous = rows[index - 1];
        const auto& current = rows[index];
        if (current.session_id <= previous.session_id ||
            current.prediction_available_at < previous.prediction_available_at ||
            current.decision_at < previous.decision_at ||
            current.decision_at < previous.realized_at ||
            current.realized_at <= previous.realized_at) {
            return false;
        }
    }
    return true;
}

std::string status_json(InfoTSReplayStatus status) {
    return std::string(infots_replay_status_name(status));
}

}  // namespace

const char* infots_replay_status_name(InfoTSReplayStatus status) noexcept {
    switch (status) {
    case InfoTSReplayStatus::OK: return "OK";
    case InfoTSReplayStatus::INVALID_SPEC: return "INVALID_SPEC";
    case InfoTSReplayStatus::INVALID_INPUT: return "INVALID_INPUT";
    case InfoTSReplayStatus::FUTURE_LEAKAGE: return "FUTURE_LEAKAGE";
    case InfoTSReplayStatus::PROVENANCE_MISMATCH: return "PROVENANCE_MISMATCH";
    case InfoTSReplayStatus::REPLAY_FAILURE: return "REPLAY_FAILURE";
    }
    return "UNKNOWN";
}

bool valid_infots_replay_spec(const InfoTSReplaySpecV1& spec) noexcept {
    return spec.schema_version == 1 && !spec.group_id.empty() &&
        (spec.group_id == "from_scratch" || spec.group_id == "fixed_augmentation" ||
         spec.group_id == "infots") && spec.fold_id > 0 && spec.fold_id <= 3 &&
        !spec.policy_id.empty() &&
        digest_like(spec.contract_sha256) && digest_like(spec.dataset_fingerprint) &&
        digest_like(spec.prediction_artifact_sha256) &&
        digest_like(spec.validation_embedding_sha256) &&
        digest_like(spec.test_embedding_sha256) &&
        digest_like(spec.source_snapshot_set_sha256) &&
        !spec.calendar_id.empty() && finite(spec.calendar_periods_per_year) &&
        spec.calendar_periods_per_year > 0.0 && spec.config_hash != 0 &&
        spec.minimum_tail_observations >= 2 && finite(spec.initial_equity) &&
        spec.initial_equity > 0.0 && spec.claim_scope == "RESEARCH_PROXY" &&
        spec.reference_price_quality == "PROXY" &&
        spec.execution_data_state == "UNAVAILABLE" &&
        spec.corporate_action_state == "UNAVAILABLE" &&
        spec.bar_reference_policy == "PROXY_BAR_REFERENCE" &&
        spec.action_policy == "NO_ACTION" && spec.lot_policy == "LOT_1" &&
        spec.limit_policy == "DISABLED" && spec.fee_policy == "ASSUMED_ZERO" &&
        spec.slippage_policy == "UNAVAILABLE";
}

std::string infots_replay_input_sha256(
    const InfoTSReplaySpecV1& spec, const std::vector<InfoTSReplayRowV1>& rows) {
    return sha256_text(build_input_payload(spec, rows));
}

InfoTSReplayResultV1 run_infots_precomputed_replay(
    const InfoTSReplaySpecV1& spec, const std::vector<InfoTSReplayRowV1>& rows) {
    InfoTSReplayResultV1 result;
    result.group_id = spec.group_id;
    result.fold_id = spec.fold_id;
    result.policy_id = spec.policy_id;
    result.row_count = rows.size();
    if (!valid_infots_replay_spec(spec)) {
        result.status = InfoTSReplayStatus::INVALID_SPEC;
        return result;
    }
    if (!valid_rows(rows)) {
        for (std::size_t index = 0; index < rows.size(); ++index) {
            const auto& row = rows[index];
            if (!valid_row_shape(row)) {
                result.status = (row.prediction_available_at > row.decision_at ||
                                 row.realized_at <= row.decision_at)
                    ? InfoTSReplayStatus::FUTURE_LEAKAGE
                    : InfoTSReplayStatus::INVALID_INPUT;
                return result;
            }
            if (index > 0 && (row.decision_at < rows[index - 1].realized_at ||
                              row.realized_at <= rows[index - 1].realized_at)) {
                result.status = InfoTSReplayStatus::FUTURE_LEAKAGE;
                return result;
            }
        }
        result.status = InfoTSReplayStatus::INVALID_INPUT;
        return result;
    }
    if (rows.size() < spec.minimum_tail_observations) {
        result.status = InfoTSReplayStatus::REPLAY_FAILURE;
        return result;
    }
    result.source_replay_sha256 = infots_replay_input_sha256(spec, rows);
    if (!spec.declared_source_replay_sha256.empty() &&
        spec.declared_source_replay_sha256 != result.source_replay_sha256) {
        result.status = InfoTSReplayStatus::PROVENANCE_MISMATCH;
        result.source_replay_sha256.clear();
        return result;
    }

    PerformanceSpecV1 performance_spec;
    performance_spec.calendar_id = spec.calendar_id;
    performance_spec.calendar_periods_per_year = spec.calendar_periods_per_year;
    performance_spec.minimum_tail_observations = spec.minimum_tail_observations;
    performance_spec.config_hash = spec.config_hash;
    ReturnLedger ledger(performance_spec);
    double equity = spec.initial_equity;
    for (const auto& row : rows) {
        const double next_equity = equity * (1.0 + row.realized_proxy_return);
        const double pnl = next_equity - equity;
        const PeriodReturnInput input{
            row.decision_at, row.realized_at, row.session_id, equity,
            next_equity, pnl, 0.0, pnl, 0.0, 0.0, 0.0, std::nullopt};
        if (ledger.append(input) != LedgerStatus::OK) {
            result.status = InfoTSReplayStatus::REPLAY_FAILURE;
            result.ledger_sha256.clear();
            return result;
        }
        equity = next_equity;
    }

    std::vector<double> returns;
    returns.reserve(rows.size());
    for (const auto& record : ledger.records()) returns.push_back(record.period_return);
    const auto metrics = compute_return_metrics(returns, performance_spec);
    if (metrics.status != AnalysisStatus::OK &&
        metrics.status != AnalysisStatus::ZERO_VARIANCE) {
        result.status = InfoTSReplayStatus::REPLAY_FAILURE;
        return result;
    }
    result.status = InfoTSReplayStatus::OK;
    result.ledger_hash = ledger.ledger_hash();
    result.ledger_sha256 = ledger.ledger_sha256();
    result.observations = metrics.observations;
    result.cumulative_return = metrics.cumulative_return;
    result.sharpe = metrics.sharpe;
    result.maximum_drawdown = metrics.maximum_drawdown;
    result.return_cvar = metrics.cvar;
    if (std::isfinite(metrics.cvar)) {
        result.expected_shortfall_loss = -metrics.cvar;
        std::vector<double> sorted(returns);
        std::sort(sorted.begin(), sorted.end());
        const std::size_t tail_count = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(sorted.size() * 0.05)));
        result.var_loss = -sorted[tail_count - 1];
    }
    result.research_comparison_eligible = true;
    result.phase_exit_eligible = false;
    result.promotion_eligible = false;
    result.artifact_json = serialize_infots_replay_artifact(result, spec);
    const std::string marker = ",\"artifact_sha256\":\"";
    const auto marker_position = result.artifact_json.rfind(marker);
    if (marker_position != std::string::npos &&
        result.artifact_json.size() >= marker_position + marker.size() + 64) {
        result.artifact_sha256 = result.artifact_json.substr(
            marker_position + marker.size(), 64);
    }
    return result;
}

std::string serialize_infots_replay_artifact(
    const InfoTSReplayResultV1& result, const InfoTSReplaySpecV1& spec) {
    if (result.status != InfoTSReplayStatus::OK ||
        result.source_replay_sha256.empty() || result.ledger_sha256.empty()) {
        return {};
    }
    std::ostringstream output;
    output << std::setprecision(17);
    output << "{\"schema_version\":1,\"role\":\"phase5_infots_precomputed_replay\""
           << ",\"status\":\"" << status_json(result.status) << "\""
           << ",\"evidence_level\":\"RESEARCH_PROXY\""
           << ",\"test_blind\":true"
           << ",\"purged_oos_guard_passed\":true"
           << ",\"group_id\":\"" << escape(spec.group_id) << "\""
           << ",\"fold_id\":" << spec.fold_id
           << ",\"minimum_tail_observations\":" << spec.minimum_tail_observations
           << ",\"policy_id\":\"" << escape(spec.policy_id) << "\""
           << ",\"prediction_outputs\":[";
    for (std::size_t index = 0; index < kInfoTSReplayPredictionOutputs.size(); ++index) {
        if (index) output << ',';
        output << '\"' << kInfoTSReplayPredictionOutputs[index] << '\"';
    }
    output << "]"
           << ",\"row_count\":" << result.row_count
           << ",\"observations\":" << result.observations
           << ",\"source_replay_sha256\":\"" << result.source_replay_sha256 << "\""
           << ",\"ledger_sha256\":\"" << result.ledger_sha256 << "\""
           << ",\"ledger_hash\":" << result.ledger_hash
           << ",\"dataset_fingerprint\":\"" << spec.dataset_fingerprint << "\""
           << ",\"contract_sha256\":\"" << spec.contract_sha256 << "\""
           << ",\"prediction_artifact_sha256\":\"" << spec.prediction_artifact_sha256 << "\""
           << ",\"validation_embedding_sha256\":\"" << spec.validation_embedding_sha256 << "\""
           << ",\"test_embedding_sha256\":\"" << spec.test_embedding_sha256 << "\""
           << ",\"source_snapshot_set_sha256\":\"" << spec.source_snapshot_set_sha256 << "\""
           << ",\"metrics\":{\"cumulative_return\":" << number(result.cumulative_return)
           << ",\"sharpe\":" << number(result.sharpe)
           << ",\"maximum_drawdown\":" << number(result.maximum_drawdown)
           << ",\"var_loss\":" << number(result.var_loss)
           << ",\"expected_shortfall_loss\":" << number(result.expected_shortfall_loss)
           << ",\"return_cvar\":" << number(result.return_cvar) << '}';
    output << ",\"claim_scope\":\"" << spec.claim_scope << "\""
           << ",\"reference_price_quality\":\"" << spec.reference_price_quality << "\""
           << ",\"execution_data_state\":\"" << spec.execution_data_state << "\""
           << ",\"corporate_action_state\":\"" << spec.corporate_action_state << "\""
           << ",\"bar_reference_policy\":\"" << spec.bar_reference_policy << "\""
           << ",\"action_policy\":\"" << spec.action_policy << "\""
           << ",\"lot_policy\":\"" << spec.lot_policy << "\""
           << ",\"limit_policy\":\"" << spec.limit_policy << "\""
           << ",\"fee_policy\":\"" << spec.fee_policy << "\""
           << ",\"slippage_policy\":\"" << spec.slippage_policy << "\""
           << ",\"research_comparison_eligible\":true"
           << ",\"phase_exit_eligible\":false"
           << ",\"promotion_eligible\":false"
           << ",\"limitations\":[\"precomputed predictions only\",\"RESEARCH_PROXY claim scope\",\"no historical execution fields\"]}";
    const std::string unsigned_artifact = output.str();
    const std::string artifact_hash =
        sha256_text(build_report_hash_payload(result, spec));
    return unsigned_artifact.substr(0, unsigned_artifact.size() - 1) +
        ",\"artifact_sha256\":\"" + artifact_hash + "\"}";
}

}  // namespace performance_analytics

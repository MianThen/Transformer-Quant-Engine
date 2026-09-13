#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "engine_common/types.h"

namespace performance_analytics {

inline constexpr std::array<const char*, 6>
    kInfoTSReplayPredictionOutputs = {
        "expected_return", "expected_volatility", "direction_probability",
        "lower_quantile", "upper_quantile", "confidence"};

enum class InfoTSReplayStatus : std::uint8_t {
    OK,
    INVALID_SPEC,
    INVALID_INPUT,
    FUTURE_LEAKAGE,
    PROVENANCE_MISMATCH,
    REPLAY_FAILURE,
};

struct InfoTSReplaySpecV1 {
    std::uint32_t schema_version{1};
    std::string group_id;
    std::uint32_t fold_id{0};
    std::string policy_id{"INFOTS-PRECOMPUTED-PROXY-V1"};
    std::string contract_sha256;
    std::string dataset_fingerprint;
    std::string prediction_artifact_sha256;
    std::string validation_embedding_sha256;
    std::string test_embedding_sha256;
    std::string source_snapshot_set_sha256;
    std::string declared_source_replay_sha256;
    std::string calendar_id{"CN-EQUITY-RESEARCH"};
    double calendar_periods_per_year{252.0};
    std::uint64_t config_hash{1};
    std::uint32_t minimum_tail_observations{20};
    double initial_equity{1.0};
    std::string claim_scope{"RESEARCH_PROXY"};
    std::string reference_price_quality{"PROXY"};
    std::string execution_data_state{"UNAVAILABLE"};
    std::string corporate_action_state{"UNAVAILABLE"};
    std::string bar_reference_policy{"PROXY_BAR_REFERENCE"};
    std::string action_policy{"NO_ACTION"};
    std::string lot_policy{"LOT_1"};
    std::string limit_policy{"DISABLED"};
    std::string fee_policy{"ASSUMED_ZERO"};
    std::string slippage_policy{"UNAVAILABLE"};
};

struct InfoTSReplayRowV1 {
    std::uint64_t session_id{0};
    engine_common::TimestampNs prediction_available_at{0};
    engine_common::TimestampNs decision_at{0};
    engine_common::TimestampNs realized_at{0};
    double realized_proxy_return{std::numeric_limits<double>::quiet_NaN()};
    std::array<double, 6> prediction_outputs{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()};
};

struct InfoTSReplayResultV1 {
    InfoTSReplayStatus status{InfoTSReplayStatus::INVALID_INPUT};
    std::string group_id;
    std::uint32_t fold_id{0};
    std::string policy_id;
    std::size_t row_count{0};
    std::string source_replay_sha256;
    std::string ledger_sha256;
    std::string artifact_sha256;
    std::uint64_t ledger_hash{0};
    std::size_t observations{0};
    double cumulative_return{std::numeric_limits<double>::quiet_NaN()};
    double sharpe{std::numeric_limits<double>::quiet_NaN()};
    double maximum_drawdown{std::numeric_limits<double>::quiet_NaN()};
    double var_loss{std::numeric_limits<double>::quiet_NaN()};
    double expected_shortfall_loss{std::numeric_limits<double>::quiet_NaN()};
    double return_cvar{std::numeric_limits<double>::quiet_NaN()};
    bool research_comparison_eligible{false};
    bool phase_exit_eligible{false};
    bool promotion_eligible{false};
    std::string artifact_json;
};

[[nodiscard]] bool valid_infots_replay_spec(
    const InfoTSReplaySpecV1&) noexcept;
[[nodiscard]] std::string infots_replay_input_sha256(
    const InfoTSReplaySpecV1&, const std::vector<InfoTSReplayRowV1>&);
[[nodiscard]] InfoTSReplayResultV1 run_infots_precomputed_replay(
    const InfoTSReplaySpecV1&, const std::vector<InfoTSReplayRowV1>&);
[[nodiscard]] std::string serialize_infots_replay_artifact(
    const InfoTSReplayResultV1&, const InfoTSReplaySpecV1&);
[[nodiscard]] const char* infots_replay_status_name(InfoTSReplayStatus) noexcept;

}  // namespace performance_analytics

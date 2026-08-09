#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "performance_analytics/infots_replay.h"

namespace {
bool check(bool value, const char* message) {
    if (!value) std::fprintf(stderr, "FAILED: %s\n", message);
    return value;
}

performance_analytics::InfoTSReplaySpecV1 spec() {
    performance_analytics::InfoTSReplaySpecV1 value;
    value.group_id = "infots";
    value.fold_id = 2;
    value.contract_sha256 = std::string(64, 'a');
    value.dataset_fingerprint = std::string(64, 'b');
    value.prediction_artifact_sha256 = std::string(64, 'c');
    value.validation_embedding_sha256 = std::string(64, 'd');
    value.test_embedding_sha256 = std::string(64, 'e');
    value.source_snapshot_set_sha256 = std::string(64, 'f');
    value.config_hash = 20260809;
    value.minimum_tail_observations = 20;
    return value;
}

std::vector<performance_analytics::InfoTSReplayRowV1> rows() {
    std::vector<performance_analytics::InfoTSReplayRowV1> result;
    result.reserve(24);
    for (std::uint64_t index = 0; index < 24; ++index) {
        performance_analytics::InfoTSReplayRowV1 row;
        row.session_id = index + 1;
        row.prediction_available_at = static_cast<std::int64_t>(100 + index * 10);
        row.decision_at = static_cast<std::int64_t>(100 + index * 10);
        row.realized_at = static_cast<std::int64_t>(110 + index * 10);
        row.realized_proxy_return = (index % 4 == 0) ? 0.01 :
            ((index % 4 == 1) ? -0.006 : ((index % 4 == 2) ? 0.004 : -0.002));
        row.prediction_outputs = {0.003, 0.02, 0.55, -0.01, 0.02, 0.8};
        result.push_back(row);
    }
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    using namespace performance_analytics;
    bool ok = true;
    const auto input_spec = spec();
    const auto input_rows = rows();
    const auto first = run_infots_precomputed_replay(input_spec, input_rows);
    const auto second = run_infots_precomputed_replay(input_spec, input_rows);
    ok &= check(first.status == InfoTSReplayStatus::OK, "valid replay");
    ok &= check(first.row_count == 24 && first.observations == 24,
                "row count and observations");
    ok &= check(std::isfinite(first.cumulative_return) &&
                    std::isfinite(first.return_cvar) &&
                    first.expected_shortfall_loss >= 0.0,
                "proxy return and CVaR values");
    ok &= check(first.research_comparison_eligible &&
                    !first.phase_exit_eligible && !first.promotion_eligible,
                "research-only gates");
    ok &= check(first.artifact_sha256.size() == 64 &&
                    first.artifact_json.find("RESEARCH_PROXY") != std::string::npos,
                "artifact provenance");
    if (argc == 3 && std::string(argv[1]) == "--output") {
        std::ofstream output(argv[2], std::ios::binary);
        output << first.artifact_json << '\n';
        ok &= check(static_cast<bool>(output), "artifact output");
    }
    ok &= check(first.artifact_json == second.artifact_json &&
                    first.source_replay_sha256 == second.source_replay_sha256 &&
                    first.ledger_sha256 == second.ledger_sha256,
                "deterministic replay");

    auto future = input_rows;
    future[5].prediction_available_at = future[5].decision_at + 1;
    const auto future_result = run_infots_precomputed_replay(input_spec, future);
    ok &= check(future_result.status == InfoTSReplayStatus::FUTURE_LEAKAGE,
                "future prediction guard");

    auto broken_provenance = input_spec;
    broken_provenance.declared_source_replay_sha256 = std::string(64, '0');
    const auto provenance_result =
        run_infots_precomputed_replay(broken_provenance, input_rows);
    ok &= check(provenance_result.status == InfoTSReplayStatus::PROVENANCE_MISMATCH,
                "declared source hash guard");

    auto broken_output = input_rows;
    broken_output[2].prediction_outputs[2] = 1.2;
    const auto output_result = run_infots_precomputed_replay(input_spec, broken_output);
    ok &= check(output_result.status == InfoTSReplayStatus::INVALID_INPUT,
                "six-output range guard");
    auto short_rows = input_rows;
    short_rows.resize(2);
    const auto short_result = run_infots_precomputed_replay(input_spec, short_rows);
    ok &= check(short_result.status == InfoTSReplayStatus::REPLAY_FAILURE &&
                    short_result.artifact_json.empty(),
                "CVaR minimum observation guard");
    if (!ok) return 1;
    std::puts("test_infots_replay: all checks passed");
    return 0;
}

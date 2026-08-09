#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "performance_analytics/policy_family_governance.h"

namespace {

std::string report_prefix(const std::string &governance_json,
                          std::uint64_t fixture_hash) {
  std::ostringstream output;
  output << "{\"schema_version\":1,\"role\":\"phase4b_policy_family_reference\""
         << ",\"status\":\"REFERENCE_ONLY\",\"evidence_level\":\"REFERENCE_ONLY\""
         << ",\"phase_exit_eligible\":false,\"promotion_eligible\":false"
         << ",\"fixture_hash\":" << fixture_hash
         << ",\"governance\":" << governance_json;
  return output.str();
}

int write_report(const std::filesystem::path &output_path) {
  using namespace performance_analytics;
  constexpr std::size_t observations = 24;
  const std::array<double, 96> returns{
      .020, -.010, .015, -.020, .025, -.005, .018, -.012, .010, -.015, .022, -.008,
      .019, -.009, .014, -.019, .024, -.004, .017, -.011, .009, -.014, .021, -.007,
      .018, -.008, .013, -.018, .023, -.003, .016, -.010, .008, -.013, .020, -.006,
      .017, -.007, .012, -.017, .022, -.002, .015, -.009, .007, -.012, .019, -.005,
      -.015, .020, -.010, .018, -.020, .015, -.012, .022, -.008, .017, -.018, .021,
      -.014, .019, -.009, .017, -.019, .014, -.011, .021, -.007, .016, -.017, .020,
      -.013, .018, -.008, .016, -.018, .013, -.010, .020, -.006, .015, -.016, .019,
      -.012, .017, -.007, .015, -.017, .012, -.009, .019, -.005, .014, -.015, .018};
  const std::array<double, 4> p_values{0.001, 0.002, 0.7, 0.8};

  PolicyFamilyGovernanceSpecV1 spec;
  spec.fdr_q_level = 0.05;
  spec.effective_trial_spec = EffectiveTrialSpec{12, 4101};
  spec.performance_spec.calendar_id = "CN-EQUITY-REFERENCE";
  spec.performance_spec.calendar_periods_per_year = 252.0;
  spec.performance_spec.minimum_return_observations = 12;
  spec.performance_spec.minimum_tail_observations = 12;
  spec.performance_spec.config_hash = 4102;
  spec.dsr_config_hash = 4103;
  spec.family_contract_hash = 4104;
  spec.onc_spec.min_clusters = 2;
  spec.onc_spec.max_clusters = 2;
  spec.onc_spec.min_cluster_size = 2;

  PolicyFamilyGovernanceInputV1 input{
      {"posterior-direct", "nco-ffv", "nco-risk-only", "nco-risk-budget"},
      returns, observations, p_values};
  const auto artifact = evaluate_policy_family_governance(input, spec);
  const auto governance_json = serialize_policy_family_governance(artifact, input);
  if (artifact.status != AnalysisStatus::OK || governance_json.empty()) return 1;

  const auto fixture_hash = artifact.artifact_hash;
  const auto prefix = report_prefix(governance_json, fixture_hash);
  const auto report_body = prefix + "}";
  const auto report_hash = sha256_text(report_body);
  const auto report = prefix + ",\"report_body_sha256\":\"" + report_hash + "\"}\n";
  const auto file_hash = sha256_text(report);
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream output(output_path, std::ios::binary);
  if (!output) return 1;
  output << report;
  output.close();
  std::ofstream sidecar(output_path.string() + ".sha256", std::ios::binary);
  if (!sidecar) return 1;
  sidecar << file_hash << "  " << output_path.filename().string() << "\n";
  std::cout << "phase4b_governance_reference: " << output_path << "\n"
            << "report_body_sha256=" << report_hash << "\n"
            << "file_sha256=" << file_hash << "\n"
            << "artifact_hash=" << fixture_hash << "\n";
  return 0;
}

}

int main(int argc, char **argv) {
  std::filesystem::path output =
      "runs/phase4b-governance-reference/phase4b_governance_reference.json";
  if (argc == 3 && std::string(argv[1]) == "--output") {
    output = argv[2];
  } else if (argc != 1) {
    std::cerr << "usage: phase4b_governance_reference [--output path]\n";
    return 2;
  }
  return write_report(output);
}

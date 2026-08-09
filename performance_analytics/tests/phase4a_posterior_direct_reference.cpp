#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "performance_analytics/return_analysis.h"
#include "portfolio_math/posterior.h"
#include "portfolio_math/posterior_direct.h"

namespace {

struct BuiltReport {
  bool valid{false};
  std::string body;
};

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_byte(std::uint64_t& hash, std::uint8_t value) {
  hash ^= value;
  hash *= kFnvPrime;
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash_byte(hash, static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffU));
  }
}

void hash_double(std::uint64_t& hash, double value) {
  hash_u64(hash, std::bit_cast<std::uint64_t>(value));
}

std::uint64_t fixture_hash(std::span<const double> scenario_values,
                           std::span<const engine_common::TimestampNs> timestamps,
                           const portfolio_math::ViewSpecV1& view,
                           const portfolio_math::PosteriorDirectOptions& options) {
  std::uint64_t hash = kFnvOffset;
  hash_u64(hash, 1);
  hash_u64(hash, scenario_values.size());
  for (double value : scenario_values) hash_double(hash, value);
  hash_u64(hash, timestamps.size());
  for (const auto timestamp : timestamps) hash_u64(hash, timestamp);
  hash_u64(hash, portfolio_math::view_spec_hash(view));
  hash_u64(hash, options.max_iterations);
  hash_double(hash, options.tolerance);
  hash_double(hash, options.risk_aversion);
  hash_double(hash, options.target_investment);
  hash_double(hash, options.max_single_weight);
  return hash;
}

void json_doubles(std::ostringstream& output, std::span<const double> values) {
  output << '[' << std::setprecision(17);
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
}

void json_timestamps(
    std::ostringstream& output,
    std::span<const engine_common::TimestampNs> timestamps) {
  output << '[';
  for (std::size_t index = 0; index < timestamps.size(); ++index) {
    if (index != 0) output << ',';
    output << timestamps[index];
  }
  output << ']';
}

BuiltReport build_report() {
  constexpr std::size_t scenario_count = 8;
  constexpr std::size_t asset_count = 3;
  const std::array<double, scenario_count * asset_count> scenario_values{
      0.020, 0.004, -0.006,
      0.012, 0.009,  0.001,
      -0.008, 0.016, 0.010,
      0.018, -0.004, 0.013,
      0.006, 0.011, 0.003,
      -0.004, 0.007, 0.018,
      0.014, 0.002, -0.002,
      0.009, 0.013, 0.006};
  const std::array<engine_common::TimestampNs, scenario_count> timestamps{
      10, 20, 30, 40, 50, 60, 70, 80};
  const auto prior = portfolio_math::build_prior_scenario_artifact(
      {scenario_values.data(), scenario_count, asset_count, asset_count},
      timestamps, 80, 90);

  portfolio_math::ViewSpecV1 view;
  view.view_id = "phase4a-reference-asset-1-mean";
  view.available_at = 90;
  view.loading = {1.0, 0.0, 0.0};
  view.target = 0.014;
  view.confidence = 0.75;
  view.observation_variance = 0.0004;
  view.confidence_mapping_hash = 4401;
  view.source_artifact_hash = 4402;
  const std::array<portfolio_math::ViewSpecV1, 1> views{view};

  const auto gaussian = portfolio_math::apply_gaussian_mean_views(prior, views);
  const auto fully_flexible = portfolio_math::apply_ffv_mean_views(prior, views);
  portfolio_math::PosteriorDirectOptions options;
  options.max_iterations = 10'000;
  options.tolerance = 1e-10;
  options.risk_aversion = 3.0;
  options.target_investment = 1.0;
  options.max_single_weight = 0.70;
  const auto comparison = portfolio_math::compare_posterior_direct_policies(
      gaussian, fully_flexible, options);
  const auto input_hash = fixture_hash(scenario_values, timestamps, view, options);

  std::ostringstream body;
  body << "{\"schema_version\":1"
       << ",\"role\":\"phase4a_posterior_direct_reference\""
       << ",\"status\":\"REFERENCE_ONLY\""
       << ",\"evidence_level\":\"REFERENCE_ONLY\""
       << ",\"phase_exit_eligible\":false"
       << ",\"promotion_eligible\":false"
       << ",\"winner_selected\":false"
       << ",\"fixture_hash\":" << input_hash
       << ",\"fit_start\":10,\"fit_end\":80"
       << ",\"available_at\":80,\"decision_at\":90"
       << ",\"scenario_count\":" << scenario_count
       << ",\"asset_count\":" << asset_count
       << ",\"scenario_timestamps\":";
  json_timestamps(body, timestamps);
  body << ",\"scenario_values\":";
  json_doubles(body, scenario_values);
  body << ",\"view\":" << portfolio_math::serialize_view_spec(view)
       << ",\"prior_scenario_hash\":" << prior.scenario_hash
       << ",\"gaussian_posterior\":"
       << portfolio_math::serialize_posterior_scenario_artifact(gaussian)
       << ",\"fully_flexible_posterior\":"
       << portfolio_math::serialize_posterior_scenario_artifact(fully_flexible)
       << ",\"downstream_options\":{\"max_iterations\":"
       << options.max_iterations << ",\"tolerance\":"
       << std::setprecision(17) << options.tolerance
       << ",\"risk_aversion\":" << options.risk_aversion
       << ",\"target_investment\":" << options.target_investment
       << ",\"max_single_weight\":" << options.max_single_weight << '}'
       << ",\"policy_comparison\":"
       << portfolio_math::serialize_posterior_direct_policy_comparison(comparison)
       << ",\"limitations\":[\"fixed synthetic fixture\",\"no formal OOS\",\"no costs or execution fields\"]}";
  const bool valid = prior.status == portfolio_math::PosteriorStatus::OK &&
                     portfolio_math::valid_prior_scenario_artifact(prior) &&
                     gaussian.status == portfolio_math::PosteriorStatus::OK &&
                     fully_flexible.status == portfolio_math::PosteriorStatus::OK &&
                     comparison.status == portfolio_math::OptimizationStatus::OK &&
                     !comparison.winner_selected;
  return {valid, body.str()};
}

int write_report(const std::filesystem::path& output_path) {
  const auto built_report = build_report();
  if (!built_report.valid) return 1;
  const std::string& report_body = built_report.body;
  const std::string report_hash = performance_analytics::sha256_text(report_body);
  const std::string report = report_body.substr(0, report_body.size() - 1) +
                             ",\"report_body_sha256\":\"" + report_hash + "\"}\n";
  const std::string file_hash = performance_analytics::sha256_text(report);
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
  std::cout << "phase4a_posterior_direct_reference: " << output_path << "\n"
            << "report_body_sha256=" << report_hash << "\n"
            << "file_sha256=" << file_hash << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path output =
      "runs/phase4a-posterior-direct-reference/phase4a_posterior_direct_reference.json";
  if (argc == 3 && std::string(argv[1]) == "--output") {
    output = argv[2];
  } else if (argc != 1) {
    std::cerr << "usage: phase4a_posterior_direct_reference [--output path]\n";
    return 2;
  }
  return write_report(output);
}

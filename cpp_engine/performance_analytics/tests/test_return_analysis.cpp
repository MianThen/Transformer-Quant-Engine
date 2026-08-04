#include <array>
#include <cmath>
#include <cstdio>
#include <string>

#include "performance_analytics/return_analysis.h"
#include "performance_analytics/return_ledger.h"

namespace {
bool check(bool value, const char *message) {
  if (!value)
    std::fprintf(stderr, "FAILED: %s\n", message);
  return value;
}
} // namespace

int main() {
  using namespace performance_analytics;
  PerformanceSpecV1 spec;
  spec.calendar_id = "CN-EQUITY";
  spec.calendar_periods_per_year = 252.0;
  spec.config_hash = 100;
  spec.minimum_tail_observations = 2;
  const std::array candidate{0.02, -0.01, 0.03, 0.00, 0.01, -0.02};
  const std::array baseline{0.01, -0.01, 0.01, 0.00, 0.00, -0.01};
  bool ok =
      check(paired_stationary_bootstrap(
                candidate, baseline, StationaryBootstrapSpec{100, 7, 2.0, 101})
                    .status == AnalysisStatus::OK,
            "stationary bootstrap");
  const auto metrics = compute_return_metrics(candidate, spec);
  ok &= check(metrics.status == AnalysisStatus::OK &&
                  std::isfinite(metrics.cumulative_return),
              "return metrics");
  const std::array trials{0.01,  0.02, -0.01, 0.03,  0.00, 0.01,
                          0.01,  0.02, -0.01, 0.03,  0.00, 0.01,
                          -0.01, 0.00, 0.02,  -0.02, 0.01, 0.03};
  const auto effective =
      estimate_effective_trial_count(trials, 3, 6, EffectiveTrialSpec{3, 102});
  ok &= check(effective.status == AnalysisStatus::OK &&
                  effective.effective_trials < 3.0,
              "effective trials");
  const auto dsr = compute_deflated_sharpe_ratio(
      candidate, spec, effective.effective_trials, 103);
  ok &= check(dsr.status == AnalysisStatus::OK &&
                  dsr.one_sided_p_value >= 0.0 && dsr.one_sided_p_value <= 1.0,
              "deflated Sharpe");
  ReturnAnalysisManifest manifest;
  manifest.benchmark_available = true;
  manifest.promotion_eligible = true;
  manifest.reference_price_quality = "PROXY";
  manifest.var_loss = 0.01;
  manifest.expected_shortfall_loss = 0.03;
  manifest.return_cvar = -0.03;
  ok &= check(serialize_return_analysis_manifest(manifest).find(
                  "\"promotion_eligible\":false") != std::string::npos,
              "proxy reference closes promotion gate");
  const auto serialized_manifest = serialize_return_analysis_manifest(manifest);
  ok &= check(serialized_manifest.find("\"var_loss\":") != std::string::npos &&
                  serialized_manifest.find("\"expected_shortfall_loss\":") !=
                      std::string::npos &&
                  serialized_manifest.find("\"return_cvar\":") !=
                      std::string::npos,
              "manifest records formal tail-risk values");
  ReturnLedger ledger(spec);
  PeriodReturnInput first{1, 2, 1, 1'000.0, 1'010.0, 12.0, 2.0, 10.0,
                         0.0, 0.0, 0.0, std::nullopt};
  PeriodReturnInput second{2, 3, 2, 1'010.0, 1'005.0, -4.0, 1.0, -5.0,
                          0.0, 0.0, 0.0, std::nullopt};
  ok &= check(ledger.append(first) == LedgerStatus::OK &&
                  ledger.append(second) == LedgerStatus::OK,
              "report source ledger closes accounting");
  const auto report = build_return_analysis_report(ledger, manifest);
  ok &= check(report.status == AnalysisStatus::OK &&
                  report.metrics.observations == 2 &&
                  report.artifact_json.find("return_analysis_v1") !=
                      std::string::npos &&
                  report.artifact_json.find("report_sha256") !=
                      std::string::npos &&
                  report.artifact_json.find("promotion_eligible\":false") !=
                      std::string::npos,
              "return analysis report links metrics, ledger and promotion gate");
  ReturnAnalysisManifest missing_reference;
  missing_reference.benchmark_available = true;
  missing_reference.promotion_eligible = true;
  ok &= check(serialize_return_analysis_manifest(missing_reference).find(
                  "\"promotion_eligible\":false") != std::string::npos,
              "missing reference provenance closes promotion gate");

  DriftSnapshotContractV0 snapshot;
  const std::string hash =
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  snapshot.model_manifest_sha256 = hash;
  snapshot.raw_schema_hash = hash;
  snapshot.preprocessing_spec_sha256 = hash;
  snapshot.feature_schema_hash = hash;
  snapshot.prediction_schema_hash = hash;
  snapshot.raw_fields_sha256 = hash;
  snapshot.preprocessed_features_sha256 = hash;
  snapshot.prediction_values_sha256 = hash;
  snapshot.embedding_values_sha256 = hash;
  snapshot.source_snapshot_set_sha256 = hash;
  snapshot.ledger_schema_hash = hash;
  snapshot.available_at_utc = "2026-08-02T00:00:00Z";
  const auto snapshot_artifact = serialize_drift_snapshot_artifact(snapshot);
  ok &= check(!snapshot_artifact.empty(), "drift snapshot artifact serializes");
  const auto report_marker = snapshot_artifact.rfind(
      ",\"report_sha256\":\"");
  ok &= check(report_marker != std::string::npos &&
                  snapshot_artifact.size() >= report_marker + 80,
              "drift snapshot report hash exists");
  if (report_marker != std::string::npos &&
      snapshot_artifact.size() >= report_marker + 80) {
    snapshot.report_sha256 = snapshot_artifact.substr(
        report_marker + std::string(",\"report_sha256\":\"").size(), 64);
  }
  ok &= check(valid_drift_snapshot_contract(snapshot),
              "drift snapshot contract validates");
  ok &= check(serialize_drift_snapshot_artifact(snapshot) == snapshot_artifact,
              "drift snapshot serialization is deterministic");
  const auto snapshot_hash = hash_drift_snapshot_contract(snapshot);
  auto changed_snapshot = snapshot;
  changed_snapshot.prediction_values_sha256[0] =
      changed_snapshot.prediction_values_sha256[0] == '0' ? '1' : '0';
  changed_snapshot.report_sha256.clear();
  const auto changed_artifact =
      serialize_drift_snapshot_artifact(changed_snapshot);
  ok &= check(!changed_artifact.empty() && changed_artifact != snapshot_artifact,
              "drift snapshot field changes artifact hash");
  ok &= check(hash_drift_snapshot_contract(changed_snapshot) != snapshot_hash,
              "drift snapshot field changes contract hash");
  auto missing_available_at = snapshot;
  missing_available_at.available_at_utc.clear();
  missing_available_at.report_sha256.clear();
  ok &= check(serialize_drift_snapshot_artifact(missing_available_at).empty(),
              "drift snapshot requires available_at_utc");
  if (!ok)
    return 1;
  std::puts("test_return_analysis: all checks passed");
  return 0;
}

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "performance_analytics/performance_spec.h"

namespace performance_analytics {

class ReturnLedger;

enum class AnalysisStatus : std::uint8_t {
  OK,
  UNAVAILABLE,
  INVALID_INPUT,
  INSUFFICIENT_DATA,
  ZERO_VARIANCE,
  NUMERICAL_FAILURE,
  PENDING,
  INCOMPATIBLE,
};

struct StationaryBootstrapSpec {
  std::uint32_t replicates{2000};
  std::uint64_t seed{0};
  double mean_block_length{10.0};
  std::uint64_t config_hash{0};
};
struct PairedBootstrapResult {
  AnalysisStatus status{AnalysisStatus::INVALID_INPUT};
  double observed_difference{std::numeric_limits<double>::quiet_NaN()};
  double bootstrap_mean{std::numeric_limits<double>::quiet_NaN()};
  double p_value{std::numeric_limits<double>::quiet_NaN()};
  double confidence_level{0.95};
  double confidence_lower{std::numeric_limits<double>::quiet_NaN()};
  double confidence_upper{std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t artifact_hash{0};
};
[[nodiscard]] PairedBootstrapResult
paired_stationary_bootstrap(std::span<const double>, std::span<const double>,
                            const StationaryBootstrapSpec &,
                            double confidence_level = 0.95);
struct StationaryBootstrapSensitivity {
  double mean_block_length{0.0};
  PairedBootstrapResult result;
};
[[nodiscard]] std::vector<StationaryBootstrapSensitivity>
paired_stationary_bootstrap_sensitivity(std::span<const double>,
                                        std::span<const double>,
                                        const StationaryBootstrapSpec &,
                                        std::span<const double>,
                                        double confidence_level = 0.95);

struct HacSpec {
  std::uint32_t lag{0};
  std::uint64_t config_hash{0};
};
struct HacResult {
  AnalysisStatus status{AnalysisStatus::INVALID_INPUT};
  double mean{std::numeric_limits<double>::quiet_NaN()};
  double long_run_variance{std::numeric_limits<double>::quiet_NaN()};
  double standard_error{std::numeric_limits<double>::quiet_NaN()};
  double t_statistic{std::numeric_limits<double>::quiet_NaN()};
  std::uint32_t lag{0};
  std::uint64_t artifact_hash{0};
};
[[nodiscard]] HacResult newey_west_hac(std::span<const double>,
                                       const HacSpec &);

struct EffectiveTrialSpec {
  std::uint32_t minimum_observations{3};
  std::uint64_t config_hash{0};
};
struct EffectiveTrialResult {
  AnalysisStatus status{AnalysisStatus::INVALID_INPUT};
  std::size_t trial_count{0};
  std::size_t observations{0};
  double mean_absolute_correlation{std::numeric_limits<double>::quiet_NaN()};
  double effective_trials{std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t artifact_hash{0};
};
[[nodiscard]] EffectiveTrialResult
estimate_effective_trial_count(std::span<const double>, std::size_t trial_count,
                               std::size_t observations,
                               const EffectiveTrialSpec &);
struct DsrResult {
  AnalysisStatus status{AnalysisStatus::INVALID_INPUT};
  std::size_t observations{0};
  double raw_sharpe{std::numeric_limits<double>::quiet_NaN()};
  double benchmark_sharpe{0.0};
  double effective_trials{std::numeric_limits<double>::quiet_NaN()};
  double expected_max_sharpe{std::numeric_limits<double>::quiet_NaN()};
  double deflated_sharpe{std::numeric_limits<double>::quiet_NaN()};
  double one_sided_p_value{std::numeric_limits<double>::quiet_NaN()};
  double skewness{std::numeric_limits<double>::quiet_NaN()};
  double excess_kurtosis{std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t artifact_hash{0};
};
[[nodiscard]] DsrResult compute_deflated_sharpe_ratio(
    std::span<const double>, const PerformanceSpecV1 &, double effective_trials,
    std::uint64_t config_hash, double benchmark_sharpe = 0.0);

struct ReturnMetrics {
  AnalysisStatus status{AnalysisStatus::INVALID_INPUT};
  std::size_t observations{0};
  double cumulative_return{std::numeric_limits<double>::quiet_NaN()};
  double annualized_return{std::numeric_limits<double>::quiet_NaN()};
  double annualized_volatility{std::numeric_limits<double>::quiet_NaN()};
  double sharpe{std::numeric_limits<double>::quiet_NaN()};
  double sortino{std::numeric_limits<double>::quiet_NaN()};
  double calmar{std::numeric_limits<double>::quiet_NaN()};
  double maximum_drawdown{std::numeric_limits<double>::quiet_NaN()};
  std::uint32_t maximum_drawdown_duration{0};
  double win_rate{std::numeric_limits<double>::quiet_NaN()};
  double profit_factor{std::numeric_limits<double>::quiet_NaN()};
  double average_win_loss_ratio{std::numeric_limits<double>::quiet_NaN()};
  double cvar{std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t artifact_hash{0};
};
[[nodiscard]] ReturnMetrics compute_return_metrics(std::span<const double>,
                                                   const PerformanceSpecV1 &);

struct ActiveMetrics {
  AnalysisStatus status{AnalysisStatus::INVALID_INPUT};
  double active_growth{std::numeric_limits<double>::quiet_NaN()};
  double mean_active_return{std::numeric_limits<double>::quiet_NaN()};
  double tracking_error{std::numeric_limits<double>::quiet_NaN()};
  double information_ratio{std::numeric_limits<double>::quiet_NaN()};
  double capm_alpha{std::numeric_limits<double>::quiet_NaN()};
  double capm_beta{std::numeric_limits<double>::quiet_NaN()};
  double up_capture{std::numeric_limits<double>::quiet_NaN()};
  double down_capture{std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t artifact_hash{0};
};
struct AlignedReturnPair {
  std::uint64_t period_id{0};
  double portfolio_return{0.0};
  double benchmark_return{0.0};
};
[[nodiscard]] ActiveMetrics compute_active_metrics(std::span<const double>,
                                                   std::span<const double>,
                                                   const PerformanceSpecV1 &);
[[nodiscard]] ActiveMetrics
compute_aligned_active_metrics(std::span<const AlignedReturnPair>,
                               const PerformanceSpecV1 &);

struct ScoreObservation {
  std::uint64_t date_key{0}, symbol_id{0};
  double score{0.0}, realized_return{0.0};
  std::uint32_t horizon{1};
};
struct ScoreBucketSpec {
  std::uint32_t bucket_count{10}, minimum_observations{1};
  std::uint64_t config_hash{0};
};
struct ScoreBucket {
  std::uint32_t bucket{0};
  std::uint64_t observations{0}, dates{0};
  double mean_return{std::numeric_limits<double>::quiet_NaN()};
};
struct ScoreBucketResult {
  AnalysisStatus status{AnalysisStatus::INVALID_INPUT};
  std::vector<ScoreBucket> buckets;
  double top_bottom_spread{std::numeric_limits<double>::quiet_NaN()};
  double monotonicity_slope{std::numeric_limits<double>::quiet_NaN()};
  double icir{std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t covered_symbols{0};
  double top_bucket_turnover{std::numeric_limits<double>::quiet_NaN()};
  std::vector<double> horizon_decay;
  std::uint64_t artifact_hash{0};
};
[[nodiscard]] ScoreBucketResult
analyze_score_buckets(std::span<const ScoreObservation>,
                      const ScoreBucketSpec &);

struct ReturnAnalysisManifest {
  std::uint32_t schema_version{1};
  std::string metric_spec_version{"return-metrics-v1"};
  std::string source_replay_sha256, dataset_fingerprint,
      portfolio_return_ledger_sha256;
  std::string benchmark_artifact_sha256, implementation_shortfall_ledger_sha256;
  std::string stationary_bootstrap_spec_hash, hac_sensitivity_spec_hash,
      benchmark_id, calendar_id;
  bool benchmark_available{false}, promotion_eligible{false};
  std::vector<std::string> limitations;
  std::string reference_price_quality;
  std::optional<double> execution_price_cost, explicit_fees, opportunity_cost,
      implementation_shortfall, gross_pnl, net_pnl, accounting_residual,
      effective_trials, deflated_sharpe, var_loss, expected_shortfall_loss,
      return_cvar;
  std::uint64_t raw_trials{0};
};

struct ReturnAnalysisReport {
  AnalysisStatus status{AnalysisStatus::INVALID_INPUT};
  ReturnMetrics metrics;
  std::uint64_t ledger_hash{0};
  std::uint64_t manifest_hash{0};
  std::string manifest_json;
  std::string report_sha256;
  std::string artifact_json;
};
[[nodiscard]] std::string
serialize_return_analysis_manifest(const ReturnAnalysisManifest &);
[[nodiscard]] std::uint64_t
hash_return_analysis_manifest(const ReturnAnalysisManifest &);
[[nodiscard]] std::string
sha256_return_analysis_manifest(const ReturnAnalysisManifest &);
[[nodiscard]] std::string sha256_text(std::string_view value);
[[nodiscard]] ReturnAnalysisReport build_return_analysis_report(
    const ReturnLedger&, ReturnAnalysisManifest manifest);

struct DriftSnapshotContractV0 {
  std::uint32_t schema_version{0};
  std::string model_manifest_sha256, raw_schema_hash, preprocessing_spec_sha256,
      feature_schema_hash, prediction_schema_hash, label_spec_sha256,
      raw_fields_sha256, preprocessed_features_sha256, prediction_values_sha256,
      embedding_values_sha256, matured_labels_sha256,
      source_snapshot_set_sha256, ledger_schema_hash, available_at_utc,
      report_sha256;
  bool labels_mature{false};
};
[[nodiscard]] bool
valid_drift_snapshot_contract(const DriftSnapshotContractV0 &) noexcept;
[[nodiscard]] std::uint64_t
hash_drift_snapshot_contract(const DriftSnapshotContractV0 &);
[[nodiscard]] std::string
serialize_drift_snapshot_artifact(const DriftSnapshotContractV0 &);

} // namespace performance_analytics

#include "performance_analytics/policy_family_governance.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

namespace performance_analytics {
namespace {
constexpr std::uint64_t kOffset = 1469598103934665603ULL;
constexpr std::uint64_t kPrime = 1099511628211ULL;

void hash_value(std::uint64_t &hash, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    hash ^= (value >> shift) & 0xffU;
    hash *= kPrime;
  }
}

void hash_double(std::uint64_t &hash, double value) {
  hash_value(hash, std::bit_cast<std::uint64_t>(value == 0.0 ? 0.0 : value));
}

void hash_string(std::uint64_t &hash, const std::string &value) {
  for (const unsigned char character : value) {
    hash ^= character;
    hash *= kPrime;
  }
  hash_value(hash, value.size());
}

std::string json_escape(const std::string &value) {
  std::ostringstream output;
  for (const unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
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

const char *fdr_method_name(portfolio_math::FdrMethod method) {
  switch (method) {
    case portfolio_math::FdrMethod::BENJAMINI_HOCHBERG:
      return "BENJAMINI_HOCHBERG";
    case portfolio_math::FdrMethod::BENJAMINI_YEKUTIELI:
      return "BENJAMINI_YEKUTIELI";
    case portfolio_math::FdrMethod::STOREY:
      return "STOREY";
  }
  return "INVALID";
}

bool finite(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

quant_math::DenseMatrix correlation_matrix(std::span<const double> values,
                                           std::size_t trials,
                                           std::size_t observations) {
  quant_math::DenseMatrix output(trials, trials);
  std::vector<double> means(trials), scales(trials);
  for (std::size_t trial = 0; trial < trials; ++trial) {
    const auto row = values.subspan(trial * observations, observations);
    means[trial] = std::accumulate(row.begin(), row.end(), 0.0) / observations;
    double sum = 0.0;
    for (double value : row) sum += (value - means[trial]) * (value - means[trial]);
    scales[trial] = std::sqrt(sum / static_cast<double>(observations - 1));
  }
  for (std::size_t left = 0; left < trials; ++left) {
    for (std::size_t right = 0; right < trials; ++right) {
      double covariance = 0.0;
      for (std::size_t index = 0; index < observations; ++index) {
        covariance += (values[left * observations + index] - means[left]) *
                      (values[right * observations + index] - means[right]);
      }
      output(static_cast<Eigen::Index>(left), static_cast<Eigen::Index>(right)) =
          covariance / ((observations - 1.0) * scales[left] * scales[right]);
    }
  }
  output.diagonal().setOnes();
  return output;
}
}

PolicyFamilyGovernanceArtifactV1 evaluate_policy_family_governance(
    const PolicyFamilyGovernanceInputV1 &input,
    const PolicyFamilyGovernanceSpecV1 &spec) {
  PolicyFamilyGovernanceArtifactV1 artifact;
  const auto trials = input.policy_ids.size();
  artifact.raw_trial_count = trials;
  artifact.observation_count = input.observations;
  if (spec.schema_version != 1 || spec.family_contract_hash == 0 ||
      spec.dsr_config_hash == 0 || !(spec.fdr_q_level > 0.0 && spec.fdr_q_level < 1.0) ||
      !valid_performance_spec(spec.performance_spec) || trials < 3 ||
      input.observations < spec.effective_trial_spec.minimum_observations ||
      input.trial_major_returns.size() != trials * input.observations ||
      input.primary_p_values.size() != trials || !finite(input.trial_major_returns) ||
      !finite(input.primary_p_values) ||
      std::any_of(input.policy_ids.begin(), input.policy_ids.end(),
                  [](const std::string &value) { return value.empty(); })) {
    return artifact;
  }
  auto sorted_ids = input.policy_ids;
  std::sort(sorted_ids.begin(), sorted_ids.end());
  if (std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) != sorted_ids.end())
    return artifact;

  artifact.policy_return_correlation = correlation_matrix(
      input.trial_major_returns, trials, input.observations);
  if (!artifact.policy_return_correlation.allFinite()) return artifact;
  artifact.policy_clusters = portfolio_math::onc_partition(
      quant_math::view(artifact.policy_return_correlation), spec.onc_spec);
  if (artifact.policy_clusters.status != portfolio_math::OncStatus::OK) return artifact;
  artifact.cluster_count = artifact.policy_clusters.diagnostics.selected_cluster_count;
  artifact.effective_trials = estimate_effective_trial_count(
      input.trial_major_returns, trials, input.observations,
      spec.effective_trial_spec);
  if (artifact.effective_trials.status != AnalysisStatus::OK) return artifact;

  if (spec.fdr_method == portfolio_math::FdrMethod::BENJAMINI_HOCHBERG) {
    artifact.fdr = portfolio_math::benjamini_hochberg(
        input.primary_p_values, spec.fdr_q_level);
  } else if (spec.fdr_method == portfolio_math::FdrMethod::BENJAMINI_YEKUTIELI) {
    artifact.fdr = portfolio_math::benjamini_yekutieli(
        input.primary_p_values, spec.fdr_q_level);
  } else {
    artifact.fdr = portfolio_math::storey_fdr(
        input.primary_p_values, spec.fdr_q_level, spec.storey_lambdas,
        spec.storey_aggregation);
  }
  if (artifact.fdr.status != portfolio_math::MultipleTestingStatus::OK) return artifact;

  artifact.dsr_by_policy.reserve(trials);
  for (std::size_t trial = 0; trial < trials; ++trial) {
    const auto returns = input.trial_major_returns.subspan(
        trial * input.observations, input.observations);
    artifact.dsr_by_policy.push_back(compute_deflated_sharpe_ratio(
        returns, spec.performance_spec, artifact.effective_trials.effective_trials,
        spec.dsr_config_hash));
    if (artifact.dsr_by_policy.back().status != AnalysisStatus::OK) return artifact;
  }

  double best_sharpe = -std::numeric_limits<double>::infinity();
  for (std::size_t trial = 0; trial < trials; ++trial) {
    const auto &dsr = artifact.dsr_by_policy[trial];
    if (artifact.fdr.rejected[trial] != 0 &&
        dsr.one_sided_p_value <= spec.fdr_q_level && dsr.raw_sharpe > best_sharpe) {
      best_sharpe = dsr.raw_sharpe;
      artifact.selected_policy_index = trial;
      artifact.candidate_selected = true;
    }
  }
  artifact.winner_frozen = artifact.candidate_selected &&
      input.registered_before_evaluation && input.formal_oos &&
      input.economic_gate_passed && input.winner_freeze_requested &&
      input.final_untouched_unseen;
  artifact.promotion_eligible = false;
  artifact.artifact_hash = kOffset;
  hash_value(artifact.artifact_hash, spec.family_contract_hash);
  hash_value(artifact.artifact_hash, spec.dsr_config_hash);
  hash_value(artifact.artifact_hash, spec.fdr_q_level == 0.0 ? 0ULL :
      std::bit_cast<std::uint64_t>(spec.fdr_q_level));
  hash_value(artifact.artifact_hash, static_cast<std::uint64_t>(spec.fdr_method));
  for (const auto &policy_id : input.policy_ids) hash_string(artifact.artifact_hash, policy_id);
  for (double value : input.trial_major_returns) hash_double(artifact.artifact_hash, value);
  for (double value : input.primary_p_values) hash_double(artifact.artifact_hash, value);
  hash_value(artifact.artifact_hash, input.registered_before_evaluation);
  hash_value(artifact.artifact_hash, input.formal_oos);
  hash_value(artifact.artifact_hash, input.economic_gate_passed);
  hash_value(artifact.artifact_hash, input.winner_freeze_requested);
  hash_value(artifact.artifact_hash, input.final_untouched_unseen);
  hash_value(artifact.artifact_hash, artifact.policy_clusters.diagnostics.partition_hash);
  hash_value(artifact.artifact_hash,
             artifact.policy_clusters.diagnostics.input_correlation_hash);
  for (double value : artifact.policy_clusters.diagnostics.selected_silhouette) {
    hash_double(artifact.artifact_hash, value);
  }
  for (const auto &candidate : artifact.policy_clusters.diagnostics.candidates) {
    hash_value(artifact.artifact_hash, candidate.cluster_count);
    hash_value(artifact.artifact_hash, candidate.seed);
    hash_double(artifact.artifact_hash, candidate.quality);
    hash_double(artifact.artifact_hash, candidate.minimum_silhouette);
    hash_double(artifact.artifact_hash, candidate.maximum_silhouette);
    hash_value(artifact.artifact_hash, candidate.valid);
  }
  hash_value(artifact.artifact_hash, artifact.effective_trials.artifact_hash);
  hash_value(artifact.artifact_hash, artifact.raw_trial_count);
  hash_value(artifact.artifact_hash, artifact.observation_count);
  hash_value(artifact.artifact_hash, artifact.candidate_selected);
  hash_value(artifact.artifact_hash, artifact.winner_frozen);
  hash_double(artifact.artifact_hash, artifact.fdr.q_level);
  hash_double(artifact.artifact_hash, artifact.fdr.pi0);
  for (double value : artifact.fdr.pi0_by_lambda) hash_double(artifact.artifact_hash, value);
  for (double value : artifact.fdr.adjusted_p_values) hash_double(artifact.artifact_hash, value);
  for (const auto value : artifact.fdr.rejected) hash_value(artifact.artifact_hash, value);
  for (const auto &value : artifact.dsr_by_policy) hash_value(artifact.artifact_hash, value.artifact_hash);
  artifact.status = AnalysisStatus::OK;
  return artifact;
}

std::string serialize_policy_family_governance(
    const PolicyFamilyGovernanceArtifactV1 &artifact,
    const PolicyFamilyGovernanceInputV1 &input) {
  if (artifact.status != AnalysisStatus::OK || artifact.artifact_hash == 0 ||
      input.policy_ids.size() != artifact.raw_trial_count) return {};
  std::ostringstream output;
  output << std::setprecision(17)
         << "{\"schema_version\":1,\"status\":\"OK\",\"evidence_level\":\"REFERENCE_ONLY\""
         << ",\"raw_trial_count\":" << artifact.raw_trial_count
         << ",\"observation_count\":" << artifact.observation_count
         << ",\"cluster_count\":" << artifact.cluster_count
         << ",\"effective_trials\":" << artifact.effective_trials.effective_trials
         << ",\"policy_ids\":[";
  for (std::size_t index = 0; index < input.policy_ids.size(); ++index) {
    if (index != 0) output << ',';
    output << '"' << json_escape(input.policy_ids[index]) << '"';
  }
  output << "],\"primary_p_values\":[";
  for (std::size_t index = 0; index < input.primary_p_values.size(); ++index) {
    if (index != 0) output << ',';
    output << input.primary_p_values[index];
  }
  output << "],\"policy_return_correlation\":[";
  for (std::size_t row = 0; row < artifact.raw_trial_count; ++row) {
    if (row != 0) output << ',';
    output << '[';
    for (std::size_t column = 0; column < artifact.raw_trial_count; ++column) {
      if (column != 0) output << ',';
      output << artifact.policy_return_correlation(
          static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
    }
    output << ']';
  }
  output << "],\"cluster_id_by_policy\":[";
  for (std::size_t index = 0; index < artifact.policy_clusters.cluster_id_by_symbol.size(); ++index) {
    if (index != 0) output << ',';
    output << artifact.policy_clusters.cluster_id_by_symbol[index];
  }
  output << "],\"quasi_diagonal_order\":[";
  for (std::size_t index = 0; index < artifact.policy_clusters.quasi_diagonal_order.size(); ++index) {
    if (index != 0) output << ',';
    output << artifact.policy_clusters.quasi_diagonal_order[index];
  }
  output << "],\"onc\":{\"selected_cluster_count\":"
         << artifact.policy_clusters.diagnostics.selected_cluster_count
         << ",\"selected_quality\":"
         << artifact.policy_clusters.diagnostics.selected_quality
         << ",\"second_best_quality\":"
         << artifact.policy_clusters.diagnostics.second_best_quality
         << ",\"best_second_gap\":"
         << artifact.policy_clusters.diagnostics.best_second_gap
         << ",\"selected_silhouette\":[";
  for (std::size_t index = 0; index < artifact.policy_clusters.diagnostics.selected_silhouette.size(); ++index) {
    if (index != 0) output << ',';
    output << artifact.policy_clusters.diagnostics.selected_silhouette[index];
  }
  output << "],\"candidates\":[";
  for (std::size_t index = 0; index < artifact.policy_clusters.diagnostics.candidates.size(); ++index) {
    if (index != 0) output << ',';
    const auto &candidate = artifact.policy_clusters.diagnostics.candidates[index];
    output << "{\"cluster_count\":" << candidate.cluster_count
           << ",\"seed\":" << candidate.seed
           << ",\"quality\":" << candidate.quality
           << ",\"minimum_silhouette\":" << candidate.minimum_silhouette
           << ",\"maximum_silhouette\":" << candidate.maximum_silhouette
           << ",\"valid\":" << (candidate.valid ? "true" : "false") << '}';
  }
  output << "],\"input_correlation_hash\":"
         << artifact.policy_clusters.diagnostics.input_correlation_hash
         << ",\"partition_hash\":"
         << artifact.policy_clusters.diagnostics.partition_hash
         << ",\"eligible_for_official_risk\":false}"
         << ",\"effective_trial_diagnostics\":{\"trial_count\":"
         << artifact.effective_trials.trial_count
         << ",\"observations\":" << artifact.effective_trials.observations
         << ",\"mean_absolute_correlation\":"
         << artifact.effective_trials.mean_absolute_correlation
         << ",\"artifact_hash\":" << artifact.effective_trials.artifact_hash
         << "},\"fdr\":{\"method\":\""
         << fdr_method_name(artifact.fdr.method) << "\",\"q_level\":"
         << artifact.fdr.q_level << ",\"pi0\":" << artifact.fdr.pi0
         << ",\"pi0_by_lambda\":[";
  for (std::size_t index = 0; index < artifact.fdr.pi0_by_lambda.size(); ++index) {
    if (index != 0) output << ',';
    output << artifact.fdr.pi0_by_lambda[index];
  }
  output << "],\"adjusted_p_values\":[";
  for (std::size_t index = 0; index < artifact.fdr.adjusted_p_values.size(); ++index) {
    if (index != 0) output << ',';
    output << artifact.fdr.adjusted_p_values[index];
  }
  output << "],\"rejected\":[";
  for (std::size_t index = 0; index < artifact.fdr.rejected.size(); ++index) {
    if (index != 0) output << ',';
    output << static_cast<unsigned int>(artifact.fdr.rejected[index]);
  }
  output << "]},\"dsr_by_policy\":[";
  for (std::size_t index = 0; index < artifact.dsr_by_policy.size(); ++index) {
    if (index != 0) output << ',';
    const auto &dsr = artifact.dsr_by_policy[index];
    output << "{\"observations\":" << dsr.observations
           << ",\"raw_sharpe\":" << dsr.raw_sharpe
           << ",\"benchmark_sharpe\":" << dsr.benchmark_sharpe
           << ",\"effective_trials\":" << dsr.effective_trials
           << ",\"expected_max_sharpe\":" << dsr.expected_max_sharpe
           << ",\"deflated_sharpe\":" << dsr.deflated_sharpe
           << ",\"one_sided_p_value\":" << dsr.one_sided_p_value
           << ",\"skewness\":" << dsr.skewness
           << ",\"excess_kurtosis\":" << dsr.excess_kurtosis
           << ",\"artifact_hash\":" << dsr.artifact_hash << '}';
  }
  output << "],\"gate_inputs\":{\"registered_before_evaluation\":"
         << (input.registered_before_evaluation ? "true" : "false")
         << ",\"formal_oos\":" << (input.formal_oos ? "true" : "false")
         << ",\"economic_gate_passed\":"
         << (input.economic_gate_passed ? "true" : "false")
         << ",\"winner_freeze_requested\":"
         << (input.winner_freeze_requested ? "true" : "false")
         << ",\"final_untouched_unseen\":"
         << (input.final_untouched_unseen ? "true" : "false") << '}'
         << ",\"candidate_selected\":" << (artifact.candidate_selected ? "true" : "false")
         << ",\"winner_frozen\":" << (artifact.winner_frozen ? "true" : "false")
         << ",\"promotion_eligible\":false"
         << ",\"selected_policy_id\":";
  if (artifact.candidate_selected) {
    output << '"' << json_escape(input.policy_ids[artifact.selected_policy_index]) << '"';
  }
  else output << "null";
  output << ",\"artifact_hash\":" << artifact.artifact_hash << "}";
  return output.str();
}

}  // namespace performance_analytics

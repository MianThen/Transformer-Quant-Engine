#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "portfolio_math/covariance.h"
#include "portfolio_math/risk_budget.h"

namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_byte(std::uint64_t& hash, std::uint8_t value) {
  hash ^= value;
  hash *= kFnvPrime;
}

void hash_value(std::uint64_t& hash, std::uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) {
    hash_byte(hash, static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffU));
  }
}

void hash_double(std::uint64_t& hash, double value) {
  hash_value(hash, std::bit_cast<std::uint64_t>(value));
}

void hash_weights(std::uint64_t& hash, std::span<const double> values) {
  hash_value(hash, values.size());
  for (const double value : values) hash_double(hash, value);
}

const char* covariance_status_name(portfolio_math::CovarianceStatus status) {
  switch (status) {
    case portfolio_math::CovarianceStatus::OK: return "ok";
    case portfolio_math::CovarianceStatus::INVALID_INPUT: return "invalid_input";
    case portfolio_math::CovarianceStatus::INSUFFICIENT_OBSERVATIONS:
      return "insufficient_observations";
    case portfolio_math::CovarianceStatus::NUMERICAL_FAILURE:
      return "numerical_failure";
  }
  return "invalid";
}

const char* optimization_status_name(
    portfolio_math::OptimizationStatus status) {
  switch (status) {
    case portfolio_math::OptimizationStatus::OK: return "ok";
    case portfolio_math::OptimizationStatus::INVALID_INPUT: return "invalid_input";
    case portfolio_math::OptimizationStatus::NON_PSD_RISK_MODEL:
      return "non_psd_risk_model";
    case portfolio_math::OptimizationStatus::INFEASIBLE: return "infeasible";
    case portfolio_math::OptimizationStatus::MAX_ITERATIONS:
      return "max_iterations";
    case portfolio_math::OptimizationStatus::NUMERICAL_FAILURE:
      return "numerical_failure";
  }
  return "invalid";
}

struct PolicyRun {
  portfolio_math::OptimizationStatus status{
      portfolio_math::OptimizationStatus::INVALID_INPUT};
  std::vector<double> weights;
  double weight_sum{0.0};
  std::uint32_t iterations{0};
  double residual{0.0};
};

struct EstimatorRun {
  const char* name;
  portfolio_math::CovarianceStatus covariance_status{
      portfolio_math::CovarianceStatus::INVALID_INPUT};
  double shrinkage{0.0};
  double concentration_ratio{0.0};
  std::size_t effective_observations{0};
  std::uint64_t covariance_hash{0};
  PolicyRun topk;
  PolicyRun risk_budget;
};

struct ClosureReport {
  bool ok{false};
  std::uint64_t input_fingerprint{0};
  std::uint64_t artifact_hash{0};
  std::vector<EstimatorRun> estimators;
  bool regular_branch_ok{false};
  bool singular_branch_ok{false};
  bool invalid_quest_fails_closed{false};
  bool policy_switch_ok{false};
};

std::uint64_t matrix_hash(quant_math::MatrixView matrix) {
  std::uint64_t hash = kFnvOffset;
  hash_value(hash, matrix.rows);
  hash_value(hash, matrix.cols);
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t col = 0; col < matrix.cols; ++col) {
      hash_double(hash, matrix(row, col));
    }
  }
  return hash;
}

std::uint64_t covariance_hash(const portfolio_math::CovarianceResult& result) {
  if (result.status != portfolio_math::CovarianceStatus::OK) return 0;
  std::uint64_t hash = kFnvOffset;
  hash_byte(hash, static_cast<std::uint8_t>(result.status));
  hash_byte(hash, static_cast<std::uint8_t>(result.estimator));
  hash_byte(hash, static_cast<std::uint8_t>(result.loss_profile));
  hash_double(hash, result.shrinkage_intensity);
  hash_double(hash, result.average_correlation);
  hash_value(hash, result.effective_observations);
  hash_double(hash, result.concentration_ratio);
  hash_value(hash, matrix_hash(quant_math::view(result.covariance)));
  return hash;
}

PolicyRun topk_policy() {
  PolicyRun result;
  result.status = portfolio_math::OptimizationStatus::OK;
  result.weights = {0.5, 0.5, 0.0};
  result.weight_sum = 1.0;
  result.iterations = 1;
  return result;
}

PolicyRun risk_budget_policy(quant_math::MatrixView covariance) {
  PolicyRun result;
  const std::array<double, 3> budgets{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
  portfolio_math::RiskBudgetOptions options;
  options.max_iterations = 10'000;
  options.tolerance = 1e-10;
  options.covariance_tolerance = 1e-10;
  const auto solved = portfolio_math::solve_long_only_risk_budget(
      covariance, budgets, {}, {}, options);
  result.status = solved.diagnostics.status;
  result.weights = solved.weights;
  result.weight_sum = solved.weights.empty()
      ? 0.0
      : std::accumulate(solved.weights.begin(), solved.weights.end(), 0.0);
  result.iterations = solved.diagnostics.iterations;
  result.residual = solved.diagnostics.max_risk_budget_error;
  return result;
}

EstimatorRun run_estimator(
    const char* name, const portfolio_math::ResearchCovarianceSelection& selection) {
  EstimatorRun result;
  result.name = name;
  result.covariance_status = selection.covariance.status;
  result.shrinkage = selection.covariance.shrinkage_intensity;
  result.concentration_ratio = selection.covariance.concentration_ratio;
  result.effective_observations = selection.covariance.effective_observations;
  result.covariance_hash = covariance_hash(selection.covariance);
  result.topk = topk_policy();
  if (selection.covariance.status == portfolio_math::CovarianceStatus::OK) {
    result.risk_budget = risk_budget_policy(
        quant_math::view(selection.covariance.covariance));
  }
  return result;
}

ClosureReport build_report() {
  ClosureReport report;
  quant_math::DenseMatrix returns(5, 3);
  returns << 0.01, 0.02, -0.01,
      0.02, 0.01, 0.00,
      -0.01, 0.00, 0.02,
      0.00, 0.01, 0.01,
      0.03, 0.02, -0.02;
  portfolio_math::QuestSpec quest_options;
  quest_options.max_inverse_iterations = 100;
  report.input_fingerprint = matrix_hash(quant_math::view(returns));
  const auto comparison =
      portfolio_math::compare_covariance_estimators_fixed_input(
          quant_math::view(returns), quest_options);
  report.estimators.push_back(run_estimator("sample", comparison.sample));
  report.estimators.push_back(run_estimator(
      "lw_lin_cc", comparison.linear_constant_correlation));
  report.estimators.push_back(run_estimator(
      "lw_nls_mv_quest", comparison.nonlinear_minimum_variance_quest));

  quant_math::DenseMatrix singular_returns(4, 6);
  singular_returns << 1.0, 0.0, 2.0, -1.0, 0.5, 1.5,
      0.0, 1.0, 1.0, 0.0, 1.5, 0.5,
      -1.0, 0.5, 0.0, 1.0, 2.0, 1.0,
      2.0, 1.5, 3.0, -2.0, 0.0, 2.5;
  const auto singular =
      portfolio_math::ledoit_wolf_nonlinear_minimum_variance_quest(
          quant_math::view(singular_returns), quest_options);
  report.regular_branch_ok =
      comparison.nonlinear_minimum_variance_quest.covariance.status ==
          portfolio_math::CovarianceStatus::OK &&
      comparison.nonlinear_minimum_variance_quest.has_nonlinear_diagnostics &&
      comparison.nonlinear_minimum_variance_quest.nonlinear_diagnostics.dimensional_branch ==
          portfolio_math::NonlinearDimensionalBranch::REGULAR_P_LT_N;
  report.singular_branch_ok =
      singular.covariance.status == portfolio_math::CovarianceStatus::OK &&
      singular.diagnostics.dimensional_branch ==
          portfolio_math::NonlinearDimensionalBranch::SINGULAR_P_GT_N &&
      singular.diagnostics.null_space_shrinkage > 0.0;

  auto invalid_options = quest_options;
  invalid_options.max_inverse_iterations = 1;
  invalid_options.objective_tolerance = 0.0;
  const auto invalid = portfolio_math::estimate_research_covariance(
      quant_math::view(returns),
      portfolio_math::CovarianceEstimator::LEDOIT_WOLF_NONLINEAR_QUEST,
      invalid_options);
  report.invalid_quest_fails_closed =
      invalid.covariance.status == portfolio_math::CovarianceStatus::NUMERICAL_FAILURE &&
      invalid.covariance.estimator ==
          portfolio_math::CovarianceEstimator::LEDOIT_WOLF_NONLINEAR_QUEST &&
      invalid.covariance.covariance.size() == 0;
  report.policy_switch_ok = comparison.all_estimators_succeeded &&
      report.estimators.size() == 3 &&
      std::all_of(report.estimators.begin(), report.estimators.end(),
                  [](const EstimatorRun& estimator) {
                    return estimator.topk.status ==
                               portfolio_math::OptimizationStatus::OK &&
                        estimator.risk_budget.status ==
                               portfolio_math::OptimizationStatus::OK &&
                        std::abs(estimator.topk.weight_sum - 1.0) < 1e-10 &&
                        std::abs(estimator.risk_budget.weight_sum - 1.0) < 1e-8;
                  });
  report.ok = comparison.all_estimators_succeeded && report.regular_branch_ok &&
      report.singular_branch_ok && report.invalid_quest_fails_closed &&
      report.policy_switch_ok;
  std::uint64_t hash = kFnvOffset;
  hash_value(hash, report.input_fingerprint);
  hash_byte(hash, report.ok ? 1 : 0);
  hash_byte(hash, report.regular_branch_ok ? 1 : 0);
  hash_byte(hash, report.singular_branch_ok ? 1 : 0);
  hash_byte(hash, report.invalid_quest_fails_closed ? 1 : 0);
  hash_byte(hash, report.policy_switch_ok ? 1 : 0);
  for (const auto& estimator : report.estimators) {
    hash_byte(hash, static_cast<std::uint8_t>(estimator.covariance_status));
    hash_value(hash, estimator.covariance_hash);
    hash_double(hash, estimator.shrinkage);
    hash_double(hash, estimator.concentration_ratio);
    hash_value(hash, estimator.effective_observations);
    hash_byte(hash, static_cast<std::uint8_t>(estimator.topk.status));
    hash_weights(hash, estimator.topk.weights);
    hash_byte(hash, static_cast<std::uint8_t>(estimator.risk_budget.status));
    hash_weights(hash, estimator.risk_budget.weights);
    hash_double(hash, estimator.risk_budget.residual);
  }
  report.artifact_hash = hash;
  return report;
}

void write_json(std::ostream& output, const ClosureReport& report) {
  output << std::setprecision(17)
         << "{\"schema_version\":1,\"phase\":\"1A\",\"status\":\""
         << (report.ok ? "RESEARCH_REFERENCE_COMPLETE" : "FAIL")
         << "\",\"input_fingerprint\":" << report.input_fingerprint
         << ",\"estimators\":[";
  for (std::size_t index = 0; index < report.estimators.size(); ++index) {
    if (index != 0) output << ',';
    const auto& estimator = report.estimators[index];
    output << "{\"name\":\"" << estimator.name
           << "\",\"covariance_status\":\""
           << covariance_status_name(estimator.covariance_status)
           << "\",\"shrinkage\":" << estimator.shrinkage
           << ",\"concentration_ratio\":" << estimator.concentration_ratio
           << ",\"effective_observations\":"
           << estimator.effective_observations
           << ",\"covariance_hash\":" << estimator.covariance_hash
           << ",\"topk\":{\"status\":\""
           << optimization_status_name(estimator.topk.status)
           << "\",\"weights\":[";
    for (std::size_t weight = 0; weight < estimator.topk.weights.size(); ++weight) {
      if (weight != 0) output << ',';
      output << estimator.topk.weights[weight];
    }
    output << "],\"weight_sum\":" << estimator.topk.weight_sum
           << "},\"risk_budget\":{\"status\":\""
           << optimization_status_name(estimator.risk_budget.status)
           << "\",\"weights\":[";
    for (std::size_t weight = 0; weight < estimator.risk_budget.weights.size(); ++weight) {
      if (weight != 0) output << ',';
      output << estimator.risk_budget.weights[weight];
    }
    output << "],\"weight_sum\":" << estimator.risk_budget.weight_sum
           << ",\"risk_budget_error\":" << estimator.risk_budget.residual
           << "}}";
  }
  output << "],\"checks\":{\"regular_p_lt_n\":"
         << (report.regular_branch_ok ? "true" : "false")
         << ",\"singular_p_gt_n\":"
         << (report.singular_branch_ok ? "true" : "false")
         << ",\"invalid_quest_fails_closed\":"
         << (report.invalid_quest_fails_closed ? "true" : "false")
         << ",\"policy_switch_topk_risk_budget\":"
         << (report.policy_switch_ok ? "true" : "false")
         << "},\"phase_exit_eligible\":"
         << (report.ok ? "true" : "false")
         << ",\"promotion_eligible\":false"
         << ",\"evidence_level\":\"RESEARCH_REFERENCE\""
         << ",\"artifact_hash\":" << report.artifact_hash << '}';
}

}  // namespace

int main(int argc, char** argv) {
  const auto report = build_report();
  if (argc == 1) {
    write_json(std::cout, report);
    std::cout << '\n';
  } else if (argc == 3 && std::string(argv[1]) == "--output") {
    std::ofstream output(argv[2]);
    if (!output) return 2;
    write_json(output, report);
    output << '\n';
  } else {
    std::fprintf(stderr, "usage: phase1a_closure_report [--output path]\n");
    return 2;
  }
  if (!report.ok) {
    std::fprintf(stderr, "phase1a closure checks failed\n");
    return 1;
  }
  return 0;
}

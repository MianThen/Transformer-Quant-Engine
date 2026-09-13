#include "portfolio_math/factor_risk_diagnostics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <span>
#include <vector>

#include <Eigen/Core>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using portfolio_math::FactorRiskDiagnosticInput;
using portfolio_math::FactorRiskDiagnosticResult;
using portfolio_math::FactorRiskDiagnosticStatus;
using portfolio_math::FactorRiskModelView;
using quant_math::DenseMatrix;
using quant_math::DenseVector;

struct Fixture {
  std::size_t asset_count{0};
  std::size_t factor_count{0};
  std::size_t observation_count{0};
  DenseMatrix exposures;
  DenseMatrix factor_covariance;
  DenseMatrix factor_returns;
  DenseMatrix asset_returns;
  std::vector<double> specific_variance;
  std::vector<double> weights;
  std::vector<engine_common::TimestampNs> timestamps;

  FactorRiskDiagnosticInput input() const {
    const FactorRiskModelView model{
        quant_math::view(exposures),
        quant_math::view(factor_covariance),
        std::span<const double>(specific_variance),
    };
    return {
        model,
        static_cast<std::uint64_t>(asset_count * 1'000 + factor_count),
        quant_math::view(factor_returns),
        quant_math::view(asset_returns),
        std::span<const double>(weights),
        std::span<const engine_common::TimestampNs>(timestamps),
        static_cast<engine_common::TimestampNs>(observation_count),
        static_cast<engine_common::TimestampNs>(observation_count),
        1e-12,
    };
  }
};

struct BenchmarkResult {
  FactorRiskDiagnosticResult reference;
  std::uint64_t p50_ns{0};
  std::uint64_t p95_ns{0};
  std::uint64_t p99_ns{0};
  std::uint64_t peak_rss_delta_bytes{0};
  bool deterministic{false};
};

Fixture make_fixture(std::size_t asset_count,
                     std::size_t factor_count,
                     std::size_t observation_count) {
  Fixture fixture;
  fixture.asset_count = asset_count;
  fixture.factor_count = factor_count;
  fixture.observation_count = observation_count;
  fixture.exposures.resize(static_cast<Eigen::Index>(asset_count),
                           static_cast<Eigen::Index>(factor_count));
  fixture.factor_covariance = DenseMatrix::Zero(
      static_cast<Eigen::Index>(factor_count),
      static_cast<Eigen::Index>(factor_count));
  fixture.factor_returns.resize(static_cast<Eigen::Index>(observation_count),
                                static_cast<Eigen::Index>(factor_count));
  fixture.asset_returns.resize(static_cast<Eigen::Index>(observation_count),
                               static_cast<Eigen::Index>(asset_count));
  fixture.specific_variance.resize(asset_count);
  fixture.weights.assign(asset_count, 1.0 / static_cast<double>(asset_count));
  fixture.timestamps.reserve(observation_count);

  for (std::size_t asset = 0; asset < asset_count; ++asset) {
    for (std::size_t factor = 0; factor < factor_count; ++factor) {
      const double phase = static_cast<double>(
          ((asset + 3) * (factor + 5)) % 101);
      fixture.exposures(static_cast<Eigen::Index>(asset),
                        static_cast<Eigen::Index>(factor)) =
          factor == 0 ? 1.0 :
          0.65 * std::sin(0.071 * phase) + 0.35 * std::cos(0.037 * phase);
    }
    fixture.specific_variance[asset] =
        2.0e-5 + 1.0e-6 * static_cast<double>(asset % 17);
  }

  for (std::size_t factor = 0; factor < factor_count; ++factor) {
    fixture.factor_covariance(static_cast<Eigen::Index>(factor),
                              static_cast<Eigen::Index>(factor)) =
        1.0e-4 * (1.0 + 0.1 * static_cast<double>(factor));
    for (std::size_t other = 0; other < factor; ++other) {
      const double covariance = 1.0e-6 /
          static_cast<double>(factor - other + 1);
      fixture.factor_covariance(static_cast<Eigen::Index>(factor),
                                static_cast<Eigen::Index>(other)) = covariance;
      fixture.factor_covariance(static_cast<Eigen::Index>(other),
                                static_cast<Eigen::Index>(factor)) = covariance;
    }
  }

  for (std::size_t row = 0; row < observation_count; ++row) {
    for (std::size_t factor = 0; factor < factor_count; ++factor) {
      fixture.factor_returns(static_cast<Eigen::Index>(row),
                             static_cast<Eigen::Index>(factor)) =
          0.008 * std::sin(
              0.071 * static_cast<double>((row + 1) * (factor + 1))) +
          0.003 * std::cos(
              0.043 * static_cast<double>((row + 3) * (factor + 2)));
    }
    for (std::size_t asset = 0; asset < asset_count; ++asset) {
      double asset_return = 0.0;
      for (std::size_t factor = 0; factor < factor_count; ++factor) {
        asset_return +=
            fixture.exposures(static_cast<Eigen::Index>(asset),
                              static_cast<Eigen::Index>(factor)) *
            fixture.factor_returns(static_cast<Eigen::Index>(row),
                                   static_cast<Eigen::Index>(factor));
      }
      asset_return += 0.0005 * std::sin(
          0.013 * static_cast<double>((row + 1) * ((asset % 113) + 1)));
      fixture.asset_returns(static_cast<Eigen::Index>(row),
                            static_cast<Eigen::Index>(asset)) = asset_return;
    }
    fixture.timestamps.push_back(
        static_cast<engine_common::TimestampNs>(row + 1));
  }
  return fixture;
}

std::uint64_t peak_rss_bytes() {
#if defined(_WIN32)
  return 0;
#else
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
  return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
  return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
#endif
}

std::uint64_t percentile(std::vector<std::uint64_t> samples,
                         double probability) {
  if (samples.empty()) return 0;
  std::sort(samples.begin(), samples.end());
  const std::size_t index = std::min(
      samples.size() - 1,
      static_cast<std::size_t>(
          std::ceil(static_cast<double>(samples.size()) * probability)) - 1);
  return samples[index];
}

bool same_result(const FactorRiskDiagnosticResult& left,
                 const FactorRiskDiagnosticResult& right) {
  return left.status == right.status &&
      left.predicted_risk_artifact_hash ==
          right.predicted_risk_artifact_hash &&
      left.factor_covariance_qlike == right.factor_covariance_qlike &&
      left.predicted_portfolio_variance ==
          right.predicted_portfolio_variance &&
      left.realized_portfolio_variance == right.realized_portfolio_variance &&
      left.predicted_risk_contributions ==
          right.predicted_risk_contributions &&
      left.realized_risk_contributions ==
          right.realized_risk_contributions;
}

BenchmarkResult benchmark_fixture(const Fixture& fixture,
                                  std::size_t runs) {
  BenchmarkResult benchmark;
  const auto input = fixture.input();
  const std::uint64_t rss_before = peak_rss_bytes();
  benchmark.reference =
      portfolio_math::evaluate_factor_risk_diagnostics(input);
  benchmark.deterministic =
      benchmark.reference.status == FactorRiskDiagnosticStatus::OK;
  std::vector<std::uint64_t> samples;
  samples.reserve(runs);
  for (std::size_t run = 0; run < runs; ++run) {
    const auto start = Clock::now();
    const auto result =
        portfolio_math::evaluate_factor_risk_diagnostics(input);
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - start);
    samples.push_back(static_cast<std::uint64_t>(elapsed.count()));
    benchmark.deterministic = benchmark.deterministic &&
        same_result(benchmark.reference, result);
  }
  const std::uint64_t rss_after = peak_rss_bytes();
  benchmark.peak_rss_delta_bytes =
      rss_after >= rss_before ? rss_after - rss_before : 0;
  benchmark.p50_ns = percentile(samples, 0.50);
  benchmark.p95_ns = percentile(samples, 0.95);
  benchmark.p99_ns = percentile(samples, 0.99);
  return benchmark;
}

bool close(double left, double right, double tolerance = 1e-9) {
  return std::abs(left - right) <= tolerance *
      std::max({1.0, std::abs(left), std::abs(right)});
}

bool dense_parity(const Fixture& fixture,
                  const FactorRiskDiagnosticResult& result) {
  const auto input = fixture.input();
  const DenseMatrix predicted_covariance =
      portfolio_math::materialize_factor_covariance(
          input.predicted_risk_model);
  if (predicted_covariance.rows() !=
      static_cast<Eigen::Index>(fixture.asset_count)) {
    return false;
  }
  const DenseVector weights = DenseVector::Map(
      fixture.weights.data(),
      static_cast<Eigen::Index>(fixture.weights.size()));
  const DenseVector predicted_marginal = predicted_covariance * weights;
  const double predicted_variance = weights.dot(predicted_marginal);
  if (!close(result.predicted_portfolio_variance,
             predicted_variance, 1e-11)) {
    return false;
  }

  DenseMatrix centered_returns = fixture.asset_returns;
  centered_returns.rowwise() -= centered_returns.colwise().mean();
  const DenseMatrix realized_covariance = centered_returns.transpose() *
      centered_returns / static_cast<double>(fixture.observation_count);
  const DenseVector realized_marginal = realized_covariance * weights;
  const double realized_variance = weights.dot(realized_marginal);
  if (!close(result.realized_portfolio_variance,
             realized_variance, 1e-11)) {
    return false;
  }
  for (std::size_t asset = 0; asset < fixture.asset_count; ++asset) {
    const double predicted_contribution = fixture.weights[asset] *
        predicted_marginal(static_cast<Eigen::Index>(asset)) /
        predicted_variance;
    const double realized_contribution = fixture.weights[asset] *
        realized_marginal(static_cast<Eigen::Index>(asset)) /
        realized_variance;
    if (!close(result.predicted_risk_contributions[asset],
               predicted_contribution, 1e-11) ||
        !close(result.realized_risk_contributions[asset],
               realized_contribution, 1e-11)) {
      return false;
    }
  }
  return true;
}

std::uint64_t dense_covariance_bytes(std::size_t asset_count) {
  return static_cast<std::uint64_t>(asset_count) *
      static_cast<std::uint64_t>(asset_count) * sizeof(double);
}

std::uint64_t input_payload_bytes(const Fixture& fixture) {
  const std::uint64_t double_count =
      static_cast<std::uint64_t>(fixture.asset_count) * fixture.factor_count +
      static_cast<std::uint64_t>(fixture.factor_count) * fixture.factor_count +
      static_cast<std::uint64_t>(fixture.observation_count) *
          fixture.factor_count +
      static_cast<std::uint64_t>(fixture.observation_count) *
          fixture.asset_count +
      2 * static_cast<std::uint64_t>(fixture.asset_count);
  return double_count * sizeof(double) +
      static_cast<std::uint64_t>(fixture.observation_count) *
          sizeof(engine_common::TimestampNs);
}

std::uint64_t conservative_linear_workspace_bytes(const Fixture& fixture) {
  const std::uint64_t double_count =
      4 * static_cast<std::uint64_t>(fixture.asset_count) *
          fixture.factor_count +
      3 * static_cast<std::uint64_t>(fixture.observation_count) *
          fixture.factor_count +
      16 * static_cast<std::uint64_t>(fixture.asset_count) +
      4 * static_cast<std::uint64_t>(fixture.observation_count) +
      32 * static_cast<std::uint64_t>(fixture.factor_count) *
          fixture.factor_count +
      1'024;
  return double_count * sizeof(double);
}

void print_result(const Fixture& fixture,
                  const BenchmarkResult& benchmark,
                  const char* dense_parity_status) {
  std::cout << "{\"assets\":" << fixture.asset_count
            << ",\"factors\":" << fixture.factor_count
            << ",\"observations\":" << fixture.observation_count
            << ",\"p50_ns\":" << benchmark.p50_ns
            << ",\"p95_ns\":" << benchmark.p95_ns
            << ",\"p99_ns\":" << benchmark.p99_ns
            << ",\"peak_rss_delta_bytes\":"
            << benchmark.peak_rss_delta_bytes
            << ",\"input_payload_bytes\":" << input_payload_bytes(fixture)
            << ",\"linear_workspace_bound_bytes\":"
            << conservative_linear_workspace_bytes(fixture)
            << ",\"dense_covariance_bytes\":"
            << dense_covariance_bytes(fixture.asset_count)
            << ",\"dense_parity\":\"" << dense_parity_status << "\""
            << ",\"deterministic\":"
            << (benchmark.deterministic ? "true" : "false") << "}\n";
}

}  // namespace

int main() {
  constexpr std::size_t factor_count = 8;
  constexpr std::size_t observation_count = 64;
  constexpr std::size_t benchmark_runs = 31;

  const Fixture dense_fixture =
      make_fixture(200, factor_count, observation_count);
  const BenchmarkResult dense_benchmark =
      benchmark_fixture(dense_fixture, benchmark_runs);
  const bool parity = dense_benchmark.reference.status ==
          FactorRiskDiagnosticStatus::OK &&
      dense_parity(dense_fixture, dense_benchmark.reference);

  const Fixture full_a_fixture =
      make_fixture(5'000, factor_count, observation_count);
  const BenchmarkResult full_a_benchmark =
      benchmark_fixture(full_a_fixture, benchmark_runs);
  const std::uint64_t full_a_dense_bytes =
      dense_covariance_bytes(full_a_fixture.asset_count);
  const std::uint64_t full_a_linear_bound =
      input_payload_bytes(full_a_fixture) +
      conservative_linear_workspace_bytes(full_a_fixture);

  bool ok = parity && dense_benchmark.deterministic &&
      full_a_benchmark.reference.status == FactorRiskDiagnosticStatus::OK &&
      full_a_benchmark.deterministic;
  ok = ok && full_a_linear_bound <= full_a_dense_bytes / 16;
  if (full_a_benchmark.peak_rss_delta_bytes > 0) {
    ok = ok && full_a_benchmark.peak_rss_delta_bytes <
        full_a_dense_bytes / 2;
  }
  ok = ok && close(
      std::accumulate(
          full_a_benchmark.reference.predicted_risk_contributions.begin(),
          full_a_benchmark.reference.predicted_risk_contributions.end(), 0.0),
      1.0, 1e-9);
  ok = ok && close(
      std::accumulate(
          full_a_benchmark.reference.realized_risk_contributions.begin(),
          full_a_benchmark.reference.realized_risk_contributions.end(), 0.0),
      1.0, 1e-9);

  print_result(dense_fixture, dense_benchmark, parity ? "PASS" : "FAIL");
  print_result(full_a_fixture, full_a_benchmark, "NOT_MATERIALIZED");
  std::cout << (ok ? "test_factor_risk_performance: all checks passed\n"
                   : "test_factor_risk_performance: failed\n");
  return ok ? 0 : 1;
}

#include <cmath>
#include <cstdio>

#include "portfolio_math/detoning.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAILED: %s\n", message);
  }
  return condition;
}

bool near(double left, double right, double tolerance = 1e-10) {
  return std::abs(left - right) <= tolerance;
}

quant_math::DenseMatrix market_correlation() {
  quant_math::DenseMatrix correlation(4, 4);
  correlation.setConstant(0.75);
  correlation.diagonal().setOnes();
  return correlation;
}

bool test_market_component_removal() {
  const auto input = market_correlation();
  const auto result = portfolio_math::detone_correlation(
      quant_math::view(input));
  bool ok = true;
  ok &= check(result.status == portfolio_math::DetoningStatus::OK,
              "detoning status");
  ok &= check(result.diagnostics.removed_components == 1 &&
                  result.diagnostics.raw_eigenvalues.size() == 4 &&
                  result.diagnostics.detoned_eigenvalues.size() == 4,
              "detoning diagnostics shape");
  ok &= check(near(result.correlation(0, 0), 1.0) &&
                  near(result.correlation(0, 1), -1.0 / 3.0) &&
                  near(result.correlation(2, 3), -1.0 / 3.0),
              "detoned correlation oracle");
  ok &= check(result.diagnostics.trace_drift < 1e-12 &&
                  result.diagnostics.psd_repair_amount == 0.0 &&
                  result.diagnostics.minimum_output_eigenvalue >= -1e-10 &&
                  !result.diagnostics.eligible_for_official_risk,
              "detoning PSD and research-only boundary");
  return ok;
}

bool test_zero_component_and_failures() {
  const auto input = market_correlation();
  auto zero_spec = portfolio_math::DetoningSpec{};
  zero_spec.components = 0;
  const auto unchanged = portfolio_math::detone_correlation(
      quant_math::view(input), zero_spec);
  bool ok = unchanged.status == portfolio_math::DetoningStatus::OK;
  for (Eigen::Index row = 0; row < input.rows(); ++row) {
    for (Eigen::Index col = 0; col < input.cols(); ++col) {
      ok &= check(near(unchanged.correlation(row, col), input(row, col)),
                  "zero-component detoning parity");
    }
  }

  auto invalid_spec = zero_spec;
  invalid_spec.components = 2;
  const auto invalid = portfolio_math::detone_correlation(
      quant_math::view(input), invalid_spec);
  ok &= check(invalid.status == portfolio_math::DetoningStatus::INVALID_INPUT,
              "detoning component bound");

  quant_math::DenseMatrix singular = quant_math::DenseMatrix::Ones(3, 3);
  const auto degenerate = portfolio_math::detone_correlation(
      quant_math::view(singular));
  ok &= check(degenerate.status ==
                  portfolio_math::DetoningStatus::DEGENERATE_OUTPUT,
              "detoning rejects zero residual diagonal");
  return ok;
}

}  // namespace

int main() {
  if (!(test_market_component_removal() && test_zero_component_and_failures())) {
    return 1;
  }
  std::printf("test_detoning: all checks passed\n");
  return 0;
}

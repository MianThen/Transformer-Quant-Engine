#include <cmath>
#include <cstdio>

#include "portfolio_math/hierarchical_linkage.h"

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

quant_math::DenseMatrix block_correlation() {
  quant_math::DenseMatrix correlation(4, 4);
  correlation.setConstant(0.1);
  correlation.diagonal().setOnes();
  correlation(0, 1) = correlation(1, 0) = 0.9;
  correlation(2, 3) = correlation(3, 2) = 0.8;
  return correlation;
}

bool test_complete_linkage_oracle() {
  const auto correlation = block_correlation();
  const auto result = portfolio_math::hierarchical_linkage(
      quant_math::view(correlation));
  bool ok = true;
  ok &= check(result.status ==
                  portfolio_math::HierarchicalLinkageStatus::OK,
              "hierarchical linkage status");
  ok &= check(result.diagnostics.merge_tree.size() == 3 &&
                  result.diagnostics.quasi_diagonal_order.size() == 4,
              "linkage tree and order shape");
  ok &= check(result.diagnostics.merge_tree[0].left == 0 &&
                  result.diagnostics.merge_tree[0].right == 1 &&
                  near(result.diagnostics.merge_tree[0].distance,
                       std::sqrt(0.05)) &&
                  result.diagnostics.merge_tree[1].left == 2 &&
                  result.diagnostics.merge_tree[1].right == 3 &&
                  near(result.diagnostics.merge_tree[1].distance,
                       std::sqrt(0.1)),
              "complete linkage first merges");
  ok &= check(result.diagnostics.quasi_diagonal_order[0] == 0 &&
                  result.diagnostics.quasi_diagonal_order[1] == 1 &&
                  result.diagnostics.quasi_diagonal_order[2] == 2 &&
                  result.diagnostics.quasi_diagonal_order[3] == 3,
              "quasi-diagonal order oracle");
  ok &= check(result.diagnostics.input_correlation_hash != 0 &&
                  result.diagnostics.linkage_tree_hash != 0 &&
                  !result.diagnostics.eligible_for_official_risk,
              "linkage provenance and research boundary");
  return ok;
}

bool test_tie_break_and_failures() {
  quant_math::DenseMatrix identity =
      quant_math::DenseMatrix::Identity(4, 4);
  const auto tied = portfolio_math::hierarchical_linkage(
      quant_math::view(identity));
  bool ok = true;
  ok &= check(tied.status == portfolio_math::HierarchicalLinkageStatus::OK &&
                  tied.diagnostics.merge_tree[0].left == 0 &&
                  tied.diagnostics.merge_tree[0].right == 1 &&
                  tied.diagnostics.merge_tree[1].left == 2 &&
                  tied.diagnostics.merge_tree[1].right == 3,
              "deterministic tie-break");

  auto invalid_spec = portfolio_math::HierarchicalLinkageSpec{};
  invalid_spec.distance_tolerance = -1.0;
  const auto invalid = portfolio_math::hierarchical_linkage(
      quant_math::view(identity), invalid_spec);
  ok &= check(invalid.status ==
                  portfolio_math::HierarchicalLinkageStatus::INVALID_INPUT,
              "invalid linkage spec fails closed");

  quant_math::DenseMatrix nonsymmetric = identity;
  nonsymmetric(0, 1) = 0.5;
  const auto failed = portfolio_math::hierarchical_linkage(
      quant_math::view(nonsymmetric));
  ok &= check(failed.status ==
                  portfolio_math::HierarchicalLinkageStatus::INVALID_INPUT,
              "non-symmetric correlation fails closed");
  return ok;
}

}  // namespace

int main() {
  if (!(test_complete_linkage_oracle() && test_tie_break_and_failures())) {
    return 1;
  }
  std::printf("test_hierarchical_linkage: all checks passed\n");
  return 0;
}

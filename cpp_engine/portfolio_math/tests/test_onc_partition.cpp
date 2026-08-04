#include <cmath>
#include <cstdio>

#include "portfolio_math/onc_partition.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

quant_math::DenseMatrix block_correlation() {
  quant_math::DenseMatrix correlation(4, 4);
  correlation.setConstant(0.1);
  correlation.diagonal().setOnes();
  correlation(0, 1) = correlation(1, 0) = 0.9;
  correlation(2, 3) = correlation(3, 2) = 0.8;
  return correlation;
}

quant_math::DenseMatrix three_block_correlation() {
  quant_math::DenseMatrix correlation(6, 6);
  correlation.setConstant(0.05);
  correlation.diagonal().setOnes();
  correlation(0, 1) = correlation(1, 0) = 0.90;
  correlation(2, 3) = correlation(3, 2) = 0.88;
  correlation(4, 5) = correlation(5, 4) = 0.86;
  return correlation;
}

bool same_partition_after_permutation(
    const std::vector<std::uint32_t>& first,
    const std::vector<std::uint32_t>& second,
    const std::size_t permutation[4]) {
  for (std::size_t left = 0; left < 4; ++left) {
    for (std::size_t right = 0; right < 4; ++right) {
      const bool first_same = first[permutation[left]] == first[permutation[right]];
      const bool second_same = second[left] == second[right];
      if (first_same != second_same) return false;
    }
  }
  return true;
}

bool test_block_recovery_and_determinism() {
  const auto correlation = block_correlation();
  auto spec = portfolio_math::OncSpec{};
  spec.min_clusters = 2;
  spec.max_clusters = 3;
  spec.min_cluster_size = 2;
  spec.repeats = 3;
  spec.seeds = {17, 29, 43};
  const auto first = portfolio_math::onc_partition(
      quant_math::view(correlation), spec);
  const auto second = portfolio_math::onc_partition(
      quant_math::view(correlation), spec);
  bool ok = true;
  ok &= check(first.status == portfolio_math::OncStatus::OK,
              "ONC status");
  ok &= check(first.diagnostics.selected_cluster_count == 2 &&
                  first.cluster_id_by_symbol.size() == 4 &&
                  first.diagnostics.selected_quality > 0.0,
              "ONC block recovery");
  ok &= check(first.cluster_id_by_symbol[0] == first.cluster_id_by_symbol[1] &&
                  first.cluster_id_by_symbol[2] == first.cluster_id_by_symbol[3] &&
                  first.cluster_id_by_symbol[0] != first.cluster_id_by_symbol[2],
              "ONC partition membership");
  ok &= check(first.diagnostics.partition_hash ==
                  second.diagnostics.partition_hash &&
                  first.diagnostics.candidates.size() == 3 &&
                  first.diagnostics.best_second_gap >= 0.0 &&
                  !first.diagnostics.eligible_for_official_risk,
              "ONC deterministic diagnostics");
  return ok;
}

bool test_permutation_and_failures() {
  const auto source = block_correlation();
  quant_math::DenseMatrix permuted(source.rows(), source.cols());
  const std::size_t permutation[4]{2, 3, 0, 1};
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      permuted(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
          source(static_cast<Eigen::Index>(permutation[row]),
                 static_cast<Eigen::Index>(permutation[col]));
    }
  }
  const auto first = portfolio_math::onc_partition(quant_math::view(source));
  const auto second =
      portfolio_math::onc_partition(quant_math::view(permuted));
  bool ok = check(first.status == portfolio_math::OncStatus::OK &&
                      second.status == portfolio_math::OncStatus::OK,
                  "ONC permutation status");
  if (ok) {
    ok &= check(same_partition_after_permutation(
                    first.cluster_id_by_symbol, second.cluster_id_by_symbol,
                    permutation),
                "ONC permutation partition equivalence");
  }

  auto invalid_spec = portfolio_math::OncSpec{};
  invalid_spec.repeats = 5;
  invalid_spec.seeds = {1, 2, 3, 4};
  const auto invalid = portfolio_math::onc_partition(
      quant_math::view(source), invalid_spec);
  ok &= check(invalid.status == portfolio_math::OncStatus::INVALID_INPUT,
              "ONC seed/repeat contract");

  auto impossible = portfolio_math::OncSpec{};
  impossible.min_cluster_size = 3;
  impossible.min_clusters = 2;
  impossible.max_clusters = 2;
  const auto failed = portfolio_math::onc_partition(
      quant_math::view(source), impossible);
  ok &= check(failed.status == portfolio_math::OncStatus::INVALID_INPUT,
              "ONC min-cluster-size guard");
  return ok;
}

bool test_k_search() {
  const auto correlation = three_block_correlation();
  auto spec = portfolio_math::OncSpec{};
  spec.min_clusters = 2;
  spec.max_clusters = 3;
  spec.min_cluster_size = 2;
  spec.repeats = 2;
  spec.seeds = {17, 29};
  const auto result = portfolio_math::onc_partition(
      quant_math::view(correlation), spec);
  bool ok = true;
  ok &= check(result.status == portfolio_math::OncStatus::OK &&
                  result.diagnostics.selected_cluster_count == 3 &&
                  result.diagnostics.candidates.size() == 4,
              "ONC K search selects three blocks");
  if (ok) {
    ok &= check(result.cluster_id_by_symbol[0] == result.cluster_id_by_symbol[1] &&
                    result.cluster_id_by_symbol[2] == result.cluster_id_by_symbol[3] &&
                    result.cluster_id_by_symbol[4] == result.cluster_id_by_symbol[5],
                "ONC three-block partition");
  }
  return ok;
}

}  // namespace

int main() {
  if (!(test_block_recovery_and_determinism() &&
        test_permutation_and_failures() && test_k_search())) {
    return 1;
  }
  std::printf("test_onc_partition: all checks passed\n");
  return 0;
}

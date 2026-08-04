#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "quant_math/matrix.h"

namespace portfolio_math {

struct OncSpec {
  std::uint32_t min_clusters{2};
  std::uint32_t max_clusters{4};
  std::uint32_t min_cluster_size{2};
  std::uint32_t max_iterations{100};
  std::uint32_t repeats{4};
  double distance_tolerance{1e-12};
  double symmetry_tolerance{1e-10};
  double psd_tolerance{1e-10};
  std::vector<std::uint64_t> seeds{17, 29, 43, 71};
};

enum class OncStatus {
  OK,
  INVALID_INPUT,
  NUMERICAL_FAILURE,
  NO_VALID_PARTITION,
};

struct OncCandidateDiagnostic {
  std::uint32_t cluster_count{0};
  std::uint64_t seed{0};
  double quality{0.0};
  double minimum_silhouette{0.0};
  double maximum_silhouette{0.0};
  bool valid{false};
};

struct OncDiagnostics {
  std::uint32_t selected_cluster_count{0};
  double selected_quality{0.0};
  double second_best_quality{0.0};
  double best_second_gap{0.0};
  std::vector<double> selected_silhouette;
  std::vector<OncCandidateDiagnostic> candidates;
  std::uint64_t input_correlation_hash{0};
  std::uint64_t partition_hash{0};
  bool eligible_for_official_risk{false};
};

struct OncPartitionResult {
  OncStatus status{OncStatus::INVALID_INPUT};
  std::vector<std::uint32_t> cluster_id_by_symbol;
  std::vector<std::uint32_t> quasi_diagonal_order;
  OncDiagnostics diagnostics;
};

[[nodiscard]] bool valid_onc_spec(const OncSpec& spec) noexcept;

[[nodiscard]] OncPartitionResult onc_partition(
    quant_math::MatrixView correlation, OncSpec spec = {});

}  // namespace portfolio_math

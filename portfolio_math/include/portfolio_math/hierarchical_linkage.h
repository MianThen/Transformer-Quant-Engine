#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "quant_math/matrix.h"

namespace portfolio_math {

enum class LinkageMethod : std::uint8_t {
  COMPLETE,
};

struct HierarchicalLinkageSpec {
  LinkageMethod method{LinkageMethod::COMPLETE};
  double distance_tolerance{1e-12};
  double symmetry_tolerance{1e-10};
  double psd_tolerance{1e-10};
};

enum class HierarchicalLinkageStatus {
  OK,
  INVALID_INPUT,
  NUMERICAL_FAILURE,
};

struct HierarchicalMerge {
  std::uint32_t left{0};
  std::uint32_t right{0};
  std::uint32_t cluster_size{0};
  double distance{0.0};
};

struct HierarchicalLinkageDiagnostics {
  LinkageMethod method{LinkageMethod::COMPLETE};
  std::vector<double> pairwise_distances;
  std::vector<HierarchicalMerge> merge_tree;
  std::vector<std::uint32_t> quasi_diagonal_order;
  std::uint64_t input_correlation_hash{0};
  std::uint64_t linkage_tree_hash{0};
  bool eligible_for_official_risk{false};
};

struct HierarchicalLinkageResult {
  HierarchicalLinkageStatus status{HierarchicalLinkageStatus::INVALID_INPUT};
  HierarchicalLinkageDiagnostics diagnostics;
};

[[nodiscard]] bool valid_hierarchical_linkage_spec(
    const HierarchicalLinkageSpec& spec) noexcept;

[[nodiscard]] HierarchicalLinkageResult hierarchical_linkage(
    quant_math::MatrixView correlation,
    HierarchicalLinkageSpec spec = {});

}  // namespace portfolio_math

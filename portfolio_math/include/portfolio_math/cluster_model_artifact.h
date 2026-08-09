#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "engine_common/types.h"
#include "portfolio_math/hierarchical_linkage.h"
#include "portfolio_math/onc_partition.h"

namespace portfolio_math {

enum class ClusterModelKind : std::uint8_t {
  HRP_HIERARCHICAL_LINKAGE,
  ONC_PARTITION,
};

enum class ClusterCorrelationSource : std::uint8_t {
  RAW,
  DENOISED,
  DENOISED_DETONED,
};

struct ClusterModelArtifact {
  std::uint32_t schema_version{1};
  ClusterModelKind kind{ClusterModelKind::HRP_HIERARCHICAL_LINKAGE};
  ClusterCorrelationSource correlation_source{ClusterCorrelationSource::RAW};
  std::string official_risk_model_sha256;
  std::string denoised_risk_sha256;
  std::uint32_t detone_components{0};
  LinkageMethod linkage_method{LinkageMethod::COMPLETE};
  std::string merge_tree_sha256;
  bool has_onc_spec{false};
  OncSpec onc_spec{};
  std::vector<engine_common::SymbolId> symbols;
  std::vector<std::uint32_t> cluster_id_by_symbol;
  std::vector<std::uint32_t> quasi_diagonal_order;
  std::uint32_t cluster_count{0};
  double quality{0.0};
  std::vector<double> silhouette;
  double stability_score{0.0};
  std::string cluster_id_by_symbol_sha256;
  std::string quasi_diagonal_order_sha256;
  engine_common::TimestampNs fit_start{0};
  engine_common::TimestampNs fit_end{0};
  engine_common::TimestampNs available_at{0};
  std::uint64_t artifact_hash{0};
};

[[nodiscard]] bool valid_cluster_model_artifact(
    const ClusterModelArtifact& artifact) noexcept;

[[nodiscard]] bool finalize_cluster_model_artifact(
    ClusterModelArtifact& artifact) noexcept;

[[nodiscard]] std::uint64_t cluster_model_artifact_hash(
    const ClusterModelArtifact& artifact) noexcept;

[[nodiscard]] std::string serialize_cluster_model_artifact(
    const ClusterModelArtifact& artifact);

}  // namespace portfolio_math

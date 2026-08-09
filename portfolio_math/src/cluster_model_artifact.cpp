#include "portfolio_math/cluster_model_artifact.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

namespace portfolio_math {
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

void hash_string(std::uint64_t& hash, const std::string& value) {
  hash_value(hash, value.size());
  for (const unsigned char character : value) hash_byte(hash, character);
}

bool digest_like(const std::string& value, bool allow_empty = false) {
  if (value.empty()) return allow_empty;
  if (value.size() != 64) return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f') ||
           (character >= 'A' && character <= 'F');
  });
}

const char* kind_name(ClusterModelKind kind) {
  switch (kind) {
    case ClusterModelKind::HRP_HIERARCHICAL_LINKAGE:
      return "hrp_hierarchical_linkage";
    case ClusterModelKind::ONC_PARTITION:
      return "onc_partition";
  }
  return "invalid";
}

const char* source_name(ClusterCorrelationSource source) {
  switch (source) {
    case ClusterCorrelationSource::RAW:
      return "raw";
    case ClusterCorrelationSource::DENOISED:
      return "denoised";
    case ClusterCorrelationSource::DENOISED_DETONED:
      return "denoised_detoned";
  }
  return "invalid";
}

const char* linkage_name(LinkageMethod method) {
  return method == LinkageMethod::COMPLETE ? "complete" : "invalid";
}

bool valid_order(const std::vector<std::uint32_t>& order,
                 std::size_t expected_size) {
  if (order.size() != expected_size) return false;
  std::vector<bool> seen(expected_size, false);
  for (const auto value : order) {
    if (value >= expected_size || seen[value]) return false;
    seen[value] = true;
  }
  return true;
}

bool valid_base(const ClusterModelArtifact& artifact) {
  if (artifact.schema_version != 1 || artifact.symbols.empty() ||
      artifact.fit_start <= 0 || artifact.fit_start > artifact.fit_end ||
      artifact.fit_end > artifact.available_at ||
      !digest_like(artifact.official_risk_model_sha256) ||
      !std::isfinite(artifact.quality) ||
      !std::isfinite(artifact.stability_score) ||
      artifact.stability_score < 0.0 || artifact.stability_score > 1.0 ||
      !valid_order(artifact.quasi_diagonal_order, artifact.symbols.size()) ||
      !digest_like(artifact.quasi_diagonal_order_sha256)) {
    return false;
  }
  if (!std::is_sorted(artifact.symbols.begin(), artifact.symbols.end()) ||
      std::adjacent_find(artifact.symbols.begin(), artifact.symbols.end()) !=
          artifact.symbols.end()) {
    return false;
  }
  if (artifact.correlation_source == ClusterCorrelationSource::RAW) {
    if (!artifact.denoised_risk_sha256.empty() || artifact.detone_components != 0) {
      return false;
    }
  } else if (!digest_like(artifact.denoised_risk_sha256) ||
             (artifact.correlation_source ==
                  ClusterCorrelationSource::DENOISED &&
              artifact.detone_components != 0) ||
             (artifact.correlation_source ==
                  ClusterCorrelationSource::DENOISED_DETONED &&
              artifact.detone_components != 1)) {
    return false;
  }
  if (artifact.kind == ClusterModelKind::HRP_HIERARCHICAL_LINKAGE) {
    return artifact.linkage_method == LinkageMethod::COMPLETE &&
           !artifact.merge_tree_sha256.empty() &&
           digest_like(artifact.merge_tree_sha256) && !artifact.has_onc_spec &&
           artifact.cluster_count == 0 && artifact.cluster_id_by_symbol.empty() &&
           artifact.cluster_id_by_symbol_sha256.empty() &&
           artifact.silhouette.empty();
  }
  if (artifact.kind != ClusterModelKind::ONC_PARTITION ||
      !artifact.has_onc_spec || !valid_onc_spec(artifact.onc_spec) ||
      artifact.cluster_count < artifact.onc_spec.min_clusters ||
      artifact.cluster_count > artifact.onc_spec.max_clusters ||
      artifact.cluster_id_by_symbol.size() != artifact.symbols.size() ||
      !digest_like(artifact.cluster_id_by_symbol_sha256) ||
      artifact.silhouette.size() != artifact.symbols.size() ||
      artifact.quality < -1.0 || artifact.quality > 1.0 ||
      !artifact.merge_tree_sha256.empty()) {
    return false;
  }
  std::vector<std::uint32_t> cluster_sizes(artifact.cluster_count, 0U);
  for (const auto cluster : artifact.cluster_id_by_symbol) {
    if (cluster >= artifact.cluster_count) return false;
    ++cluster_sizes[cluster];
  }
  if (std::any_of(cluster_sizes.begin(), cluster_sizes.end(),
                  [&](std::uint32_t size) {
                    return size < artifact.onc_spec.min_cluster_size;
                  })) {
    return false;
  }
  return std::all_of(artifact.silhouette.begin(), artifact.silhouette.end(),
                     [](double value) {
                       return std::isfinite(value) && value >= -1.0 &&
                              value <= 1.0;
                     });
}

void hash_base(std::uint64_t& hash, const ClusterModelArtifact& artifact) {
  hash_value(hash, artifact.schema_version);
  hash_value(hash, static_cast<std::uint8_t>(artifact.kind));
  hash_value(hash, static_cast<std::uint8_t>(artifact.correlation_source));
  hash_string(hash, artifact.official_risk_model_sha256);
  hash_string(hash, artifact.denoised_risk_sha256);
  hash_value(hash, artifact.detone_components);
  hash_value(hash, static_cast<std::uint8_t>(artifact.linkage_method));
  hash_string(hash, artifact.merge_tree_sha256);
  hash_value(hash, artifact.has_onc_spec ? 1U : 0U);
  if (artifact.has_onc_spec) {
    hash_value(hash, artifact.onc_spec.min_clusters);
    hash_value(hash, artifact.onc_spec.max_clusters);
    hash_value(hash, artifact.onc_spec.min_cluster_size);
    hash_value(hash, artifact.onc_spec.max_iterations);
    hash_value(hash, artifact.onc_spec.repeats);
    hash_double(hash, artifact.onc_spec.distance_tolerance);
    hash_double(hash, artifact.onc_spec.symmetry_tolerance);
    hash_double(hash, artifact.onc_spec.psd_tolerance);
    for (const auto seed : artifact.onc_spec.seeds) hash_value(hash, seed);
  }
  for (const auto symbol : artifact.symbols) hash_value(hash, symbol);
  for (const auto cluster : artifact.cluster_id_by_symbol) {
    hash_value(hash, cluster);
  }
  for (const auto order : artifact.quasi_diagonal_order) hash_value(hash, order);
  hash_value(hash, artifact.cluster_count);
  hash_double(hash, artifact.quality);
  for (const auto value : artifact.silhouette) hash_double(hash, value);
  hash_double(hash, artifact.stability_score);
  hash_string(hash, artifact.cluster_id_by_symbol_sha256);
  hash_string(hash, artifact.quasi_diagonal_order_sha256);
  hash_value(hash, static_cast<std::uint64_t>(artifact.fit_start));
  hash_value(hash, static_cast<std::uint64_t>(artifact.fit_end));
  hash_value(hash, static_cast<std::uint64_t>(artifact.available_at));
}

std::string json_escape(const std::string& value) {
  std::string escaped;
  for (const char character : value) {
    if (character == '\\' || character == '"') escaped.push_back('\\');
    escaped.push_back(character);
  }
  return escaped;
}

template <typename Value>
void append_array(std::ostringstream& output, const std::vector<Value>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
}

std::string unsigned_json(const ClusterModelArtifact& artifact) {
  std::ostringstream output;
  output << std::setprecision(17);
  output << "{\"schema_version\":" << artifact.schema_version
         << ",\"kind\":\"" << kind_name(artifact.kind)
         << "\",\"correlation_source\":\""
         << source_name(artifact.correlation_source)
         << "\",\"official_risk_model_sha256\":\""
         << json_escape(artifact.official_risk_model_sha256)
         << "\",\"denoised_risk_sha256\":\""
         << json_escape(artifact.denoised_risk_sha256)
         << "\",\"detone_components\":" << artifact.detone_components
         << ",\"linkage_method\":\"" << linkage_name(artifact.linkage_method)
         << "\",\"merge_tree_sha256\":\""
         << json_escape(artifact.merge_tree_sha256) << "\""
         << ",\"symbols\":";
  append_array(output, artifact.symbols);
  if (artifact.has_onc_spec) {
    output << ",\"onc_spec\":{\"min_clusters\":"
           << artifact.onc_spec.min_clusters
           << ",\"max_clusters\":" << artifact.onc_spec.max_clusters
           << ",\"min_cluster_size\":"
           << artifact.onc_spec.min_cluster_size
           << ",\"max_iterations\":"
           << artifact.onc_spec.max_iterations << ",\"repeats\":"
           << artifact.onc_spec.repeats << ",\"seeds\":";
    append_array(output, artifact.onc_spec.seeds);
    output << '}';
  } else {
    output << ",\"onc_spec\":null";
  }
  output << ",\"cluster_id_by_symbol\":";
  append_array(output, artifact.cluster_id_by_symbol);
  output << ",\"quasi_diagonal_order\":";
  append_array(output, artifact.quasi_diagonal_order);
  output << ",\"cluster_count\":" << artifact.cluster_count
         << ",\"quality\":" << artifact.quality << ",\"silhouette\":";
  append_array(output, artifact.silhouette);
  output << ",\"stability_score\":" << artifact.stability_score
         << ",\"fit_start\":" << artifact.fit_start
         << ",\"fit_end\":" << artifact.fit_end
         << ",\"available_at\":" << artifact.available_at << '}';
  return output.str();
}

}  // namespace

bool valid_cluster_model_artifact(
    const ClusterModelArtifact& artifact) noexcept {
  if (!valid_base(artifact) || artifact.artifact_hash == 0) return false;
  return artifact.artifact_hash == cluster_model_artifact_hash(artifact);
}

bool finalize_cluster_model_artifact(ClusterModelArtifact& artifact) noexcept {
  if (!valid_base(artifact)) return false;
  artifact.artifact_hash = cluster_model_artifact_hash(artifact);
  return artifact.artifact_hash != 0;
}

std::uint64_t cluster_model_artifact_hash(
    const ClusterModelArtifact& artifact) noexcept {
  if (!valid_base(artifact)) return 0;
  std::uint64_t hash = kFnvOffset;
  hash_base(hash, artifact);
  return hash;
}

std::string serialize_cluster_model_artifact(
    const ClusterModelArtifact& artifact) {
  if (!valid_cluster_model_artifact(artifact)) return {};
  std::ostringstream output;
  output << unsigned_json(artifact);
  output.seekp(-1, std::ios_base::end);
  output << ",\"cluster_id_by_symbol_sha256\":\""
         << json_escape(artifact.cluster_id_by_symbol_sha256)
         << "\",\"quasi_diagonal_order_sha256\":\""
         << json_escape(artifact.quasi_diagonal_order_sha256)
         << "\",\"artifact_hash\":" << artifact.artifact_hash << '}';
  return output.str();
}

}  // namespace portfolio_math

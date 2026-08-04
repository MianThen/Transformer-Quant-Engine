#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "engine_common/model_types.h"

namespace qbt::ml {

struct ModelDescriptor {
  std::string model_id;
  std::string model_version;
  engine_common::FeatureProfile feature_profile =
      engine_common::FeatureProfile::BAR_V1;
  uint64_t feature_schema_hash = 0;
  uint64_t model_version_hash = 0;
  uint32_t lookback = 0;
  uint32_t feature_count = 0;
  uint32_t static_feature_count = 0;
  uint32_t max_batch_size = 0;
  engine_common::RankingScoreMode ranking_score_mode =
      engine_common::RankingScoreMode::RAW_RETURN;
  float ranking_risk_floor = 1e-4F;
  uint32_t ranking_cutoff = 0;
  uint64_t ranking_score_spec_hash = 0;
  uint64_t label_spec_hash = 0;
};

struct ModelArtifact {
  std::filesystem::path root;
  std::filesystem::path model_path;
  ModelDescriptor descriptor;
};

} // namespace qbt::ml

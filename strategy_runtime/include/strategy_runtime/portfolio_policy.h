#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "engine_common/model_types.h"
#include "engine_common/strategy.h"

namespace qbt::strategy {

struct LongOnlyPolicyConfig {
  uint32_t max_positions = 20;
  float max_position_weight = 0.05F;
  float minimum_expected_return = 0.0F;
  float minimum_ranking_score = std::numeric_limits<float>::quiet_NaN();
  float minimum_confidence = 0.0F;
  engine_common::RankingScoreMode ranking_score_mode =
      engine_common::RankingScoreMode::RAW_RETURN;
  float ranking_risk_floor = 1e-4F;
};

[[nodiscard]] double
ranking_score(const engine_common::ModelPrediction &prediction,
              const LongOnlyPolicyConfig &config) noexcept;

class LongOnlyTopKPolicy {
public:
  explicit LongOnlyTopKPolicy(LongOnlyPolicyConfig config);

  std::span<const engine_common::TargetPosition>
  build(const engine_common::PredictionBatch &predictions,
        const engine_common::MarketFrameBatchView &market,
        const engine_common::PortfolioView &portfolio);

private:
  LongOnlyPolicyConfig config_;
  std::vector<engine_common::TargetPosition> targets_;
};

} // namespace qbt::strategy

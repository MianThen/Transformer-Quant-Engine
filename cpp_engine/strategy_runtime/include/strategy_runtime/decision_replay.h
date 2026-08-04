#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "engine_common/model_types.h"
#include "engine_common/replay.h"
#include "engine_common/strategy.h"
#include "strategy_runtime/risk_manager.h"

namespace qbt::strategy {

struct DecisionTiming {
    int64_t feature_ns = 0;
    int64_t queue_ns = 0;
    int64_t infer_ns = 0;
    int64_t policy_ns = 0;
    int64_t risk_ns = 0;
};

class DecisionReplayRecorder {
public:
    explicit DecisionReplayRecorder(const std::string& path);

    [[nodiscard]] bool enabled() const noexcept { return writer_ != nullptr; }
    void record_feature(uint64_t decision_id,
                        const engine_common::FeatureBatchView& features,
                        uint64_t model_version_hash);
    void record_predictions(uint64_t decision_id,
                            uint64_t feature_schema_hash,
                            const engine_common::PredictionBatch& predictions);
    void record_targets(uint64_t decision_id, engine_common::TimestampNs timestamp,
                        uint64_t feature_schema_hash, uint64_t model_version_hash,
                        std::span<const engine_common::TargetPosition> targets);
    void record_risk(uint64_t decision_id, engine_common::TimestampNs timestamp,
                     uint64_t feature_schema_hash, uint64_t model_version_hash,
                     const engine_common::OrderIntent& intent,
                     RiskDecision decision);
    void record_order(uint64_t decision_id, engine_common::TimestampNs timestamp,
                      uint64_t feature_schema_hash, uint64_t model_version_hash,
                      const engine_common::OrderIntent& intent);
    void record_summary(uint64_t decision_id, engine_common::TimestampNs timestamp,
                        uint64_t feature_schema_hash, uint64_t model_version_hash,
                        uint32_t predictions, uint32_t targets, uint32_t approved,
                        uint32_t rejected, engine_common::StrategyStatus status,
                        const DecisionTiming& timing);
    void record_execution(const engine_common::ExecutionEvent& execution,
                          uint64_t feature_schema_hash,
                          uint64_t model_version_hash);

private:
    std::unique_ptr<engine_common::ReplayWriter> writer_;
};

}  // namespace qbt::strategy

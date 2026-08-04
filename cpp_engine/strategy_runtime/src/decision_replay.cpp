#include "strategy_runtime/decision_replay.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>

#include "engine_common/normalized_replay.h"

namespace qbt::strategy {
namespace {

engine_common::NormalizedPayloadPrefix prefix(
    size_t size, uint64_t decision_id, engine_common::TimestampNs timestamp,
    uint64_t feature_schema_hash, uint64_t model_version_hash) {
    engine_common::NormalizedPayloadPrefix value;
    value.payload_size = static_cast<uint32_t>(size);
    value.feature_schema_hash = feature_schema_hash;
    value.model_version_hash = model_version_hash;
    value.asof_timestamp = timestamp;
    value.decision_id = decision_id;
    return value;
}

uint64_t hash_bytes(uint64_t value, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < size; ++index) {
        value ^= bytes[index];
        value *= 1099511628211ULL;
    }
    return value;
}

}  // namespace

DecisionReplayRecorder::DecisionReplayRecorder(const std::string& path) {
    if (!path.empty()) {
        const std::filesystem::path destination(path);
        if (!destination.parent_path().empty()) {
            std::filesystem::create_directories(destination.parent_path());
        }
        writer_ = std::make_unique<engine_common::ReplayWriter>(path);
    }
}

void DecisionReplayRecorder::record_feature(
    uint64_t decision_id, const engine_common::FeatureBatchView& features,
    uint64_t model_version_hash) {
    if (!writer_) return;
    engine_common::NormalizedFeatureBatchRecord record;
    record.prefix = prefix(sizeof(record), decision_id, features.asof_timestamp,
                           features.feature_schema_hash, model_version_hash);
    uint64_t hash = 1469598103934665603ULL;
    hash = hash_bytes(hash, features.symbols.data(),
                      features.symbols.size_bytes());
    hash = hash_bytes(hash, features.values.data(), features.values.size_bytes());
    hash = hash_bytes(hash, features.valid_mask.data(),
                      features.valid_mask.size_bytes());
    hash = hash_bytes(hash, features.static_values.data(),
                      features.static_values.size_bytes());
    record.feature_batch_hash = hash;
    record.batch_size = features.batch_size;
    record.lookback = features.lookback;
    record.feature_count = features.feature_count;
    record.valid_token_count = static_cast<uint32_t>(std::count_if(
        features.valid_mask.begin(), features.valid_mask.end(),
        [](uint8_t value) { return value != 0; }));
    writer_->append(engine_common::ReplayRecordType::FEATURE_BATCH,
                    features.asof_timestamp, engine_common::normalized_bytes(record));
}

void DecisionReplayRecorder::record_predictions(
    uint64_t decision_id, uint64_t feature_schema_hash,
    const engine_common::PredictionBatch& predictions) {
    if (!writer_) return;
    for (size_t index = 0; index < predictions.size; ++index) {
        const auto& value = predictions.values[index];
        engine_common::NormalizedPredictionRecord record;
        record.prefix = prefix(sizeof(record), decision_id, predictions.asof_timestamp,
                               feature_schema_hash, predictions.model_version_hash);
        record.symbol_id = value.symbol_id;
        record.expected_return = value.expected_return;
        record.expected_volatility = value.expected_volatility;
        record.direction_probability = value.direction_probability;
        record.lower_quantile = value.lower_quantile;
        record.upper_quantile = value.upper_quantile;
        record.confidence = value.confidence;
        record.prediction_flags = value.flags;
        writer_->append(engine_common::ReplayRecordType::PREDICTION_BATCH,
                        predictions.asof_timestamp,
                        engine_common::normalized_bytes(record));
    }
}

void DecisionReplayRecorder::record_targets(
    uint64_t decision_id, engine_common::TimestampNs timestamp,
    uint64_t feature_schema_hash, uint64_t model_version_hash,
    std::span<const engine_common::TargetPosition> targets) {
    if (!writer_) return;
    for (const auto& value : targets) {
        engine_common::NormalizedTargetPositionRecord record;
        record.prefix = prefix(sizeof(record), decision_id, timestamp,
                               feature_schema_hash, model_version_hash);
        record.symbol_id = value.symbol_id;
        record.target_quantity = value.target_quantity;
        record.target_weight = value.target_weight;
        writer_->append(engine_common::ReplayRecordType::TARGET_POSITION_BATCH,
                        timestamp, engine_common::normalized_bytes(record));
    }
}

void DecisionReplayRecorder::record_risk(
    uint64_t decision_id, engine_common::TimestampNs timestamp,
    uint64_t feature_schema_hash, uint64_t model_version_hash,
    const engine_common::OrderIntent& intent, RiskDecision decision) {
    if (!writer_) return;
    engine_common::NormalizedRiskDecisionRecord record;
    record.prefix = prefix(sizeof(record), decision_id, timestamp,
                           feature_schema_hash, model_version_hash);
    record.client_order_id = intent.client_order_id;
    record.symbol_id = intent.symbol_id;
    record.reject_reason = static_cast<uint16_t>(decision);
    record.approved = decision == RiskDecision::APPROVED ? 1 : 0;
    record.side = static_cast<uint8_t>(intent.side);
    record.quantity = intent.quantity;
    writer_->append(engine_common::ReplayRecordType::RISK_DECISION, timestamp,
                    engine_common::normalized_bytes(record));
}

void DecisionReplayRecorder::record_order(
    uint64_t decision_id, engine_common::TimestampNs timestamp,
    uint64_t feature_schema_hash, uint64_t model_version_hash,
    const engine_common::OrderIntent& intent) {
    if (!writer_) return;
    engine_common::NormalizedOrderIntentRecord record;
    record.prefix = prefix(sizeof(record), decision_id, timestamp,
                           feature_schema_hash, model_version_hash);
    record.symbol_id = intent.symbol_id;
    record.side = static_cast<uint8_t>(intent.side);
    record.type = static_cast<uint8_t>(intent.type);
    record.quantity = intent.quantity;
    writer_->append(engine_common::ReplayRecordType::ORDER_INTENT, timestamp,
                    engine_common::normalized_bytes(record));
}

void DecisionReplayRecorder::record_summary(
    uint64_t decision_id, engine_common::TimestampNs timestamp,
    uint64_t feature_schema_hash, uint64_t model_version_hash,
    uint32_t predictions, uint32_t targets, uint32_t approved, uint32_t rejected,
    engine_common::StrategyStatus status, const DecisionTiming& timing) {
    if (!writer_) return;
    engine_common::NormalizedDecisionSummaryRecord record;
    record.prefix = prefix(sizeof(record), decision_id, timestamp,
                           feature_schema_hash, model_version_hash);
    record.prediction_count = predictions;
    record.target_count = targets;
    record.approved_order_count = approved;
    record.risk_rejection_count = rejected;
    record.strategy_status = static_cast<uint16_t>(status);
    record.feature_ns = timing.feature_ns;
    record.queue_ns = timing.queue_ns;
    record.infer_ns = timing.infer_ns;
    record.policy_ns = timing.policy_ns;
    record.risk_ns = timing.risk_ns;
    writer_->append(engine_common::ReplayRecordType::METRIC, timestamp,
                    engine_common::normalized_bytes(record));
}

void DecisionReplayRecorder::record_execution(
    const engine_common::ExecutionEvent& execution,
    uint64_t feature_schema_hash, uint64_t model_version_hash) {
    if (!writer_) return;
    engine_common::NormalizedStrategyExecutionRecord record;
    record.prefix = prefix(sizeof(record), execution.decision_id, execution.timestamp,
                           feature_schema_hash, model_version_hash);
    record.client_order_id = execution.client_order_id;
    record.execution_id = execution.execution_id;
    record.last_quantity = execution.last_quantity;
    record.cumulative_quantity = execution.cumulative_quantity;
    record.last_price = execution.last_price;
    record.status = static_cast<uint8_t>(execution.status);
    record.reject_reason = static_cast<uint8_t>(execution.reject_reason);
    writer_->append(engine_common::ReplayRecordType::EXECUTION, execution.timestamp,
                    engine_common::normalized_bytes(record));
}

}  // namespace qbt::strategy

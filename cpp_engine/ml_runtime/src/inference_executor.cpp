#include "ml_runtime/inference_executor.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace qbt::ml {

InferenceExecutor::InferenceExecutor(std::unique_ptr<IInferenceBackend> backend)
    : backend_(std::move(backend)) {
    if (!backend_) throw std::invalid_argument("inference backend is required");
}

InferenceStatus InferenceExecutor::load(
    const ModelArtifact& artifact, const RuntimeOptions& options) {
    reset();
    descriptor_ = artifact.descriptor;
    options_ = options;
    const uint32_t maximum = options_.max_batch_size != 0
        ? options_.max_batch_size : descriptor_.max_batch_size;
    if (maximum == 0 || options_.deadline_ns < 0) {
        return InferenceStatus::INVALID_INPUT;
    }
    auto status = backend_->load(artifact, options_);
    if (status == InferenceStatus::OK) status = backend_->warmup();
    if (status != InferenceStatus::OK) {
        backend_->reset();
        descriptor_ = {};
        options_ = {};
        metrics_.last_status = status;
        return status;
    }
    loaded_ = true;
    metrics_.last_status = InferenceStatus::OK;
    return InferenceStatus::OK;
}

InferenceStatus InferenceExecutor::infer(
    const engine_common::FeatureBatchView& input,
    engine_common::PredictionBatch& output) noexcept {
    output.asof_timestamp = 0;
    output.model_version_hash = 0;
    output.size = 0;
    ++metrics_.calls;
    if (!loaded_) return fail(InferenceStatus::NOT_LOADED, output);
    if (!input.valid_shape() || output.values.size() < input.batch_size ||
        input.feature_schema_hash != descriptor_.feature_schema_hash ||
        input.lookback != descriptor_.lookback ||
        input.feature_count != descriptor_.feature_count ||
        input.static_feature_count != descriptor_.static_feature_count) {
        return fail(InferenceStatus::INVALID_INPUT, output);
    }
    const uint32_t maximum = options_.max_batch_size != 0
        ? options_.max_batch_size : descriptor_.max_batch_size;
    const auto started = std::chrono::steady_clock::now();
    const size_t dynamic_row =
        static_cast<size_t>(input.lookback) * input.feature_count;
    const size_t static_row = input.static_feature_count;
    for (uint32_t offset = 0; offset < input.batch_size; offset += maximum) {
        const uint32_t count = std::min(maximum, input.batch_size - offset);
        engine_common::FeatureBatchView chunk{
            input.asof_timestamp, input.feature_schema_hash,
            input.symbols.subspan(offset, count),
            input.values.subspan(static_cast<size_t>(offset) * dynamic_row,
                                 static_cast<size_t>(count) * dynamic_row),
            input.valid_mask.subspan(static_cast<size_t>(offset) * input.lookback,
                                     static_cast<size_t>(count) * input.lookback),
            input.static_values.subspan(static_cast<size_t>(offset) * static_row,
                                        static_cast<size_t>(count) * static_row),
            count, input.lookback, input.feature_count, input.static_feature_count};
        engine_common::PredictionBatch chunk_output{
            0, 0, output.values.subspan(offset, count), 0};
        ++metrics_.chunks;
        const auto status = backend_->infer(chunk, chunk_output);
        if (status != InferenceStatus::OK) return fail(status, output);
        if (!chunk_output.valid_shape() || chunk_output.size != count ||
            chunk_output.asof_timestamp != input.asof_timestamp ||
            chunk_output.model_version_hash != descriptor_.model_version_hash) {
            return fail(InferenceStatus::BACKEND_ERROR, output);
        }
        if (options_.deadline_ns > 0 &&
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count() >
                options_.deadline_ns) {
            return fail(InferenceStatus::TIMEOUT, output);
        }
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    metrics_.last_infer_ns = elapsed;
    metrics_.total_infer_ns += elapsed;
    metrics_.last_status = InferenceStatus::OK;
    output.asof_timestamp = input.asof_timestamp;
    output.model_version_hash = descriptor_.model_version_hash;
    output.size = input.batch_size;
    return InferenceStatus::OK;
}

void InferenceExecutor::reset() noexcept {
    backend_->reset();
    descriptor_ = {};
    options_ = {};
    loaded_ = false;
}

const ModelDescriptor& InferenceExecutor::descriptor() const noexcept {
    return descriptor_;
}

InferenceStatus InferenceExecutor::fail(
    InferenceStatus status, engine_common::PredictionBatch& output) noexcept {
    ++metrics_.failures;
    if (status == InferenceStatus::TIMEOUT) ++metrics_.timeouts;
    metrics_.last_status = status;
    output.asof_timestamp = 0;
    output.model_version_hash = 0;
    output.size = 0;
    return status;
}

}  // namespace qbt::ml

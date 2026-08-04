#pragma once

#include <cstdint>
#include <memory>

#include "ml_runtime/inference_backend.h"

namespace qbt::ml {

struct InferenceExecutorMetrics {
    uint64_t calls = 0;
    uint64_t chunks = 0;
    uint64_t failures = 0;
    uint64_t timeouts = 0;
    int64_t total_infer_ns = 0;
    int64_t last_infer_ns = 0;
    InferenceStatus last_status = InferenceStatus::NOT_LOADED;
};

class InferenceExecutor {
public:
    explicit InferenceExecutor(std::unique_ptr<IInferenceBackend> backend);

    InferenceStatus load(const ModelArtifact& artifact,
                         const RuntimeOptions& options);
    InferenceStatus infer(const engine_common::FeatureBatchView& input,
                          engine_common::PredictionBatch& output) noexcept;
    void reset() noexcept;

    [[nodiscard]] const ModelDescriptor& descriptor() const noexcept;
    [[nodiscard]] const InferenceExecutorMetrics& metrics() const noexcept {
        return metrics_;
    }

private:
    InferenceStatus fail(InferenceStatus status,
                         engine_common::PredictionBatch& output) noexcept;

    std::unique_ptr<IInferenceBackend> backend_;
    ModelDescriptor descriptor_;
    RuntimeOptions options_;
    InferenceExecutorMetrics metrics_;
    bool loaded_ = false;
};

}  // namespace qbt::ml

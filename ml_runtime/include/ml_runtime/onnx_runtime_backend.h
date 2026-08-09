#pragma once

#include <memory>

#include "ml_runtime/inference_backend.h"

namespace qbt::ml {

class OnnxRuntimeBackend final : public IInferenceBackend {
public:
    OnnxRuntimeBackend();
    ~OnnxRuntimeBackend() override;
    OnnxRuntimeBackend(const OnnxRuntimeBackend&) = delete;
    OnnxRuntimeBackend& operator=(const OnnxRuntimeBackend&) = delete;

    InferenceStatus load(const ModelArtifact& artifact,
                         const RuntimeOptions& options) override;
    InferenceStatus warmup() override;
    InferenceStatus infer(
        const engine_common::FeatureBatchView& input,
        engine_common::PredictionBatch& output) noexcept override;
    void reset() noexcept override;
    const ModelDescriptor& descriptor() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qbt::ml

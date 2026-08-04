#pragma once

#include "ml_runtime/inference_backend.h"

namespace qbt::ml {

class MockInferenceBackend final : public IInferenceBackend {
public:
    InferenceStatus load(const ModelArtifact& artifact,
                         const RuntimeOptions& options) override;
    InferenceStatus warmup() override;
    InferenceStatus infer(
        const engine_common::FeatureBatchView& input,
        engine_common::PredictionBatch& output) noexcept override;
    void reset() noexcept override;
    const ModelDescriptor& descriptor() const noexcept override { return descriptor_; }

private:
    ModelDescriptor descriptor_;
    RuntimeOptions options_;
    bool loaded_ = false;
};

}  // namespace qbt::ml

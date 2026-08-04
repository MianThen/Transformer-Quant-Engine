#pragma once

#include <cstdint>

#include "engine_common/model_types.h"
#include "ml_runtime/model_artifact.h"

namespace qbt::ml {

enum class InferenceStatus : uint8_t {
    OK,
    NOT_LOADED,
    SCHEMA_MISMATCH,
    INVALID_INPUT,
    OUTPUT_OVERFLOW,
    TIMEOUT,
    BACKEND_ERROR,
    NON_FINITE_OUTPUT,
};

struct RuntimeOptions {
    uint32_t intra_op_threads = 1;
    uint32_t inter_op_threads = 1;
    uint32_t max_batch_size = 0;
    int64_t deadline_ns = 0;
    bool enable_arena = true;
};

class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;
    virtual InferenceStatus load(const ModelArtifact& artifact,
                                 const RuntimeOptions& options) = 0;
    virtual InferenceStatus warmup() = 0;
    virtual InferenceStatus infer(
        const engine_common::FeatureBatchView& input,
        engine_common::PredictionBatch& output) noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual const ModelDescriptor& descriptor() const noexcept = 0;
};

}  // namespace qbt::ml

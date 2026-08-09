#include "ml_runtime/onnx_runtime_backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace qbt::ml {
namespace {

constexpr std::array<const char*, 6> kOutputNames{
    "expected_return", "expected_volatility", "direction_probability",
    "lower_quantile", "upper_quantile", "confidence"};

bool valid_descriptor(const ModelDescriptor& value) {
    return value.feature_schema_hash != 0 && value.model_version_hash != 0 &&
        value.lookback > 0 && value.feature_count > 0;
}

}  // namespace

struct OnnxRuntimeBackend::Impl {
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "qbt_ml"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    ModelDescriptor descriptor;
    RuntimeOptions options;
    std::vector<std::string> input_name_storage;
    std::vector<std::string> output_name_storage;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
};

OnnxRuntimeBackend::OnnxRuntimeBackend() : impl_(std::make_unique<Impl>()) {}
OnnxRuntimeBackend::~OnnxRuntimeBackend() = default;

InferenceStatus OnnxRuntimeBackend::load(
    const ModelArtifact& artifact, const RuntimeOptions& options) {
    reset();
    if (!valid_descriptor(artifact.descriptor) || artifact.model_path.empty() ||
        !std::filesystem::is_regular_file(artifact.model_path)) {
        return InferenceStatus::INVALID_INPUT;
    }
    try {
        impl_->options = options;
        impl_->descriptor = artifact.descriptor;
        impl_->session_options.SetIntraOpNumThreads(
            static_cast<int>(std::max<uint32_t>(options.intra_op_threads, 1)));
        impl_->session_options.SetInterOpNumThreads(
            static_cast<int>(std::max<uint32_t>(options.inter_op_threads, 1)));
        impl_->session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (!options.enable_arena) impl_->session_options.DisableCpuMemArena();
        impl_->session = std::make_unique<Ort::Session>(
            impl_->environment, artifact.model_path.c_str(), impl_->session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t index = 0; index < impl_->session->GetInputCount(); ++index) {
            auto name = impl_->session->GetInputNameAllocated(index, allocator);
            impl_->input_name_storage.emplace_back(name.get());
        }
        for (size_t index = 0; index < impl_->session->GetOutputCount(); ++index) {
            auto name = impl_->session->GetOutputNameAllocated(index, allocator);
            impl_->output_name_storage.emplace_back(name.get());
        }
        const std::unordered_set<std::string> actual_inputs(
            impl_->input_name_storage.begin(), impl_->input_name_storage.end());
        const size_t expected_inputs = artifact.descriptor.static_feature_count == 0 ? 2 : 3;
        if (actual_inputs.size() != expected_inputs || actual_inputs.count("features") == 0 ||
            actual_inputs.count("valid_mask") == 0 ||
            (artifact.descriptor.static_feature_count != 0 &&
             actual_inputs.count("static_features") == 0)) {
            reset();
            return InferenceStatus::SCHEMA_MISMATCH;
        }
        const std::unordered_set<std::string> actual_outputs(
            impl_->output_name_storage.begin(), impl_->output_name_storage.end());
        if (actual_outputs.size() != kOutputNames.size()) {
            reset();
            return InferenceStatus::SCHEMA_MISMATCH;
        }
        for (const char* name : kOutputNames) {
            if (actual_outputs.count(name) == 0) {
                reset();
                return InferenceStatus::SCHEMA_MISMATCH;
            }
        }
        impl_->input_names = {"features", "valid_mask"};
        if (artifact.descriptor.static_feature_count != 0) {
            impl_->input_names.push_back("static_features");
        }
        impl_->output_names.assign(kOutputNames.begin(), kOutputNames.end());
        return InferenceStatus::OK;
    } catch (const Ort::Exception&) {
        reset();
        return InferenceStatus::BACKEND_ERROR;
    }
}

InferenceStatus OnnxRuntimeBackend::warmup() {
    if (!impl_->session) return InferenceStatus::NOT_LOADED;
    const auto& descriptor = impl_->descriptor;
    std::vector<engine_common::SymbolId> symbols(1, 0);
    std::vector<float> values(
        static_cast<size_t>(descriptor.lookback) * descriptor.feature_count, 0.0F);
    std::vector<uint8_t> mask(descriptor.lookback, 1);
    std::vector<float> static_values(descriptor.static_feature_count, 0.0F);
    engine_common::FeatureBatchView input{
        1, descriptor.feature_schema_hash, symbols, values, mask, static_values,
        1, descriptor.lookback, descriptor.feature_count,
        descriptor.static_feature_count};
    std::vector<engine_common::ModelPrediction> predictions(1);
    engine_common::PredictionBatch output{0, 0, predictions, 0};
    return infer(input, output);
}

InferenceStatus OnnxRuntimeBackend::infer(
    const engine_common::FeatureBatchView& input,
    engine_common::PredictionBatch& output) noexcept {
    if (!impl_->session) return InferenceStatus::NOT_LOADED;
    const auto& descriptor = impl_->descriptor;
    if (!input.valid_shape()) return InferenceStatus::INVALID_INPUT;
    if (input.feature_schema_hash != descriptor.feature_schema_hash ||
        input.lookback != descriptor.lookback ||
        input.feature_count != descriptor.feature_count ||
        input.static_feature_count != descriptor.static_feature_count) {
        return InferenceStatus::SCHEMA_MISMATCH;
    }
    const uint32_t maximum = impl_->options.max_batch_size != 0
        ? impl_->options.max_batch_size : descriptor.max_batch_size;
    if ((maximum != 0 && input.batch_size > maximum) ||
        output.values.size() < input.batch_size) {
        return InferenceStatus::OUTPUT_OVERFLOW;
    }
    try {
        const auto started = std::chrono::steady_clock::now();
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 3> feature_shape{
            input.batch_size, input.lookback, input.feature_count};
        const std::array<int64_t, 2> mask_shape{input.batch_size, input.lookback};
        std::vector<Ort::Value> tensors;
        tensors.reserve(3);
        tensors.push_back(Ort::Value::CreateTensor<float>(
            memory, const_cast<float*>(input.values.data()), input.values.size(),
            feature_shape.data(), feature_shape.size()));
        tensors.push_back(Ort::Value::CreateTensor<uint8_t>(
            memory, const_cast<uint8_t*>(input.valid_mask.data()), input.valid_mask.size(),
            mask_shape.data(), mask_shape.size()));
        const std::array<int64_t, 2> static_shape{
            input.batch_size, input.static_feature_count};
        if (input.static_feature_count != 0) {
            tensors.push_back(Ort::Value::CreateTensor<float>(
                memory, const_cast<float*>(input.static_values.data()),
                input.static_values.size(), static_shape.data(), static_shape.size()));
        }
        auto results = impl_->session->Run(
            Ort::RunOptions{nullptr}, impl_->input_names.data(), tensors.data(), tensors.size(),
            impl_->output_names.data(), impl_->output_names.size());
        if (impl_->options.deadline_ns > 0 &&
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count() >
                impl_->options.deadline_ns) {
            return InferenceStatus::TIMEOUT;
        }
        if (results.size() != kOutputNames.size()) return InferenceStatus::BACKEND_ERROR;
        std::array<const float*, 6> columns{};
        for (size_t index = 0; index < results.size(); ++index) {
            if (!results[index].IsTensor()) return InferenceStatus::SCHEMA_MISMATCH;
            const auto info = results[index].GetTensorTypeAndShapeInfo();
            if (info.GetElementCount() != input.batch_size ||
                info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                return InferenceStatus::SCHEMA_MISMATCH;
            }
            columns[index] = results[index].GetTensorData<float>();
        }
        for (uint32_t row = 0; row < input.batch_size; ++row) {
            auto& prediction = output.values[row];
            const size_t mask_offset = static_cast<size_t>(row) * input.lookback;
            const bool any_valid = std::any_of(
                input.valid_mask.begin() + static_cast<std::ptrdiff_t>(mask_offset),
                input.valid_mask.begin() +
                    static_cast<std::ptrdiff_t>(mask_offset + input.lookback),
                [](uint8_t value) { return value != 0; });
            if (!any_valid || input.valid_mask[mask_offset + input.lookback - 1] == 0) {
                prediction = {};
                prediction.symbol_id = input.symbols[row];
                prediction.asof_timestamp = input.asof_timestamp;
                prediction.flags = engine_common::INSUFFICIENT_HISTORY;
                continue;
            }
            prediction = {input.symbols[row], input.asof_timestamp,
                          columns[0][row], columns[1][row], columns[2][row],
                          columns[3][row], columns[4][row], columns[5][row],
                          engine_common::PREDICTION_VALID};
            if (!prediction.finite()) return InferenceStatus::NON_FINITE_OUTPUT;
        }
        output.asof_timestamp = input.asof_timestamp;
        output.model_version_hash = descriptor.model_version_hash;
        output.size = input.batch_size;
        return InferenceStatus::OK;
    } catch (...) {
        return InferenceStatus::BACKEND_ERROR;
    }
}

void OnnxRuntimeBackend::reset() noexcept {
    impl_->session.reset();
    impl_->descriptor = {};
    impl_->options = {};
    impl_->input_name_storage.clear();
    impl_->output_name_storage.clear();
    impl_->input_names.clear();
    impl_->output_names.clear();
    impl_->session_options = Ort::SessionOptions{};
}

const ModelDescriptor& OnnxRuntimeBackend::descriptor() const noexcept {
    return impl_->descriptor;
}

}  // namespace qbt::ml

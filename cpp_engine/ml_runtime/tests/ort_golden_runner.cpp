#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "ml_runtime/onnx_runtime_backend.h"
#include "strategy_runtime/portfolio_policy.h"

namespace {

template <class Value>
std::vector<Value> read_values(const std::filesystem::path& path, size_t count) {
    std::vector<Value> values(count);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(Value)));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("golden file size mismatch: " + path.string());
    }
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "usage: qbt_ort_golden MODEL FEATURES MASK EXPECTED TARGETS N T F\n";
        return 2;
    }
    try {
        const uint32_t batch = static_cast<uint32_t>(std::stoul(argv[6]));
        const uint32_t lookback = static_cast<uint32_t>(std::stoul(argv[7]));
        const uint32_t feature_count = static_cast<uint32_t>(std::stoul(argv[8]));
        if (batch == 0 || lookback == 0 || feature_count == 0) return 2;
        const size_t feature_values = static_cast<size_t>(batch) * lookback * feature_count;
        const size_t mask_values = static_cast<size_t>(batch) * lookback;
        auto features = read_values<float>(argv[2], feature_values);
        auto mask = read_values<uint8_t>(argv[3], mask_values);
        const auto expected = read_values<float>(argv[4], static_cast<size_t>(batch) * 6);
        std::vector<engine_common::SymbolId> symbols(batch);
        for (uint32_t index = 0; index < batch; ++index) symbols[index] = index + 1;

        qbt::ml::ModelArtifact artifact;
        artifact.model_path = argv[1];
        artifact.descriptor.feature_schema_hash = 1;
        artifact.descriptor.model_version_hash = 1;
        artifact.descriptor.lookback = lookback;
        artifact.descriptor.feature_count = feature_count;
        artifact.descriptor.max_batch_size = batch;
        qbt::ml::RuntimeOptions options;
        options.max_batch_size = batch;
        qbt::ml::OnnxRuntimeBackend backend;
        const auto loaded = backend.load(artifact, options);
        if (loaded != qbt::ml::InferenceStatus::OK) {
            std::cerr << "backend load failed: " << static_cast<int>(loaded) << '\n';
            return 3;
        }
        const engine_common::FeatureBatchView input{
            1, 1, symbols, features, mask, {}, batch, lookback, feature_count, 0};
        std::vector<engine_common::ModelPrediction> predictions(batch);
        engine_common::PredictionBatch output{0, 0, predictions, 0};
        const auto status = backend.infer(input, output);
        if (status != qbt::ml::InferenceStatus::OK || output.size != batch) {
            std::cerr << "backend inference failed: " << static_cast<int>(status) << '\n';
            return 4;
        }

        double max_absolute = 0.0;
        double max_relative = 0.0;
        std::vector<std::pair<float, uint32_t>> expected_rank;
        std::vector<std::pair<float, uint32_t>> actual_rank;
        expected_rank.reserve(batch);
        actual_rank.reserve(batch);
        for (uint32_t row = 0; row < batch; ++row) {
            const std::array<float, 6> actual{
                predictions[row].expected_return, predictions[row].expected_volatility,
                predictions[row].direction_probability, predictions[row].lower_quantile,
                predictions[row].upper_quantile, predictions[row].confidence};
            for (size_t column = 0; column < actual.size(); ++column) {
                const double reference = expected[column * batch + row];
                const double absolute = std::abs(static_cast<double>(actual[column]) - reference);
                const double relative = absolute / std::max(std::abs(reference), 1.0e-12);
                max_absolute = std::max(max_absolute, absolute);
                max_relative = std::max(max_relative, relative);
                if (absolute > 1.0e-5 + 1.0e-4 * std::abs(reference)) {
                    std::cerr << "parity mismatch row=" << row << " output=" << column
                              << " expected=" << reference << " actual=" << actual[column]
                              << '\n';
                    return 5;
                }
            }
            expected_rank.emplace_back(expected[row], row + 1);
            actual_rank.emplace_back(predictions[row].expected_return, row + 1);
        }
        const auto rank_order = [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) return lhs.first > rhs.first;
            return lhs.second < rhs.second;
        };
        std::sort(expected_rank.begin(), expected_rank.end(), rank_order);
        std::sort(actual_rank.begin(), actual_rank.end(), rank_order);
        const size_t top_k = std::min<size_t>(3, batch);
        if (batch > top_k &&
            std::abs(expected_rank[top_k - 1].first - expected_rank[top_k].first) <= 1.0e-5F) {
            std::cerr << "golden top-k boundary is a near tie\n";
            return 6;
        }
        for (size_t index = 0; index < top_k; ++index) {
            if (expected_rank[index].second != actual_rank[index].second) {
                std::cerr << "top-k decision mismatch at rank=" << index << '\n';
                return 7;
            }
        }
        std::vector<engine_common::MarketBar> bars(batch);
        for (uint32_t row = 0; row < batch; ++row) {
            bars[row].symbol_id = row + 1;
            bars[row].timestamp = 1;
            bars[row].close = 10.0 + static_cast<double>(row + 1);
            bars[row].lot_size = 100;
            bars[row].flags = engine_common::MARKET_LISTED |
                engine_common::MARKET_DATA_TRUSTED;
        }
        qbt::strategy::LongOnlyPolicyConfig policy_config;
        policy_config.max_positions = 3;
        policy_config.max_position_weight = 0.05F;
        policy_config.minimum_expected_return = -1.0e9F;
        policy_config.minimum_confidence = 0.0F;
        qbt::strategy::LongOnlyTopKPolicy policy(policy_config);
        const auto targets = policy.build(
            output, {1, bars, true}, {{}, 1'000'000.0, 1'000'000.0, 0.0, 0.0});
        std::ifstream target_input(argv[5]);
        size_t expected_target_count = 0;
        target_input >> expected_target_count;
        if (!target_input || expected_target_count != targets.size()) return 8;
        for (size_t index = 0; index < targets.size(); ++index) {
            uint32_t symbol = 0;
            int64_t quantity = 0;
            float weight = 0.0F;
            target_input >> symbol >> quantity >> weight;
            if (!target_input || symbol != targets[index].symbol_id ||
                quantity != targets[index].target_quantity ||
                std::abs(weight - targets[index].target_weight) > 1.0e-7F) {
                std::cerr << "target position mismatch at index=" << index << '\n';
                return 8;
            }
        }
        std::cout << "{\"status\":\"PASS\",\"batch_size\":" << batch
                  << ",\"max_absolute_error\":" << max_absolute
                  << ",\"max_relative_error\":" << max_relative
                  << ",\"top_k\":" << top_k
                  << ",\"top_k_match\":true,\"target_positions_match\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}

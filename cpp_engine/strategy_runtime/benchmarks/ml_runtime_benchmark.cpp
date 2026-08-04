#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif

#include <onnxruntime_cxx_api.h>

#include "ml_runtime/artifact_loader.h"
#include "ml_runtime/onnx_runtime_backend.h"
#include "strategy_runtime/model_strategy_runtime.h"

#ifndef QBT_BENCHMARK_BUILD_TYPE
#define QBT_BENCHMARK_BUILD_TYPE "unknown"
#endif
#ifndef QBT_BENCHMARK_COMPILER_ID
#define QBT_BENCHMARK_COMPILER_ID "unknown"
#endif
#ifndef QBT_BENCHMARK_COMPILER_VERSION
#define QBT_BENCHMARK_COMPILER_VERSION "unknown"
#endif
#ifndef QBT_BENCHMARK_LTO
#define QBT_BENCHMARK_LTO 0
#endif

namespace {

struct StageSamples {
    std::vector<int64_t> feature;
    std::vector<int64_t> infer;
    std::vector<int64_t> policy;
    std::vector<int64_t> risk;
    std::vector<int64_t> end_to_end;
    uint64_t peak_rss_bytes = 0;
};

uint64_t peak_rss_bytes() {
#if defined(__APPLE__) || defined(__linux__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<uint64_t>(usage.ru_maxrss);
#else
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
#else
    return 0;
#endif
}

int64_t percentile(std::vector<int64_t> values, double quantile) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const size_t index = std::min(
        static_cast<size_t>(std::ceil(quantile * values.size())) - 1,
        values.size() - 1);
    return values[index];
}

std::string argument(int argc, char** argv, const std::string& name,
                     const std::string& fallback = {}) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return argv[index + 1];
    }
    return fallback;
}

StageSamples benchmark(
    const qbt::ml::ModelArtifact& artifact, uint32_t batch_size,
    uint32_t iterations, uint32_t chunk_size) {
    qbt::strategy::ModelStrategyConfig config;
    config.artifact = artifact;
    config.runtime_options.intra_op_threads = 1;
    config.runtime_options.inter_op_threads = 1;
    config.runtime_options.max_batch_size = chunk_size;
    config.policy.max_positions = batch_size;
    config.policy.max_position_weight = std::min(0.01F, 0.8F / batch_size);
    config.policy.minimum_expected_return = -1.0F;
    config.policy.minimum_confidence = 0.0F;
    config.policy.max_turnover_weight = 1.0F;
    config.risk.max_order_quantity = 10'000'000;
    config.risk.max_gross_exposure = 2.0;
    config.risk.max_net_exposure = 2.0;
    config.risk.max_orders_per_batch = batch_size;
    config.max_order_intents = batch_size;
    qbt::strategy::ModelStrategyRuntime runtime(
        std::make_unique<qbt::ml::OnnxRuntimeBackend>(), std::move(config));
    engine_common::StrategySessionContext context;
    context.feature_schema_hash = artifact.descriptor.feature_schema_hash;
    context.model_version_hash = artifact.descriptor.model_version_hash;
    if (runtime.start(context) != engine_common::StrategyStatus::OK) {
        throw std::runtime_error("benchmark model runtime failed to start");
    }

    std::vector<engine_common::MarketBar> bars(batch_size);
    std::vector<engine_common::PortfolioItem> items(batch_size);
    std::vector<engine_common::OrderIntent> intents(batch_size);
    engine_common::PortfolioView portfolio{
        items, 1.0e9, 1.0e9, 0.0, 0.0, 1.0e9};
    StageSamples samples;
    for (auto* values : {&samples.feature, &samples.infer, &samples.policy,
                         &samples.risk, &samples.end_to_end}) {
        values->reserve(iterations);
    }
    const uint32_t total_frames = 61 + iterations;
    for (uint32_t frame = 1; frame <= total_frames; ++frame) {
        for (uint32_t symbol = 0; symbol < batch_size; ++symbol) {
            const double close = 10.0 + symbol * 0.001 + frame * 0.01;
            auto& bar = bars[symbol];
            bar = {};
            bar.symbol_id = symbol;
            bar.timestamp = frame;
            bar.open = close * 0.999;
            bar.high = close * 1.01;
            bar.low = close * 0.99;
            bar.close = close;
            bar.volume = 100'000 + frame + symbol;
            bar.lot_size = 1;
            bar.flags = engine_common::MARKET_LISTED |
                        engine_common::MARKET_DATA_TRUSTED;
            items[symbol].symbol_id = symbol;
            items[symbol].mark_price = close;
        }
        engine_common::MarketFrameBatchView market{frame, bars, true};
        engine_common::OrderIntentBuffer output{intents, 0};
        const auto before = runtime.metrics();
        const auto started = std::chrono::steady_clock::now();
        const auto status = runtime.on_market_batch(market, portfolio, output);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started).count();
        if (status != engine_common::StrategyStatus::OK) {
            throw std::runtime_error("benchmark inference failed");
        }
        if (frame > 61) {
            const auto& after = runtime.metrics();
            samples.feature.push_back(after.total_feature_ns - before.total_feature_ns);
            samples.infer.push_back(after.total_infer_ns - before.total_infer_ns);
            samples.policy.push_back(after.total_policy_ns - before.total_policy_ns);
            samples.risk.push_back(after.total_risk_ns - before.total_risk_ns);
            samples.end_to_end.push_back(elapsed);
        }
    }
    runtime.stop();
    samples.peak_rss_bytes = peak_rss_bytes();
    return samples;
}

void write_stage(std::ostream& output, const char* name,
                 const std::vector<int64_t>& values) {
    output << '\"' << name << "\":{\"p50_ns\":" << percentile(values, 0.50)
           << ",\"p99_ns\":" << percentile(values, 0.99) << '}';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto artifact_path = argument(argc, argv, "--artifact");
        const auto output_path = argument(argc, argv, "--output");
        const auto hardware = argument(argc, argv, "--hardware", "unknown");
        const auto revision = argument(argc, argv, "--revision", "unknown");
        const auto iterations = static_cast<uint32_t>(
            std::stoul(argument(argc, argv, "--iterations", "20")));
        const auto chunk_size = static_cast<uint32_t>(
            std::stoul(argument(argc, argv, "--chunk-size", "512")));
        if (artifact_path.empty() || output_path.empty() || iterations == 0 ||
            chunk_size == 0) {
            throw std::invalid_argument(
                "usage: qbt_ml_benchmark --artifact PATH --output PATH "
                "[--iterations N] [--chunk-size N] [--hardware TEXT] "
                "[--revision SHA]");
        }
        const auto loaded = qbt::ml::ArtifactLoader{}.load(artifact_path, chunk_size);
        if (!loaded) throw std::runtime_error(loaded.message);
        const std::array<uint32_t, 4> batches{1, 64, 512, 4096};
        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("cannot open benchmark output");
        output << "{\n  \"schema_version\":1,\n  \"hardware\":\"" << hardware
               << "\",\n  \"revision\":\"" << revision
               << "\",\n  \"execution_provider\":\"CPUExecutionProvider\","
                  "\n  \"onnxruntime_version\":\"" << Ort::GetVersionString()
               << "\",\n  \"build_type\":\"" << QBT_BENCHMARK_BUILD_TYPE
               << "\",\n  \"compiler_id\":\"" << QBT_BENCHMARK_COMPILER_ID
               << "\",\n  \"compiler_version\":\"" << QBT_BENCHMARK_COMPILER_VERSION
               << "\",\n  \"lto_enabled\":"
               << (QBT_BENCHMARK_LTO ? "true" : "false")
               << ",\n  \"intra_op_threads\":1,\n  \"inter_op_threads\":1,"
                  "\n  \"chunk_size\":" << chunk_size
               << ",\n  \"iterations\":" << iterations << ",\n  \"results\":[\n";
        for (size_t index = 0; index < batches.size(); ++index) {
            const auto samples = benchmark(
                loaded.artifact, batches[index], iterations, chunk_size);
            output << "    {\"batch_size\":" << batches[index]
                   << ",\"peak_rss_bytes\":" << samples.peak_rss_bytes << ',';
            write_stage(output, "feature", samples.feature); output << ',';
            write_stage(output, "infer", samples.infer); output << ',';
            write_stage(output, "policy", samples.policy); output << ',';
            write_stage(output, "risk", samples.risk); output << ',';
            write_stage(output, "end_to_end", samples.end_to_end);
            output << '}' << (index + 1 == batches.size() ? "\n" : ",\n");
        }
        output << "  ]\n}\n";
        std::cout << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

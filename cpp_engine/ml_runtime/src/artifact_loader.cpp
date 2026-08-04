#include "ml_runtime/artifact_loader.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "ml_runtime/bar_v1_feature_pipeline.h"

namespace qbt::ml {
namespace {

constexpr std::array<uint32_t, 64> kRound{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

class Sha256 {
public:
    void update(const uint8_t* data, size_t size) {
        total_ += size;
        while (size != 0) {
            const size_t count = std::min(size, block_.size() - used_);
            std::copy_n(data, count, block_.data() + used_);
            data += count;
            size -= count;
            used_ += count;
            if (used_ == block_.size()) {
                transform(block_.data());
                used_ = 0;
            }
        }
    }

    std::string finish() {
        const uint64_t bits = static_cast<uint64_t>(total_) * 8;
        block_[used_++] = 0x80;
        if (used_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(used_), block_.end(), 0);
            transform(block_.data());
            used_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(used_), block_.begin() + 56, 0);
        for (size_t index = 0; index < 8; ++index) {
            block_[63 - index] = static_cast<uint8_t>(bits >> (index * 8));
        }
        transform(block_.data());
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (uint32_t value : state_) output << std::setw(8) << value;
        return output.str();
    }

private:
    void transform(const uint8_t* data) {
        std::array<uint32_t, 64> words{};
        for (size_t index = 0; index < 16; ++index) {
            words[index] = (static_cast<uint32_t>(data[index * 4]) << 24) |
                (static_cast<uint32_t>(data[index * 4 + 1]) << 16) |
                (static_cast<uint32_t>(data[index * 4 + 2]) << 8) |
                data[index * 4 + 3];
        }
        for (size_t index = 16; index < words.size(); ++index) {
            const uint32_t s0 = std::rotr(words[index - 15], 7) ^
                std::rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const uint32_t s1 = std::rotr(words[index - 2], 17) ^
                std::rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        auto [a, b, c, d, e, f, g, h] = state_;
        for (size_t index = 0; index < words.size(); ++index) {
            const uint32_t s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + choice + kRound[index] + words[index];
            const uint32_t s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + majority;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        const std::array<uint32_t, 8> work{a, b, c, d, e, f, g, h};
        for (size_t index = 0; index < state_.size(); ++index) state_[index] += work[index];
    }

    std::array<uint32_t, 8> state_{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<uint8_t, 64> block_{};
    size_t used_ = 0;
    size_t total_ = 0;
};

nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    return nlohmann::json::parse(input);
}

ArtifactLoadResult failure(ArtifactLoadStatus status, std::string message) {
    return {status, {}, std::move(message)};
}

bool is_sha256(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

bool is_supported_frequency(const std::string& value) {
    if (value == "1d") return true;
    if (value.size() < 2 || value.back() != 'm' || value.front() == '0') return false;
    return std::all_of(value.begin(), value.end() - 1, [](char c) {
        return c >= '0' && c <= '9';
    });
}

uint64_t hash64(const std::string& value) {
    uint64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + 16, result, 16);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + 16) {
        throw std::invalid_argument("invalid SHA-256 prefix");
    }
    return result;
}

bool valid_outputs(const nlohmann::json& outputs, bool require_v2_semantics,
                   uint32_t& horizon) {
    constexpr std::array<std::string_view, 6> names{
        "expected_return", "expected_volatility", "direction_probability",
        "lower_quantile", "upper_quantile", "confidence"};
    constexpr std::array<std::string_view, 6> units{
        "log_return", "return_std", "probability", "log_return", "log_return",
        "probability"};
    if (!outputs.is_array() || outputs.size() != names.size()) return false;
    horizon = 0;
    for (size_t index = 0; index < names.size(); ++index) {
        const auto& output = outputs[index];
        if (!output.is_object() || output.at("name") != names[index]) return false;
        if (!require_v2_semantics) continue;
        const uint32_t output_horizon = output.at("horizon_bars").get<uint32_t>();
        if (output_horizon == 0 || (horizon != 0 && horizon != output_horizon) ||
            output.at("unit") != units[index]) {
            return false;
        }
        horizon = output_horizon;
    }
    if (require_v2_semantics &&
        (outputs[3].at("quantile").get<double>() != 0.10 ||
         outputs[4].at("quantile").get<double>() != 0.90)) {
        return false;
    }
    return true;
}

}  // namespace

std::string sha256_text(const std::string& value) {
    Sha256 digest;
    digest.update(reinterpret_cast<const uint8_t*>(value.data()), value.size());
    return digest.finish();
}

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    Sha256 digest;
    std::array<uint8_t, 1024 * 1024> buffer{};
    while (input) {
        input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        const auto count = input.gcount();
        if (count > 0) digest.update(buffer.data(), static_cast<size_t>(count));
    }
    if (!input.eof()) throw std::runtime_error("failed reading " + path.string());
    return digest.finish();
}

ArtifactLoadResult ArtifactLoader::load(
    const std::filesystem::path& source, uint32_t max_batch_size) const noexcept {
    try {
        const auto root = std::filesystem::absolute(source);
        const auto manifest_path = root / "manifest.json";
        const auto model_path = root / "model.onnx";
        const auto schema_path = root / "feature_schema.json";
        for (const auto& path : {manifest_path, model_path, schema_path, root / "metrics.json"}) {
            if (!std::filesystem::is_regular_file(path)) {
                return failure(ArtifactLoadStatus::MISSING_FILE, "missing " + path.filename().string());
            }
        }
        const auto manifest = read_json(manifest_path);
        const int version = manifest.at("schema_version").get<int>();
        const auto frequency = version == 2
            ? manifest.at("frequency").get<std::string>()
            : manifest.value("frequency", "1d");
        const auto calendar_id = manifest.at("calendar_id").get<std::string>();
        if ((version != 1 && version != 2) ||
            calendar_id.empty() || !is_supported_frequency(frequency) ||
            manifest.at("feature_profile") != "BAR_V1" ||
            manifest.at("execution_alignment") != "NEXT_OPEN" ||
            manifest.at("input_dtype") != "float32" || manifest.at("input_layout") != "NTF" ||
            manifest.at("onnx_opset").get<uint32_t>() < 17 ||
            manifest.at("minimum_runtime_version") != "1.17" ||
            manifest.at("preferred_provider") != "CPUExecutionProvider") {
            return failure(ArtifactLoadStatus::UNSUPPORTED_PROTOCOL, "unsupported manifest protocol");
        }
        const std::string model_hash = manifest.at("model_sha256");
        const std::string schema_hash = manifest.at("feature_schema_sha256");
        if (!is_sha256(model_hash) || !is_sha256(schema_hash)) {
            return failure(ArtifactLoadStatus::INVALID_VALUE, "manifest SHA-256 format invalid");
        }
        if (sha256_file(model_path) != model_hash || sha256_file(schema_path) != schema_hash) {
            return failure(ArtifactLoadStatus::HASH_MISMATCH, "model or feature schema hash mismatch");
        }
        uint32_t output_horizon = 0;
        if (!valid_outputs(manifest.at("outputs"), version == 2, output_horizon)) {
            return failure(ArtifactLoadStatus::UNSUPPORTED_PROTOCOL,
                           "unsupported manifest output protocol");
        }
        if (version == 2) {
            if (!manifest.at("dynamic_batch").get<bool>() ||
                manifest.at("model_family") != "TemporalTransformer" ||
                manifest.at("architecture_version") != "V1.1" ||
                manifest.at("normalization_method") != "mean_std" ||
                manifest.at("calibration_method") != "platt_validation_only") {
                return failure(ArtifactLoadStatus::UNSUPPORTED_PROTOCOL, "unsupported manifest V2 protocol");
            }
            const std::array<std::pair<const char*, const char*>, 4> lineage{{
                {"label_spec.json", "label_spec_sha256"},
                {"normalization.json", "normalization_sha256"},
                {"calibration.json", "calibration_sha256"},
                {"leakage_report.json", "leakage_report_sha256"}}};
            for (const auto& [filename, field] : lineage) {
                const auto path = root / filename;
                if (!std::filesystem::is_regular_file(path))
                    return failure(ArtifactLoadStatus::MISSING_FILE, "missing " + std::string(filename));
                const auto expected_hash = manifest.at(field).get<std::string>();
                if (!is_sha256(expected_hash))
                    return failure(ArtifactLoadStatus::INVALID_VALUE,
                                   std::string(field) + " format invalid");
                if (sha256_file(path) != expected_hash)
                    return failure(ArtifactLoadStatus::HASH_MISMATCH, std::string(filename) + " hash mismatch");
            }
            const auto fingerprint = manifest.at("training_dataset_fingerprint").get<std::string>();
            if (!is_sha256(fingerprint)) {
                return failure(ArtifactLoadStatus::INVALID_VALUE,
                               "training dataset fingerprint format invalid");
            }
            const auto label_spec = read_json(root / "label_spec.json");
            const auto normalization = read_json(root / "normalization.json");
            const auto calibration = read_json(root / "calibration.json");
            const auto leakage = read_json(root / "leakage_report.json");
            if (normalization.at("method") != manifest.at("normalization_method") ||
                calibration.at("method") != manifest.at("calibration_method") ||
                label_spec.at("horizon_bars").get<uint32_t>() != output_horizon ||
                leakage.at("status") != "PASS" ||
                leakage.at("dataset_fingerprint") != fingerprint) {
                return failure(ArtifactLoadStatus::SCHEMA_MISMATCH,
                               "manifest V2 lineage semantics mismatch");
            }
        }
        const auto schema = read_json(schema_path);
        if (schema.at("schema_version") != 1 || schema.at("profile") != "BAR_V1" ||
            schema.at("layout") != "NTF" || schema.at("value_dtype") != "float32") {
            return failure(ArtifactLoadStatus::SCHEMA_MISMATCH, "unsupported feature schema");
        }
        const auto names = schema.at("feature_names").get<std::vector<std::string>>();
        if (names.size() != kBarV1FeatureNames.size())
            return failure(ArtifactLoadStatus::SCHEMA_MISMATCH, "BAR_V1 feature count mismatch");
        for (size_t index = 0; index < names.size(); ++index) {
            if (names[index] != kBarV1FeatureNames[index])
                return failure(ArtifactLoadStatus::SCHEMA_MISMATCH, "BAR_V1 feature order mismatch");
        }
        ModelArtifact artifact;
        artifact.root = root;
        artifact.model_path = model_path;
        auto& descriptor = artifact.descriptor;
        descriptor.model_id = manifest.at("model_id");
        descriptor.model_version = manifest.at("model_version");
        descriptor.frequency = frequency;
        descriptor.calendar_id = calendar_id;
        descriptor.feature_schema_hash = hash64(schema_hash);
        descriptor.model_version_hash = hash64(sha256_text(descriptor.model_version));
        descriptor.lookback = manifest.at("lookback");
        descriptor.feature_count = manifest.at("feature_count");
        descriptor.static_feature_count = manifest.at("static_feature_count");
        descriptor.max_batch_size = max_batch_size;
        descriptor.minimum_valid_tokens = version == 2 ? manifest.at("minimum_valid_tokens").get<uint32_t>() : 1;
        descriptor.manifest_version = static_cast<uint16_t>(version);
        descriptor.dynamic_batch = version == 2 && manifest.at("dynamic_batch").get<bool>();
        if (descriptor.lookback == 0 || descriptor.feature_count != kBarV1FeatureCount ||
            descriptor.static_feature_count != 0 || descriptor.max_batch_size == 0 ||
            descriptor.minimum_valid_tokens == 0 ||
            descriptor.minimum_valid_tokens > descriptor.lookback) {
            return failure(ArtifactLoadStatus::INVALID_VALUE, "artifact shape or limits invalid");
        }
        return {ArtifactLoadStatus::OK, std::move(artifact), {}};
    } catch (const nlohmann::json::exception& error) {
        return failure(ArtifactLoadStatus::INVALID_JSON, error.what());
    } catch (const std::exception& error) {
        return failure(ArtifactLoadStatus::IO_ERROR, error.what());
    }
}

}  // namespace qbt::ml

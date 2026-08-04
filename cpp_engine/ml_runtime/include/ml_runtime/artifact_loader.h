#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "ml_runtime/model_artifact.h"

namespace qbt::ml {

enum class ArtifactLoadStatus : uint8_t {
    OK,
    MISSING_FILE,
    INVALID_JSON,
    UNSUPPORTED_PROTOCOL,
    HASH_MISMATCH,
    SCHEMA_MISMATCH,
    INVALID_VALUE,
    IO_ERROR,
};

struct ArtifactLoadResult {
    ArtifactLoadStatus status = ArtifactLoadStatus::INVALID_VALUE;
    ModelArtifact artifact;
    std::string message;

    [[nodiscard]] explicit operator bool() const noexcept {
        return status == ArtifactLoadStatus::OK;
    }
};

class ArtifactLoader {
public:
    [[nodiscard]] ArtifactLoadResult load(
        const std::filesystem::path& root, uint32_t max_batch_size) const noexcept;
};

[[nodiscard]] std::string sha256_text(const std::string& value);
[[nodiscard]] std::string sha256_file(const std::filesystem::path& path);

}  // namespace qbt::ml

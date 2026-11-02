#ifndef MC_GATE_REPLAY_H
#define MC_GATE_REPLAY_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>
#include <filesystem>

#include "mc_gate_sha256.h"

namespace mc_gate {

inline void put_f64_le(std::vector<unsigned char>& out, double v) {
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<unsigned char>((bits >> (8 * i)) & 0xFFu));
    }
}

inline double get_f64_le(const unsigned char* p) {
    uint64_t bits = 0;
    for (int i = 0; i < 8; ++i) {
        bits |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
    double v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

inline bool write_replay_dataset(const char* dir, int dataset_id,
                                 const std::vector<double>& data) {
    const std::filesystem::path dir_path(dir);
    std::error_code ec;
    std::filesystem::create_directories(dir_path, ec);
    if (ec) {
        return false;
    }

    const std::string filename = "dataset_" + std::to_string(dataset_id) + ".bin";
    const std::filesystem::path file_path = dir_path / filename;

    std::vector<unsigned char> payload;
    payload.reserve(data.size() * 8);
    for (double v : data) {
        put_f64_le(payload, v);
    }

    const std::string hex = sha256_hex(payload.data(), payload.size());
    std::vector<unsigned char> digest(32);
    for (int i = 0; i < 32; ++i) {
        const unsigned int byte_val = std::stoul(hex.substr(2 * i, 2), nullptr, 16);
        digest[i] = static_cast<unsigned char>(byte_val);
    }

    std::FILE* fp = std::fopen(file_path.string().c_str(), "wb");
    if (fp == nullptr) {
        return false;
    }

    bool ok = true;
    if (!payload.empty()) {
        const std::size_t want = payload.size();
        const std::size_t got = std::fwrite(payload.data(), 1, want, fp);
        if (got != want) {
            ok = false;
        }
    }
    if (ok) {
        const std::size_t want = digest.size();
        const std::size_t got = std::fwrite(digest.data(), 1, want, fp);
        if (got != want) {
            ok = false;
        }
    }

    if (std::fclose(fp) != 0) {
        ok = false;
    }
    return ok;
}

inline bool verify_replay_dataset(const char* path, std::vector<double>& out) {
    std::FILE* fp = std::fopen(path, "rb");
    if (fp == nullptr) {
        return false;
    }

    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::fclose(fp);
        return false;
    }
    const long file_len = std::ftell(fp);
    if (file_len < 0) {
        std::fclose(fp);
        return false;
    }
    if (std::fseek(fp, 0, SEEK_SET) != 0) {
        std::fclose(fp);
        return false;
    }

    const std::size_t len = static_cast<std::size_t>(file_len);
    if (len < 32 || (len - 32) % 8 != 0) {
        std::fclose(fp);
        return false;
    }

    std::vector<unsigned char> buf(len);
    const std::size_t want = len;
    const std::size_t got = std::fread(buf.data(), 1, want, fp);
    std::fclose(fp);
    if (got != want) {
        return false;
    }

    const std::size_t data_len = len - 32;
    const std::string hex = sha256_hex(buf.data(), data_len);
    std::vector<unsigned char> digest(32);
    for (int i = 0; i < 32; ++i) {
        const unsigned int byte_val = std::stoul(hex.substr(2 * i, 2), nullptr, 16);
        digest[i] = static_cast<unsigned char>(byte_val);
    }

    const unsigned char* file_digest = buf.data() + data_len;
    if (std::memcmp(digest.data(), file_digest, 32) != 0) {
        return false;
    }

    out.clear();
    out.reserve(data_len / 8);
    for (std::size_t i = 0; i < data_len; i += 8) {
        out.push_back(get_f64_le(buf.data() + i));
    }
    return true;
}

} // namespace mc_gate

#ifdef MC_GATE_REPLAY_SELFTEST

#include <cstdio>

int main() {
    using namespace mc_gate;

    const std::string dir = "/tmp/qgate/C15/replay";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    const std::vector<std::vector<double>> datasets = {
        {1.0, -2.5, 3.25, 1e-8, -0.0, 1e308, -1e-308, 3.141592653589793},
        {0.0, 1.0},
        {-1.0, -1.0, -1.0, -1.0, -1.0}
    };

    for (int id = 0; id < 3; ++id) {
        write_replay_dataset(dir.c_str(), id, datasets[id]);
    }

    int file_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("dataset_", 0) == 0 && name.size() > 4 &&
            name.compare(name.size() - 4, 4, ".bin") == 0) {
            ++file_count;
        }
    }
    std::printf("REPLAY files %d\n", file_count);

    for (int id = 0; id < 3; ++id) {
        const std::string path = dir + "/dataset_" + std::to_string(id) + ".bin";
        std::FILE* fp = std::fopen(path.c_str(), "rb");
        if (fp == nullptr) {
            continue;
        }
        if (std::fseek(fp, 0, SEEK_END) != 0) {
            std::fclose(fp);
            continue;
        }
        const long file_len = std::ftell(fp);
        if (file_len < 0) {
            std::fclose(fp);
            continue;
        }
        if (std::fseek(fp, 0, SEEK_SET) != 0) {
            std::fclose(fp);
            continue;
        }
        const std::size_t len = static_cast<std::size_t>(file_len);
        std::vector<unsigned char> buf(len);
        const std::size_t want = len;
        const std::size_t got = std::fread(buf.data(), 1, want, fp);
        std::fclose(fp);
        if (got != want) {
            continue;
        }
        const std::size_t data_len = len - 32;
        const std::string hex = sha256_hex(buf.data(), data_len);
        std::printf("REPLAY %d sha %s\n", id, hex.c_str());
        std::printf("REPLAY %d bytes %zu\n", id, len);
    }

    int extra_count = 0;
    for (int k = 3; k <= 9; ++k) {
        const std::string path = dir + "/dataset_" + std::to_string(k) + ".bin";
        if (std::filesystem::exists(path)) {
            ++extra_count;
        }
    }
    std::printf("REPLAY extra %d\n", extra_count);

    for (int id = 0; id < 3; ++id) {
        const std::string path = dir + "/dataset_" + std::to_string(id) + ".bin";
        std::vector<double> out;
        const bool ok = verify_replay_dataset(path.c_str(), out);
        std::printf("VERIFY %d %d\n", id, ok ? 1 : 0);
    }

    {
        const std::string path = dir + "/dataset_1.bin";
        std::FILE* fp = std::fopen(path.c_str(), "rb");
        if (fp != nullptr) {
            if (std::fseek(fp, 0, SEEK_END) == 0) {
                const long file_len = std::ftell(fp);
                if (file_len >= 0) {
                    if (std::fseek(fp, 0, SEEK_SET) == 0) {
                        const std::size_t len = static_cast<std::size_t>(file_len);
                        std::vector<unsigned char> buf(len);
                        const std::size_t want = len;
                        const std::size_t got = std::fread(buf.data(), 1, want, fp);
                        if (got == want && len > 15) {
                            buf[15] ^= 0xFFu;
                        }
                        std::fclose(fp);
                        const std::string corrupt_path = dir + "/corrupt.bin";
                        std::FILE* cf = std::fopen(corrupt_path.c_str(), "wb");
                        if (cf != nullptr) {
                            if (!buf.empty()) {
                                const std::size_t cw = buf.size();
                                const std::size_t cg = std::fwrite(buf.data(), 1, cw, cf);
                                (void)cg;
                            }
                            std::fclose(cf);
                        }
                    }
                }
            } else {
                std::fclose(fp);
            }
        }
        const std::string corrupt_path = dir + "/corrupt.bin";
        std::vector<double> out;
        const bool ok = verify_replay_dataset(corrupt_path.c_str(), out);
        std::printf("VERIFY corrupt %d\n", ok ? 1 : 0);
    }

    std::printf("REPLAY_DONE\n");
    return 0;
}

#endif // MC_GATE_REPLAY_SELFTEST

#endif // MC_GATE_REPLAY_H

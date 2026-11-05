// mc_gate_seed.h — header-only, C++20, namespace mc_gate
#ifndef MC_GATE_SEED_H
#define MC_GATE_SEED_H

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace mc_gate {

constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

inline std::uint64_t splitmix64(std::uint64_t x) {
    std::uint64_t z = x + kGolden;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline std::uint64_t fnv1a64(const std::string& s) {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    for (unsigned char b : s) h = (h ^ b) * 0x100000001B3ULL;
    return h;
}

inline std::string dataset_key(const char* composition, double S, int N, int T,
                               double rho, const char* dist, double p, int ds) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "v1|%s|S=%.4f|N=%d|T=%d|rho=%.2f|%s|p=%.1f|ds=%d",
                  composition, S, N, T, rho, dist, p, ds);
    return std::string(buf);
}

inline std::uint64_t seed_of(const std::string& key, int rep, int batch,
                             const char* stream, std::uint64_t run_seed) {
    std::string s = key + "|rep=" + std::to_string(rep)
                  + "|batch=" + std::to_string(batch)
                  + "|stream=" + stream;
    return splitmix64(fnv1a64(s) ^ splitmix64(run_seed));
}

inline std::string hex16(std::uint64_t v) {
    static const char* digits = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[i] = digits[v & 0xF];
        v >>= 4;
    }
    return out;
}

} // namespace mc_gate

#ifdef MC_GATE_SEED_SELFTEST
int main() {
    using namespace mc_gate;

    const std::uint64_t run_seed = 20261027ULL;

    const std::string keyA = dataset_key("null", 0.0, 266, 20, 0.30, "t5", 50.4, 0);
    const std::string keyB = dataset_key("one_good", 1.2, 266, 4, 0.30, "t5", 50.4, 0);
    const std::string keyR = dataset_key("null", 0.0, 266, 20, 0.30, "t5", 50.4, 3);

    std::printf("KEY A %s\n", keyA.c_str());
    std::printf("KEY B %s\n", keyB.c_str());
    std::printf("KEY R %s\n", keyR.c_str());

    std::printf("FNV A %s\n", hex16(fnv1a64(keyA)).c_str());
    std::printf("FNV B %s\n", hex16(fnv1a64(keyB)).c_str());
    std::printf("FNV R %s\n", hex16(fnv1a64(keyR)).c_str());

    std::printf("SEED A 0 0 ret %s\n", hex16(seed_of(keyA, 0, 0, "ret", run_seed)).c_str());
    std::printf("SEED A 0 0 hold %s\n", hex16(seed_of(keyA, 0, 0, "hold", run_seed)).c_str());
    std::printf("SEED A 0 0 assign %s\n", hex16(seed_of(keyA, 0, 0, "assign", run_seed)).c_str());
    std::printf("SEED B 123 0 ret %s\n", hex16(seed_of(keyB, 123, 0, "ret", run_seed)).c_str());
    std::printf("SEED A 7 1 ret %s\n", hex16(seed_of(keyA, 7, 1, "ret", run_seed)).c_str());
    std::printf("SEED R 0 0 ret %s\n", hex16(seed_of(keyR, 0, 0, "ret", run_seed)).c_str());

    std::printf("SEED_TREE_DONE\n");
    return 0;
}
#endif // MC_GATE_SEED_SELFTEST

#endif // MC_GATE_SEED_H

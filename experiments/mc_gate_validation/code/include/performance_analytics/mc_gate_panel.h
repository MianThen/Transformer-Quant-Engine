#ifndef MC_GATE_PANEL_H
#define MC_GATE_PANEL_H

#include "mc_gate_seed.h"
#include "mc_gate_dgp.h"
#include "mc_gate_sha256.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

namespace mc_gate {

inline ReturnMatrix generate_panel(Family fam, Composition comp, int T, int N,
                                   double rho, double S,
                                   const std::string& dataset_key,
                                   int rep, int batch, std::uint64_t run_seed) {
    const std::uint64_t s = seed_of(dataset_key, rep, batch, "ret", run_seed);
    return generate(fam, comp, T, N, rho, S, s);
}

} // namespace mc_gate

#ifdef MC_GATE_PANEL_SELFTEST

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cstdint>

int main() {
    using namespace mc_gate;

    const std::string key = "v1|null|S=0.0000|N=126|T=4|rho=0.30|gauss|p=50.4|ds=0";
    const std::uint64_t run_seed = 20261027ULL;
    const int rep = 0;
    const int batch = 0;
    const int T = 4;
    const int N = 126;
    const double rho = 0.30;
    const double S = 0.0;
    const Family fam = Family::Gauss;
    const Composition comp = Composition::Null;

    // Step 1: print seed
    const std::uint64_t seed = seed_of(key, rep, batch, "ret", run_seed);
    std::printf("SEED %s\n", hex16(seed).c_str());

    // Step 2: generate panel and print first 8 doubles as bit patterns
    ReturnMatrix m = generate_panel(fam, comp, T, N, rho, S, key, rep, batch, run_seed);
    std::printf("HEAD8");
    for (int i = 0; i < 8; ++i) {
        std::uint64_t bits;
        std::memcpy(&bits, &m.r[i], 8);
        std::printf(" %016llx", static_cast<unsigned long long>(bits));
    }
    std::printf("\n");

    // Step 3: SHA-256 of all bytes in m.r
    const std::string hash = sha256_hex(m.r.data(), m.r.size() * sizeof(double));
    std::printf("SHA256 %s\n", hash.c_str());

    std::printf("PANEL_DONE\n");
    return 0;
}

#endif // MC_GATE_PANEL_SELFTEST

#endif // MC_GATE_PANEL_H

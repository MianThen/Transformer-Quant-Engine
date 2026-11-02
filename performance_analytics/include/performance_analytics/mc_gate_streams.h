#ifndef MC_GATE_STREAMS_H
#define MC_GATE_STREAMS_H

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "mc_gate_seed.h"
#include "mc_gate_dgp.h"
#include "mc_gate_sha256.h"

namespace mc_gate {

inline int kappa_of(const std::string& dataset_key, int rep, int batch,
                    std::uint64_t run_seed, int T) {
    std::uint64_t s = seed_of(dataset_key, rep, batch, "assign", run_seed);
    Pcg64 rng(s);
    return (T <= 0) ? 0 : (int)(rng.next() % (std::uint64_t)T);
}

inline ReturnMatrix hold_path(Family fam, int T, int N_hold, double rho,
                              const std::string& dataset_key, int rep, int batch,
                              std::uint64_t run_seed) {
    std::uint64_t s = seed_of(dataset_key, rep, batch, "hold", run_seed);
    return generate(fam, Composition::Null, T, N_hold, rho, 0.0, s);
}

} // namespace mc_gate

#ifdef MC_GATE_STREAMS_SELFTEST

#include <cstdio>

int main() {
    using namespace mc_gate;

    const std::string key = "v1|null|S=0.0000|N=126|T=4|rho=0.30|gauss|p=50.4|ds=0";
    const std::uint64_t run_seed = 20261027ULL;
    const int T = 4;
    const int N_hold = 2000;
    const double rho = 0.30;

    for (int rep = 0; rep < 4; ++rep) {
        int k = kappa_of(key, rep, 0, run_seed, T);
        std::printf("KAPPA %d %d\n", rep, k);
    }

    ReturnMatrix m = hold_path(Family::Gauss, T, N_hold, rho, key, 0, 0, run_seed);

    std::uint64_t bits[8];
    for (int i = 0; i < 8; ++i) {
        std::memcpy(&bits[i], &m.r[i], sizeof(std::uint64_t));
    }
    std::printf("HOLD8 %016llx %016llx %016llx %016llx %016llx %016llx %016llx %016llx\n",
                (unsigned long long)bits[0], (unsigned long long)bits[1],
                (unsigned long long)bits[2], (unsigned long long)bits[3],
                (unsigned long long)bits[4], (unsigned long long)bits[5],
                (unsigned long long)bits[6], (unsigned long long)bits[7]);

    std::string h = sha256_hex(m.r.data(), m.r.size() * sizeof(double));
    std::printf("HOLD_SHA %s\n", h.c_str());

    std::printf("STREAMS_DONE\n");
    return 0;
}

#endif // MC_GATE_STREAMS_SELFTEST

#endif // MC_GATE_STREAMS_H

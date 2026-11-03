#include "mc_gate_seed.h"
#include "mc_gate_dgp.h"
#include "mc_gate_streams.h"
#include "mc_gate_adapter.h"
#include "mc_gate_baselines.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <span>

int main() {
    using namespace mc_gate;

    const int T = 100;
    const int N = 126;
    const double rho = 0.30;
    const double S = 50.0;
    const std::uint64_t run_seed = 20261027ULL;
    const int reps = 20;

    std::string key = dataset_key("one_good", 50.0, 126, 100, 0.30, "t5", 50.4, 0);

    int hits = 0;

    for (int rep = 0; rep < reps; ++rep) {
        int kappa = kappa_of(key, rep, 0, run_seed, T);
        std::uint64_t s = seed_of(key, rep, 0, "ret", run_seed);
        ReturnMatrix m = generate_with_kappa(Family::T5, Composition::OneGood, T, N, rho, S, s, kappa);

        std::vector<double> flat(static_cast<std::size_t>(T) * static_cast<std::size_t>(N));
        for (int i = 0; i < T; ++i) {
            for (int t = 0; t < N; ++t) {
                flat[static_cast<std::size_t>(i) * N + t] = m.at(i, t);
            }
        }

        ResearchWindow w{
            std::span<const double>(flat.data(), flat.size()),
            static_cast<std::size_t>(T),
            static_cast<std::size_t>(N),
            0
        };

        Decision d = decide_from_stats(make_window_stats(w, false), RuleConfig{GateMode::Full, 0.05, 0.05});

        std::printf("CANARY %d %d %lld %d\n", rep, d.selected ? 1 : 0, static_cast<long long>(d.index), kappa);

        if (d.selected && static_cast<long long>(d.index) == kappa) {
            ++hits;
        }
    }

    std::printf("CANARY_OK %d/20\n", hits);
    std::printf("CANARY_DONE\n");

    return 0;
}

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

    constexpr int T = 100;
    constexpr int N = 126;
    constexpr double rho = 0.30;
    constexpr double S = 50.0;
    constexpr int kappa = 7;
    constexpr std::uint64_t run_seed = 20261027ULL;
    constexpr int REPS = 10;

    const std::string key = dataset_key("one_good", 50.0, 126, 100, 0.30, "t5", 50.4, 0);

    int okA = 0;
    int okB = 0;

    // A: AC-17 permutation invariance
    for (int rep = 0; rep < REPS; ++rep) {
        std::uint64_t s = seed_of(key, rep, 0, "ret", run_seed);
        ReturnMatrix m = generate_with_kappa(Family::T5, Composition::OneGood, T, N, rho, S, s, kappa);

        std::span<const double> sp0(m.r.data(), m.r.size());
        ResearchWindow w0{ sp0, (std::size_t)T, (std::size_t)N, 0 };
        Decision d0 = decide_from_stats(make_window_stats(w0, false), RuleConfig{ GateMode::Full, 0.05, 0.05 });

        std::vector<double> rev((std::size_t)T * (std::size_t)N);
        for (int i = 0; i < T; ++i)
            for (int t = 0; t < N; ++t)
                rev[(std::size_t)i * N + t] = m.at(T - 1 - i, t);

        std::span<const double> sp1(rev.data(), rev.size());
        ResearchWindow w1{ sp1, (std::size_t)T, (std::size_t)N, 0 };
        Decision d1 = decide_from_stats(make_window_stats(w1, false), RuleConfig{ GateMode::Full, 0.05, 0.05 });

        long long i0 = (long long)d0.index;
        long long i1 = (long long)d1.index;
        int ok = (d0.selected && d1.selected && i0 == 7 && i1 == (T - 1 - 7)) ? 1 : 0;
        okA += ok;

        std::printf("PERM %d %lld %lld %d\n", rep, i0, i1, ok);
    }

    // B: AC-16 deployment path does not pollute decision data
    for (int rep = 0; rep < REPS; ++rep) {
        std::uint64_t s = seed_of(key, rep, 0, "ret", run_seed);
        ReturnMatrix ma = generate_with_kappa(Family::T5, Composition::OneGood, T, N, rho, S, s, kappa);

        ReturnMatrix hp = hold_path(Family::T5, T, 2000, rho, key, rep, 0, run_seed);
        (void)hp;

        ReturnMatrix mb = generate_with_kappa(Family::T5, Composition::OneGood, T, N, rho, S, s, kappa);

        int same = (ma.r.size() == mb.r.size() && std::memcmp(ma.r.data(), mb.r.data(),
                                                              ma.r.size() * sizeof(double)) == 0) ? 1 : 0;
        okB += same;

        std::printf("HOLD %d %d %zu\n", rep, same, ma.r.size());
    }

    std::printf("AC17_OK %d/10\n", okA);
    std::printf("AC16_OK %d/10\n", okB);
    std::printf("INVAR_DONE\n");

    return 0;
}

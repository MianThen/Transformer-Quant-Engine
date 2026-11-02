#include "mc_gate_seed.h"
#include "mc_gate_dgp.h"
#include "mc_gate_adapter.h"
#include "mc_gate_baselines.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <span>
#include <cmath>
#include <algorithm>

int main() {
    using namespace mc_gate;

    constexpr int T = 100;
    constexpr int N = 126;
    constexpr double rho = 0.30;
    constexpr double S = 1.2;
    constexpr std::uint64_t run_seed = 20261027ULL;
    constexpr int REPS = 2000;
    constexpr double Q = 0.05;

    std::string key = dataset_key("one_good", 1.2, 126, 100, 0.30, "t5", 50.4, 0);

    double thr_sum = 0.0;
    double te_sum = 0.0;
    int nsel = 0;
    double lo_sel = 1e18;
    double hi_nosel = -1e18;

    for (int rep = 0; rep < REPS; ++rep) {
        std::uint64_t s = seed_of(key, rep, 0, "ret", run_seed);
        ReturnMatrix m = generate_with_kappa(Family::T5, Composition::OneGood, T, N, rho, S, s, 0);

        std::span<const double> sp(m.r.data(), m.r.size());
        ResearchWindow w{ sp, static_cast<std::size_t>(T), static_cast<std::size_t>(N), 0 };
        WindowStats st = make_window_stats(w, false);

        double te = gate_teff(static_cast<double>(T), st.m);
        double thr = gate_threshold_sr(te, static_cast<double>(T), static_cast<double>(N), Q);

        double mx = *std::max_element(st.sr.begin(), st.sr.end());

        bool sel = decide_from_stats(st, RuleConfig{ GateMode::Full, Q, Q }).selected;

        thr_sum += thr;
        te_sum += te;

        if (sel) {
            ++nsel;
            if (mx < lo_sel) lo_sel = mx;
        } else {
            if (mx > hi_nosel) hi_nosel = mx;
        }
    }

    double thr_mean = thr_sum / static_cast<double>(REPS);
    double teff_mean = te_sum / static_cast<double>(REPS);

    printf("H4 lo_sel=%.6f hi_nosel=%.6f nsel=%d reps=%d\n", lo_sel, hi_nosel, nsel, REPS);
    printf("THR mean=%.6f teff_mean=%.6f\n", thr_mean, teff_mean);
    printf("GAP lo_gap=%.6f hi_gap=%.6f\n", lo_sel - thr_mean, hi_nosel - thr_mean);
    printf("H4_DONE\n");

    return 0;
}

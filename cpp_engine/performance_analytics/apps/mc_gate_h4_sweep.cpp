// AC-13：ALT 9 个 S 点上的门槛一致性（|threshold_sr − SR_req(m_measured)| ≤ 0.10）
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string>
#include <vector>
#include <span>
#include <algorithm>
#include "mc_gate_seed.h"
#include "mc_gate_dgp.h"
#include "mc_gate_adapter.h"

int main() {
    using namespace mc_gate;
    constexpr int T = 100;
    constexpr int N = 126;
    constexpr double rho = 0.30;
    constexpr std::uint64_t run_seed = 20261027ULL;
    constexpr int REPS = 1200;
    constexpr double Q = 0.05;
    constexpr int MIN_SEL = 10;   // 少于这么多选中就统计不可靠，跳过断言
    const double Ss[9] = {-0.60, -0.30, 0.15, 0.30, 0.45, 0.60, 0.90, 1.20, 1.80};

    int checked = 0, passed = 0, skipped = 0;
    for (int si = 0; si < 9; ++si) {
        const double S = Ss[si];
        std::string key = dataset_key("one_good", S, N, T, rho, "t5", 50.4, 0);
        double thr_sum = 0.0, te_sum = 0.0, lo_sel = 1e18, hi_nosel = -1e18;
        int nsel = 0;
        for (int rep = 0; rep < REPS; ++rep) {
            std::uint64_t s = seed_of(key, rep, 0, "ret", run_seed);
            ReturnMatrix m = generate_with_kappa(Family::T5, Composition::OneGood, T, N, rho, S, s, 0);
            std::span<const double> sp(m.r.data(), m.r.size());
            ResearchWindow w{ sp, static_cast<std::size_t>(T), static_cast<std::size_t>(N), 0 };
            WindowStats st = make_window_stats(w, false);
            const double te = gate_teff(static_cast<double>(T), st.m);
            const double thr = gate_threshold_sr(te, static_cast<double>(T), static_cast<double>(N), Q);
            const double mx = *std::max_element(st.sr.begin(), st.sr.end());
            const bool sel = decide_from_stats(st, RuleConfig{ GateMode::Full, Q, Q }).selected;
            thr_sum += thr; te_sum += te;
            if (sel) { ++nsel; if (mx < lo_sel) lo_sel = mx; }
            else     { if (mx > hi_nosel) hi_nosel = mx; }
        }
        const double thr_mean = thr_sum / REPS;
        const double teff_mean = te_sum / REPS;
        if (nsel < MIN_SEL) {
            ++skipped;
            std::printf("S13 %.2f skip nsel=%d thr=%.6f teff=%.4f\n", S, nsel, thr_mean, teff_mean);
            continue;
        }
        const double lo_gap = lo_sel - thr_mean;
        const double hi_gap = hi_nosel - thr_mean;
        const bool ok = (std::fabs(lo_gap) <= 0.10) && (std::fabs(hi_gap) <= 0.10);
        ++checked;
        if (ok) ++passed;
        std::printf("S13 %.2f lo_gap=%.6f hi_gap=%.6f nsel=%d thr=%.6f teff=%.4f ok=%d\n",
                    S, lo_gap, hi_gap, nsel, thr_mean, teff_mean, ok ? 1 : 0);
    }
    std::printf("AC13 checked=%d passed=%d skipped=%d tol=0.10\n", checked, passed, skipped);
    std::printf("AC13_OK %d\n", (checked >= 6 && passed == checked) ? 1 : 0);
    std::printf("H4SWEEP_DONE\n");
    return (checked >= 6 && passed == checked) ? 0 : 1;
}

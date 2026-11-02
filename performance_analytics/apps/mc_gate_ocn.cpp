#include "mc_gate_dgp.h"
#include "mc_gate_baselines.h"
#include "mc_gate_adapter.h"
#include "mc_gate_onc.h"

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <cmath>

static std::uint64_t sm(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static std::vector<double> corr_of(const mc_gate::ReturnMatrix& m) {
    const int T = m.T;
    const int N = m.N;
    std::vector<double> mu(T, 0.0), sd(T, 0.0);
    for (int i = 0; i < T; i++) {
        double s = 0;
        for (int t = 0; t < N; t++) s += m.at(i, t);
        mu[(std::size_t)i] = s / N;
        double v = 0;
        for (int t = 0; t < N; t++) {
            double d = m.at(i, t) - mu[(std::size_t)i];
            v += d * d;
        }
        sd[(std::size_t)i] = std::sqrt(v / (N - 1));
    }
    std::vector<double> c((std::size_t)T * (std::size_t)T, 0.0);
    for (int a = 0; a < T; a++) {
        c[(std::size_t)a * T + a] = 1.0;
        for (int b = a + 1; b < T; b++) {
            double s = 0;
            for (int t = 0; t < N; t++)
                s += (m.at(a, t) - mu[(std::size_t)a]) * (m.at(b, t) - mu[(std::size_t)b]);
            double den = (N - 1) * sd[(std::size_t)a] * sd[(std::size_t)b];
            double r = den > 0.0 ? s / den : 0.0;
            if (r > 1.0) r = 1.0;
            if (r < -1.0) r = -1.0;
            c[(std::size_t)a * T + b] = r;
            c[(std::size_t)b * T + a] = r;
        }
    }
    return c;
}

int main() {
    using namespace mc_gate;

    const int N = 126;
    const int REPS = 400;
    const double Q = 0.05;
    OncSpec spec;

    const Composition comps[2] = { Composition::Null, Composition::OneGood };
    const double Ss[2] = { 0.0, 1.8 };
    const int Ts[3] = { 4, 20, 100 };
    const double rhos[3] = { 0.0, 0.3, 0.7 };

    for (int ci = 0; ci < 2; ci++) {
        const Composition comp = comps[ci];
        const double S = Ss[ci];
        const int is_one_good = (comp == Composition::OneGood) ? 1 : 0;

        for (int ti = 0; ti < 3; ti++) {
            const int T = Ts[ti];

            for (int ri = 0; ri < 3; ri++) {
                const double rho = rhos[ri];
                const int rho10 = (int)(rho * 10.0 + 0.5);

                int abstain = 0;
                int sel_all = 0;
                int sel_cond = 0;
                int ok = 0;

                for (int rep = 0; rep < REPS; rep++) {
                    std::uint64_t seed = sm(9137ULL ^ ((std::uint64_t)T << 36) ^
                                           ((std::uint64_t)(rho * 10) << 20) ^
                                           (std::uint64_t)rep);
                    ReturnMatrix m = generate(Family::T5, comp, T, N, rho, S, seed);
                    std::vector<double> corr = corr_of(m);
                    OncResult oc = onc_partition(corr, (std::uint32_t)T, spec);
                    if (oc.status != OncStatus::OK) {
                        ++abstain;
                        continue;
                    }
                    ++ok;

                    std::vector<double> flat((std::size_t)T * (std::size_t)N);
                    for (int i = 0; i < T; i++)
                        for (int t = 0; t < N; t++)
                            flat[(std::size_t)i * N + t] = m.at(i, t);

                    ResearchWindow w{ std::span<const double>(flat.data(), flat.size()),
                                      (std::size_t)T, (std::size_t)N, 0 };
                    Decision d = decide_from_stats(make_window_stats(w, false),
                                                   RuleConfig{ GateMode::Full, Q, Q });
                    if (d.selected) {
                        ++sel_all;
                        ++sel_cond;
                    }
                }

                double abstain_rate = (double)abstain / (double)REPS;
                double sel_all_rate = (double)sel_all / (double)REPS;
                double sel_cond_rate = (ok > 0) ? (double)sel_cond / (double)ok : 0.0;

                std::printf("OCNROW %d %d %d %.1f %.4f %.4f\n",
                            T, is_one_good, rho10,
                            abstain_rate, sel_all_rate, sel_cond_rate);
            }
        }
    }

    std::printf("OCN_DONE\n");
    return 0;
}

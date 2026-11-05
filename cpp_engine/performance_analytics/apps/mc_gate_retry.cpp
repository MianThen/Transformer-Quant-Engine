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

int main() {
    using namespace mc_gate;

    constexpr int T = 20;
    constexpr int N = 266;
    constexpr double rho = 0.30;
    constexpr double S = 0.0;
    constexpr std::uint64_t run_seed = 20261027ULL;
    constexpr int REPS = 1000;

    // Step 1: K=1
    int sel1 = 0;
    for (int rep = 0; rep < REPS; ++rep) {
        std::string key = dataset_key("null", S, N, T, rho, "t5", 50.4, 0);
        std::uint64_t s = seed_of(key, rep, 0, "ret", run_seed);
        int kp = 0;
        ReturnMatrix m = generate_with_kappa(Family::T5, Composition::Null, T, N, rho, S, s, kp);
        std::span<const double> sp(m.r.data(), m.r.size());
        ResearchWindow w{sp, static_cast<std::size_t>(T), static_cast<std::size_t>(N), 0};
        WindowStats ws = make_window_stats(w);
        RuleConfig rc{GateMode::Full, 0.05, 0.05};
        Decision d = decide_from_stats(ws, rc);
        if (d.selected) {
            ++sel1;
        }
    }
    std::printf("RET1 fpr=%.4f reps=%d\n", static_cast<double>(sel1) / REPS, REPS);

    // Step 2: K=10
    int sel10 = 0;
    for (int rep = 0; rep < REPS; ++rep) {
        bool any = false;
        for (int ds = 0; ds < 10; ++ds) {
            std::string key = dataset_key("null", S, N, T, rho, "t5", 50.4, ds);
            std::uint64_t s = seed_of(key, rep, 0, "ret", run_seed);
            ReturnMatrix m = generate_with_kappa(Family::T5, Composition::Null, T, N, rho, S, s, 0);
            std::span<const double> sp(m.r.data(), m.r.size());
            ResearchWindow w{sp, static_cast<std::size_t>(T), static_cast<std::size_t>(N), 0};
            WindowStats ws = make_window_stats(w);
            RuleConfig rc{GateMode::Full, 0.05, 0.05};
            Decision d = decide_from_stats(ws, rc);
            if (d.selected) {
                any = true;
                break;
            }
        }
        if (any) {
            ++sel10;
        }
    }
    std::printf("RET10 fpr=%.4f reps=%d\n", static_cast<double>(sel10) / REPS, REPS);

    // Step 3: Predicted upper bound and ratio
    double p1 = static_cast<double>(sel1) / REPS;
    double pred = 1.0 - std::pow(1.0 - p1, 10);
    std::printf("PRED 1-(1-%.4f)^10 = %.4f\n", p1, pred);
    double ratio = (sel1 > 0) ? static_cast<double>(sel10) / sel1 : 0.0;
    std::printf("RATIO %.2f\n", ratio);
    std::printf("RETRY_DONE\n");

    return 0;
}

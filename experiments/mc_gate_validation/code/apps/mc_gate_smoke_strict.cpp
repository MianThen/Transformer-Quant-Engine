// AC-25：smoke（3 单元 × 50 reps + canary）60 s 内 --strict 通过
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <span>
#include <chrono>
#include <algorithm>
#include "mc_gate_seed.h"
#include "mc_gate_dgp.h"
#include "mc_gate_adapter.h"
#include "mc_gate_baselines.h"

int main() {
    using namespace mc_gate;
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();

    constexpr std::uint64_t run_seed = 20261027ULL;
    constexpr int REPS = 50;
    struct Cell { int id; int T; int N; double S; const char* comp; };
    const Cell cells[3] = {
        {7001,   4, 126, 0.0, "null"},
        {7002,  20, 266, 0.6, "one_good"},
        {7003, 100, 126, 0.0, "null"},
    };

    long long records = 0;
    int failures = 0;
    for (const Cell& c : cells) {
        const Composition cmp = (c.S == 0.0) ? Composition::Null : Composition::OneGood;
        const std::string key = dataset_key(c.comp, c.S, c.N, c.T, 0.30, "t5", 50.4, 0);
        for (int rep = 0; rep < REPS; ++rep) {
            const std::uint64_t s = seed_of(key, rep, 0, "ret", run_seed);
            ReturnMatrix m = generate_with_kappa(Family::T5, cmp, c.T, c.N, 0.30, c.S, s, 0);
            std::span<const double> sp(m.r.data(), m.r.size());
            ResearchWindow w{ sp, static_cast<std::size_t>(c.T), static_cast<std::size_t>(c.N), 0 };
            WindowStats st = make_window_stats(w, false);
            // AC-41 维度绑定
            if (st.T != c.T || st.N != c.N || st.sr.size() != static_cast<std::size_t>(c.T)) {
                std::printf("STRICT_FAIL ac41 cell=%d\n", c.id); ++failures;
            }
            const Decision d = decide_from_stats(st, RuleConfig{ GateMode::Full, 0.05, 0.05 });
            const RuleResult r0 = rule_b0(m), r1 = rule_b1(m, 0.05), r2 = rule_b2(m);
            const RuleResult r3 = rule_b3(m), r4 = rule_b4_cscv(m, 6, 0.5);
            (void)d; (void)r0; (void)r1; (void)r2; (void)r3; (void)r4;
            records += 6;
        }
    }
    // canary：S=50, kappa=7，必须选中且下标正确
    const std::string ckey = dataset_key("one_good", 50.0, 126, 100, 0.30, "t5", 50.4, 0);
    ReturnMatrix cm = generate_with_kappa(Family::T5, Composition::OneGood, 100, 126, 0.30, 50.0,
                                          seed_of(ckey, 0, 0, "ret", run_seed), 7);
    std::span<const double> csp(cm.r.data(), cm.r.size());
    ResearchWindow cw{ csp, 100, 126, 0 };
    const Decision cd = decide_from_stats(make_window_stats(cw, false), RuleConfig{ GateMode::Full, 0.05, 0.05 });
    const int canary_ok = (cd.selected && static_cast<long long>(cd.index) == 7) ? 1 : 0;
    if (!canary_ok) ++failures;

    const double elapsed = std::chrono::duration<double>(clk::now() - t0).count();
    const long long want = 3LL * REPS * 6LL;
    if (records != want) { std::printf("STRICT_FAIL records=%lld want=%lld\n", records, want); ++failures; }
    if (elapsed > 60.0) { std::printf("STRICT_FAIL elapsed=%.2f > 60\n", elapsed); ++failures; }

    std::printf("STRICT cells=3 reps=%d records=%lld canary=%d elapsed=%.3f\n", REPS, records, canary_ok, elapsed);
    std::printf("STRICT_OK %d\n", failures == 0 ? 1 : 0);
    std::printf("STRICT_DONE\n");
    return failures == 0 ? 0 : 1;
}

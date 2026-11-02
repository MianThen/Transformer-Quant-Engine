// mc_gate_scorer.h
#ifndef MC_GATE_SCORER_H
#define MC_GATE_SCORER_H

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <utility>
#include <vector>
#include <algorithm>

#include "mc_gate_decide.h"

namespace mc_gate {

struct PropStat  { long long k; long long n; };
struct DeltaStat { double delta; long long b; long long c; long long n; };
struct CellStat  { long long k; long long n; double abstain_rate; bool ocn_constrained; };
struct VerdictInput  { std::vector<CellStat> nul; std::vector<DeltaStat> delta; };
struct VerdictResult { const char* verdict; bool high_abstain; bool ocn_constrained; };

inline std::pair<double,double> wilson_ci(long long k, long long n, double z) {
    if (n <= 0) return {0.0, 0.0};
    double p = (double)k / (double)n;
    double den = 1.0 + z*z/(double)n;
    double center = (p + z*z/(2.0*(double)n)) / den;
    double half = z * std::sqrt( p*(1.0-p)/(double)n + z*z/(4.0*(double)n*(double)n) ) / den;
    double lo = center - half;
    double hi = center + half;
    if (lo < 0.0) lo = 0.0;
    if (hi > 1.0) hi = 1.0;
    return {lo, hi};
}

inline std::pair<double,double> mcnemar_ci(long long b, long long c, long long n, double z) {
    if (n <= 0) return {0.0, 0.0};
    double d = (double)(c - b) / (double)n;
    double var = ( (double)(b + c) - (double)(c-b)*(double)(c-b)/(double)n ) / ((double)n * (double)n);
    if (var < 0.0) var = 0.0;
    double se = std::sqrt(var);
    return {d - z*se, d + z*se};
}

inline bool ci_contains(double lo, double hi, double x) {
    return lo <= x && x <= hi;
}

inline VerdictResult compute_verdict(const VerdictInput& in, double p0, double delta_thr, double z) {
    bool invalid = false;
    bool undecided = false;
    bool dominated = false;
    bool delta_und = false;
    bool high_abstain = false;
    bool ocn_constrained = false;

    for (const auto& c : in.nul) {
        auto ci = wilson_ci(c.k, c.n, z);
        if (ci.first > p0) invalid = true;
        if (ci_contains(ci.first, ci.second, p0)) undecided = true;
        if (c.abstain_rate > 0.01) high_abstain = true;
        if (c.ocn_constrained) ocn_constrained = true;
    }

    for (const auto& d : in.delta) {
        if (d.delta < delta_thr) dominated = true;
        auto ci = mcnemar_ci(d.b, d.c, d.n, z);
        if (ci_contains(ci.first, ci.second, delta_thr)) delta_und = true;
    }

    const char* verdict;
    if (invalid) verdict = "V3";
    else if (undecided || delta_und) verdict = "VU";
    else if (dominated) verdict = "V2";
    else verdict = "V1";

    return {verdict, high_abstain, ocn_constrained};
}

} // namespace mc_gate

#ifdef MC_GATE_SCORER_SELFTEST
int main() {
    using namespace mc_gate;
    double z = gate_norm_ppf(0.975);
    double p0 = 0.065;
    double delta_thr = -0.05;

    {
        auto [lo, hi] = wilson_ci(100, 1000, z);
        std::printf("WILSON %lld %lld %.6f %.6f\n", 100LL, 1000LL, lo, hi);
    }
    {
        auto [lo, hi] = wilson_ci(40, 1000, z);
        std::printf("WILSON %lld %lld %.6f %.6f\n", 40LL, 1000LL, lo, hi);
    }
    {
        auto [lo, hi] = wilson_ci(60, 1000, z);
        std::printf("WILSON %lld %lld %.6f %.6f\n", 60LL, 1000LL, lo, hi);
    }
    {
        auto [lo, hi] = wilson_ci(65, 1000, z);
        std::printf("WILSON %lld %lld %.6f %.6f\n", 65LL, 1000LL, lo, hi);
    }
    {
        auto [lo, hi] = wilson_ci(0, 1000, z);
        std::printf("WILSON %lld %lld %.6f %.6f\n", 0LL, 1000LL, lo, hi);
    }
    {
        auto [lo, hi] = wilson_ci(1000, 1000, z);
        std::printf("WILSON %lld %lld %.6f %.6f\n", 1000LL, 1000LL, lo, hi);
    }

    {
        auto [lo, hi] = mcnemar_ci(150, 50, 1000, z);
        std::printf("MCNEMAR %lld %lld %lld %.6f %.6f\n", 150LL, 50LL, 1000LL, lo, hi);
    }
    {
        auto [lo, hi] = mcnemar_ci(55, 45, 1000, z);
        std::printf("MCNEMAR %lld %lld %lld %.6f %.6f\n", 55LL, 45LL, 1000LL, lo, hi);
    }
    {
        auto [lo, hi] = mcnemar_ci(10, 10, 1000, z);
        std::printf("MCNEMAR %lld %lld %lld %.6f %.6f\n", 10LL, 10LL, 1000LL, lo, hi);
    }
    {
        auto [lo, hi] = mcnemar_ci(500, 400, 1000, z);
        std::printf("MCNEMAR %lld %lld %lld %.6f %.6f\n", 500LL, 400LL, 1000LL, lo, hi);
    }

    {
        VerdictInput in;
        in.nul = { {100, 1000, 0.0, false} };
        in.delta = { {-0.10, 150, 50, 1000} };
        auto r = compute_verdict(in, p0, delta_thr, z);
        std::printf("VERDICT S1 %s %d %d\n", r.verdict, (int)r.high_abstain, (int)r.ocn_constrained);
    }
    {
        VerdictInput in;
        in.nul = { {40, 1000, 0.0, false} };
        in.delta = { {-0.10, 150, 50, 1000} };
        auto r = compute_verdict(in, p0, delta_thr, z);
        std::printf("VERDICT S2 %s %d %d\n", r.verdict, (int)r.high_abstain, (int)r.ocn_constrained);
    }
    {
        VerdictInput in;
        in.nul = { {60, 1000, 0.0, false} };
        in.delta = { {-0.01, 55, 45, 1000} };
        auto r = compute_verdict(in, p0, delta_thr, z);
        std::printf("VERDICT S3 %s %d %d\n", r.verdict, (int)r.high_abstain, (int)r.ocn_constrained);
    }
    {
        VerdictInput in;
        in.nul = { {40, 1000, 0.0, false} };
        in.delta = { {-0.01, 55, 45, 1000} };
        auto r = compute_verdict(in, p0, delta_thr, z);
        std::printf("VERDICT S4 %s %d %d\n", r.verdict, (int)r.high_abstain, (int)r.ocn_constrained);
    }
    {
        VerdictInput in;
        in.nul = { {40, 1000, 0.02, true} };
        in.delta = { {-0.01, 55, 45, 1000} };
        auto r = compute_verdict(in, p0, delta_thr, z);
        std::printf("VERDICT S5 %s %d %d\n", r.verdict, (int)r.high_abstain, (int)r.ocn_constrained);
    }

    std::printf("SCORER_DONE\n");
    return 0;
}
#endif
#endif

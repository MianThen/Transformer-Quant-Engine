// mc_gate_decide.h
// Header-only C++20 module for MC gate decision logic.
//
// Only standard headers allowed: <cmath> <cstdio> <cstddef> <vector> <algorithm>

#ifndef MC_GATE_DECIDE_H
#define MC_GATE_DECIDE_H

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <vector>
#include <algorithm>

namespace mc_gate {

// ---------------------------------------------------------------------------
// Constants (names fixed by spec to avoid collisions with other headers)
// ---------------------------------------------------------------------------
constexpr double kGateP     = 50.4;
constexpr double kGateScale = 7.099295739719547;
constexpr double kGateQ     = 0.05;
constexpr double kGateAlpha = 0.05;

// ---------------------------------------------------------------------------
// Function 1: gate_norm_ppf
//   Standard normal quantile via bisection on the CDF.
//   CDF(z) = 0.5 * erfc(-z / sqrt(2))
// ---------------------------------------------------------------------------
inline double gate_norm_ppf(double u) {
    double lo = -40.0;
    double hi =  40.0;
    for (int i = 0; i < 200; ++i) {
        double mid = (lo + hi) * 0.5;
        double cdf = 0.5 * std::erfc(-mid / std::sqrt(2.0));
        if (cdf < u) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return (lo + hi) * 0.5;
}

// ---------------------------------------------------------------------------
// Function 2: gate_teff
//   v = T / (1 + (T-1)*m); clamp to [1, T]
// ---------------------------------------------------------------------------
inline double gate_teff(double T, double m) {
    double v = T / (1.0 + (T - 1.0) * m);
    if (v < 1.0) return 1.0;
    if (v > T)   return T;
    return v;
}

// ---------------------------------------------------------------------------
// Function 3: gate_dsr_thr
// ---------------------------------------------------------------------------
inline double gate_dsr_thr(double teff, double N, double q) {
    return kGateScale * (std::sqrt(2.0 * std::log(teff) / N)
                         + gate_norm_ppf(1.0 - q) / std::sqrt(N - 1.0));
}

// ---------------------------------------------------------------------------
// Function 4: gate_bh_thr
// ---------------------------------------------------------------------------
inline double gate_bh_thr(double T, double N, double q) {
    return gate_norm_ppf(1.0 - q / T) / std::sqrt(N / kGateP);
}

// ---------------------------------------------------------------------------
// Function 5: gate_threshold_sr
// ---------------------------------------------------------------------------
inline double gate_threshold_sr(double teff, double T, double N, double q) {
    return std::max(gate_dsr_thr(teff, N, q), gate_bh_thr(T, N, q));
}

// ---------------------------------------------------------------------------
// Function 6: gate_p_one_sided
//   One-sided p-value: 1 - Phi(t), t = sr * sqrt(T / kGateP)
//   1 - Phi(t) = 0.5 * erfc(t / sqrt(2))
// ---------------------------------------------------------------------------
inline double gate_p_one_sided(double sr, double obs) {
    // obs = 期数（observations）；v1.4.5 前误用 trials，导致 p 值被放大、FDR 误杀
    double t = sr * std::sqrt(obs / kGateP);
    return 0.5 * std::erfc(t / std::sqrt(2.0));
}

// ---------------------------------------------------------------------------
// Enum + name helper
// ---------------------------------------------------------------------------
enum class GateMode { Full, DsrOnly, FdrOnly, RawT };

inline const char* gate_mode_name(GateMode m) {
    switch (m) {
        case GateMode::Full:    return "Full";
        case GateMode::DsrOnly: return "DsrOnly";
        case GateMode::FdrOnly: return "FdrOnly";
        case GateMode::RawT:    return "RawT";
    }
    return "Unknown";
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------
struct WindowStats {
    std::vector<double> sr;
    double m;
    int T;
    int N;
    bool abstain;
};

struct RuleConfig {
    GateMode mode;
    double q;
    double alpha;
};

struct Decision {
    bool selected;
    std::size_t index;
    bool abstain;
    const char* reason;
};

// ---------------------------------------------------------------------------
// Core decision function
// ---------------------------------------------------------------------------
inline Decision decide_from_stats(const WindowStats& w, const RuleConfig& cfg) {
    const std::size_t n = w.sr.size();

    // amax: index of max sr (ties -> smallest index)
    std::size_t amax = 0;
    if (n > 0) {
        amax = 0;
        for (std::size_t i = 1; i < n; ++i) {
            if (w.sr[i] > w.sr[amax]) amax = i;
        }
    }

    const double te_m = gate_teff(static_cast<double>(w.T), w.m);
    const double te_1 = 1.0;

    const double thr_full = gate_threshold_sr(te_m, static_cast<double>(w.T),
                                               static_cast<double>(w.N), cfg.q);
    const double thr_fdr  = gate_threshold_sr(te_1, static_cast<double>(w.T),
                                               static_cast<double>(w.N), cfg.q);

    // Abstain takes priority over everything
    if (w.abstain) {
        return {false, 0, true, "ocn_abstain"};
    }

    // ---- RawT mode ----
    if (cfg.mode == GateMode::RawT) {
        std::size_t best = 0;
        bool found = false;
        for (std::size_t i = 0; i < n; ++i) {
            if (gate_p_one_sided(w.sr[i], static_cast<double>(w.N)) < cfg.alpha) {
                if (!found || w.sr[i] > w.sr[best]) {
                    best = i;
                    found = true;
                }
            }
        }
        if (!found) {
            return {false, amax, false, "raw_t_rejected"};
        }
        return {true, best, false, "selected"};
    }

    // ---- Non-RawT modes ----
    const double thr = (cfg.mode == GateMode::FdrOnly) ? thr_fdr : thr_full;

    // Candidate set: sr[i] > thr
    std::vector<std::size_t> candidates;
    candidates.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (w.sr[i] > thr) candidates.push_back(i);
    }

    if (candidates.empty()) {
        return {false, amax, false, "below_threshold"};
    }

    // ---- DsrOnly: additional one-sided p-value filter ----
    if (cfg.mode == GateMode::DsrOnly) {
        std::size_t best = 0;
        bool found = false;
        for (std::size_t i : candidates) {
            if (gate_p_one_sided(w.sr[i], static_cast<double>(w.N)) <= cfg.q) {
                if (!found || w.sr[i] > w.sr[best]) {
                    best = i;
                    found = true;
                }
            }
        }
        if (!found) {
            return {false, amax, false, "fdr_rejected"};
        }
        return {true, best, false, "selected"};
    }

    // ---- Full / FdrOnly: Benjamini-Hochberg procedure ----
    // Build (p, index) pairs for all n strategies
    std::vector<std::pair<double, std::size_t>> pidx;
    pidx.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        pidx.emplace_back(gate_p_one_sided(w.sr[i], static_cast<double>(w.N)), i);
    }

    // Sort by p ascending; ties broken by original index ascending
    std::sort(pidx.begin(), pidx.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first < b.first;
                  return a.second < b.second;
              });

    // Find largest r (1-based) such that p_(r) <= q * r / n
    int k = 0;
    for (int r = 1; r <= static_cast<int>(n); ++r) {
        if (pidx[r - 1].first <= cfg.q * static_cast<double>(r) / static_cast<double>(n)) {
            k = r;
        }
    }

    // Survivors: first k in sorted order, then filter by sr > thr
    std::size_t best = 0;
    bool found = false;
    for (int r = 0; r < k; ++r) {
        std::size_t idx = pidx[r].second;
        if (w.sr[idx] > thr) {
            if (!found || w.sr[idx] > w.sr[best]) {
                best = idx;
                found = true;
            }
        }
    }

    if (!found) {
        return {false, amax, false, "fdr_rejected"};
    }
    return {true, best, false, "selected"};
}

} // namespace mc_gate

// ---------------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------------
#ifdef MC_GATE_DECIDE_SELFTEST

#include <cstdio>

int main() {
    using namespace mc_gate;

    // Helper: print GATE_THR line
    auto print_gate_thr = [](const char* tag, double te_m, double thr_full, double thr_fdr) {
        std::printf("GATE_THR %s %.6f %.6f %.6f\n", tag, te_m, thr_full, thr_fdr);
    };

    // Helper: print DECIDE line
    auto print_decide = [](const char* tag, GateMode mode, const Decision& d) {
        std::printf("DECIDE %s %s %d %zu %d %s\n",
                    tag, gate_mode_name(mode),
                    d.selected ? 1 : 0,
                    d.index,
                    d.abstain ? 1 : 0,
                    d.reason);
    };

    // ---- S1 ----
    {
        WindowStats w1{ {7.29, 1.0, -2.0, 0.5}, 0.3, 4, 126, false };
        const double te_m  = gate_teff(static_cast<double>(w1.T), w1.m);
        const double thr_f = gate_threshold_sr(te_m, static_cast<double>(w1.T),
                                               static_cast<double>(w1.N), kGateQ);
        const double thr_d = gate_threshold_sr(1.0, static_cast<double>(w1.T),
                                               static_cast<double>(w1.N), kGateQ);
        print_gate_thr("S1", te_m, thr_f, thr_d);

        RuleConfig cfg{GateMode::Full,    kGateQ, kGateAlpha};
        print_decide("S1", GateMode::Full,    decide_from_stats(w1, cfg));
        cfg.mode = GateMode::DsrOnly;
        print_decide("S1", GateMode::DsrOnly, decide_from_stats(w1, cfg));
        cfg.mode = GateMode::FdrOnly;
        print_decide("S1", GateMode::FdrOnly, decide_from_stats(w1, cfg));
        cfg.mode = GateMode::RawT;
        print_decide("S1", GateMode::RawT,    decide_from_stats(w1, cfg));
    }

    // ---- S2 ----
    {
        WindowStats w2{ {12.0, 1.0, -2.0, 0.5}, 0.3, 100, 126, false };
        const double te_m  = gate_teff(static_cast<double>(w2.T), w2.m);
        const double thr_f = gate_threshold_sr(te_m, static_cast<double>(w2.T),
                                               static_cast<double>(w2.N), kGateQ);
        const double thr_d = gate_threshold_sr(1.0, static_cast<double>(w2.T),
                                               static_cast<double>(w2.N), kGateQ);
        print_gate_thr("S2", te_m, thr_f, thr_d);

        RuleConfig cfg{GateMode::Full,    kGateQ, kGateAlpha};
        print_decide("S2", GateMode::Full,    decide_from_stats(w2, cfg));
        cfg.mode = GateMode::DsrOnly;
        print_decide("S2", GateMode::DsrOnly, decide_from_stats(w2, cfg));
        cfg.mode = GateMode::FdrOnly;
        print_decide("S2", GateMode::FdrOnly, decide_from_stats(w2, cfg));
        cfg.mode = GateMode::RawT;
        print_decide("S2", GateMode::RawT,    decide_from_stats(w2, cfg));
    }

    // ---- S3 ----
    {
        WindowStats w3{ {1.0, 0.9, -2.0, 0.5}, 0.3, 4, 126, false };
        const double te_m  = gate_teff(static_cast<double>(w3.T), w3.m);
        const double thr_f = gate_threshold_sr(te_m, static_cast<double>(w3.T),
                                               static_cast<double>(w3.N), kGateQ);
        const double thr_d = gate_threshold_sr(1.0, static_cast<double>(w3.T),
                                               static_cast<double>(w3.N), kGateQ);
        print_gate_thr("S3", te_m, thr_f, thr_d);

        RuleConfig cfg{GateMode::Full,    kGateQ, kGateAlpha};
        print_decide("S3", GateMode::Full,    decide_from_stats(w3, cfg));
        cfg.mode = GateMode::DsrOnly;
        print_decide("S3", GateMode::DsrOnly, decide_from_stats(w3, cfg));
        cfg.mode = GateMode::FdrOnly;
        print_decide("S3", GateMode::FdrOnly, decide_from_stats(w3, cfg));
        cfg.mode = GateMode::RawT;
        print_decide("S3", GateMode::RawT,    decide_from_stats(w3, cfg));
    }

    // ---- S1_Abort ----
    {
        WindowStats wa{ {7.29, 1.0, -2.0, 0.5}, 0.3, 4, 126, true };
        RuleConfig cfg{GateMode::Full, kGateQ, kGateAlpha};
        print_decide("S1_Abort", GateMode::Full, decide_from_stats(wa, cfg));
    }

    std::printf("DECIDE_DONE\n");
    return 0;
}

#endif // MC_GATE_DECIDE_SELFTEST

#endif // MC_GATE_DECIDE_H

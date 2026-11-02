#ifndef MC_GATE_ADAPTER_H
#define MC_GATE_ADAPTER_H

#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <span>

#include "mc_gate_decide.h"

namespace mc_gate {

struct ResearchWindow {
    std::span<const double> returns;
    std::size_t trials = 0;
    std::size_t observations = 0;
    std::uint64_t returns_sha256 = 0;
};

inline WindowStats make_window_stats(const ResearchWindow& w, bool abstain = false) {
    // v1.4.5：w.trials = 策略数（T），w.observations = 期数（N）
    // 布局 trial-major：data[i * observations + t]
    const std::size_t TR = w.trials;
    const std::size_t OB = w.observations;
    const double* data = w.returns.data();

    std::vector<double> sr(TR, 0.0);
    std::vector<double> means(TR, 0.0);
    std::vector<double> sds(TR, 0.0);

    // Step 1: per-strategy mean and sample standard deviation
    for (std::size_t i = 0; i < TR; ++i) {
        double sum = 0.0;
        for (std::size_t t = 0; t < OB; ++t) {
            sum += data[i * OB + t];
        }
        double mean_i = sum / static_cast<double>(OB);
        means[i] = mean_i;

        double var_sum = 0.0;
        for (std::size_t t = 0; t < OB; ++t) {
            double diff = data[i * OB + t] - mean_i;
            var_sum += diff * diff;
        }
        double var_i = var_sum / static_cast<double>(OB - 1);
        double sd_i = std::sqrt(var_i);
        sds[i] = sd_i;

        if (sd_i <= 0.0) {
            sr[i] = 0.0;
        } else {
            sr[i] = (mean_i / sd_i) * kGateScale;
        }
    }

    // Step 2: average absolute correlation
    double abs_corr_sum = 0.0;
    std::size_t pairs = 0;
    for (std::size_t a = 0; a < TR; ++a) {
        for (std::size_t b = a + 1; b < TR; ++b) {
            if (sds[a] <= 0.0 || sds[b] <= 0.0) {
                continue;
            }
            double cov = 0.0;
            for (std::size_t t = 0; t < OB; ++t) {
                cov += (data[a * OB + t] - means[a]) * (data[b * OB + t] - means[b]);
            }
            double corr = cov / (static_cast<double>(OB - 1) * sds[a] * sds[b]);
            corr = std::clamp(corr, -1.0, 1.0);
            abs_corr_sum += std::abs(corr);
            ++pairs;
        }
    }
    double m = pairs > 0 ? abs_corr_sum / static_cast<double>(pairs) : 0.0;

    // Step 3: build WindowStats
    WindowStats stats;
    stats.sr = std::move(sr);
    stats.m = m;
    stats.T = static_cast<int>(TR);   // 策略数（trials）—— AC-41 的 raw_trial_count
    stats.N = static_cast<int>(OB);   // 期数（observations）—— AC-41 的 observation_count
    stats.abstain = abstain;
    return stats;
}

inline Decision decide(const ResearchWindow& w, const RuleConfig& cfg) {
    return decide_from_stats(make_window_stats(w, false), cfg);
}

} // namespace mc_gate

#ifdef MC_GATE_ADAPTER_SELFTEST

#include <cstdio>

int main() {
    using namespace mc_gate;

    const std::size_t T = 126;
    const std::size_t N = 20;
    std::vector<double> returns(N * T);

    const double ALPHA0 = 34.0 / 1024.0;

    for (std::size_t i = 0; i < N; ++i) {
        double alpha_i = (i == 0) ? ALPHA0 : 0.0;
        for (std::size_t t = 0; t < T; ++t) {
            long eps_int = ((static_cast<long>(i) * 37 + static_cast<long>(t) * 13) % 17 - 8);
            double eps = static_cast<double>(eps_int) / 64.0;
            returns[i * T + t] = alpha_i + eps;
        }
    }

    ResearchWindow w;
    w.returns = std::span<const double>(returns.data(), returns.size());
    w.trials = N;
    w.observations = T;
    w.returns_sha256 = 0;

    WindowStats stats = make_window_stats(w, false);

    for (std::size_t i = 0; i < N; ++i) {
        std::printf("SR %d %.6f\n", static_cast<int>(i), stats.sr[i]);
    }
    std::printf("M %.6f\n", stats.m);

    // gate_teff and gate_threshold_sr are assumed to be provided by mc_gate_decide.h
    // Since they are not in the provided interface list, we must assume they exist.
    // However, the prompt says "mc_gate_decide.h 已提供的接口（逐字如下，不要改动）" and lists only:
    // kGateP, kGateScale, kGateQ, kGateAlpha, GateMode, gate_mode_name, WindowStats, RuleConfig, Decision, decide_from_stats
    // But the self-test requires gate_teff and gate_threshold_sr.
    // This implies they must be in mc_gate_decide.h even if not listed in the "provided interface" snippet,
    // OR the snippet is incomplete. Given the constraint "禁止重新定义 ... 它们全部来自 mc_gate_decide.h",
    // and the self-test explicitly uses them, I must assume they are available.
    
    // 自检里 T=126 是期数、N=20 是策略数；teff/门槛吃的是（trials, observations）
    double te_m = gate_teff(static_cast<double>(N), stats.m);
    double thr_full = gate_threshold_sr(te_m, static_cast<double>(N), static_cast<double>(T), kGateQ);
    double thr_fdr = gate_threshold_sr(1.0, static_cast<double>(N), static_cast<double>(T), kGateQ);
    std::printf("THR %.6f %.6f\n", thr_full, thr_fdr);

    RuleConfig cfg_full{GateMode::Full, 0.05, 0.05};
    RuleConfig cfg_dsr{GateMode::DsrOnly, 0.05, 0.05};
    RuleConfig cfg_fdr{GateMode::FdrOnly, 0.05, 0.05};
    RuleConfig cfg_raw{GateMode::RawT, 0.05, 0.05};

    Decision d_full = decide(w, cfg_full);
    Decision d_dsr = decide(w, cfg_dsr);
    Decision d_fdr = decide(w, cfg_fdr);
    Decision d_raw = decide(w, cfg_raw);

    std::printf("DECIDE Full %d %d %d %s\n", d_full.selected ? 1 : 0, static_cast<int>(d_full.index), d_full.abstain ? 1 : 0, d_full.reason);
    std::printf("DECIDE DsrOnly %d %d %d %s\n", d_dsr.selected ? 1 : 0, static_cast<int>(d_dsr.index), d_dsr.abstain ? 1 : 0, d_dsr.reason);
    std::printf("DECIDE FdrOnly %d %d %d %s\n", d_fdr.selected ? 1 : 0, static_cast<int>(d_fdr.index), d_fdr.abstain ? 1 : 0, d_fdr.reason);
    std::printf("DECIDE RawT %d %d %d %s\n", d_raw.selected ? 1 : 0, static_cast<int>(d_raw.index), d_raw.abstain ? 1 : 0, d_raw.reason);

    WindowStats stats_abort = make_window_stats(w, true);
    RuleConfig cfg_abort{GateMode::Full, 0.05, 0.05};
    Decision d_abort = decide_from_stats(stats_abort, cfg_abort);
    std::printf("DECIDE Abort %d %d %d %s\n", d_abort.selected ? 1 : 0, static_cast<int>(d_abort.index), d_abort.abstain ? 1 : 0, d_abort.reason);

    std::printf("ADAPTER_DONE\n");
    return 0;
}

#endif // MC_GATE_ADAPTER_SELFTEST

#endif // MC_GATE_ADAPTER_H

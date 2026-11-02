#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstddef>
#include <vector>
#include <functional>
#include "mc_gate_dgp.h"

namespace mc_gate {

double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double strategy_sr(const ReturnMatrix& R, int i) {
    if (R.N <= 1) return 0.0;
    double sum = 0.0;
    for (int t = 0; t < R.N; ++t) sum += R.at(i, t);
    double mean = sum / R.N;
    double var = 0.0;
    for (int t = 0; t < R.N; ++t) {
        double d = R.at(i, t) - mean;
        var += d * d;
    }
    var /= (R.N - 1);
    double sd = std::sqrt(var);
    if (sd == 0.0) return 0.0;
    return (mean / sd) * kScale;
}

double strategy_sr_window(const ReturnMatrix& R, int i, int t0, int t1) {
    int n = t1 - t0;
    if (n < 2) return 0.0;
    double sum = 0.0;
    for (int t = t0; t < t1; ++t) sum += R.at(i, t);
    double mean = sum / n;
    double var = 0.0;
    for (int t = t0; t < t1; ++t) {
        double d = R.at(i, t) - mean;
        var += d * d;
    }
    var /= (n - 1);
    double sd = std::sqrt(var);
    if (sd == 0.0) return 0.0;
    return (mean / sd) * kScale;
}

double t_stat(double sr, double T) {
    return sr * std::sqrt(T / kP);
}

double p_one_sided(double sr, double T) {
    return 1.0 - norm_cdf(t_stat(sr, T));
}

struct RuleResult {
    bool selected;
    int index;
    double sr;
    int window;
    double pbo;
};

RuleResult rule_b0(const ReturnMatrix& R) {
    int best_i = 0;
    double best_sr = strategy_sr(R, 0);
    for (int i = 1; i < R.T; ++i) {
        double s = strategy_sr(R, i);
        if (s > best_sr) {
            best_sr = s;
            best_i = i;
        }
    }
    return {best_sr > 0.0, best_i, best_sr, -1, -1.0};
}

RuleResult rule_b1(const ReturnMatrix& R, double alpha) {
    int best_i = 0;
    double best_sr = strategy_sr(R, 0);
    for (int i = 1; i < R.T; ++i) {
        double s = strategy_sr(R, i);
        if (s > best_sr) {
            best_sr = s;
            best_i = i;
        }
    }
    double p = p_one_sided(best_sr, R.N);
    return {p < alpha, best_i, best_sr, -1, -1.0};
}

RuleResult rule_b2(const ReturnMatrix& R) {
    int b0 = 0;
    int b1 = R.N / 3;
    int b2 = 2 * R.N / 3;
    int b3 = R.N;
    int bnd[4] = {b0, b1, b2, b3};
    int best_i = 0;
    int best_w = 0;
    double best_sr = strategy_sr_window(R, 0, bnd[0], bnd[1]);
    for (int w = 0; w < 3; ++w) {
        for (int i = 0; i < R.T; ++i) {
            double s = strategy_sr_window(R, i, bnd[w], bnd[w + 1]);
            if (s > best_sr) {
                best_sr = s;
                best_i = i;
                best_w = w;
            }
        }
    }
    return {best_sr > 0.0, best_i, best_sr, best_w, -1.0};
}

RuleResult rule_b3(const ReturnMatrix& R) {
    int mid = R.N / 2;
    int best_i = 0;
    double best_sr = strategy_sr_window(R, 0, 0, mid);
    for (int i = 1; i < R.T; ++i) {
        double s = strategy_sr_window(R, i, 0, mid);
        if (s > best_sr) {
            best_sr = s;
            best_i = i;
        }
    }
    double oos_sr = strategy_sr_window(R, best_i, mid, R.N);
    return {oos_sr > 0.0, best_i, oos_sr, -1, -1.0};
}

RuleResult rule_b4_cscv(const ReturnMatrix& R, int S, double pbo_max) {
    int BS = R.N / S;
    int half = S / 2;
    int total_combos = 1;
    for (int i = 0; i < S; ++i) {
        total_combos = total_combos * (i + 1) / (i + 1);
    }
    // Compute C(S, S/2)
    int C = 1;
    for (int i = 0; i < half; ++i) {
        C = C * (S - i) / (i + 1);
    }
    
    int overfit = 0;
    int max_mask = (1 << S);
    for (int mask = 0; mask < max_mask; ++mask) {
        int bits = 0;
        for (int b = 0; b < S; ++b) {
            if (mask & (1 << b)) ++bits;
        }
        if (bits != half) continue;
        
        // IS stats
        double sum_is[1024] = {0};
        double sumsq_is[1024] = {0};
        int n_is[1024] = {0};
        // OOS stats
        double sum_oos[1024] = {0};
        double sumsq_oos[1024] = {0};
        int n_oos[1024] = {0};
        
        for (int i = 0; i < R.T; ++i) {
            sum_is[i] = 0.0; sumsq_is[i] = 0.0; n_is[i] = 0;
            sum_oos[i] = 0.0; sumsq_oos[i] = 0.0; n_oos[i] = 0;
        }
        
        for (int b = 0; b < S; ++b) {
            int start = b * BS;
            int end = (b + 1) * BS;
            bool is_is = (mask & (1 << b)) != 0;
            for (int t = start; t < end; ++t) {
                for (int i = 0; i < R.T; ++i) {
                    double x = R.at(i, t);
                    if (is_is) {
                        sum_is[i] += x;
                        sumsq_is[i] += x * x;
                        n_is[i]++;
                    } else {
                        sum_oos[i] += x;
                        sumsq_oos[i] += x * x;
                        n_oos[i]++;
                    }
                }
            }
        }
        
        double sr_is[1024];
        for (int i = 0; i < R.T; ++i) {
            if (n_is[i] < 2) {
                sr_is[i] = 0.0;
            } else {
                double mean = sum_is[i] / n_is[i];
                double var = (sumsq_is[i] - sum_is[i] * sum_is[i] / n_is[i]) / (n_is[i] - 1);
                if (var < 0) var = 0;
                double sd = std::sqrt(var);
                if (sd == 0.0) sr_is[i] = 0.0;
                else sr_is[i] = (mean / sd) * kScale;
            }
        }
        
        int best_i = 0;
        double best_sr_is = sr_is[0];
        for (int i = 1; i < R.T; ++i) {
            if (sr_is[i] > best_sr_is) {
                best_sr_is = sr_is[i];
                best_i = i;
            }
        }
        
        double sr_oos[1024];
        for (int i = 0; i < R.T; ++i) {
            if (n_oos[i] < 2) {
                sr_oos[i] = 0.0;
            } else {
                double mean = sum_oos[i] / n_oos[i];
                double var = (sumsq_oos[i] - sum_oos[i] * sum_oos[i] / n_oos[i]) / (n_oos[i] - 1);
                if (var < 0) var = 0;
                double sd = std::sqrt(var);
                if (sd == 0.0) sr_oos[i] = 0.0;
                else sr_oos[i] = (mean / sd) * kScale;
            }
        }
        
        double sr_oos_best = sr_oos[best_i];
        int rank = 1;
        for (int i = 0; i < R.T; ++i) {
            if (sr_oos[i] < sr_oos_best) ++rank;
        }
        double omega = static_cast<double>(rank) / (R.T + 1);
        if (std::log(omega / (1.0 - omega)) < 0.0) {
            ++overfit;
        }
    }
    
    double pbo = static_cast<double>(overfit) / C;
    
    int best_i = 0;
    double best_sr = strategy_sr(R, 0);
    for (int i = 1; i < R.T; ++i) {
        double s = strategy_sr(R, i);
        if (s > best_sr) {
            best_sr = s;
            best_i = i;
        }
    }
    
    return {pbo < pbo_max && best_sr > 0.0, best_i, best_sr, -1, pbo};
}

} // namespace mc_gate

#ifdef MC_GATE_BASE_SELFTEST
int main() {
    using namespace mc_gate;
    
    ReturnMatrix R;
    R.N = 12;
    R.T = 4;
    R.r.resize(R.N * R.T);
    
    double data[12][4] = {
        {-0.0109, -0.0177, 0.0295, 0.0216},
        {0.0425, -0.0049, 0.0392, 0.0315},
        {0.0377, 0.0185, 0.0107, 0.0594},
        {-0.0000, -0.0350, 0.0443, -0.0017},
        {0.0148, -0.0143, -0.0031, 0.0333},
        {-0.0195, 0.0209, 0.0030, 0.0119},
        {0.0272, 0.0107, -0.0371, -0.0078},
        {0.0078, -0.0090, 0.0119, 0.0052},
        {0.0081, 0.0117, 0.0049, -0.0149},
        {0.0301, 0.0115, 0.0079, -0.0171},
        {-0.0083, 0.0016, -0.0211, -0.0040},
        {0.0151, -0.0170, -0.0053, -0.0094}
    };
    
    for (int t = 0; t < R.N; ++t) {
        for (int i = 0; i < R.T; ++i) {
            R.r[i * R.N + t] = data[t][i];   // trial-major：data 是 [期][策略]
        }
    }
    
    for (int i = 0; i < R.T; ++i) {
        std::printf("SR %d %.6f\n", i, strategy_sr(R, i));
    }
    
    RuleResult r0 = rule_b0(R);
    std::printf("B0 %d %d %.6f\n", r0.selected ? 1 : 0, r0.index, r0.sr);
    
    RuleResult r1 = rule_b1(R, 0.05);
    std::printf("B1 %d %d %.6f %.6f\n", r1.selected ? 1 : 0, r1.index, r1.sr, p_one_sided(r1.sr, R.N));
    
    RuleResult r2 = rule_b2(R);
    std::printf("B2 %d %d %.6f %d\n", r2.selected ? 1 : 0, r2.index, r2.sr, r2.window);
    
    RuleResult r3 = rule_b3(R);
    std::printf("B3 %d %d %.6f\n", r3.selected ? 1 : 0, r3.index, r3.sr);
    
    RuleResult r4 = rule_b4_cscv(R, 4, 0.5);
    std::printf("B4 %d %d %.6f %.6f\n", r4.selected ? 1 : 0, r4.index, r4.sr, r4.pbo);
    
    std::printf("BASE_DONE\n");
    return 0;
}
#endif

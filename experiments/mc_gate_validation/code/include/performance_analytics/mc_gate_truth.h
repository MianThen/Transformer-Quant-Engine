#ifndef MC_GATE_TRUTH_H
#define MC_GATE_TRUTH_H

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace mc_gate {

constexpr double kP = 50.4;
constexpr double kQ = 0.05;

double norm_ppf(double u) {
    if (u <= 0.0) return -40.0;
    if (u >= 1.0) return 40.0;
    double lo = -40.0;
    double hi = 40.0;
    for (int i = 0; i < 200; ++i) {
        double mid = (lo + hi) / 2.0;
        double cdf = 0.5 * std::erfc(-mid / std::sqrt(2.0));
        if (cdf < u) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return (lo + hi) / 2.0;
}

double t_eff(double T, double m) {
    double v = T / (1.0 + (T - 1.0) * m);
    if (v < 1.0) return 1.0;
    if (v > T) return T;
    return v;
}

double sr_req_dsr(double T_eff, double N) {
    return std::sqrt(kP) * (std::sqrt(2.0 * std::log(T_eff) / N) + norm_ppf(1.0 - kQ) / std::sqrt(N - 1.0));
}

double sr_req_bh(double T, double N) {
    return norm_ppf(1.0 - kQ / T) / std::sqrt(N / kP);
}

double sr_req(double T, double N, double m) {
    return std::max(sr_req_dsr(t_eff(T, m), N), sr_req_bh(T, N));
}

double c_kappa(const char* family) {
    if (family == nullptr) return 0.5;
    if (std::strcmp(family, "gauss") == 0) return 0.5;
    if (std::strcmp(family, "t5") == 0) return 2.0;
    if (std::strcmp(family, "garch") == 0) return 3.5;
    if (std::strcmp(family, "shortvol") == 0) return 4.25;
    return 0.5;
}

double var_single(double S, double N, double ck) {
    return (kP + ck * S * S) / N;
}

double se_mean(double S, double N, double ck, int R) {
    return std::sqrt(var_single(S, N, ck) / (double)R);
}

double z_cell(double sample_sr, double S, double N, double ck, int R) {
    return (sample_sr - S) / se_mean(S, N, ck, R);
}

} // namespace mc_gate

#ifdef MC_GATE_TRUTH_SELFTEST
int main() {
    using namespace mc_gate;

    double us[4] = {0.95, 0.975, 0.99, 0.9875};
    for (int i = 0; i < 4; ++i) {
        std::printf("NORMPPF %.4f %.15f\n", us[i], norm_ppf(us[i]));
    }

    int Ns[3] = {126, 266, 520};
    int Ts[3] = {4, 20, 100};

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int N = Ns[i];
            int T = Ts[j];
            double m = 0.3;
            double te = t_eff((double)T, m);
            double dsr = sr_req_dsr(te, (double)N);
            double bh = sr_req_bh((double)T, (double)N);
            double req = sr_req((double)T, (double)N, m);
            std::printf("TEFF %d %d %.6f %.6f %.6f %.6f\n", T, N, te, dsr, bh, req);
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int N = Ns[i];
            int T = Ts[j];
            double m = 0.8 / std::sqrt((double)N);
            double te = t_eff((double)T, m);
            std::printf("TEFF0 %d %d %.6f\n", T, N, te);
        }
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int N = Ns[i];
            int T = Ts[j];
            double m = 0.7;
            double te = t_eff((double)T, m);
            std::printf("TEFF7 %d %d %.6f\n", T, N, te);
        }
    }

    const char* fams[4] = {"gauss", "t5", "garch", "shortvol"};
    for (int i = 0; i < 4; ++i) {
        std::printf("CKAPPA %s %.2f\n", fams[i], c_kappa(fams[i]));
    }

    double S = 1.8;
    int N = 126;
    double ck = 2.0;
    int R = 1;
    double sample_sr = 1.8;
    std::printf("ZCELL %.12f\n", z_cell(sample_sr, S, (double)N, ck, R));

    std::printf("TRUTH_DONE\n");
    return 0;
}
#endif

#endif // MC_GATE_TRUTH_H

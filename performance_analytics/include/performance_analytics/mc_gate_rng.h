#include <cstdint>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace mc_gate {

struct Pcg64 {
    unsigned __int128 state_;
    unsigned __int128 inc_;

    explicit Pcg64(std::uint64_t seed) {
        state_ = 0;
        inc_ = ((unsigned __int128)seed << 1) | 1;
        step_();
        state_ += seed;
        step_();
    }

    std::uint64_t next() {
        unsigned __int128 old = state_;
        step_();
        std::uint64_t hi = (std::uint64_t)(old >> 64);
        std::uint64_t lo = (std::uint64_t)old;
        std::uint64_t xs = hi ^ lo;
        int r = (int)(hi >> 58);
        std::uint64_t rot = ((std::uint64_t)r) & 63ULL;
        return (xs >> rot) | (xs << ((64ULL - rot) & 63ULL));
    }

    double next_double() {
        return (double)(next() >> 11) * 0x1.0p-53;
    }

private:
    void step_() {
        const unsigned __int128 MULT = ((unsigned __int128)0x2360ED051FC65DA4ULL << 64) | 0x4385DF649FCCF645ULL;
        const unsigned __int128 INC = ((unsigned __int128)0x5851F42D4C957F2DULL << 64) | 0x14057B7EF767814FULL;
        state_ = state_ * MULT + INC;
    }
};

double normal_z(Pcg64& rng) {
    static thread_local double cache = 0.0;
    static thread_local bool has_cache = false;

    if (has_cache) {
        has_cache = false;
        return cache;
    }

    double u1 = rng.next_double();
    double u2 = rng.next_double();
    if (u1 < 1e-300) u1 = 1e-300;

    double r = std::sqrt(-2.0 * std::log(u1));
    double th = 6.283185307179586 * u2;

    double z1 = r * std::cos(th);
    double z2 = r * std::sin(th);

    cache = z2;
    has_cache = true;

    return z1;
}

double t5_z(Pcg64& rng) {
    double z0 = normal_z(rng);
    double chi2_5 = 0.0;
    for (int i = 0; i < 5; ++i) {
        double zi = normal_z(rng);
        chi2_5 += zi * zi;
    }
    double t = z0 / std::sqrt(chi2_5 / 5.0);
    return t * std::sqrt(3.0 / 5.0);
}

struct Garch {
    double alpha;
    double beta;
    double omega;
    double sigma2;
    double a_prev_sq;

    Garch() : alpha(0.08), beta(0.90), omega(1.0 - 0.08 - 0.90), sigma2(1.0), a_prev_sq(0.0) {}

    double step(Pcg64& rng) {
        sigma2 = omega + alpha * a_prev_sq + beta * sigma2;
        double sigma = std::sqrt(sigma2);
        double z = t5_z(rng);
        double a = sigma * z;
        a_prev_sq = a * a;
        return a;
    }
};

struct ShortVol {
    double a;
    double b;
    double norm;

    ShortVol() : a(1.0), b(0.0), norm(1.0) {
        double lo = 0.0;
        double hi = 5.0;
        for (int i = 0; i < 200; ++i) {
            double mid = (lo + hi) / 2.0;
            double skew_mid = -(6.0 * a * a * mid + 8.0 * mid * mid * mid) / std::pow(a * a + 2.0 * mid * mid, 1.5);
            if (skew_mid > -2.5) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        b = (lo + hi) / 2.0;
        norm = std::sqrt(a * a + 2.0 * b * b);
    }

    double z(Pcg64& rng) {
        double u = normal_z(rng);
        double w = u * u - 1.0;
        return (a * u - b * w) / norm;
    }
};

} // namespace mc_gate

#ifdef MC_GATE_RNG_SELFTEST
int main() {
    {
        mc_gate::Pcg64 g(42);
        for (int i = 0; i < 8; ++i)
            std::printf("PCG42 %016llx\n", (unsigned long long)g.next());
    }
    {
        mc_gate::Pcg64 g(12345);
        for (int i = 0; i < 8; ++i)
            std::printf("PCG12345 %016llx\n", (unsigned long long)g.next());
    }
    {
        mc_gate::Pcg64 g(7);
        const int N = 2000000;
        double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        for (int i = 0; i < N; ++i) {
            double x = mc_gate::t5_z(g);
            s1 += x;
            s2 += x * x;
            s3 += x * x * x;
            s4 += x * x * x * x;
        }
        double m1 = s1 / N, m2 = s2 / N, m3 = s3 / N, m4 = s4 / N;
        double var = m2 - m1 * m1;
        double mu4 = m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 * m1 * m1 * m1;
        std::printf("T5VAR %.9f\n", var);
        std::printf("T5KURT %.6f\n", mu4 / (var * var) - 3.0);
    }
    {
        mc_gate::ShortVol sv;
        std::printf("SVB %.12f\n", sv.b);
        mc_gate::Pcg64 g(11);
        const int N = 2000000;
        double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        for (int i = 0; i < N; ++i) {
            double x = sv.z(g);
            s1 += x;
            s2 += x * x;
            s3 += x * x * x;
            s4 += x * x * x * x;
        }
        double m1 = s1 / N, m2 = s2 / N, m3 = s3 / N, m4 = s4 / N;
        double var = m2 - m1 * m1;
        std::printf("SVVAR %.9f\n", var);
        std::printf("SVSKEW %.6f\n", (m3 - 3 * m1 * m2 + 2 * m1 * m1 * m1) / (var * std::sqrt(var)));
    }
    {
        mc_gate::Garch gh;
        mc_gate::Pcg64 g(13);
        const int N = 2000000;
        double sa = 0;
        for (int i = 0; i < N; ++i) {
            double a = gh.step(g);
            sa += a * a;
        }
        std::printf("GARCHOMEGA %.12f\n", gh.omega);
        std::printf("GARCHVAR %.6f\n", sa / N);
    }
    return 0;
}
#endif

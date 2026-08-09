#include <cmath>
#include <cstdio>
#include <limits>

#include "quant_math/matrix.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
    return condition;
}

}  // namespace

int main() {
    bool ok = true;
    quant_math::DenseMatrix psd(2, 2);
    psd << 2.0, -1.0, -1.0, 2.0;
    double minimum_eigenvalue = 0.0;
    ok &= check(quant_math::validate_finite(quant_math::view(psd)).ok, "finite matrix");
    ok &= check(quant_math::validate_symmetric(quant_math::view(psd), 1e-12).ok,
                "symmetric matrix");
    ok &= check(quant_math::is_positive_semidefinite(psd, 1e-12, &minimum_eigenvalue),
                "PSD matrix");
    ok &= check(std::abs(minimum_eigenvalue - 1.0) < 1e-12, "minimum eigenvalue");

    psd(0, 1) = 0.0;
    ok &= check(!quant_math::validate_symmetric(quant_math::view(psd), 1e-12).ok,
                "asymmetric rejection");
    psd(0, 1) = std::numeric_limits<double>::quiet_NaN();
    ok &= check(!quant_math::validate_finite(quant_math::view(psd)).ok, "NaN rejection");
    if (!ok) return 1;
    std::printf("test_quant_math: all checks passed\n");
    return 0;
}

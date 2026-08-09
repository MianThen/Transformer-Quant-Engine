#include <array>
#include <cmath>
#include <cstdio>

#include "portfolio_math/multiple_testing.h"

namespace {

bool check(bool condition, const char* message) {
    if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
    return condition;
}

bool near(double left, double right, double tolerance = 1e-12) {
    return std::abs(left - right) <= tolerance;
}

}  // namespace

int main() {
    bool ok = true;
    const std::array<double, 5> p_values{0.01, 0.04, 0.03, 0.002, 0.80};
    const auto bh = portfolio_math::benjamini_hochberg(p_values, 0.05);
    const std::array<double, 5> bh_golden{0.025, 0.05, 0.05, 0.01, 0.80};
    ok &= check(bh.status == portfolio_math::MultipleTestingStatus::OK,
                "BH status");
    for (std::size_t index = 0; index < p_values.size(); ++index) {
        ok &= check(near(bh.adjusted_p_values[index], bh_golden[index]),
                    "BH adjusted p-value golden");
    }
    const auto by = portfolio_math::benjamini_yekutieli(p_values, 0.05);
    ok &= check(by.status == portfolio_math::MultipleTestingStatus::OK &&
                by.adjusted_p_values[0] >= bh.adjusted_p_values[0],
                "BY is at least as conservative as BH");

    const std::array<double, 3> lambdas{0.25, 0.50, 0.75};
    const auto storey = portfolio_math::storey_fdr(
        p_values, 0.05, lambdas,
        portfolio_math::StoreyAggregation::MEAN_PRE_REGISTERED_LAMBDAS);
    ok &= check(storey.status == portfolio_math::MultipleTestingStatus::OK &&
                near(storey.pi0, 22.0 / 45.0) &&
                storey.pi0_by_lambda.size() == lambdas.size(),
                "Storey pi0 report");

    const std::array<double, 5> permuted{0.80, 0.002, 0.03, 0.01, 0.04};
    const auto permuted_bh = portfolio_math::benjamini_hochberg(permuted, 0.05);
    ok &= check(near(permuted_bh.adjusted_p_values[0], bh.adjusted_p_values[4]) &&
                near(permuted_bh.adjusted_p_values[1], bh.adjusted_p_values[3]) &&
                near(permuted_bh.adjusted_p_values[3], bh.adjusted_p_values[0]),
                "BH permutation invariance");
    const std::array<double, 2> all_rejected{0.0, 0.0};
    const std::array<double, 2> none_rejected{1.0, 1.0};
    const auto all = portfolio_math::benjamini_hochberg(all_rejected, 0.05);
    const auto none = portfolio_math::benjamini_hochberg(none_rejected, 0.05);
    ok &= check(all.rejected[0] == 1 && all.rejected[1] == 1 &&
                none.rejected[0] == 0 && none.rejected[1] == 0,
                "BH all/none rejection boundaries");
    const std::array<double, 1> zero_lambda{0.5};
    const auto zero_pi0 = portfolio_math::storey_fdr(
        all_rejected, 0.05, zero_lambda);
    ok &= check(zero_pi0.status == portfolio_math::MultipleTestingStatus::OK &&
                zero_pi0.pi0 == 0.0 && zero_pi0.adjusted_p_values[0] == 0.0,
                "Storey pi0 zero boundary is preserved");

    const std::array<double, 8> differences{
        0.02, 0.01, -0.01, 0.03, 0.02, 0.00, 0.01, 0.04};
    portfolio_math::BlockBootstrapSpec bootstrap_spec;
    bootstrap_spec.block_length = 3;
    bootstrap_spec.resamples = 999;
    bootstrap_spec.seed = 1234567;
    bootstrap_spec.config_hash = 99;
    const auto first = portfolio_math::paired_block_bootstrap_mean(
        differences, bootstrap_spec);
    const auto second = portfolio_math::paired_block_bootstrap_mean(
        differences, bootstrap_spec);
    ok &= check(first.status == portfolio_math::MultipleTestingStatus::OK &&
                first.one_sided_positive_p_value == second.one_sided_positive_p_value &&
                first.exceedances == second.exceedances &&
                first.replay_hash == second.replay_hash,
                "block bootstrap seed reproducibility");
    ok &= check(near(first.observed_mean, 0.015), "paired mean statistic");
    ok &= check(first.exceedances == 0 &&
                near(first.one_sided_positive_p_value, 0.001),
                "cross-language block-bootstrap golden");

    const std::array<double, 2> invalid_p_values{0.1, 1.1};
    ok &= check(portfolio_math::benjamini_hochberg(invalid_p_values, 0.05).status ==
                    portfolio_math::MultipleTestingStatus::INVALID_INPUT,
                "invalid p-value rejection");
    if (!ok) return 1;
    std::printf("test_multiple_testing: all checks passed\n");
    return 0;
}

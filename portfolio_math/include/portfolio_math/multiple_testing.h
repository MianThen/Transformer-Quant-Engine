#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace portfolio_math {

enum class MultipleTestingStatus : std::uint8_t {
    OK,
    INVALID_INPUT,
    INSUFFICIENT_OBSERVATIONS,
    NUMERICAL_FAILURE,
};

enum class FdrMethod : std::uint8_t {
    BENJAMINI_HOCHBERG,
    BENJAMINI_YEKUTIELI,
    STOREY,
};

enum class StoreyAggregation : std::uint8_t {
    MEAN_PRE_REGISTERED_LAMBDAS,
    MAXIMUM_PRE_REGISTERED_LAMBDAS,
};

struct FdrResult {
    MultipleTestingStatus status{MultipleTestingStatus::INVALID_INPUT};
    FdrMethod method{FdrMethod::BENJAMINI_HOCHBERG};
    double q_level{0.0};
    double pi0{1.0};
    std::vector<double> pi0_by_lambda;
    std::vector<double> adjusted_p_values;
    std::vector<std::uint8_t> rejected;
};

[[nodiscard]] FdrResult benjamini_hochberg(
    std::span<const double> p_values, double q_level);

[[nodiscard]] FdrResult benjamini_yekutieli(
    std::span<const double> p_values, double q_level);

[[nodiscard]] FdrResult storey_fdr(
    std::span<const double> p_values,
    double q_level,
    std::span<const double> pre_registered_lambdas,
    StoreyAggregation aggregation = StoreyAggregation::MEAN_PRE_REGISTERED_LAMBDAS);

struct BlockBootstrapSpec {
    std::uint32_t block_length{0};
    std::uint32_t resamples{0};
    std::uint64_t seed{0};
    bool studentized{false};
    std::uint64_t config_hash{0};
};

struct BlockBootstrapResult {
    MultipleTestingStatus status{MultipleTestingStatus::INVALID_INPUT};
    double observed_mean{0.0};
    double observed_statistic{0.0};
    double one_sided_positive_p_value{1.0};
    std::uint32_t exceedances{0};
    std::uint64_t replay_hash{0};
};

[[nodiscard]] BlockBootstrapResult paired_block_bootstrap_mean(
    std::span<const double> paired_differences,
    const BlockBootstrapSpec& spec);

}  // namespace portfolio_math

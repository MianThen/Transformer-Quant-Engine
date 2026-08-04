#include "portfolio_math/multiple_testing.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
#include <utility>

namespace portfolio_math {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_value(std::uint64_t& hash, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffU;
        hash *= kFnvPrime;
    }
}

class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() noexcept {
        std::uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

private:
    std::uint64_t state_;
};

bool valid_probabilities(std::span<const double> values) {
    return !values.empty() && std::all_of(values.begin(), values.end(), [](double value) {
        return std::isfinite(value) && value >= 0.0 && value <= 1.0;
    });
}

FdrResult adjusted_fdr(
    std::span<const double> p_values,
    double q_level,
    FdrMethod method,
    double multiplier) {
    FdrResult result;
    result.method = method;
    result.q_level = q_level;
    if (!valid_probabilities(p_values) || !(q_level > 0.0 && q_level < 1.0) ||
        !(multiplier >= 0.0) || !std::isfinite(multiplier)) {
        return result;
    }
    std::vector<std::pair<double, std::size_t>> ordered;
    ordered.reserve(p_values.size());
    for (std::size_t index = 0; index < p_values.size(); ++index) {
        ordered.emplace_back(p_values[index], index);
    }
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
        if (left.first != right.first) return left.first < right.first;
        return left.second < right.second;
    });
    result.adjusted_p_values.assign(p_values.size(), 1.0);
    result.rejected.assign(p_values.size(), 0);
    double running_minimum = 1.0;
    for (std::size_t offset = ordered.size(); offset > 0; --offset) {
        const std::size_t rank = offset;
        const double adjusted = std::min(
            1.0, ordered[offset - 1].first * static_cast<double>(ordered.size()) *
                     multiplier / static_cast<double>(rank));
        running_minimum = std::min(running_minimum, adjusted);
        result.adjusted_p_values[ordered[offset - 1].second] = running_minimum;
    }
    for (std::size_t index = 0; index < p_values.size(); ++index) {
        result.rejected[index] = result.adjusted_p_values[index] <= q_level ? 1 : 0;
    }
    result.status = MultipleTestingStatus::OK;
    return result;
}

double sample_standard_error(std::span<const double> values, double mean) {
    if (values.size() < 2) return 0.0;
    double sum = 0.0;
    for (double value : values) {
        const double delta = value - mean;
        sum += delta * delta;
    }
    return std::sqrt(sum / static_cast<double>(values.size() - 1) /
                     static_cast<double>(values.size()));
}

}  // namespace

FdrResult benjamini_hochberg(std::span<const double> p_values, double q_level) {
    return adjusted_fdr(p_values, q_level, FdrMethod::BENJAMINI_HOCHBERG, 1.0);
}

FdrResult benjamini_yekutieli(std::span<const double> p_values, double q_level) {
    double harmonic = 0.0;
    for (std::size_t index = 1; index <= p_values.size(); ++index) {
        harmonic += 1.0 / static_cast<double>(index);
    }
    return adjusted_fdr(p_values, q_level, FdrMethod::BENJAMINI_YEKUTIELI, harmonic);
}

FdrResult storey_fdr(
    std::span<const double> p_values,
    double q_level,
    std::span<const double> pre_registered_lambdas,
    StoreyAggregation aggregation) {
    FdrResult result;
    result.method = FdrMethod::STOREY;
    result.q_level = q_level;
    if (!valid_probabilities(p_values) || !(q_level > 0.0 && q_level < 1.0) ||
        pre_registered_lambdas.empty() ||
        !std::is_sorted(pre_registered_lambdas.begin(), pre_registered_lambdas.end()) ||
        std::adjacent_find(pre_registered_lambdas.begin(),
                           pre_registered_lambdas.end()) != pre_registered_lambdas.end()) {
        return result;
    }
    std::vector<double> pi0_by_lambda;
    pi0_by_lambda.reserve(pre_registered_lambdas.size());
    for (double lambda : pre_registered_lambdas) {
        if (!(lambda >= 0.0 && lambda < 1.0) || !std::isfinite(lambda)) return result;
        const auto count = std::count_if(p_values.begin(), p_values.end(), [&](double value) {
            return value > lambda;
        });
        pi0_by_lambda.push_back(std::clamp(
            static_cast<double>(count) /
                (static_cast<double>(p_values.size()) * (1.0 - lambda)),
            0.0, 1.0));
    }
    double pi0 = 1.0;
    if (aggregation == StoreyAggregation::MAXIMUM_PRE_REGISTERED_LAMBDAS) {
        pi0 = *std::max_element(pi0_by_lambda.begin(), pi0_by_lambda.end());
    } else {
        pi0 = std::accumulate(pi0_by_lambda.begin(), pi0_by_lambda.end(), 0.0) /
              static_cast<double>(pi0_by_lambda.size());
    }
    result = adjusted_fdr(p_values, q_level, FdrMethod::STOREY,
                          pi0);
    result.pi0 = pi0;
    result.pi0_by_lambda = std::move(pi0_by_lambda);
    return result;
}

BlockBootstrapResult paired_block_bootstrap_mean(
    std::span<const double> paired_differences,
    const BlockBootstrapSpec& spec) {
    BlockBootstrapResult result;
    if (paired_differences.size() < 2 || spec.block_length == 0 ||
        spec.block_length > paired_differences.size() || spec.resamples == 0 ||
        spec.seed == 0 || spec.config_hash == 0 ||
        !std::all_of(paired_differences.begin(), paired_differences.end(), [](double value) {
            return std::isfinite(value);
        })) {
        return result;
    }
    result.observed_mean = std::accumulate(
        paired_differences.begin(), paired_differences.end(), 0.0) /
        static_cast<double>(paired_differences.size());
    const double observed_standard_error =
        sample_standard_error(paired_differences, result.observed_mean);
    if (spec.studentized && !(observed_standard_error > 0.0)) {
        result.status = MultipleTestingStatus::NUMERICAL_FAILURE;
        return result;
    }
    result.observed_statistic = spec.studentized
        ? result.observed_mean / observed_standard_error
        : result.observed_mean;

    std::vector<double> centered(paired_differences.size());
    std::transform(paired_differences.begin(), paired_differences.end(), centered.begin(),
                   [&](double value) { return value - result.observed_mean; });
    std::vector<double> sample(paired_differences.size());
    SplitMix64 random(spec.seed);
    for (std::uint32_t repetition = 0; repetition < spec.resamples; ++repetition) {
        std::size_t written = 0;
        while (written < sample.size()) {
            const std::size_t start = random.next() % centered.size();
            for (std::uint32_t offset = 0;
                 offset < spec.block_length && written < sample.size(); ++offset) {
                sample[written++] = centered[(start + offset) % centered.size()];
            }
        }
        const double bootstrap_mean =
            std::accumulate(sample.begin(), sample.end(), 0.0) /
            static_cast<double>(sample.size());
        double statistic = bootstrap_mean;
        if (spec.studentized) {
            const double standard_error = sample_standard_error(sample, bootstrap_mean);
            statistic = standard_error > 0.0 ? bootstrap_mean / standard_error : 0.0;
        }
        if (statistic >= result.observed_statistic) ++result.exceedances;
    }
    result.one_sided_positive_p_value =
        (1.0 + static_cast<double>(result.exceedances)) /
        (1.0 + static_cast<double>(spec.resamples));
    result.replay_hash = kFnvOffset;
    hash_value(result.replay_hash, spec.config_hash);
    hash_value(result.replay_hash, spec.seed);
    hash_value(result.replay_hash, spec.block_length);
    hash_value(result.replay_hash, spec.resamples);
    hash_value(result.replay_hash,
               std::bit_cast<std::uint64_t>(result.one_sided_positive_p_value));
    result.status = MultipleTestingStatus::OK;
    return result;
}

}  // namespace portfolio_math

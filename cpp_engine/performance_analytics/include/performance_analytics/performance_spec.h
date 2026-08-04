#pragma once

#include <cstdint>
#include <string>

namespace performance_analytics {

enum class ReturnFrequency : std::uint8_t {
    MINUTE,
    DAILY,
    WEEKLY,
    MONTHLY,
    CUSTOM_SESSION,
};

enum class TurnoverConvention : std::uint8_t {
    ONE_WAY,
    TWO_WAY,
};

struct PerformanceSpecV1 {
    std::uint32_t schema_version{1};
    ReturnFrequency frequency{ReturnFrequency::DAILY};
    std::string calendar_id;
    double calendar_periods_per_year{0.0};
    double annual_risk_free_rate{0.0};
    std::string benchmark_id;
    TurnoverConvention turnover_convention{TurnoverConvention::ONE_WAY};
    bool annualize_turnover{false};
    double accounting_absolute_tolerance{1e-8};
    double accounting_relative_tolerance{1e-10};
    std::uint32_t minimum_return_observations{2};
    std::uint32_t minimum_tail_observations{20};
    std::uint64_t config_hash{0};
};

[[nodiscard]] bool valid_performance_spec(const PerformanceSpecV1& spec) noexcept;

}  // namespace performance_analytics

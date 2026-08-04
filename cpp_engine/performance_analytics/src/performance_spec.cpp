#include "performance_analytics/performance_spec.h"

#include <cmath>

namespace performance_analytics {

bool valid_performance_spec(const PerformanceSpecV1& spec) noexcept {
    return spec.schema_version == 1 && !spec.calendar_id.empty() &&
        std::isfinite(spec.calendar_periods_per_year) &&
        spec.calendar_periods_per_year > 0.0 &&
        std::isfinite(spec.annual_risk_free_rate) &&
        std::isfinite(spec.accounting_absolute_tolerance) &&
        spec.accounting_absolute_tolerance >= 0.0 &&
        std::isfinite(spec.accounting_relative_tolerance) &&
        spec.accounting_relative_tolerance >= 0.0 &&
        spec.minimum_return_observations >= 2 &&
        spec.minimum_tail_observations >= spec.minimum_return_observations &&
        spec.config_hash != 0;
}

}  // namespace performance_analytics

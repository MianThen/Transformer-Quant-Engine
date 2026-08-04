#include "performance_analytics/period_contribution_coordinator.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace performance_analytics {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_value(std::uint64_t& hash, std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffU;
        hash *= kFnvPrime;
    }
}

void hash_double(std::uint64_t& hash, double value) {
    const double normalized = value == 0.0 ? 0.0 : value;
    hash_value(hash, std::bit_cast<std::uint64_t>(normalized));
}

bool valid_security(const PeriodSecuritySnapshot& security) {
    return std::isfinite(security.mark_price) && security.mark_price > 0.0;
}

bool valid_fill(const ContributionFillInput& fill,
                engine_common::TimestampNs period_start) {
    const bool valid_side = fill.side == engine_common::Side::BUY ||
        fill.side == engine_common::Side::SELL;
    return fill.execution_id != 0 && valid_side && fill.quantity > 0 &&
        std::isfinite(fill.fill_price) && fill.fill_price > 0.0 &&
        std::isfinite(fill.explicit_fee) && fill.explicit_fee >= 0.0 &&
        fill.timestamp >= period_start;
}

bool valid_action(const ContributionCorporateActionInput& action,
                  engine_common::TimestampNs period_start) {
    return action.action_id != 0 && action.timestamp >= period_start &&
        std::isfinite(action.cash_amount);
}

bool add_quantity(engine_common::Quantity& total,
                  engine_common::Quantity increment) {
    if ((increment > 0 &&
         total > std::numeric_limits<engine_common::Quantity>::max() - increment) ||
        (increment < 0 &&
         total < std::numeric_limits<engine_common::Quantity>::min() - increment)) {
        return false;
    }
    total += increment;
    return true;
}

bool sort_and_validate_securities(std::vector<PeriodSecuritySnapshot>& securities) {
    std::sort(securities.begin(), securities.end(),
              [](const auto& left, const auto& right) {
        return left.symbol_id < right.symbol_id;
    });
    for (std::size_t index = 0; index < securities.size(); ++index) {
        if (!valid_security(securities[index]) ||
            (index > 0 &&
             securities[index - 1].symbol_id == securities[index].symbol_id)) {
            return false;
        }
    }
    return true;
}

struct WorkingSecurity {
    bool has_beginning{false};
    bool has_ending{false};
    PeriodSecuritySnapshot beginning;
    PeriodSecuritySnapshot ending;
    engine_common::Quantity corporate_action_quantity_delta{0};
    double corporate_action_cash{0.0};
};

}  // namespace

PeriodContributionCoordinator::PeriodContributionCoordinator(PerformanceSpecV1 spec)
    : ledger_(spec), coordinator_hash_(kFnvOffset) {
    hash_value(coordinator_hash_, spec.config_hash);
}

PeriodCoordinatorStatus PeriodContributionCoordinator::fail(
    PeriodCoordinatorStatus status) {
    beginning_securities_.clear();
    fills_.clear();
    corporate_actions_.clear();
    open_ = false;
    failed_ = true;
    last_status_ = status;
    return last_status_;
}

PeriodCoordinatorStatus PeriodContributionCoordinator::open_period(
    const PeriodOpenSnapshot& snapshot) {
    if (failed_) return PeriodCoordinatorStatus::FAILED;
    if (finalized_) return fail(PeriodCoordinatorStatus::FINALIZED);
    if (open_) return fail(PeriodCoordinatorStatus::PERIOD_ALREADY_OPEN);
    if (snapshot.period_start <= 0 || snapshot.session_id == 0 ||
        !std::isfinite(snapshot.starting_equity) ||
        snapshot.starting_equity <= 0.0 ||
        !std::isfinite(snapshot.starting_cash)) {
        return fail(PeriodCoordinatorStatus::INVALID_INPUT);
    }
    beginning_securities_.assign(snapshot.securities.begin(),
                                 snapshot.securities.end());
    if (!sort_and_validate_securities(beginning_securities_)) {
        return fail(PeriodCoordinatorStatus::INVALID_INPUT);
    }
    period_start_ = snapshot.period_start;
    session_id_ = snapshot.session_id;
    starting_equity_ = snapshot.starting_equity;
    starting_cash_ = snapshot.starting_cash;
    fills_.clear();
    corporate_actions_.clear();
    open_ = true;
    last_status_ = PeriodCoordinatorStatus::OK;
    return last_status_;
}

PeriodCoordinatorStatus PeriodContributionCoordinator::record_fill(
    const ContributionFillInput& fill) {
    if (failed_) return PeriodCoordinatorStatus::FAILED;
    if (finalized_) return fail(PeriodCoordinatorStatus::FINALIZED);
    if (!open_) return fail(PeriodCoordinatorStatus::NO_OPEN_PERIOD);
    if (!valid_fill(fill, period_start_)) {
        return fail(PeriodCoordinatorStatus::INVALID_INPUT);
    }
    if (std::find(execution_ids_.begin(), execution_ids_.end(), fill.execution_id) !=
        execution_ids_.end()) {
        return fail(PeriodCoordinatorStatus::DUPLICATE_EXECUTION);
    }
    execution_ids_.push_back(fill.execution_id);
    fills_.push_back(fill);
    last_status_ = PeriodCoordinatorStatus::OK;
    return last_status_;
}

PeriodCoordinatorStatus PeriodContributionCoordinator::record_corporate_action(
    const ContributionCorporateActionInput& action) {
    if (failed_) return PeriodCoordinatorStatus::FAILED;
    if (finalized_) return fail(PeriodCoordinatorStatus::FINALIZED);
    if (!open_) return fail(PeriodCoordinatorStatus::NO_OPEN_PERIOD);
    if (!valid_action(action, period_start_)) {
        return fail(PeriodCoordinatorStatus::INVALID_INPUT);
    }
    if (std::find(corporate_action_ids_.begin(), corporate_action_ids_.end(),
                  action.action_id) != corporate_action_ids_.end()) {
        return fail(PeriodCoordinatorStatus::DUPLICATE_CORPORATE_ACTION);
    }
    corporate_action_ids_.push_back(action.action_id);
    corporate_actions_.push_back(action);
    last_status_ = PeriodCoordinatorStatus::OK;
    return last_status_;
}

PeriodCoordinatorStatus PeriodContributionCoordinator::close_period(
    const PeriodCloseSnapshot& snapshot) {
    if (failed_) return PeriodCoordinatorStatus::FAILED;
    if (finalized_) return fail(PeriodCoordinatorStatus::FINALIZED);
    if (!open_) return fail(PeriodCoordinatorStatus::NO_OPEN_PERIOD);
    if (snapshot.period_end <= period_start_ ||
        !std::isfinite(snapshot.ending_equity) || snapshot.ending_equity < 0.0 ||
        !std::isfinite(snapshot.ending_cash) ||
        !std::isfinite(snapshot.cash_interest) ||
        !std::isfinite(snapshot.external_cash_flow)) {
        return fail(PeriodCoordinatorStatus::INVALID_INPUT);
    }
    if (std::any_of(fills_.begin(), fills_.end(), [&](const auto& fill) {
            return fill.timestamp > snapshot.period_end;
        }) ||
        std::any_of(corporate_actions_.begin(), corporate_actions_.end(),
                    [&](const auto& action) {
            return action.timestamp > snapshot.period_end;
        })) {
        return fail(PeriodCoordinatorStatus::INVALID_INPUT);
    }

    std::vector<PeriodSecuritySnapshot> ending_securities(
        snapshot.securities.begin(), snapshot.securities.end());
    if (!sort_and_validate_securities(ending_securities)) {
        return fail(PeriodCoordinatorStatus::INVALID_INPUT);
    }
    std::map<engine_common::SymbolId, WorkingSecurity> working;
    for (const auto& security : beginning_securities_) {
        auto& value = working[security.symbol_id];
        value.has_beginning = true;
        value.beginning = security;
    }
    for (const auto& security : ending_securities) {
        auto& value = working[security.symbol_id];
        value.has_ending = true;
        value.ending = security;
    }
    for (const auto& fill : fills_) {
        if (working.find(fill.symbol_id) == working.end()) {
            return fail(PeriodCoordinatorStatus::INVALID_INPUT);
        }
    }
    for (const auto& action : corporate_actions_) {
        const auto found = working.find(action.symbol_id);
        if (found == working.end()) {
            return fail(PeriodCoordinatorStatus::INVALID_INPUT);
        }
        if (!add_quantity(found->second.corporate_action_quantity_delta,
                          action.quantity_delta)) {
            return fail(PeriodCoordinatorStatus::INVALID_INPUT);
        }
        found->second.corporate_action_cash += action.cash_amount;
    }

    std::vector<SecurityContributionInput> securities;
    securities.reserve(working.size());
    for (const auto& [symbol_id, value] : working) {
        const auto beginning_quantity = value.has_beginning
            ? value.beginning.quantity : engine_common::Quantity{0};
        const auto ending_quantity = value.has_ending
            ? value.ending.quantity : engine_common::Quantity{0};
        if (!value.has_beginning && beginning_quantity != 0) {
            return fail(PeriodCoordinatorStatus::INVALID_INPUT);
        }
        if (!value.has_ending && ending_quantity != 0) {
            return fail(PeriodCoordinatorStatus::INVALID_INPUT);
        }
        securities.push_back({
            symbol_id,
            value.has_beginning ? value.beginning.pit_industry_id
                                : value.ending.pit_industry_id,
            beginning_quantity,
            ending_quantity,
            value.corporate_action_quantity_delta,
            value.has_beginning ? value.beginning.mark_price
                                : value.ending.mark_price,
            value.has_ending ? value.ending.mark_price
                             : value.beginning.mark_price,
            value.corporate_action_cash,
        });
    }
    const PeriodContributionInput input{
        period_start_, snapshot.period_end, session_id_,
        starting_equity_, snapshot.ending_equity,
        starting_cash_, snapshot.ending_cash,
        snapshot.cash_interest, snapshot.external_cash_flow,
        securities, fills_,
    };
    if (ledger_.append(input) != ContributionStatus::OK) {
        return fail(PeriodCoordinatorStatus::LEDGER_REJECTED);
    }

    std::sort(corporate_actions_.begin(), corporate_actions_.end(),
              [](const auto& left, const auto& right) {
        if (left.timestamp != right.timestamp) return left.timestamp < right.timestamp;
        return left.action_id < right.action_id;
    });
    std::uint64_t period_hash = kFnvOffset;
    hash_value(period_hash, static_cast<std::uint64_t>(period_start_));
    hash_value(period_hash, static_cast<std::uint64_t>(snapshot.period_end));
    hash_value(period_hash, session_id_);
    hash_value(period_hash, ledger_.records().back().record_hash);
    for (const auto& action : corporate_actions_) {
        hash_value(period_hash, action.action_id);
        hash_value(period_hash, action.symbol_id);
        hash_value(period_hash, static_cast<std::uint64_t>(action.timestamp));
        hash_value(period_hash, static_cast<std::uint64_t>(action.quantity_delta));
        hash_double(period_hash, action.cash_amount);
    }
    hash_value(coordinator_hash_, period_hash);
    corporate_action_history_.insert(corporate_action_history_.end(),
                                     corporate_actions_.begin(),
                                     corporate_actions_.end());
    beginning_securities_.clear();
    fills_.clear();
    corporate_actions_.clear();
    open_ = false;
    last_status_ = PeriodCoordinatorStatus::OK;
    return last_status_;
}

PeriodCoordinatorStatus PeriodContributionCoordinator::finalize(
    const PeriodCloseSnapshot& snapshot) {
    const auto status = close_period(snapshot);
    if (status != PeriodCoordinatorStatus::OK) return status;
    finalized_ = true;
    return last_status_;
}

}  // namespace performance_analytics

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "engine_common/model_types.h"
#include "engine_common/strategy.h"

namespace qbt::strategy {

class LongOnlyOrderPlanner {
public:
    explicit LongOnlyOrderPlanner(size_t maximum_intents);

    [[nodiscard]] std::span<const engine_common::OrderIntent> build(
        std::span<const engine_common::TargetPosition> targets,
        const engine_common::PortfolioView& portfolio,
        engine_common::TimestampNs timestamp,
        uint64_t decision_id);

private:
    size_t maximum_intents_;
    std::vector<engine_common::OrderIntent> intents_;
    std::vector<const engine_common::PortfolioItem*> portfolio_by_symbol_;
};

}  // namespace qbt::strategy

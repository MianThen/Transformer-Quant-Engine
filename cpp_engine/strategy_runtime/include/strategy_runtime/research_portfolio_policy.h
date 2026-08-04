#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "engine_common/model_types.h"
#include "engine_common/strategy.h"
#include "portfolio_math/risk_model.h"
#include "portfolio_math/types.h"
#include "strategy_runtime/portfolio_policy.h"

namespace qbt::strategy {

enum class ResearchPortfolioPolicyKind : std::uint8_t {
    TOPK_EQUAL_WEIGHT,
    RISK_BUDGET,
};

struct ResearchPortfolioPolicyConfig {
    ResearchPortfolioPolicyKind kind{ResearchPortfolioPolicyKind::TOPK_EQUAL_WEIGHT};
    LongOnlyPolicyConfig selection;
    double target_investment{1.0};
    std::uint64_t policy_config_hash{0};
};

struct ResearchPortfolioPolicyDiagnostics {
    portfolio_math::OptimizationStatus status{
        portfolio_math::OptimizationStatus::INVALID_INPUT};
    ResearchPortfolioPolicyKind policy{ResearchPortfolioPolicyKind::TOPK_EQUAL_WEIGHT};
    std::uint32_t selected_symbols{0};
    bool hold_current_weights{true};
    std::uint64_t risk_model_hash{0};
    std::uint64_t policy_config_hash{0};
    portfolio_math::OptimizationDiagnostics solver;
};

class ResearchPortfolioPolicy {
public:
    explicit ResearchPortfolioPolicy(ResearchPortfolioPolicyConfig config);

    std::span<const engine_common::TargetPosition> build(
        const engine_common::PredictionBatch& predictions,
        const engine_common::MarketFrameBatchView& market,
        const engine_common::PortfolioView& portfolio,
        const portfolio_math::DenseRiskModelView* risk_model = nullptr);

    [[nodiscard]] const ResearchPortfolioPolicyDiagnostics& diagnostics() const noexcept {
        return diagnostics_;
    }

private:
    ResearchPortfolioPolicyConfig config_;
    LongOnlyTopKPolicy topk_;
    std::vector<engine_common::TargetPosition> targets_;
    ResearchPortfolioPolicyDiagnostics diagnostics_;
};

}  // namespace qbt::strategy

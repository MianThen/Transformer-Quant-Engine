#include "strategy_runtime/research_portfolio_policy.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "portfolio_math/risk_budget.h"

namespace qbt::strategy {
namespace {

const engine_common::MarketBar *
find_bar(engine_common::SymbolId symbol,
         const engine_common::MarketFrameBatchView &market) {
  const auto found =
      std::find_if(market.bars.begin(), market.bars.end(),
                   [&](const auto &bar) { return bar.symbol_id == symbol; });
  return found == market.bars.end() ? nullptr : &*found;
}

bool valid_price(const engine_common::MarketBar *bar) {
  return bar != nullptr && std::isfinite(bar->close) && bar->close > 0.0 &&
         bar->lot_size > 0 &&
         (bar->flags & engine_common::MARKET_LISTED) != 0 &&
         (bar->flags & engine_common::MARKET_SUSPENDED) == 0;
}

} // namespace

ResearchPortfolioPolicy::ResearchPortfolioPolicy(
    ResearchPortfolioPolicyConfig config)
    : config_(config), topk_(config.selection) {
  if (!(config_.target_investment > 0.0 && config_.target_investment <= 1.0) ||
      !std::isfinite(config_.target_investment) ||
      config_.policy_config_hash == 0) {
    throw std::invalid_argument(
        "invalid research portfolio policy configuration");
  }
}

std::span<const engine_common::TargetPosition> ResearchPortfolioPolicy::build(
    const engine_common::PredictionBatch &predictions,
    const engine_common::MarketFrameBatchView &market,
    const engine_common::PortfolioView &portfolio,
    const portfolio_math::DenseRiskModelView *risk_model) {
  targets_.clear();
  diagnostics_ = {};
  diagnostics_.policy = config_.kind;
  diagnostics_.policy_config_hash = config_.policy_config_hash;
  if (config_.kind == ResearchPortfolioPolicyKind::TOPK_EQUAL_WEIGHT) {
    const auto targets = topk_.build(predictions, market, portfolio);
    targets_.assign(targets.begin(), targets.end());
    diagnostics_.status = portfolio_math::OptimizationStatus::OK;
    diagnostics_.selected_symbols = static_cast<std::uint32_t>(targets_.size());
    diagnostics_.hold_current_weights = false;
    return targets_;
  }

  if (!predictions.valid_shape() ||
      predictions.asof_timestamp != market.asof_timestamp ||
      !market.data_trusted || !(portfolio.equity > 0.0) ||
      risk_model == nullptr ||
      !portfolio_math::valid_dense_risk_model(*risk_model,
                                              predictions.asof_timestamp)) {
    return targets_;
  }
  diagnostics_.risk_model_hash = risk_model->artifact_hash;
  std::unordered_map<engine_common::SymbolId, std::size_t> risk_index;
  risk_index.reserve(risk_model->symbols.size());
  for (std::size_t index = 0; index < risk_model->symbols.size(); ++index) {
    risk_index.emplace(risk_model->symbols[index], index);
  }
  for (const auto &item : portfolio.items) {
    if (item.position_quantity != 0 && risk_index.count(item.symbol_id) == 0) {
      return targets_;
    }
  }

  std::vector<const engine_common::ModelPrediction *> ranked;
  std::unordered_set<engine_common::SymbolId> seen_predictions;
  ranked.reserve(predictions.size);
  for (std::size_t index = 0; index < predictions.size; ++index) {
    const auto &prediction = predictions.values[index];
    if (!seen_predictions.insert(prediction.symbol_id).second)
      return targets_;
    const double score = ranking_score(prediction, config_.selection);
    const double minimum_score =
        std::isnan(config_.selection.minimum_ranking_score)
            ? config_.selection.minimum_expected_return
            : config_.selection.minimum_ranking_score;
    if ((prediction.flags & engine_common::PREDICTION_VALID) == 0 ||
        !std::isfinite(score) || score < minimum_score ||
        prediction.confidence < config_.selection.minimum_confidence) {
      continue;
    }
    if (risk_index.count(prediction.symbol_id) == 0 ||
        !valid_price(find_bar(prediction.symbol_id, market))) {
      return targets_;
    }
    ranked.push_back(&prediction);
  }
  std::sort(
      ranked.begin(), ranked.end(), [&](const auto *left, const auto *right) {
        const double left_score = ranking_score(*left, config_.selection);
        const double right_score = ranking_score(*right, config_.selection);
        if (left_score != right_score) {
          return left_score > right_score;
        }
        return left->symbol_id < right->symbol_id;
      });
  if (ranked.size() > config_.selection.max_positions) {
    ranked.resize(config_.selection.max_positions);
  }
  diagnostics_.selected_symbols = static_cast<std::uint32_t>(ranked.size());
  if (ranked.empty())
    return targets_;

  quant_math::DenseMatrix covariance(static_cast<Eigen::Index>(ranked.size()),
                                     static_cast<Eigen::Index>(ranked.size()));
  std::vector<double> budgets(ranked.size(),
                              1.0 / static_cast<double>(ranked.size()));
  std::vector<double> lower_bounds(ranked.size(), 0.0);
  std::vector<double> upper_bounds(
      ranked.size(),
      static_cast<double>(config_.selection.max_position_weight));
  std::vector<double> current_weights(ranked.size(), 0.0);
  for (std::size_t row = 0; row < ranked.size(); ++row) {
    const auto source_row = risk_index.at(ranked[row]->symbol_id);
    for (std::size_t col = 0; col < ranked.size(); ++col) {
      covariance(static_cast<Eigen::Index>(row),
                 static_cast<Eigen::Index>(col)) =
          risk_model->covariance(source_row,
                                 risk_index.at(ranked[col]->symbol_id));
    }
    const auto current = std::find_if(
        portfolio.items.begin(), portfolio.items.end(), [&](const auto &item) {
          return item.symbol_id == ranked[row]->symbol_id;
        });
    if (current != portfolio.items.end()) {
      const auto *bar = find_bar(ranked[row]->symbol_id, market);
      current_weights[row] = static_cast<double>(current->position_quantity) *
                             bar->close / portfolio.equity;
    }
  }
  portfolio_math::RiskBudgetOptions options;
  options.target_investment = config_.target_investment;
  auto solved = portfolio_math::solve_bounded_long_only_risk_budget(
      quant_math::view(covariance), budgets, lower_bounds, upper_bounds,
      current_weights, {}, options);
  diagnostics_.solver = solved.diagnostics;
  diagnostics_.status = solved.diagnostics.status;
  if (solved.diagnostics.status != portfolio_math::OptimizationStatus::OK) {
    return targets_;
  }

  std::unordered_set<engine_common::SymbolId> selected;
  selected.reserve(ranked.size());
  for (const auto *prediction : ranked)
    selected.insert(prediction->symbol_id);
  for (const auto &item : portfolio.items) {
    if (item.position_quantity != 0 && selected.count(item.symbol_id) == 0) {
      targets_.push_back({item.symbol_id, 0, 0.0F});
    }
  }
  for (std::size_t index = 0; index < ranked.size(); ++index) {
    const auto *bar = find_bar(ranked[index]->symbol_id, market);
    const double notional = portfolio.equity * solved.weights[index];
    const auto raw_quantity =
        static_cast<engine_common::Quantity>(std::floor(notional / bar->close));
    const auto quantity = raw_quantity - raw_quantity % bar->lot_size;
    targets_.push_back({ranked[index]->symbol_id,
                        std::max<engine_common::Quantity>(quantity, 0),
                        static_cast<float>(solved.weights[index])});
  }
  std::sort(targets_.begin(), targets_.end(),
            [](const auto &left, const auto &right) {
              return left.symbol_id < right.symbol_id;
            });
  diagnostics_.hold_current_weights = false;
  return targets_;
}

} // namespace qbt::strategy

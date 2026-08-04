#include <array>
#include <cstdio>
#include <utility>
#include <vector>

#include "portfolio_math/policy_to_order.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

portfolio_math::SinglePeriodReconcilerResult reconciled(
    std::vector<double> weights) {
  portfolio_math::SinglePeriodReconcilerResult result;
  result.target_weights = std::move(weights);
  result.diagnostics.status = portfolio_math::OptimizationStatus::OK;
  return result;
}

std::array<engine_common::MarketBar, 2> market() {
  std::array<engine_common::MarketBar, 2> bars{};
  bars[0].symbol_id = 1;
  bars[0].timestamp = 100;
  bars[0].close = 10.0;
  bars[0].lot_size = 5;
  bars[0].flags = engine_common::MARKET_LISTED |
      engine_common::MARKET_DATA_TRUSTED;
  bars[1].symbol_id = 2;
  bars[1].timestamp = 100;
  bars[1].close = 20.0;
  bars[1].lot_size = 10;
  bars[1].flags = engine_common::MARKET_LISTED |
      engine_common::MARKET_DATA_TRUSTED;
  return bars;
}

bool test_weight_to_orders() {
  const auto bars = market();
  const std::array<engine_common::SymbolId, 2> symbols{1, 2};
  const std::array<engine_common::PortfolioItem, 2> portfolio{
      engine_common::PortfolioItem{1, 40, 40, 0.0, 10.0, 0, 0},
      engine_common::PortfolioItem{2, 30, 30, 0.0, 20.0, 0, 0}};
  std::array<engine_common::OrderIntent, 4> buffer{};
  engine_common::OrderIntentBuffer output{buffer, 0};
  portfolio_math::PolicyOrderOptions options;
  options.decision_at = 100;
  options.max_order_quantity = 1'000;
  const auto result = portfolio_math::build_policy_order_intents(
      reconciled({0.6, 0.4}), symbols, portfolio, bars, 1'000.0, output,
      options);
  bool ok = check(result.diagnostics.status == portfolio_math::PolicyOrderStatus::OK,
                  "policy-to-order status");
  ok &= check(output.size == 2 && result.diagnostics.emitted_order_count == 2,
              "policy-to-order count");
  ok &= check(output.values[0].symbol_id == 1 &&
                  output.values[0].side == engine_common::Side::BUY &&
                  output.values[0].quantity == 20,
              "policy-to-order buy");
  ok &= check(output.values[1].symbol_id == 2 &&
                  output.values[1].side == engine_common::Side::SELL &&
                  output.values[1].quantity == 10,
              "policy-to-order sell");
  ok &= check(result.diagnostics.buy_quantity == 20 &&
                  result.diagnostics.sell_quantity == 10,
              "policy-to-order diagnostics");
  return ok;
}

bool test_quantization_and_liquidation() {
  auto bars = market();
  bars[0].close = 11.0;
  const std::array<engine_common::SymbolId, 1> symbols{1};
  const std::array<engine_common::PortfolioItem, 2> portfolio{
      engine_common::PortfolioItem{1, 0, 0, 0.0, 11.0, 0, 0},
      engine_common::PortfolioItem{2, 30, 30, 0.0, 20.0, 0, 0}};
  std::array<engine_common::OrderIntent, 4> buffer{};
  engine_common::OrderIntentBuffer output{buffer, 0};
  portfolio_math::PolicyOrderOptions options;
  options.decision_at = 100;
  const auto result = portfolio_math::build_policy_order_intents(
      reconciled({0.63}), symbols, portfolio, bars, 1'000.0, output, options);
  bool ok = check(result.diagnostics.status == portfolio_math::PolicyOrderStatus::OK,
                  "quantization status");
  ok &= check(output.size == 2 && output.values[0].symbol_id == 1 &&
                  output.values[0].quantity == 55 &&
                  output.values[1].symbol_id == 2 &&
                  output.values[1].side == engine_common::Side::SELL &&
                  output.values[1].quantity == 30,
              "lot quantization and liquidation");
  return ok;
}

bool test_failure_closure() {
  const auto bars = market();
  const std::array<engine_common::SymbolId, 2> symbols{1, 2};
  const std::array<engine_common::PortfolioItem, 0> portfolio{};
  std::array<engine_common::OrderIntent, 2> buffer{};
  engine_common::OrderIntentBuffer output{buffer, 0};
  engine_common::OrderIntent existing;
  (void)output.push(existing);
  portfolio_math::PolicyOrderOptions options;
  options.decision_at = 100;
  auto failed = reconciled({0.5, 0.5});
  failed.diagnostics.status = portfolio_math::OptimizationStatus::INFEASIBLE;
  bool ok = check(portfolio_math::build_policy_order_intents(
                      failed, symbols, portfolio, bars, 1'000.0, output, options)
                      .diagnostics.status == portfolio_math::PolicyOrderStatus::RECONCILER_FAILED &&
                  output.size == 0,
              "reconciler failure closes orders");

  auto untrusted_bars = bars;
  untrusted_bars[0].flags = engine_common::MARKET_LISTED;
  (void)output.push(existing);
  const auto untrusted_result = portfolio_math::build_policy_order_intents(
      reconciled({0.5, 0.5}), symbols, portfolio, untrusted_bars, 1'000.0,
      output, options);
  ok &= check(untrusted_result.diagnostics.status ==
                  portfolio_math::PolicyOrderStatus::DATA_UNTRUSTED &&
                  output.size == 0,
              "untrusted market closes orders");

  const std::array<engine_common::PortfolioItem, 2> locked_portfolio{
      engine_common::PortfolioItem{1, 100, 0, 0.0, 10.0, 0, 0},
      engine_common::PortfolioItem{2, 0, 0, 0.0, 20.0, 0, 0}};
  ok &= check(portfolio_math::build_policy_order_intents(
                  reconciled({0.0, 1.0}), symbols, locked_portfolio, bars,
                  1'000.0, output, options)
                  .diagnostics.status == portfolio_math::PolicyOrderStatus::INSUFFICIENT_POSITION &&
                  output.size == 0,
              "locked position closes orders");
  return ok;
}

}  // namespace

int main() {
  if (!(test_weight_to_orders() && test_quantization_and_liquidation() &&
        test_failure_closure())) {
    return 1;
  }
  std::printf("test_policy_to_order: all checks passed\n");
  return 0;
}

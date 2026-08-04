#include "portfolio_math/policy_to_order.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace portfolio_math {
namespace {

struct QuantityState {
  const engine_common::PortfolioItem* item{nullptr};
};

bool add_quantity(engine_common::Quantity left,
                  engine_common::Quantity right,
                  engine_common::Quantity& result) {
  if ((right > 0 && left > std::numeric_limits<engine_common::Quantity>::max() - right) ||
      (right < 0 && left < std::numeric_limits<engine_common::Quantity>::min() - right)) {
    return false;
  }
  result = left + right;
  return true;
}

bool valid_market_bar(const engine_common::MarketBar& bar,
                      const PolicyOrderOptions& options) {
  return bar.timestamp == options.decision_at &&
      std::isfinite(bar.close) && bar.close > 0.0 && bar.lot_size > 0 &&
      (bar.flags & engine_common::MARKET_LISTED) != 0 &&
      (bar.flags & engine_common::MARKET_SUSPENDED) == 0 &&
      (!options.require_trusted_market ||
       (bar.flags & engine_common::MARKET_DATA_TRUSTED) != 0);
}

const engine_common::MarketBar* find_market(
    engine_common::SymbolId symbol,
    std::span<const engine_common::MarketBar> market) {
  const auto found = std::find_if(
      market.begin(), market.end(), [&](const auto& bar) {
        return bar.symbol_id == symbol;
      });
  return found == market.end() ? nullptr : &*found;
}

const engine_common::PortfolioItem* find_portfolio(
    engine_common::SymbolId symbol,
    const std::unordered_map<engine_common::SymbolId, QuantityState>& states) {
  const auto found = states.find(symbol);
  return found == states.end() ? nullptr : found->second.item;
}

PolicyOrderResult fail(engine_common::OrderIntentBuffer& output,
                       PolicyOrderStatus status,
                       std::uint32_t target_count) {
  output.clear();
  PolicyOrderResult result;
  result.diagnostics.status = status;
  result.diagnostics.target_count = target_count;
  return result;
}

}  // namespace

PolicyOrderResult build_policy_order_intents(
    const SinglePeriodReconcilerResult& reconciled,
    std::span<const engine_common::SymbolId> symbols,
    std::span<const engine_common::PortfolioItem> portfolio,
    std::span<const engine_common::MarketBar> market,
    double equity,
    engine_common::OrderIntentBuffer& output,
    PolicyOrderOptions options) {
  output.clear();
  const auto target_count = static_cast<std::uint32_t>(symbols.size());
  if (reconciled.diagnostics.status != OptimizationStatus::OK) {
    return fail(output, PolicyOrderStatus::RECONCILER_FAILED, target_count);
  }
  if (symbols.empty() || reconciled.target_weights.size() != symbols.size() ||
      !std::isfinite(equity) || equity <= 0.0 || options.decision_at <= 0 ||
      options.max_order_quantity <= 0) {
    return fail(output, PolicyOrderStatus::INVALID_INPUT, target_count);
  }

  std::unordered_set<engine_common::SymbolId> seen_symbols;
  seen_symbols.reserve(symbols.size());
  double target_sum = 0.0;
  for (std::size_t index = 0; index < symbols.size(); ++index) {
    if (!seen_symbols.insert(symbols[index]).second ||
        !std::isfinite(reconciled.target_weights[index]) ||
        reconciled.target_weights[index] < 0.0 ||
        reconciled.target_weights[index] > 1.0) {
      return fail(output, PolicyOrderStatus::INVALID_INPUT, target_count);
    }
    target_sum += reconciled.target_weights[index];
  }
  if (!std::isfinite(target_sum) || target_sum > 1.0 + 1e-8) {
    return fail(output, PolicyOrderStatus::INVALID_INPUT, target_count);
  }

  std::unordered_map<engine_common::SymbolId, QuantityState> states;
  states.reserve(portfolio.size());
  for (const auto& item : portfolio) {
    if (item.position_quantity < 0 || item.sellable_quantity < 0 ||
        item.active_buy_quantity < 0 ||
        item.active_sell_quantity < 0 ||
        item.active_sell_quantity > item.position_quantity ||
        !states.emplace(item.symbol_id, QuantityState{&item}).second) {
      return fail(output, PolicyOrderStatus::INVALID_INPUT, target_count);
    }
  }
  std::unordered_set<engine_common::SymbolId> seen_market_symbols;
  seen_market_symbols.reserve(market.size());
  for (const auto& bar : market) {
    if (!std::isfinite(bar.close) || bar.close <= 0.0 || bar.lot_size <= 0) {
      return fail(output, PolicyOrderStatus::INVALID_INPUT, target_count);
    }
    if (!seen_market_symbols.insert(bar.symbol_id).second) {
      return fail(output, PolicyOrderStatus::INVALID_INPUT, target_count);
    }
  }

  std::vector<engine_common::OrderIntent> intents;
  intents.reserve(symbols.size() + portfolio.size());
  PolicyOrderResult result;
  result.diagnostics.status = PolicyOrderStatus::OK;
  result.diagnostics.target_count = target_count;

  for (std::size_t index = 0; index < symbols.size(); ++index) {
    const auto symbol = symbols[index];
    const auto* current = find_portfolio(symbol, states);
    const auto current_quantity = current == nullptr ? 0 : current->position_quantity;
    const auto active_buy_quantity = current == nullptr ? 0 : current->active_buy_quantity;
    const auto active_sell_quantity = current == nullptr ? 0 : current->active_sell_quantity;
    engine_common::Quantity projected_quantity = 0;
    if (!add_quantity(current_quantity, active_buy_quantity, projected_quantity) ||
        !add_quantity(projected_quantity, -active_sell_quantity, projected_quantity) ||
        projected_quantity < 0) {
      return fail(output, PolicyOrderStatus::INVALID_INPUT, target_count);
    }

    const auto* bar = find_market(symbol, market);
    if (reconciled.target_weights[index] == 0.0 && projected_quantity == 0) {
      continue;
    }
    if (bar == nullptr) {
      return fail(output, PolicyOrderStatus::INVALID_INPUT, target_count);
    }
    if (!valid_market_bar(*bar, options)) {
      const auto status = options.require_trusted_market &&
          (bar->flags & engine_common::MARKET_DATA_TRUSTED) == 0
          ? PolicyOrderStatus::DATA_UNTRUSTED
          : PolicyOrderStatus::NOT_TRADABLE;
      return fail(output, status, target_count);
    }

    const long double target_notional =
        static_cast<long double>(equity) * reconciled.target_weights[index];
    const long double raw_quantity = std::floor(target_notional / bar->close);
    if (!(raw_quantity >= 0.0L) ||
        raw_quantity > static_cast<long double>(std::numeric_limits<engine_common::Quantity>::max())) {
      return fail(output, PolicyOrderStatus::INVALID_QUANTITY, target_count);
    }
    const auto integer_quantity = static_cast<engine_common::Quantity>(raw_quantity);
    const auto target_quantity = integer_quantity - integer_quantity % bar->lot_size;
    const auto difference = target_quantity - projected_quantity;
    if (difference == 0) continue;
    const bool buy = difference > 0;
    const auto order_quantity = buy ? difference : -difference;
    if (order_quantity <= 0) {
      return fail(output, PolicyOrderStatus::INVALID_QUANTITY, target_count);
    }
    if (order_quantity > options.max_order_quantity) {
      return fail(output, PolicyOrderStatus::ORDER_TOO_LARGE, target_count);
    }
    if (!buy) {
      const auto available_to_sell = current == nullptr
          ? 0
          : current->sellable_quantity - current->active_sell_quantity;
      if (available_to_sell < 0 || order_quantity > available_to_sell) {
        return fail(output, PolicyOrderStatus::INSUFFICIENT_POSITION, target_count);
      }
    }
    engine_common::OrderIntent intent;
    intent.symbol_id = symbol;
    intent.side = buy ? engine_common::Side::BUY : engine_common::Side::SELL;
    intent.type = engine_common::OrderType::MARKET;
    intent.time_in_force = engine_common::TimeInForce::DAY;
    intent.quantity = order_quantity;
    intent.timestamp = options.decision_at;
    intents.push_back(intent);
    if (buy) {
      result.diagnostics.buy_quantity += order_quantity;
    } else {
      result.diagnostics.sell_quantity += order_quantity;
    }
  }

  for (const auto& item : portfolio) {
    if (item.position_quantity == 0) continue;
    if (seen_symbols.count(item.symbol_id) != 0) continue;
    const auto* bar = find_market(item.symbol_id, market);
    if (bar == nullptr) return fail(output, PolicyOrderStatus::INVALID_INPUT, target_count);
    if (!valid_market_bar(*bar, options)) {
      const auto status = options.require_trusted_market &&
          (bar->flags & engine_common::MARKET_DATA_TRUSTED) == 0
          ? PolicyOrderStatus::DATA_UNTRUSTED
          : PolicyOrderStatus::NOT_TRADABLE;
      return fail(output, status, target_count);
    }
    const auto sell_quantity = item.position_quantity - item.active_sell_quantity;
    if (sell_quantity <= 0 || sell_quantity > item.sellable_quantity) continue;
    if (sell_quantity > options.max_order_quantity) {
      return fail(output, PolicyOrderStatus::ORDER_TOO_LARGE, target_count);
    }
    engine_common::OrderIntent intent;
    intent.symbol_id = item.symbol_id;
    intent.side = engine_common::Side::SELL;
    intent.type = engine_common::OrderType::MARKET;
    intent.time_in_force = engine_common::TimeInForce::DAY;
    intent.quantity = sell_quantity - sell_quantity % bar->lot_size;
    intent.timestamp = options.decision_at;
    if (intent.quantity > 0) {
      intents.push_back(intent);
      result.diagnostics.sell_quantity += intent.quantity;
    }
  }

  if (output.values.size() < intents.size()) {
    return fail(output, PolicyOrderStatus::OUTPUT_OVERFLOW, target_count);
  }
  for (const auto& intent : intents) {
    if (!output.push(intent)) return fail(output, PolicyOrderStatus::OUTPUT_OVERFLOW, target_count);
  }
  result.diagnostics.emitted_order_count = static_cast<std::uint32_t>(intents.size());
  return result;
}

}  // namespace portfolio_math

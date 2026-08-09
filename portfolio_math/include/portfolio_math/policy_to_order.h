#pragma once

#include <cstdint>
#include <span>

#include "engine_common/strategy.h"
#include "portfolio_math/reconciler.h"

namespace portfolio_math {

enum class PolicyOrderStatus : std::uint8_t {
  OK,
  INVALID_INPUT,
  RECONCILER_FAILED,
  DATA_UNTRUSTED,
  NOT_TRADABLE,
  INSUFFICIENT_POSITION,
  INVALID_QUANTITY,
  ORDER_TOO_LARGE,
  OUTPUT_OVERFLOW,
};

struct PolicyOrderOptions {
  engine_common::TimestampNs decision_at{0};
  engine_common::Quantity max_order_quantity{1'000'000};
  bool require_trusted_market{true};
};

struct PolicyOrderDiagnostics {
  PolicyOrderStatus status{PolicyOrderStatus::INVALID_INPUT};
  std::uint32_t target_count{0};
  std::uint32_t emitted_order_count{0};
  engine_common::Quantity buy_quantity{0};
  engine_common::Quantity sell_quantity{0};
  bool reference_price_proxy{true};
};

struct PolicyOrderResult {
  PolicyOrderDiagnostics diagnostics;
};

[[nodiscard]] PolicyOrderResult build_policy_order_intents(
    const SinglePeriodReconcilerResult& reconciled,
    std::span<const engine_common::SymbolId> symbols,
    std::span<const engine_common::PortfolioItem> portfolio,
    std::span<const engine_common::MarketBar> market,
    double equity,
    engine_common::OrderIntentBuffer& output,
    PolicyOrderOptions options = {});

}  // namespace portfolio_math

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"
#include "performance_analytics/period_contribution_replay_sink.h"
#include "portfolio_math/policy_to_order.h"
#include "portfolio_math/reconciler.h"
#include "portfolio_math/tail_risk.h"

namespace {

bool check(bool condition, const char* message) {
  if (!condition) std::fprintf(stderr, "FAILED: %s\n", message);
  return condition;
}

qbt::MarketSnapshot bar(const char* symbol, qbt::Timestamp timestamp,
                        double close) {
  qbt::MarketSnapshot value;
  value.symbol = symbol;
  value.timestamp = timestamp;
  value.open = close;
  value.high = close;
  value.low = close;
  value.close = close;
  value.volume = 100'000;
  value.lot_size = 1;
  value.industry = symbol[0] == 'A' ? "industry_a" : "industry_b";
  return value;
}

performance_analytics::PerformanceSpecV1 spec() {
  performance_analytics::PerformanceSpecV1 value;
  value.frequency = performance_analytics::ReturnFrequency::DAILY;
  value.calendar_id = "XSHG_PHASE2A_FIXTURE_V1";
  value.calendar_periods_per_year = 242.0;
  value.benchmark_id = "INTERNAL_FROZEN_CONTROL";
  value.config_hash = 2'026'080'300;
  return value;
}

std::vector<qbt::Order> build_orders(
    qbt::BacktestEngine& engine,
    std::span<const qbt::MarketSnapshot> batch,
    qbt::Timestamp timestamp) {
  const std::array<engine_common::SymbolId, 2> symbols{0, 1};
  std::array<engine_common::MarketBar, 2> market{};
  std::array<engine_common::PortfolioItem, 2> portfolio{};
  std::array<double, 2> current_weights{};
  const double equity = engine.get_equity();
  for (std::size_t index = 0; index < batch.size(); ++index) {
    const auto& value = batch[index];
    market[index] = {
        symbols[index], timestamp, value.open, value.high, value.low,
        value.close, value.open, value.high, value.low, value.close,
        value.volume,
        value.lot_size,
        engine_common::MARKET_LISTED | engine_common::MARKET_DATA_TRUSTED};
    const auto position = engine.get_position(value.symbol);
    portfolio[index] = {
        symbols[index], position.quantity, position.sellable_quantity,
        position.avg_cost, value.close, 0, 0};
    current_weights[index] = equity > 0.0
        ? static_cast<double>(position.quantity) * value.close / equity
        : 0.0;
  }

  portfolio_math::SinglePeriodReconcilerOptions reconciler_options;
  reconciler_options.reconciler_spec_hash = 2'026'080'301;
  reconciler_options.costs_available = true;
  reconciler_options.max_single_weight = 1.0;
  const std::array<double, 2> anchor{0.6, 0.4};
  const std::array<double, 2> penalty{1.0, 1.0};
  const std::array<double, 2> linear_cost{0.001, 0.001};
  const std::array<double, 2> quadratic_impact{0.01, 0.01};
  const std::array<std::uint32_t, 2> group_ids{0, 1};
  const std::array<double, 2> group_caps{0.6, 0.6};
  const std::array<double, 2> max_trade_weights{1.0, 1.0};
  const portfolio_math::SinglePeriodConstraintView constraints{
      group_ids, group_caps, max_trade_weights};
  const auto reconciled = portfolio_math::reconcile_single_period(
      anchor, current_weights, penalty, linear_cost, quadratic_impact,
      constraints, reconciler_options);
  if (reconciled.diagnostics.status != portfolio_math::OptimizationStatus::OK) {
    return {};
  }

  std::array<engine_common::OrderIntent, 4> intent_storage{};
  engine_common::OrderIntentBuffer intents{intent_storage, 0};
  portfolio_math::PolicyOrderOptions order_options;
  order_options.decision_at = timestamp;
  order_options.max_order_quantity = 10'000;
  const auto order_result = portfolio_math::build_policy_order_intents(
      reconciled, symbols, portfolio, market, equity, intents, order_options);
  if (order_result.diagnostics.status != portfolio_math::PolicyOrderStatus::OK) {
    return {};
  }
  std::vector<qbt::Order> orders;
  orders.reserve(intents.size);
  const std::array<std::string, 2> names{"AAA", "BBB"};
  for (std::size_t index = 0; index < intents.size; ++index) {
    const auto& intent = intents.values[index];
    qbt::Order order;
    order.symbol = names[intent.symbol_id];
    order.side = intent.side == engine_common::Side::BUY
        ? qbt::Side::BUY : qbt::Side::SELL;
    order.type = qbt::OrderType::MARKET;
    order.quantity = intent.quantity;
    order.timestamp = intent.timestamp;
    orders.push_back(std::move(order));
  }
  return orders;
}

}  // namespace

int main() {
  qbt::ExecutionConfig execution;
  execution.enforce_t_plus_one = false;
  execution.max_volume_participation = 1.0;
  qbt::BacktestEngine engine(1'000.0, qbt::FillTiming::CLOSE, execution);
  auto sink = std::make_shared<performance_analytics::PeriodContributionReplaySink>(
      spec());
  engine.set_replay_analytics_sink(sink);
  engine.set_commission_fn([](double notional, bool) {
    return notional * 0.001;
  });

  bool first_callback = true;
  engine.set_on_cross_section(
      [&](const std::vector<qbt::MarketSnapshot>& batch) {
        if (first_callback) {
          first_callback = false;
          return std::vector<qbt::Order>{};
        }
        const auto timestamp = batch.front().timestamp;
        if (timestamp != 2) return std::vector<qbt::Order>{};
        return build_orders(engine, batch, timestamp);
      });

  const std::array<std::array<double, 2>, 6> prices{{
      {10.0, 20.0}, {10.5, 19.8}, {11.0, 20.2},
      {10.8, 20.5}, {11.3, 19.9}, {11.0, 20.1}}};
  for (std::size_t index = 0; index < prices.size(); ++index) {
    const auto timestamp = static_cast<qbt::Timestamp>(index + 1);
    engine.process_market_data_batch({
        bar("AAA", timestamp, prices[index][0]),
        bar("BBB", timestamp, prices[index][1])});
    if (index == 0) {
      engine.open_performance_period(1, timestamp);
    } else {
      engine.close_performance_period(index, timestamp);
      if (index + 1 < prices.size()) {
        engine.open_performance_period(index + 1, timestamp);
      }
    }
  }
  engine.finalize(6);

  bool ok = check(!sink->failed() && sink->return_ledger().records().size() == 5,
                  "policy replay return ledger");
  const auto records = sink->return_ledger().records();
  ok &= check(engine.get_order_count() == 2 && engine.get_trade_count() == 2,
              "policy replay order and fill path");
  ok &= check(std::all_of(records.begin(), records.end(), [](const auto& record) {
                return std::abs(record.accounting_residual) <= 1e-10;
              }),
              "policy replay accounting residual");
  ok &= check(std::any_of(records.begin(), records.end(), [](const auto& record) {
                return record.explicit_fees > 0.0;
              }),
              "policy replay explicit fees");

  std::vector<double> returns;
  std::vector<engine_common::TimestampNs> timestamps;
  returns.reserve(records.size());
  timestamps.reserve(records.size());
  for (const auto& record : records) {
    returns.push_back(record.period_return);
    timestamps.push_back(record.period_end);
  }
  const std::array<engine_common::SymbolId, 2> symbols{0, 1};
  const std::array<double, 2> weights{0.6, 0.4};
  portfolio_math::TailRiskSpec tail_spec;
  tail_spec.confidence_level = 0.8;
  tail_spec.config_hash = 2'026'080'302;
  const portfolio_math::TailRiskProblemView problem{
      6, symbols, timestamps, weights, returns, {}, {}, {}, {}, nullptr,
      tail_spec};
  const auto tail = portfolio_math::estimate_tail_risk(problem);
  ok &= check(tail.status == portfolio_math::TailRiskStatus::OK &&
                  tail.return_cvar.has_value(),
              "policy replay CVaR");
  const auto tail_artifact = portfolio_math::serialize_tail_risk_artifact(
      tail, tail_spec,
      {"PHASE2A_CPP_POLICY_REPLAY_FIXTURE", "", "", "PROXY", false,
       {"REFERENCE_PRICE_PROXY_BAR_CLOSE", "FIXED_COMMISSION_PROXY"}});
  ok &= check(tail_artifact.find("\"promotion_eligible\":false") !=
                  std::string::npos,
              "policy replay CVaR promotion gate");
  if (!ok) return 1;
  std::printf("test_phase2a_policy_replay: all checks passed\n");
  return 0;
}

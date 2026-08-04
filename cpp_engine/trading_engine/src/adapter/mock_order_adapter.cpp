#include "adapter/mock_order_adapter.h"

#include <stdexcept>
#include <utility>

namespace te {

namespace {

engine_common::ExecutionStatus to_execution_status(OrderStatus status) {
    using engine_common::ExecutionStatus;
    switch (status) {
        case OrderStatus::NEW: return ExecutionStatus::NEW;
        case OrderStatus::PENDING: return ExecutionStatus::PENDING_NEW;
        case OrderStatus::ACK: return ExecutionStatus::ACKNOWLEDGED;
        case OrderStatus::PARTIALLY_FILLED: return ExecutionStatus::PARTIALLY_FILLED;
        case OrderStatus::CANCEL_PENDING: return ExecutionStatus::PENDING_CANCEL;
        case OrderStatus::REPLACE_PENDING: return ExecutionStatus::PENDING_REPLACE;
        case OrderStatus::FILLED:
        case OrderStatus::DONE: return ExecutionStatus::FILLED;
        case OrderStatus::CANCELED: return ExecutionStatus::CANCELED;
        case OrderStatus::REJECTED: return ExecutionStatus::REJECTED;
        case OrderStatus::EXPIRED: return ExecutionStatus::EXPIRED;
        case OrderStatus::SUSPENDED: return ExecutionStatus::SUSPENDED;
        case OrderStatus::RECOVERING: return ExecutionStatus::RECOVERING;
        case OrderStatus::UNKNOWN: return ExecutionStatus::UNKNOWN;
    }
    return ExecutionStatus::UNKNOWN;
}

}  // namespace

MockOrderAdapter::MockOrderAdapter(SymbolResolver symbol_resolver,
                                   GatewayConfig config)
    : symbol_resolver_(std::move(symbol_resolver)), gateway_(std::move(config)) {
    if (!symbol_resolver_) throw std::invalid_argument("symbol resolver is required");
    gateway_.set_on_exec_report([this](const ExecReport& report) { on_report(report); });
}

bool MockOrderAdapter::connect(const std::string& endpoint, uint16_t port) {
    return gateway_.connect(endpoint, port);
}

int64_t MockOrderAdapter::submit(const engine_common::OrderIntent& intent) {
    return gateway_.send_order(to_gateway_request(intent));
}

bool MockOrderAdapter::cancel(int64_t client_order_id) {
    return gateway_.cancel_order(client_order_id);
}

size_t MockOrderAdapter::poll() { return gateway_.poll_once(); }
bool MockOrderAdapter::ready() const { return gateway_.ready(); }

void MockOrderAdapter::set_on_execution(OnExecutionEvent callback) {
    on_execution_ = std::move(callback);
}

NewOrderRequest MockOrderAdapter::to_gateway_request(
    const engine_common::OrderIntent& intent) const {
    NewOrderRequest request{};
    request.client_order_id = intent.client_order_id;
    request.symbol = symbol_resolver_(intent.symbol_id);
    request.side = intent.side == engine_common::Side::BUY ? 0 : 1;
    request.quantity = intent.quantity;
    request.price = intent.limit_price;
    return request;
}

engine_common::ExecutionEvent MockOrderAdapter::to_internal_execution(
    const ExecReport& report) {
    engine_common::ExecutionEvent event{};
    event.client_order_id = report.client_order_id;
    event.execution_id = report.exec_id;
    event.status = to_execution_status(report.status);
    event.reject_reason = report.reject_reason;
    event.last_quantity = report.fill_qty;
    event.cumulative_quantity = report.cumulative_qty;
    event.last_price = report.fill_price;
    return event;
}

void MockOrderAdapter::on_report(const ExecReport& report) {
    if (on_execution_) on_execution_(to_internal_execution(report));
}

}  // namespace te

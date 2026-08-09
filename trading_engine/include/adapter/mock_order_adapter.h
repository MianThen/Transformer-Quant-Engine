#pragma once

#include "adapter/order_adapter.h"
#include "oms/order_gateway.h"

namespace te {

class MockOrderAdapter final : public IOrderAdapter {
public:
    using SymbolResolver = std::function<std::string(engine_common::SymbolId)>;

    explicit MockOrderAdapter(SymbolResolver symbol_resolver,
                              GatewayConfig config = {});

    bool connect(const std::string& endpoint, uint16_t port) override;
    int64_t submit(const engine_common::OrderIntent& intent) override;
    bool cancel(int64_t client_order_id) override;
    size_t poll() override;
    bool ready() const override;
    void set_on_execution(OnExecutionEvent callback) override;

    OrderGateway& gateway() { return gateway_; }
    const OrderGateway& gateway() const { return gateway_; }

    NewOrderRequest to_gateway_request(const engine_common::OrderIntent& intent) const;
    static engine_common::ExecutionEvent to_internal_execution(const ExecReport& report);

private:
    void on_report(const ExecReport& report);

    SymbolResolver symbol_resolver_;
    OrderGateway gateway_;
    OnExecutionEvent on_execution_;
};

}  // namespace te

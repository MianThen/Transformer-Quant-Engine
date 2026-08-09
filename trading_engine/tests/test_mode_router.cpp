#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "adapter/order_adapter.h"
#include "engine_common/replay.h"
#include "runtime/mode_router.h"

namespace {

class CountingAdapter final : public te::IOrderAdapter {
public:
    bool connect(const std::string&, uint16_t) override {
        connected = true;
        return true;
    }

    int64_t submit(const engine_common::OrderIntent&) override {
        ++submit_calls;
        return submit_calls;
    }

    bool cancel(int64_t) override {
        ++cancel_calls;
        return connected;
    }

    size_t poll() override { return 0; }
    bool ready() const override { return connected; }
    void set_on_execution(OnExecutionEvent callback) override {
        callback_ = std::move(callback);
    }

    bool connected = false;
    int64_t submit_calls = 0;
    int64_t cancel_calls = 0;

private:
    OnExecutionEvent callback_;
};

engine_common::OrderIntent intent(int64_t quantity = 100) {
    engine_common::OrderIntent value;
    value.symbol_id = 7;
    value.quantity = quantity;
    value.type = engine_common::OrderType::LIMIT;
    value.time_in_force = engine_common::TimeInForce::DAY;
    value.limit_price = 1'000;
    value.timestamp = 42;
    return value;
}

}  // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() /
                      "qbt-mode-router-shadow.replay";
    std::filesystem::remove(path);

    CountingAdapter downstream;
    {
        te::ModeRouterConfig config;
        config.max_forward_quantity = 100;
        config.canary_intent_policy = [](const auto& order) {
            return order.symbol_id == 7 && order.side == engine_common::Side::BUY;
        };
        te::ModeRouter router(downstream, config);
        router.enable_audit(path.string());
        if (!router.connect("shadow", 1)) return 1;

        if (router.submit(intent()) != -1 || downstream.submit_calls != 0 ||
            router.blocked_submits() != 1) {
            return 1;
        }

        router.set_mode(te::TradingMode::INFRA_CANARY);
        router.set_readiness(true);
        if (router.arm() != true || router.submit(intent(101)) != -1 ||
            downstream.submit_calls != 0) {
            return 1;
        }
        if (router.submit(intent()) <= 0 || downstream.submit_calls != 1 ||
            router.forwarded_submits() != 1) {
            return 1;
        }

        engine_common::OrderIntent sell = intent();
        sell.side = engine_common::Side::SELL;
        if (router.submit(sell) != -1 || downstream.submit_calls != 1) return 1;

        if (router.submit_attempts() != 4 || !router.audit_healthy()) return 1;
    }

    size_t audit_records = 0;
    engine_common::ReplayReader reader(path.string());
    engine_common::ReplayRecord record;
    while (reader.next(record)) {
        if (record.header.type !=
            engine_common::ReplayRecordType::MODE_ROUTER_DECISION) {
            return 1;
        }
        if (record.payload.size() != sizeof(te::ModeRouterAuditRecord)) return 1;
        ++audit_records;
    }
    std::filesystem::remove(path);
    if (audit_records != 4) {
        return 1;
    }
    std::printf("test_mode_router: all checks passed\n");
    return 0;
}

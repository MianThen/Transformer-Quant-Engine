#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "adapter/order_adapter.h"
#include "engine_common/replay.h"

namespace te {

enum class TradingMode : uint8_t {
    SHADOW,
    PAPER,
    INFRA_CANARY,
    MODEL_CANARY,
    LIVE,
    REDUCE_ONLY,
    KILLED,
};

const char* to_string(TradingMode mode);
bool parse_trading_mode(std::string_view text, TradingMode& mode);

enum class ModeRouterOutcome : uint8_t {
    BLOCKED,
    FORWARDED,
};

enum class ModeRouterBlockReason : uint8_t {
    NONE,
    MODE_DISALLOWS_SUBMIT,
    NOT_READY,
    NOT_ARMED,
    AUDIT_UNAVAILABLE,
    INVALID_CANARY_INTENT,
    QUANTITY_LIMIT,
    REDUCE_ONLY_POLICY,
};

#pragma pack(push, 1)
struct ModeRouterAuditRecord {
    uint8_t mode = 0;
    uint8_t outcome = 0;
    uint8_t reason = 0;
    uint8_t reserved = 0;
    int64_t client_order_id = 0;
    uint32_t symbol_id = 0;
    uint8_t side = 0;
    uint8_t order_type = 0;
    uint8_t time_in_force = 0;
    uint8_t reserved_2 = 0;
    int64_t quantity = 0;
    int64_t limit_price = 0;
    int64_t timestamp = 0;
};
#pragma pack(pop)

struct ModeRouterConfig {
    TradingMode mode = TradingMode::SHADOW;
    bool readiness = false;
    bool armed = false;
    int64_t max_forward_quantity = 0;
    std::function<bool(const engine_common::OrderIntent&)> canary_intent_policy;
    bool require_audit_before_forward = true;
};

class ModeRouter final : public IOrderAdapter {
public:
    explicit ModeRouter(IOrderAdapter& downstream, ModeRouterConfig config = {});

    bool connect(const std::string& endpoint, uint16_t port) override;
    int64_t submit(const engine_common::OrderIntent& intent) override;
    bool cancel(int64_t client_order_id) override;
    size_t poll() override;
    bool ready() const override;
    void set_on_execution(OnExecutionEvent callback) override;

    void set_mode(TradingMode mode);
    void set_readiness(bool ready);
    bool arm();
    void disarm();
    void enable_audit(const std::string& path);

    TradingMode mode() const { return config_.mode; }
    bool is_armed() const { return config_.armed; }
    bool audit_healthy() const { return audit_healthy_; }
    uint64_t submit_attempts() const { return submit_attempts_; }
    uint64_t forwarded_submits() const { return forwarded_submits_; }
    uint64_t blocked_submits() const { return blocked_submits_; }

private:
    static ModeRouterAuditRecord audit_record(
        const engine_common::OrderIntent& intent,
        TradingMode mode,
        ModeRouterOutcome outcome,
        ModeRouterBlockReason reason);
    bool write_audit(const ModeRouterAuditRecord& record);
    bool requires_arming() const;
    bool can_forward(const engine_common::OrderIntent& intent,
                     ModeRouterBlockReason& reason) const;
    void notify_audit(const ModeRouterAuditRecord& record);

    IOrderAdapter& downstream_;
    ModeRouterConfig config_;
    std::unique_ptr<engine_common::ReplayWriter> audit_writer_;
    std::function<void(const ModeRouterAuditRecord&)> on_audit_;
    bool audit_healthy_ = true;
    uint64_t submit_attempts_ = 0;
    uint64_t forwarded_submits_ = 0;
    uint64_t blocked_submits_ = 0;
};

}  // namespace te

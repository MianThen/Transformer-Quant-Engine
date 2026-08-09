#include "runtime/mode_router.h"

#include <stdexcept>
#include <string_view>
#include <utility>

namespace te {

const char* to_string(TradingMode mode) {
    switch (mode) {
        case TradingMode::SHADOW: return "SHADOW";
        case TradingMode::PAPER: return "PAPER";
        case TradingMode::INFRA_CANARY: return "INFRA_CANARY";
        case TradingMode::MODEL_CANARY: return "MODEL_CANARY";
        case TradingMode::LIVE: return "LIVE";
        case TradingMode::REDUCE_ONLY: return "REDUCE_ONLY";
        case TradingMode::KILLED: return "KILLED";
    }
    return "UNKNOWN";
}

bool parse_trading_mode(std::string_view text, TradingMode& mode) {
    if (text == "shadow") mode = TradingMode::SHADOW;
    else if (text == "paper") mode = TradingMode::PAPER;
    else if (text == "infra-canary") mode = TradingMode::INFRA_CANARY;
    else if (text == "model-canary") mode = TradingMode::MODEL_CANARY;
    else if (text == "live") mode = TradingMode::LIVE;
    else if (text == "reduce-only") mode = TradingMode::REDUCE_ONLY;
    else if (text == "killed") mode = TradingMode::KILLED;
    else return false;
    return true;
}

ModeRouter::ModeRouter(IOrderAdapter& downstream, ModeRouterConfig config)
    : downstream_(downstream), config_(std::move(config)) {
    if (config_.max_forward_quantity < 0) {
        throw std::invalid_argument("max_forward_quantity cannot be negative");
    }
    if (config_.mode == TradingMode::SHADOW ||
        config_.mode == TradingMode::PAPER ||
        config_.mode == TradingMode::KILLED ||
        config_.mode == TradingMode::REDUCE_ONLY) {
        config_.armed = false;
    }
}

bool ModeRouter::connect(const std::string& endpoint, uint16_t port) {
    return downstream_.connect(endpoint, port);
}

int64_t ModeRouter::submit(const engine_common::OrderIntent& intent) {
    ++submit_attempts_;
    ModeRouterBlockReason reason = ModeRouterBlockReason::NONE;
    if (!can_forward(intent, reason)) {
        ++blocked_submits_;
        const auto record = audit_record(intent, config_.mode,
                                         ModeRouterOutcome::BLOCKED, reason);
        // Shadow must remain usable without a journal; production forwarding does not.
        if (!write_audit(record) && requires_arming()) audit_healthy_ = false;
        notify_audit(record);
        return -1;
    }

    const auto record = audit_record(intent, config_.mode,
                                     ModeRouterOutcome::FORWARDED,
                                     ModeRouterBlockReason::NONE);
    if (!write_audit(record)) {
        audit_healthy_ = false;
        ++blocked_submits_;
        const auto blocked = audit_record(
            intent, config_.mode, ModeRouterOutcome::BLOCKED,
            ModeRouterBlockReason::AUDIT_UNAVAILABLE);
        notify_audit(blocked);
        return -1;
    }
    notify_audit(record);
    const int64_t order_id = downstream_.submit(intent);
    if (order_id <= 0) return order_id;
    ++forwarded_submits_;
    return order_id;
}

bool ModeRouter::cancel(int64_t client_order_id) {
    return downstream_.cancel(client_order_id);
}

size_t ModeRouter::poll() { return downstream_.poll(); }

bool ModeRouter::ready() const {
    return downstream_.ready() && audit_healthy_;
}

void ModeRouter::set_on_execution(OnExecutionEvent callback) {
    downstream_.set_on_execution(std::move(callback));
}

void ModeRouter::set_mode(TradingMode mode) {
    config_.mode = mode;
    config_.armed = false;
}

void ModeRouter::set_readiness(bool ready) {
    config_.readiness = ready;
    if (!ready) config_.armed = false;
}

bool ModeRouter::arm() {
    if (!requires_arming() || !config_.readiness || !audit_healthy_) {
        config_.armed = false;
        return false;
    }
    config_.armed = true;
    return true;
}

void ModeRouter::disarm() { config_.armed = false; }

void ModeRouter::enable_audit(const std::string& path) {
    audit_writer_ = std::make_unique<engine_common::ReplayWriter>(path);
    audit_healthy_ = true;
}

ModeRouterAuditRecord ModeRouter::audit_record(
    const engine_common::OrderIntent& intent,
    TradingMode mode,
    ModeRouterOutcome outcome,
    ModeRouterBlockReason reason) {
    ModeRouterAuditRecord record{};
    record.mode = static_cast<uint8_t>(mode);
    record.outcome = static_cast<uint8_t>(outcome);
    record.reason = static_cast<uint8_t>(reason);
    record.client_order_id = intent.client_order_id;
    record.symbol_id = intent.symbol_id;
    record.side = static_cast<uint8_t>(intent.side);
    record.order_type = static_cast<uint8_t>(intent.type);
    record.time_in_force = static_cast<uint8_t>(intent.time_in_force);
    record.quantity = intent.quantity;
    record.limit_price = intent.limit_price;
    record.timestamp = intent.timestamp;
    return record;
}

bool ModeRouter::write_audit(const ModeRouterAuditRecord& record) {
    if (!audit_writer_) return !config_.require_audit_before_forward;
    try {
        audit_writer_->append(
            engine_common::ReplayRecordType::MODE_ROUTER_DECISION,
            record.timestamp,
            {reinterpret_cast<const std::byte*>(&record), sizeof(record)});
        return true;
    } catch (...) {
        return false;
    }
}

bool ModeRouter::requires_arming() const {
    return config_.mode == TradingMode::INFRA_CANARY ||
           config_.mode == TradingMode::MODEL_CANARY ||
           config_.mode == TradingMode::LIVE;
}

bool ModeRouter::can_forward(const engine_common::OrderIntent& intent,
                             ModeRouterBlockReason& reason) const {
    if (!requires_arming()) {
        if (config_.mode == TradingMode::REDUCE_ONLY) {
            reason = ModeRouterBlockReason::REDUCE_ONLY_POLICY;
        } else {
            reason = ModeRouterBlockReason::MODE_DISALLOWS_SUBMIT;
        }
        return false;
    }
    if (!config_.readiness) {
        reason = ModeRouterBlockReason::NOT_READY;
        return false;
    }
    if (!config_.armed) {
        reason = ModeRouterBlockReason::NOT_ARMED;
        return false;
    }
    if (!downstream_.ready()) {
        reason = ModeRouterBlockReason::NOT_READY;
        return false;
    }
    if (config_.max_forward_quantity > 0 &&
        intent.quantity > config_.max_forward_quantity) {
        reason = ModeRouterBlockReason::QUANTITY_LIMIT;
        return false;
    }
    if ((config_.mode == TradingMode::INFRA_CANARY ||
         config_.mode == TradingMode::MODEL_CANARY) &&
        (!config_.canary_intent_policy || !config_.canary_intent_policy(intent))) {
        reason = ModeRouterBlockReason::INVALID_CANARY_INTENT;
        return false;
    }
    return true;
}

void ModeRouter::notify_audit(const ModeRouterAuditRecord& record) {
    if (on_audit_) on_audit_(record);
}

}  // namespace te

#include "oms/order_state.h"

#include <algorithm>

namespace te {

namespace {

bool allowed_transition(OrderStatus from, OrderStatus to) {
    switch (from) {
        case OrderStatus::NEW:
            return to == OrderStatus::PENDING;
        case OrderStatus::PENDING:
            return to == OrderStatus::ACK || to == OrderStatus::REJECTED ||
                   to == OrderStatus::CANCEL_PENDING || to == OrderStatus::RECOVERING ||
                   to == OrderStatus::UNKNOWN;
        case OrderStatus::ACK:
        case OrderStatus::PARTIALLY_FILLED:
            return to == OrderStatus::PARTIALLY_FILLED || to == OrderStatus::FILLED ||
                   to == OrderStatus::CANCEL_PENDING || to == OrderStatus::REPLACE_PENDING ||
                   to == OrderStatus::CANCELED || to == OrderStatus::EXPIRED ||
                   to == OrderStatus::SUSPENDED || to == OrderStatus::RECOVERING ||
                   to == OrderStatus::UNKNOWN;
        case OrderStatus::CANCEL_PENDING:
            return to == OrderStatus::CANCELED || to == OrderStatus::PARTIALLY_FILLED ||
                   to == OrderStatus::FILLED || to == OrderStatus::RECOVERING ||
                   to == OrderStatus::UNKNOWN;
        case OrderStatus::REPLACE_PENDING:
            return to == OrderStatus::ACK || to == OrderStatus::PARTIALLY_FILLED ||
                   to == OrderStatus::FILLED || to == OrderStatus::REJECTED ||
                   to == OrderStatus::RECOVERING || to == OrderStatus::UNKNOWN;
        case OrderStatus::SUSPENDED:
            return to == OrderStatus::ACK || to == OrderStatus::CANCELED ||
                   to == OrderStatus::EXPIRED || to == OrderStatus::RECOVERING;
        case OrderStatus::RECOVERING:
        case OrderStatus::UNKNOWN:
            return to == OrderStatus::ACK || to == OrderStatus::PARTIALLY_FILLED ||
                   to == OrderStatus::FILLED || to == OrderStatus::CANCELED ||
                   to == OrderStatus::REJECTED || to == OrderStatus::EXPIRED ||
                   to == OrderStatus::SUSPENDED;
        case OrderStatus::FILLED:
        case OrderStatus::CANCELED:
        case OrderStatus::REJECTED:
        case OrderStatus::EXPIRED:
        case OrderStatus::DONE:
            return false;
    }
    return false;
}

}  // namespace

bool OrderState::transition(OrderStatus next) {
    if (next == status_) return true;
    if (is_terminal() || !allowed_transition(status_, next)) return false;
    status_ = next;
    if (status_ == OrderStatus::FILLED) {
        last_fill_qty_ = quantity_ - filled_qty_;
        filled_qty_ = quantity_;
    }
    return true;
}

bool OrderState::on_fill(int64_t fill_qty, int64_t exec_id) {
    if (fill_qty <= 0) return false;
    return on_cumulative_fill(std::min(quantity_, filled_qty_ + fill_qty), exec_id);
}

bool OrderState::on_cumulative_fill(int64_t cumulative_qty, int64_t exec_id) {
    if (quantity_ <= 0 || is_terminal() || cumulative_qty <= filled_qty_ ||
        cumulative_qty > quantity_) return false;
    if (status_ != OrderStatus::ACK && status_ != OrderStatus::PARTIALLY_FILLED &&
        status_ != OrderStatus::CANCEL_PENDING && status_ != OrderStatus::REPLACE_PENDING &&
        status_ != OrderStatus::RECOVERING && status_ != OrderStatus::UNKNOWN) return false;
    if (exec_id != 0) {
        for (size_t index = 0; index < seen_exec_count_; ++index) {
            if (seen_exec_ids_[index] == exec_id) return false;
        }
        seen_exec_ids_[seen_exec_next_] = exec_id;
        seen_exec_next_ = (seen_exec_next_ + 1) % kExecDedupWindow;
        seen_exec_count_ = std::min(seen_exec_count_ + 1, kExecDedupWindow);
    }
    last_fill_qty_ = cumulative_qty - filled_qty_;
    filled_qty_ = cumulative_qty;
    status_ = filled_qty_ == quantity_ ? OrderStatus::FILLED : OrderStatus::PARTIALLY_FILLED;
    return true;
}

}  // namespace te

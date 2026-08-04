// 订单状态机测试。

#include <cstdio>

#include "oms/order_state.h"

using namespace te;

static int g_failures = 0;
#define CHECK(cond)                                                   \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s @ %d\n", #cond, __LINE__); \
            ++g_failures;                                             \
        }                                                             \
    } while (0)

void test_happy_path() {
    OrderState s(/*client_order_id=*/1, /*quantity=*/100);
    CHECK(s.status() == OrderStatus::NEW);

    CHECK(s.transition(OrderStatus::PENDING));
    CHECK(!s.transition(OrderStatus::FILLED));
    CHECK(s.transition(OrderStatus::ACK));

    CHECK(s.on_fill(40, 1001));
    CHECK(s.status() == OrderStatus::PARTIALLY_FILLED);
    CHECK(s.filled() == 40);
    CHECK(s.leaves() == 60);

    CHECK(!s.on_fill(40, 1001));
    CHECK(s.filled() == 40);
    CHECK(s.on_fill(80, 1002));
    CHECK(s.status() == OrderStatus::FILLED);
    CHECK(s.leaves() == 0);
    CHECK(s.is_terminal());
}

void test_terminal_rejects_transition() {
    OrderState s(2, 100);
    s.transition(OrderStatus::PENDING);
    s.transition(OrderStatus::REJECTED);
    CHECK(s.is_terminal());
    // 终态后任何转移都应失败(幂等保护)
    CHECK(!s.transition(OrderStatus::ACK));
}

void test_fill_requires_ack() {
    OrderState s(3, 10);
    CHECK(!s.on_fill(1, 1));
    CHECK(s.filled() == 0);
}

void test_pending_cancel_and_cumulative_fill() {
    OrderState state(4, 100);
    CHECK(state.transition(OrderStatus::PENDING));
    CHECK(state.transition(OrderStatus::ACK));
    CHECK(state.transition(OrderStatus::CANCEL_PENDING));
    CHECK(state.on_cumulative_fill(25, 2001));
    CHECK(state.last_fill() == 25);
    CHECK(!state.on_cumulative_fill(25, 2002));
    CHECK(state.transition(OrderStatus::CANCELED));
}

int main() {
    test_happy_path();
    test_terminal_rejects_transition();
    test_fill_requires_ack();
    test_pending_cancel_and_cumulative_fill();

    if (g_failures == 0) {
        std::printf("test_order_state: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_order_state: %d failure(s)\n", g_failures);
    return 1;
}

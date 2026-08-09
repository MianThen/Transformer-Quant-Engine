// SPSC 环形缓冲区测试。这个类骨架已实现,故断言可直接生效。

#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

#include "net/spsc_ring_buffer.h"

using namespace te;

static int g_failures = 0;
#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::fprintf(stderr, "FAIL: %s @ %d\n", #cond, __LINE__);       \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

void test_basic_push_pop() {
    SpscRingBuffer<int> q(8);  // 实际可用容量 7(留一格区分空/满)
    CHECK(q.empty());

    CHECK(q.push(1));
    CHECK(q.push(2));
    CHECK(q.size() == 2);
    CHECK(!q.empty());

    int v = 0;
    CHECK(q.pop(v) && v == 1);
    CHECK(q.pop(v) && v == 2);
    CHECK(q.empty());
    CHECK(q.size() == 0);
    CHECK(!q.pop(v));  // 空队列 pop 失败
}

void test_full() {
    SpscRingBuffer<int> q(4);  // 可用容量 3
    CHECK(q.push(1));
    CHECK(q.push(2));
    CHECK(q.push(3));
    CHECK(!q.push(4));  // 满
}

void test_spsc_threaded() {
    // 生产者/消费者各一线程,验证无锁传递不丢不乱
    constexpr int N = 1'000'000;
    SpscRingBuffer<int> q(1024);

    std::thread producer([&] {
        for (int i = 0; i < N; ++i) {
            while (!q.push(i)) { /* 满则自旋 */ }
        }
    });

    int expected = 0;
    std::thread consumer([&] {
        int v = 0;
        while (expected < N) {
            if (q.pop(v)) {
                if (v != expected) ++g_failures;
                ++expected;
            }
        }
    });

    producer.join();
    consumer.join();
    CHECK(expected == N);
}

void test_invalid_capacity() {
    bool threw = false;
    try {
        SpscRingBuffer<int> q(3);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

void test_bulk_and_wraparound() {
    SpscRingBuffer<int> queue(8);
    const int first[] = {1, 2, 3, 4, 5, 6};
    CHECK(queue.push_bulk(first, 6) == 6);
    int output[6]{};
    CHECK(queue.pop_bulk(output, 4) == 4);
    const int second[] = {7, 8, 9, 10};
    CHECK(queue.push_bulk(second, 4) == 4);
    CHECK(queue.pop_bulk(output, 6) == 6);
    CHECK(queue.empty());
}

int main() {
    test_basic_push_pop();
    test_full();
    test_spsc_threaded();
    test_invalid_capacity();
    test_bulk_and_wraparound();

    if (g_failures == 0) {
        std::printf("test_ring_buffer: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_ring_buffer: %d failure(s)\n", g_failures);
    return 1;
}
